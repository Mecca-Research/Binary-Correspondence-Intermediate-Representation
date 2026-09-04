"""Verified contracts for adaptive transformer research architectures.

This module is the dependency-free semantic rail for a deliberately bounded set of
architecture ideas: tied-depth LoopDeepNorm, symmetric variable widths, reference-plus-
sliding attention, ExoFormer anchors, and coarse-to-fine multi-patch processing.  It
contains no tensor framework and makes no latency claim.  Hosted PyTorch references live
under :mod:`bcir.hosted.training.adaptive_transformer` and must agree with these shapes.

The architecture features are independent research contracts, not automatic rewrites of
BCIR's stable Llama/BCIRQ8 decoder format.  Unsupported combinations fail at construction
rather than silently changing semantics.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from ..abi.streampack_abi import encode
from ..gem.streampack import StreamPack, hydrate_pipelined
from ..model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from ..verify import verify_all, verify_smart_lowering
from .cost import TargetProfile, Theta
from .provenance import hash_module
from .realize import RealizationResult, optimize

_MAX_U63 = (1 << 63) - 1
_MAX_LAYERS = 256
_MAX_WIDTH = 65536
_MAX_CONTEXT = 1 << 20
_MAX_MATERIALIZED_MASK_ELEMENTS = 1 << 20
_Q20 = 1 << 20


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _integer(value, field: str, *, minimum: int = 0, maximum: int = _MAX_U63) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _checked_add(*values: int) -> int:
    total = 0
    for value in values:
        _integer(value, "checked-add operand")
        total += value
        if total > _MAX_U63:
            raise ValueError("adaptive-transformer integer addition exceeds u63")
    return total


def _checked_mul(*values: int) -> int:
    product = 1
    for value in values:
        _integer(value, "checked-multiply operand")
        if value and product > _MAX_U63 // value:
            raise ValueError("adaptive-transformer integer multiplication exceeds u63")
        product *= value
    return product


def _int_tuple(value, field: str, *, minimum: int = 1,
               maximum: int = _MAX_U63) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field} must be a nonempty tuple")
    for index, item in enumerate(value):
        _integer(item, f"{field}[{index}]", minimum=minimum, maximum=maximum)
    return value


@dataclass(frozen=True)
class WidthScheduleSpec:
    """Per-physical-layer widths with parameter-free prefix resize semantics.

    Every block reads and writes a prefix of one fixed-width residual stream.  Inactive
    coordinates bypass narrower blocks and are therefore restored when a later block widens.
    Coordinates wider than every previous active block remain zero.  This is an oracle
    contract only: a production target still needs measured fused kernels for each shape.
    """

    widths: tuple[int, ...]
    heads: tuple[int, ...]
    ff_widths: tuple[int, ...]
    transition: str = "carry_forward"

    def __post_init__(self) -> None:
        _int_tuple(self.widths, "widths", maximum=_MAX_WIDTH)
        _int_tuple(self.heads, "heads", maximum=_MAX_WIDTH)
        _int_tuple(self.ff_widths, "ff_widths", maximum=_MAX_WIDTH * 16)
        if not len(self.widths) == len(self.heads) == len(self.ff_widths):
            raise ValueError("widths, heads, and ff_widths must have equal lengths")
        if len(self.widths) > _MAX_LAYERS:
            raise ValueError(f"width schedule exceeds {_MAX_LAYERS} physical layers")
        if self.transition != "carry_forward":
            raise ValueError("only transition='carry_forward' is admitted")
        for index, (width, heads, ff_width) in enumerate(
                zip(self.widths, self.heads, self.ff_widths)):
            if width % heads:
                raise ValueError(f"width {index} must be divisible by its head count")
            if width // heads % 2:
                raise ValueError(f"width {index} needs an even RoPE head dimension")
            if ff_width < width:
                raise ValueError(f"ff_widths[{index}] must be at least its model width")

    @classmethod
    def symmetric_u(cls, *, layers: int, outer_width: int, inner_width: int,
                    head_dim: int, ff_multiplier: int = 4,
                    quantum: int = 8) -> "WidthScheduleSpec":
        """Build a deterministic wide-edge/narrow-middle schedule.

        Widths follow a symmetric geometric interpolation and are rounded to ``quantum``.
        Fixed head dimension avoids inventing cross-layer head geometry.  This convenience
        constructor does not claim parameter matching; publication schedules should provide
        already-solved explicit widths to the dataclass.
        """
        _integer(layers, "layers", minimum=1, maximum=_MAX_LAYERS)
        _integer(outer_width, "outer_width", minimum=1, maximum=_MAX_WIDTH)
        _integer(inner_width, "inner_width", minimum=1, maximum=outer_width)
        _integer(head_dim, "head_dim", minimum=2, maximum=_MAX_WIDTH)
        _integer(ff_multiplier, "ff_multiplier", minimum=1, maximum=16)
        _integer(quantum, "quantum", minimum=1, maximum=_MAX_WIDTH)
        alignment = math.lcm(quantum, head_dim)
        if head_dim % 2 or outer_width % alignment or inner_width % alignment:
            raise ValueError("outer/inner widths must align to quantum and even head_dim")
        maximum_depth = (layers - 1) // 2
        if maximum_depth == 0 and inner_width != outer_width:
            raise ValueError("fewer than three layers cannot have a distinct inner width")
        widths = []
        for layer in range(layers):
            if maximum_depth == 0:
                width = outer_width
            else:
                depth = min(layer, layers - 1 - layer)
                ratio = (inner_width / outer_width) ** (depth / maximum_depth)
                raw = outer_width * ratio
                width = max(inner_width, int(raw / alignment + 0.5) * alignment)
            widths.append(width)
        widths[0] = outer_width
        widths[-1] = outer_width
        if layers % 2:
            widths[layers // 2] = inner_width
        heads = tuple(width // head_dim for width in widths)
        ff_widths = tuple(width * ff_multiplier for width in widths)
        return cls(tuple(widths), heads, ff_widths)

    @property
    def physical_layers(self) -> int:
        return len(self.widths)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceWindowSpec:
    """Reference-plus-sliding causal attention and bounded continuation cache."""

    prefix_tokens: int
    window_tokens: int

    def __post_init__(self) -> None:
        _integer(self.prefix_tokens, "prefix_tokens", minimum=1, maximum=_MAX_CONTEXT)
        _integer(self.window_tokens, "window_tokens", minimum=1, maximum=_MAX_CONTEXT)

    def visible_keys(self, query: int, sequence_length: int) -> tuple[int, ...]:
        _integer(sequence_length, "sequence_length", minimum=1, maximum=_MAX_CONTEXT)
        _integer(query, "query", maximum=sequence_length - 1)
        prefix = min(self.prefix_tokens, sequence_length)
        if query < prefix:
            return tuple(range(query + 1))
        recent = max(prefix, query - self.window_tokens + 1)
        return tuple(range(prefix)) + tuple(range(recent, query + 1))

    def additive_mask(self, sequence_length: int, *, masked: float = -1.0e30) -> tuple[float, ...]:
        _integer(sequence_length, "sequence_length", minimum=1, maximum=_MAX_CONTEXT)
        if isinstance(masked, bool) or not isinstance(masked, (int, float)) \
                or not math.isfinite(masked) or masked >= 0:
            raise ValueError("masked value must be a finite negative number")
        elements = _checked_mul(sequence_length, sequence_length)
        if elements > _MAX_MATERIALIZED_MASK_ELEMENTS:
            raise ValueError("reference attention mask exceeds the bounded materialization limit")
        result = [float(masked)] * elements
        for query in range(sequence_length):
            for key in self.visible_keys(query, sequence_length):
                result[query * sequence_length + key] = 0.0
        return tuple(result)

    def visible_pairs(self, sequence_length: int) -> int:
        _integer(sequence_length, "sequence_length", minimum=1, maximum=_MAX_CONTEXT)
        prefix = min(self.prefix_tokens, sequence_length)
        continuation = sequence_length - prefix
        prefix_pairs = _checked_mul(prefix, prefix + 1) // 2
        if continuation <= self.window_tokens:
            local_pairs = _checked_mul(continuation, continuation + 1) // 2
        else:
            local_pairs = _checked_add(
                _checked_mul(self.window_tokens, self.window_tokens + 1) // 2,
                _checked_mul(continuation - self.window_tokens, self.window_tokens))
        return _checked_add(prefix_pairs, _checked_mul(prefix, continuation), local_pairs)

    def cache_tokens(self, continuation_tokens: int) -> int:
        _integer(continuation_tokens, "continuation_tokens", maximum=_MAX_CONTEXT)
        return _checked_add(self.prefix_tokens, min(continuation_tokens, self.window_tokens))


@dataclass(frozen=True)
class ExogenousAnchorSpec:
    """ExoFormer projection-mixing contract from arXiv:2601.08131v4.

    ``source_depth=0`` anchors on embeddings; ``1`` anchors on the first layer output.
    H2 is deliberately not admitted because the paper reports no additional gain and it
    complicates first-use ordering.  All four anchor sources are RMS-normalized; mixed Q/K
    receive QK normalization and RoPE in the hosted reference.
    """

    mode: str = "static"
    granularity: str = "elementwise"
    source_depth: int = 0
    dynamic_hidden: int = 16

    def __post_init__(self) -> None:
        if self.mode not in ("static", "dynamic"):
            raise ValueError("anchor mode must be static or dynamic")
        if self.granularity not in ("scalar", "headwise", "elementwise"):
            raise ValueError("anchor granularity must be scalar, headwise, or elementwise")
        _integer(self.source_depth, "source_depth", maximum=1)
        _integer(self.dynamic_hidden, "dynamic_hidden", minimum=1, maximum=256)
        if self.mode == "static" and self.dynamic_hidden != 16:
            raise ValueError("static anchors retain the canonical dynamic_hidden=16 placeholder")

    @property
    def initial_base_coefficient(self) -> float:
        return 0.5 if self.mode == "static" else 1.0

    def coefficient_channels(self, width: int, heads: int) -> int:
        if self.granularity == "scalar":
            return 1
        if self.granularity == "headwise":
            return heads
        return width


@dataclass(frozen=True)
class AdaptiveLanguageSpec:
    """Content-addressed language-model architecture contract."""

    vocab_size: int
    context_length: int
    schedule: WidthScheduleSpec
    repeats: int = 1
    tied_embeddings: bool = True
    residual_mode: str = "pre_norm"
    gated_attention: bool = False
    reference_window: ReferenceWindowSpec | None = None
    anchor: ExogenousAnchorSpec | None = None

    def __post_init__(self) -> None:
        _integer(self.vocab_size, "vocab_size", minimum=2, maximum=1 << 24)
        _integer(self.context_length, "context_length", minimum=1, maximum=_MAX_CONTEXT)
        if not isinstance(self.schedule, WidthScheduleSpec):
            raise ValueError("schedule must be WidthScheduleSpec")
        _integer(self.repeats, "repeats", minimum=1, maximum=64)
        if type(self.tied_embeddings) is not bool:
            raise ValueError("tied_embeddings must be bool")
        if type(self.gated_attention) is not bool:
            raise ValueError("gated_attention must be bool")
        if self.residual_mode not in ("pre_norm", "loop_deepnorm"):
            raise ValueError("residual_mode must be pre_norm or loop_deepnorm")
        if self.tied_embeddings and self.schedule.widths[0] != self.schedule.widths[-1]:
            raise ValueError("tied embeddings require equal first and final widths")
        if self.reference_window is not None:
            if not isinstance(self.reference_window, ReferenceWindowSpec):
                raise ValueError("reference_window has the wrong type")
            if self.reference_window.prefix_tokens >= self.context_length:
                raise ValueError("reference prefix must leave at least one continuation token")
        if self.anchor is not None:
            if not isinstance(self.anchor, ExogenousAnchorSpec):
                raise ValueError("anchor has the wrong type")
            if self.residual_mode != "pre_norm":
                raise ValueError("ExoFormer is admitted only on the paper's pre-norm rail")
            if not self.gated_attention:
                raise ValueError("ExoFormer requires the paper's gated-attention baseline")
            if len(set(self.schedule.widths)) != 1 or len(set(self.schedule.heads)) != 1:
                raise ValueError("the first ExoFormer rail requires a fixed-width MHA schedule")
            if self.anchor.source_depth >= self.effective_layers:
                raise ValueError("anchor source_depth must precede a later layer")
        _checked_mul(self.effective_layers, self.context_length,
                     max(self.schedule.widths), self.vocab_size)

    @property
    def effective_layers(self) -> int:
        return _checked_mul(self.schedule.physical_layers, self.repeats)

    @property
    def loop_alpha(self) -> float:
        return math.sqrt(2.0 * self.effective_layers) \
            if self.residual_mode == "loop_deepnorm" else 1.0

    @property
    def loop_beta(self) -> float:
        return 1.0 / math.sqrt(8.0 * self.effective_layers) \
            if self.residual_mode == "loop_deepnorm" else 1.0

    def to_dict(self) -> dict:
        return {
            "schema": "bcir.adaptive_language.v1",
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "schedule": self.schedule.to_dict(),
            "repeats": self.repeats,
            "tied_embeddings": self.tied_embeddings,
            "residual_mode": self.residual_mode,
            "gated_attention": self.gated_attention,
            "reference_window": None if self.reference_window is None
            else asdict(self.reference_window),
            "anchor": None if self.anchor is None else asdict(self.anchor),
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AdaptiveCostReport:
    spec_sha256: str
    parameter_elements: int
    physical_layers: int
    effective_layers: int
    prefill_tokens: int
    prefill_attention_pairs: int
    prefill_flops: int
    decode_flops_per_token: int
    kv_cache_bytes: int
    width_transition_elements: int
    anchor_parameter_elements: int
    anchor_prefill_flops: int
    output_projection_flops: int
    uniform_baseline_prefill_flops: int
    prefill_flop_delta: int
    classification: str = "analytic-lower-bound"

    def __post_init__(self) -> None:
        if not isinstance(self.spec_sha256, str) or len(self.spec_sha256) != 64 \
                or any(character not in "0123456789abcdef" for character in self.spec_sha256):
            raise ValueError("cost report spec_sha256 must be lowercase SHA-256")
        for field in ("parameter_elements", "physical_layers", "effective_layers",
                      "prefill_tokens", "prefill_attention_pairs", "prefill_flops",
                      "decode_flops_per_token", "kv_cache_bytes",
                      "width_transition_elements", "anchor_parameter_elements",
                      "anchor_prefill_flops", "output_projection_flops",
                      "uniform_baseline_prefill_flops"):
            _integer(getattr(self, field), f"cost report {field}")
        if type(self.prefill_flop_delta) is not int:
            raise ValueError("prefill_flop_delta must be an integer")
        if self.classification != "analytic-lower-bound":
            raise ValueError("adaptive cost reports cannot claim measured latency")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def fixed_residual_view(values, rows: int, residual_width: int, active_width: int):
    """Read one block's active prefix from a fixed global residual stream."""
    _integer(rows, "rows", minimum=1, maximum=_MAX_CONTEXT)
    _integer(residual_width, "residual_width", minimum=1, maximum=_MAX_WIDTH)
    _integer(active_width, "active_width", minimum=1, maximum=residual_width)
    if not isinstance(values, (list, tuple)) or len(values) != rows * residual_width:
        raise ValueError("fixed residual length does not match rows*residual_width")
    result = [0.0] * _checked_mul(rows, active_width)
    for row in range(rows):
        source = row * residual_width
        destination = row * active_width
        for channel in range(active_width):
            result[destination + channel] = values[source + channel]
    return result


def fixed_residual_update(residual, active, rows: int,
                          residual_width: int, active_width: int):
    """Replace the active prefix while preserving every inactive coordinate.

    The returned list owns its storage.  A narrower block therefore cannot mutate carried
    coordinates, which is the paper's fixed-residual/carry-forward rule rather than the
    weaker zero-padding ablation.
    """
    _integer(rows, "rows", minimum=1, maximum=_MAX_CONTEXT)
    _integer(residual_width, "residual_width", minimum=1, maximum=_MAX_WIDTH)
    _integer(active_width, "active_width", minimum=1, maximum=residual_width)
    if not isinstance(residual, (list, tuple)) or len(residual) != rows * residual_width:
        raise ValueError("fixed residual length does not match rows*residual_width")
    if not isinstance(active, (list, tuple)) or len(active) != rows * active_width:
        raise ValueError("active block output length does not match rows*active_width")
    result = list(residual)
    for row in range(rows):
        source = row * active_width
        destination = row * residual_width
        for channel in range(active_width):
            result[destination + channel] = active[source + channel]
    return result


def _attention_pairs(spec: AdaptiveLanguageSpec, length: int) -> int:
    if spec.reference_window is None:
        return _checked_mul(length, length + 1) // 2
    return spec.reference_window.visible_pairs(length)


def _cache_tokens(spec: AdaptiveLanguageSpec, length: int) -> int:
    if spec.reference_window is None:
        return length
    prefix = min(length, spec.reference_window.prefix_tokens)
    continuation = max(0, length - prefix)
    return _checked_add(prefix, min(continuation, spec.reference_window.window_tokens))


def _layer_matmul_flops(width: int, ff_width: int, tokens: int,
                        gated_attention: bool) -> int:
    # Q, K, V, output, optional gate, plus the three SwiGLU matrices.
    projections = 5 if gated_attention else 4
    return _checked_mul(2, tokens, _checked_add(_checked_mul(projections, width, width),
                                                _checked_mul(3, width, ff_width)))


def _layer_attention_flops(width: int, pairs: int) -> int:
    # Q@K^T and probabilities@V, each one multiply-add per width and visible pair.
    return _checked_mul(4, width, pairs)


def _anchor_costs(spec: AdaptiveLanguageSpec, tokens: int) -> tuple[int, int]:
    if spec.anchor is None:
        return 0, 0
    width, heads = spec.schedule.widths[0], spec.schedule.heads[0]
    channels = spec.anchor.coefficient_channels(width, heads)
    params = _checked_add(_checked_mul(4, width, width), _checked_mul(4, width),
                          _checked_mul(spec.schedule.physical_layers, 8, channels))
    if spec.anchor.mode == "dynamic":
        per_layer = _checked_add(_checked_mul(width, spec.anchor.dynamic_hidden),
                                 _checked_mul(spec.anchor.dynamic_hidden, 8), 8)
        params = _checked_add(params,
                              _checked_mul(spec.schedule.physical_layers, per_layer))
    projection = _checked_mul(8, width, width, tokens)
    if spec.anchor.mode == "dynamic":
        per_token_layer = _checked_add(
            _checked_mul(2, width, spec.anchor.dynamic_hidden),
            _checked_mul(2, spec.anchor.dynamic_hidden, 8),
            _checked_mul(12, width))
    else:
        per_token_layer = _checked_mul(12, width)
    mixing = _checked_mul(spec.effective_layers - spec.anchor.source_depth,
                          tokens, per_token_layer)
    return params, _checked_add(projection, mixing)


def assess_adaptive_language(spec: AdaptiveLanguageSpec, *, prefill_tokens: int,
                             decode_context: int | None = None,
                             bytes_per_element: int = 2) -> AdaptiveCostReport:
    """Account exact architecture sizes and analytic operation counts.

    Parameter sizes are exact for the hosted architecture.  FLOPs count dominant
    matmuls and attention products and are therefore a lower bound, not throughput;
    dispatch, fusion, occupancy, launch cost, and heterogeneous-width kernels need measured
    evidence before promotion.
    """
    if not isinstance(spec, AdaptiveLanguageSpec):
        raise ValueError("spec must be AdaptiveLanguageSpec")
    _integer(prefill_tokens, "prefill_tokens", minimum=1, maximum=spec.context_length)
    if decode_context is None:
        decode_context = spec.context_length
    _integer(decode_context, "decode_context", minimum=1, maximum=spec.context_length)
    _integer(bytes_per_element, "bytes_per_element", minimum=1, maximum=16)

    schedule = spec.schedule
    parameters = _checked_mul(spec.vocab_size, schedule.widths[0])
    projections = 5 if spec.gated_attention else 4
    norm_parameters = 6 if spec.residual_mode == "loop_deepnorm" else 4
    for width, ff_width in zip(schedule.widths, schedule.ff_widths):
        parameters = _checked_add(parameters,
                                  _checked_mul(projections, width, width),
                                  _checked_mul(3, width, ff_width),
                                  _checked_mul(norm_parameters, width))
    parameters = _checked_add(parameters, schedule.widths[-1])
    if not spec.tied_embeddings:
        parameters = _checked_add(parameters,
                                  _checked_mul(spec.vocab_size, schedule.widths[-1]))

    anchor_parameters, anchor_prefill = _anchor_costs(spec, prefill_tokens)
    parameters = _checked_add(parameters, anchor_parameters)
    output_projection = _checked_mul(
        2, prefill_tokens, schedule.widths[-1], spec.vocab_size)
    pairs = _attention_pairs(spec, prefill_tokens)
    prefill = _checked_add(anchor_prefill, output_projection)
    transitions = 0
    previous_width = schedule.widths[0]
    for _repeat in range(spec.repeats):
        for width, ff_width in zip(schedule.widths, schedule.ff_widths):
            if width != previous_width:
                transitions = _checked_add(
                    transitions, _checked_mul(prefill_tokens, max(width, previous_width)))
            prefill = _checked_add(prefill,
                                   _layer_matmul_flops(width, ff_width, prefill_tokens,
                                                       spec.gated_attention),
                                   _layer_attention_flops(width, pairs))
            previous_width = width

    visible_decode = len(spec.reference_window.visible_keys(decode_context - 1, decode_context)) \
        if spec.reference_window is not None else decode_context
    decode = _checked_mul(2, schedule.widths[-1], spec.vocab_size)
    for width, ff_width in zip(schedule.widths, schedule.ff_widths):
        decode = _checked_add(decode,
                              _checked_mul(spec.repeats,
                                           _layer_matmul_flops(width, ff_width, 1,
                                                               spec.gated_attention)),
                              _checked_mul(spec.repeats,
                                           _layer_attention_flops(width, visible_decode)))
    if spec.anchor is not None:
        width = schedule.widths[0]
        decode = _checked_add(decode, _checked_mul(8, width, width))
        if spec.anchor.mode == "dynamic":
            mix = _checked_add(_checked_mul(2, width, spec.anchor.dynamic_hidden),
                               _checked_mul(2, spec.anchor.dynamic_hidden, 8),
                               _checked_mul(12, width))
        else:
            mix = _checked_mul(12, width)
        decode = _checked_add(decode,
                              _checked_mul(spec.effective_layers - spec.anchor.source_depth, mix))

    cache_tokens = _cache_tokens(spec, decode_context)
    kv = _checked_mul(2, bytes_per_element, cache_tokens, spec.repeats,
                      sum(schedule.widths))

    baseline_width = max(schedule.widths)
    baseline_ff = max(schedule.ff_widths)
    baseline = _checked_mul(spec.effective_layers,
                            _checked_add(_layer_matmul_flops(
                                baseline_width, baseline_ff, prefill_tokens,
                                spec.gated_attention),
                                _layer_attention_flops(baseline_width, pairs)))
    if spec.anchor is not None:
        # Compare the same anchor policy at the uniform outer width.
        baseline = _checked_add(baseline, anchor_prefill)
    baseline = _checked_add(baseline, output_projection)
    return AdaptiveCostReport(
        spec.digest, parameters, schedule.physical_layers, spec.effective_layers,
        prefill_tokens, pairs, prefill, decode, kv, transitions,
        anchor_parameters, anchor_prefill, output_projection,
        baseline, prefill - baseline)


@dataclass(frozen=True)
class MultiPatchSpec:
    """Coarse-to-fine image-token schedule inspired by MPDiT."""

    image_size: int
    input_channels: int
    hidden_width: int
    heads: int
    coarse_patch: int
    fine_patch: int
    coarse_layers: int
    fine_layers: int
    condition_tokens: int = 0

    def __post_init__(self) -> None:
        for field, maximum in (("image_size", 16384), ("input_channels", 4096),
                               ("hidden_width", _MAX_WIDTH), ("heads", _MAX_WIDTH),
                               ("coarse_patch", 1024), ("fine_patch", 1024),
                               ("coarse_layers", _MAX_LAYERS), ("fine_layers", _MAX_LAYERS)):
            _integer(getattr(self, field), field, minimum=1, maximum=maximum)
        _integer(self.condition_tokens, "condition_tokens", maximum=_MAX_CONTEXT)
        if self.hidden_width % self.heads or self.hidden_width // self.heads % 2:
            raise ValueError("multi-patch hidden width needs integral heads with even dimensions")
        if self.hidden_width % 4:
            raise ValueError("multi-patch hidden width must be divisible by four for 2D positions")
        if self.image_size % self.coarse_patch or self.image_size % self.fine_patch:
            raise ValueError("image size must be divisible by both patch sizes")
        if self.coarse_patch <= self.fine_patch \
                or self.coarse_patch % self.fine_patch:
            raise ValueError("coarse_patch must be an integer multiple larger than fine_patch")
        if self.coarse_patch // self.fine_patch > 8:
            raise ValueError("multi-patch upsample factor is bounded to 8")
        _integer(self.coarse_tokens, "coarse_tokens", minimum=1, maximum=_MAX_CONTEXT)
        _integer(self.fine_tokens, "fine_tokens", minimum=1, maximum=_MAX_CONTEXT)
        _checked_mul(self.fine_tokens, self.hidden_width, self.total_layers)

    @property
    def upsample_factor(self) -> int:
        return self.coarse_patch // self.fine_patch

    @property
    def coarse_spatial_tokens(self) -> int:
        return (self.image_size // self.coarse_patch) ** 2

    @property
    def fine_spatial_tokens(self) -> int:
        return (self.image_size // self.fine_patch) ** 2

    @property
    def coarse_tokens(self) -> int:
        return self.condition_tokens + self.coarse_spatial_tokens

    @property
    def fine_tokens(self) -> int:
        return self.condition_tokens + self.fine_spatial_tokens

    @property
    def total_layers(self) -> int:
        return self.coarse_layers + self.fine_layers

    def to_json(self) -> str:
        return _canonical({"schema": "bcir.multi_patch.v1", **asdict(self)})

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MultiPatchCostReport:
    spec_sha256: str
    parameter_elements: int
    coarse_tokens: int
    fine_tokens: int
    attention_pairs: int
    baseline_attention_pairs: int
    estimated_flops: int
    baseline_flops: int
    flop_reduction_q20: int
    classification: str = "analytic-lower-bound"

    def __post_init__(self) -> None:
        if not isinstance(self.spec_sha256, str) or len(self.spec_sha256) != 64 \
                or any(character not in "0123456789abcdef" for character in self.spec_sha256):
            raise ValueError("multi-patch report spec_sha256 must be lowercase SHA-256")
        for field in ("parameter_elements", "coarse_tokens", "fine_tokens",
                      "attention_pairs", "baseline_attention_pairs",
                      "estimated_flops", "baseline_flops", "flop_reduction_q20"):
            _integer(getattr(self, field), f"multi-patch report {field}")
        if self.flop_reduction_q20 > _Q20:
            raise ValueError("multi-patch flop reduction must be in Q20 [0, 1]")
        if self.classification != "analytic-lower-bound":
            raise ValueError("multi-patch reports cannot claim measured latency")

    def to_json(self) -> str:
        return _canonical(asdict(self))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def assess_multi_patch(spec: MultiPatchSpec) -> MultiPatchCostReport:
    if not isinstance(spec, MultiPatchSpec):
        raise ValueError("spec must be MultiPatchSpec")
    width = spec.hidden_width
    ratio2 = spec.upsample_factor ** 2
    # MultiheadAttention + 4x GELU MLP + two affine LayerNorms, including biases.
    block_parameters = _checked_add(_checked_mul(12, width, width),
                                    _checked_mul(13, width))
    coarse_embed = _checked_add(
        _checked_mul(spec.coarse_patch, spec.coarse_patch,
                     spec.input_channels, width), width)
    fine_embed = _checked_add(
        _checked_mul(spec.fine_patch, spec.fine_patch,
                     spec.input_channels, width), width)
    upsample = _checked_add(
        _checked_mul(width, width, ratio2), _checked_mul(width, ratio2),
        _checked_mul(2, width), _checked_mul(width, width), width)
    output_width = _checked_mul(spec.fine_patch, spec.fine_patch, spec.input_channels)
    parameters = _checked_add(
        coarse_embed, fine_embed,
        _checked_mul(spec.condition_tokens, width),
        _checked_mul(spec.total_layers, block_parameters),
        upsample, _checked_mul(width, output_width), output_width)
    pairs = _checked_add(_checked_mul(spec.coarse_layers, spec.coarse_tokens,
                                     spec.coarse_tokens),
                         _checked_mul(spec.fine_layers, spec.fine_tokens,
                                     spec.fine_tokens))
    baseline_pairs = _checked_mul(spec.total_layers, spec.fine_tokens, spec.fine_tokens)
    block_matmul_weights = _checked_mul(12, width, width)
    projection = _checked_mul(2, block_matmul_weights,
                              _checked_add(_checked_mul(spec.coarse_layers, spec.coarse_tokens),
                                           _checked_mul(spec.fine_layers, spec.fine_tokens)))
    attention = _checked_mul(4, width, pairs)
    flops = _checked_add(projection, attention)
    baseline_projection = _checked_mul(2, block_matmul_weights,
                                       spec.total_layers, spec.fine_tokens)
    baseline = _checked_add(baseline_projection,
                            _checked_mul(4, width, baseline_pairs))
    reduction = max(0, baseline - flops) * _Q20 // baseline
    return MultiPatchCostReport(spec.digest, parameters, spec.coarse_tokens,
                                spec.fine_tokens, pairs, baseline_pairs, flops,
                                baseline, reduction)


@dataclass(frozen=True)
class LoweredAdaptivePlan:
    report: AdaptiveCostReport
    module: Module
    realization: RealizationResult
    streampack: StreamPack
    streampack_data: bytes
    artifact_sha256: str


def lower_adaptive_language(spec: AdaptiveLanguageSpec, *, sequence_length: int,
                            batch: int = 1,
                            target: TargetProfile | None = None) -> LoweredAdaptivePlan:
    """Lower architecture work into verified claims and a provenance-carrying StreamPack."""
    if not isinstance(spec, AdaptiveLanguageSpec):
        raise ValueError("spec must be AdaptiveLanguageSpec")
    _integer(sequence_length, "sequence_length", minimum=1, maximum=spec.context_length)
    _integer(batch, "batch", minimum=1, maximum=1 << 16)
    report = assess_adaptive_language(spec, prefill_tokens=sequence_length,
                                      decode_context=sequence_length)
    rows = _checked_mul(batch, sequence_length)
    maximum_width = max(spec.schedule.widths)
    maximum_ff_width = max(spec.schedule.ff_widths)
    scratch_width = max(
        _checked_mul(5 if spec.gated_attention else 4, maximum_width),
        _checked_mul(3, maximum_ff_width))
    module = Module(name=f"adaptive-transformer:{spec.digest[:16]}")
    module.add_resource(Resource(1, Domain.RAM, 4, (rows,), name="token_ids"))
    module.add_resource(Resource(2, Domain.RAM, 4, (rows, maximum_width), name="hidden"))
    module.add_resource(Resource(3, Domain.RAM, 4, (rows, scratch_width), name="scratch"))
    module.add_resource(Resource(4, Domain.RAM, 4, (report.parameter_elements,), name="weights"))
    module.add_resource(Resource(
        5, Domain.RAM, 1, (max(1, report.kv_cache_bytes),), name="kv_cache"))
    if spec.anchor is not None:
        module.add_resource(Resource(6, Domain.RAM, 4,
                                     (4, rows, spec.schedule.widths[0]), name="anchor"))
    module.add_resource(Resource(7, Domain.RAM, 4,
                                 (rows, spec.vocab_size), name="logits"))

    phase_id = 0
    claim_id = 0

    def emit(opcode: Opcode, lane: Lane, stride: StrideClass, *, count: int,
             reads: tuple[int, ...], writes: tuple[int, ...], op: str,
             cost_class: str) -> None:
        nonlocal phase_id, claim_id
        phase_id += 1
        claim_id += 1
        claim = Claim(claim_id, opcode, lane, stride, count=max(1, count),
                      rd=reads, wr=writes, hazard="barriered", bounds="assumed_safe",
                      domain=Domain.RAM, op=op, cost_class=cost_class)
        module.add_phase(Phase(phase_id, (phase_id - 1,) if phase_id > 1 else (), [claim]))

    emit(Opcode.LOAD, Lane.GGG, StrideClass.RANDOM,
         count=_checked_mul(rows, spec.schedule.widths[0]),
         reads=(1, 4), writes=(2,), op="model.embedding.lookup",
         cost_class="bandwidth")

    def emit_anchor() -> None:
        width = spec.schedule.widths[0]
        emit(Opcode.T_MACC, Lane.T, StrideClass.TILE,
             count=_checked_mul(rows, 4, width, width), reads=(2, 4), writes=(6,),
             op="model.exogenous_anchor.project_once", cost_class="compute")

    if spec.anchor is not None and spec.anchor.source_depth == 0:
        emit_anchor()

    previous_width = spec.schedule.widths[0]
    for repeat in range(spec.repeats):
        for physical, (width, ff_width) in enumerate(
                zip(spec.schedule.widths, spec.schedule.ff_widths)):
            effective = repeat * spec.schedule.physical_layers + physical
            if width != previous_width:
                emit(Opcode.LOAD, Lane.U, StrideClass.UNIT,
                     count=_checked_mul(rows, max(width, previous_width)),
                     reads=(2,), writes=(2,), op="model.width.carry_forward_view",
                     cost_class="bandwidth")
            projection_reads = (2, 4) + ((6,) if spec.anchor is not None
                                         and effective >= spec.anchor.source_depth else ())
            emit(Opcode.T_MACC, Lane.T, StrideClass.TILE,
                 count=_checked_mul(rows, 5 if spec.gated_attention else 4,
                                    width, width),
                 reads=projection_reads, writes=(3, 5),
                 op="model.attention.project_mix" if spec.anchor is not None
                 else "model.attention.project",
                 cost_class="compute")
            pairs = _attention_pairs(spec, sequence_length)
            attention_op = "model.attention.reference_sliding" \
                if spec.reference_window is not None else "model.attention.causal"
            emit(Opcode.T_MACC, Lane.T, StrideClass.TILE,
                 count=_checked_mul(batch, 2, width, pairs), reads=(3, 5), writes=(2,),
                 op=attention_op, cost_class="compute")
            emit(Opcode.T_MACC, Lane.T, StrideClass.TILE,
                 count=_checked_mul(rows, 3, width, ff_width), reads=(2, 4), writes=(2,),
                 op="model.mlp.swiglu", cost_class="compute")
            if spec.anchor is not None and effective + 1 == spec.anchor.source_depth:
                emit_anchor()
            previous_width = width
        if repeat + 1 < spec.repeats:
            emit(Opcode.BARRIER, Lane.H, StrideClass.SCALAR, count=1,
                 reads=(2, 5), writes=(), op="model.loop.iteration_barrier",
                 cost_class="latency")

    emit(Opcode.T_MACC, Lane.T, StrideClass.TILE,
         count=_checked_mul(rows, spec.schedule.widths[-1], spec.vocab_size),
         reads=(2, 4), writes=(7,), op="model.head.logits",
         cost_class="compute")

    chosen_target = target or TargetProfile.for_host()
    realization = optimize(module, chosen_target, Theta.cool())
    pack = hydrate_pipelined(module, realization,
                             plan=f"adaptive:{spec.digest[:16]}", depth=2)
    diagnostics = verify_all(module, result=realization, pack=pack, h=chosen_target,
                             theta=Theta.cool())
    diagnostics += verify_smart_lowering(module, pack=pack)
    if diagnostics:
        message = "; ".join(f"{row.law}: {row.message}" for row in diagnostics)
        raise ValueError("adaptive-transformer lowering failed verification: " + message)
    data = encode(pack)
    artifact = _digest({"spec": spec.digest, "report": report.digest,
                        "module": hash_module(module),
                        "streampack": hashlib.sha256(data).hexdigest()})
    return LoweredAdaptivePlan(report, module, realization, pack, data, artifact)


__all__ = [
    "AdaptiveCostReport", "AdaptiveLanguageSpec", "ExogenousAnchorSpec",
    "LoweredAdaptivePlan", "MultiPatchCostReport", "MultiPatchSpec",
    "ReferenceWindowSpec", "WidthScheduleSpec", "assess_adaptive_language",
    "assess_multi_patch", "fixed_residual_update", "fixed_residual_view",
    "lower_adaptive_language",
]
