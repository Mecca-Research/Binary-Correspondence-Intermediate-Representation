"""Dependency-free contracts for BCIR byte-native model research.

The stable BCIR Llama/BCIRQ8 path consumes token ids.  This module is a separate,
bounded laboratory for models whose semantic input is a raw byte stream.  It defines
lossless UTF-8 references, incremental patch layouts, BLT diffusion/speculation
contracts, a selective-SSM shape, exact hosted parameter accounting, measured-ingest
selection, shape-checked global-weight transplantation, and verified StreamPack
lowering.  It deliberately contains no tensor framework or target-specific kernel.

Operation counts are analytic lower bounds.  A CUDA provider, production diffusion
decoder, or native model format requires measured parity evidence before promotion.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from ..abi.streampack_abi import encode
from ..gem.streampack import StreamPack, hydrate_pipelined
from ..model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from ..verify import verify_all, verify_smart_lowering
from .cost import TargetProfile, Theta
from .provenance import hash_module
from .realize import RealizationResult, optimize

_MAX_U63 = (1 << 63) - 1
_MAX_CONTEXT = 1 << 20
_MAX_WIDTH = 65536
_MAX_LAYERS = 256
_MAX_PATCH_BYTES = 4096
_MAX_STATE = 4096
_MAX_TENSOR_INVENTORY = 65536
_MAX_POLICY_ELEMENTS = 1 << 24
_Q20 = 1 << 20


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sha256(value, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower() \
            or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _integer(value, field: str, *, minimum: int = 0,
             maximum: int = _MAX_U63) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite(value, field: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or minimum is not None and result < minimum \
            or maximum is not None and result > maximum:
        raise ValueError(f"{field} is outside its valid range")
    return result


def _checked_add(*values: int) -> int:
    total = 0
    for value in values:
        _integer(value, "checked-add operand")
        total += value
        if total > _MAX_U63:
            raise ValueError("byte-model integer addition exceeds u63")
    return total


def _checked_mul(*values: int) -> int:
    product = 1
    for value in values:
        _integer(value, "checked-multiply operand")
        if value and product > _MAX_U63 // value:
            raise ValueError("byte-model integer multiplication exceeds u63")
        product *= value
    return product


@dataclass(frozen=True)
class ByteVocabularySpec:
    """Direct byte ids plus four non-byte control symbols.

    Ids 0..255 always mean the corresponding octet.  Text normalization and grapheme
    segmentation are intentionally absent: a Python string is encoded with strict UTF-8,
    while arbitrary bytes remain arbitrary bytes.
    """

    size: int = 260
    bos_id: int = 256
    eos_id: int = 257
    pad_id: int = 258
    mask_id: int = 259

    def __post_init__(self) -> None:
        if self.size != 260:
            raise ValueError("the v1 byte vocabulary is exactly 256 octets plus four controls")
        values = (self.bos_id, self.eos_id, self.pad_id, self.mask_id)
        for name, value in zip(("bos_id", "eos_id", "pad_id", "mask_id"), values):
            _integer(value, name, minimum=256, maximum=self.size - 1)
        if len(set(values)) != len(values):
            raise ValueError("byte vocabulary control ids must be distinct")
        if values != (256, 257, 258, 259):
            raise ValueError("the v1 byte vocabulary control ids are fixed")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class UnicodeByteReference:
    """Diagnostic mapping from one Unicode scalar to its exact UTF-8 byte span."""

    byte_start: int
    byte_end: int
    codepoint: int
    utf8_hex: str

    def __post_init__(self) -> None:
        _integer(self.byte_start, "byte_start", maximum=_MAX_CONTEXT)
        _integer(self.byte_end, "byte_end", minimum=self.byte_start + 1,
                 maximum=_MAX_CONTEXT)
        _integer(self.codepoint, "codepoint", maximum=0x10FFFF)
        if 0xD800 <= self.codepoint <= 0xDFFF:
            raise ValueError("Unicode references cannot contain surrogate codepoints")
        if not isinstance(self.utf8_hex, str) or len(self.utf8_hex) != 2 * (
                self.byte_end - self.byte_start):
            raise ValueError("utf8_hex length does not match its byte span")
        try:
            encoded = bytes.fromhex(self.utf8_hex)
        except ValueError as exc:
            raise ValueError("utf8_hex must contain hexadecimal bytes") from exc
        if encoded.decode("utf-8") != chr(self.codepoint):
            raise ValueError("Unicode reference bytes do not encode its codepoint")


@dataclass(frozen=True)
class RawByteSequence:
    payload: bytes
    references: tuple[UnicodeByteReference, ...] = ()

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes or len(self.payload) > _MAX_CONTEXT:
            raise ValueError(f"payload must be bytes no longer than {_MAX_CONTEXT}")
        if not isinstance(self.references, tuple) \
                or any(not isinstance(item, UnicodeByteReference) for item in self.references):
            raise ValueError("references must be UnicodeByteReference values")
        cursor = 0
        for item in self.references:
            if item.byte_start != cursor:
                raise ValueError("Unicode references must exactly partition the payload")
            if self.payload[item.byte_start:item.byte_end].hex() != item.utf8_hex:
                raise ValueError("Unicode reference does not match payload bytes")
            cursor = item.byte_end
        if self.references and cursor != len(self.payload):
            raise ValueError("Unicode references must cover the complete payload")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def to_dict(self) -> dict:
        return {"schema": "bcir.raw_bytes.v1", "length": len(self.payload),
                "sha256": self.sha256,
                "references": [asdict(item) for item in self.references]}


def encode_raw_sequence(value: str | bytes | bytearray | memoryview) -> RawByteSequence:
    """Encode text without normalization or copy a raw byte sequence.

    Character references are metadata only.  The model always trains on the octets, so
    canonically equivalent Unicode strings remain distinguishable unless a corpus policy
    explicitly normalized them before this boundary.
    """
    if isinstance(value, str):
        payload = bytearray()
        references = []
        for character in value:
            encoded = character.encode("utf-8", errors="strict")
            start = len(payload)
            if start > _MAX_CONTEXT - len(encoded):
                raise ValueError(f"UTF-8 payload exceeds {_MAX_CONTEXT} bytes")
            payload.extend(encoded)
            references.append(UnicodeByteReference(
                start, len(payload), ord(character), encoded.hex()))
        return RawByteSequence(bytes(payload), tuple(references))
    if isinstance(value, memoryview):
        if value.nbytes > _MAX_CONTEXT:
            raise ValueError(f"raw byte payload exceeds {_MAX_CONTEXT} bytes")
        return RawByteSequence(bytes(value))
    if isinstance(value, (bytes, bytearray)):
        if len(value) > _MAX_CONTEXT:
            raise ValueError(f"raw byte payload exceeds {_MAX_CONTEXT} bytes")
        return RawByteSequence(bytes(value))
    raise ValueError("raw byte input must be str or a bytes-like object")


def decode_raw_bytes(value: bytes | Sequence[int], *, errors: str = "strict") -> str:
    if errors not in ("strict", "replace"):
        raise ValueError("errors must be 'strict' or 'replace'")
    if isinstance(value, bytes):
        if len(value) > _MAX_CONTEXT:
            raise ValueError(f"raw byte payload exceeds {_MAX_CONTEXT} bytes")
        payload = value
    elif isinstance(value, Sequence):
        if len(value) > _MAX_CONTEXT:
            raise ValueError(f"raw byte sequence exceeds {_MAX_CONTEXT} bytes")
        items = tuple(value)
        if any(type(item) is not int or not 0 <= item <= 255 for item in items):
            raise ValueError("raw byte ids must be integers in [0, 255]")
        payload = bytes(items)
    else:
        raise ValueError("raw byte value must be bytes or a sequence of byte ids")
    return payload.decode("utf-8", errors=errors)


@dataclass(frozen=True)
class PatchLayout:
    byte_length: int
    starts: tuple[int, ...]
    method: str
    score_sha256: str

    def __post_init__(self) -> None:
        _integer(self.byte_length, "byte_length", minimum=1, maximum=_MAX_CONTEXT)
        if not isinstance(self.starts, tuple) or not self.starts or self.starts[0] != 0:
            raise ValueError("patch starts must be a nonempty tuple beginning at zero")
        previous = -1
        for start in self.starts:
            _integer(start, "patch start", maximum=self.byte_length - 1)
            if start <= previous:
                raise ValueError("patch starts must be strictly increasing")
            previous = start
        if self.method not in ("stride", "entropy_global", "entropy_monotonic", "learned"):
            raise ValueError("unknown patch-layout method")
        _sha256(self.score_sha256, "score_sha256")

    @property
    def lengths(self) -> tuple[int, ...]:
        ends = self.starts[1:] + (self.byte_length,)
        return tuple(end - start for start, end in zip(self.starts, ends))

    @property
    def patch_count(self) -> int:
        return len(self.starts)

    def byte_to_patch(self) -> tuple[int, ...]:
        result = [0] * self.byte_length
        for patch, (start, length) in enumerate(zip(self.starts, self.lengths)):
            result[start:start + length] = [patch] * length
        return tuple(result)

    @property
    def digest(self) -> str:
        return _digest({"byte_length": self.byte_length, "starts": self.starts,
                        "method": self.method, "score_sha256": self.score_sha256})


@dataclass(frozen=True)
class BytePatchSpec:
    """Incremental raw-byte patch policy.

    Entropy values describe the prediction of byte ``i`` from prefix ``<i``.  Learned
    scores have the same causal requirement.  The contract cannot prove how scores were
    produced, so their canonical digest is carried into every layout and artifact.
    """

    method: str = "stride"
    stride: int = 4
    min_patch_bytes: int = 1
    max_patch_bytes: int = 16
    entropy_threshold_millibits: int = 5000
    entropy_delta_millibits: int = 1000
    learned_threshold_q20: int = _Q20 // 2
    learned_window: int = 1

    def __post_init__(self) -> None:
        if self.method not in ("stride", "entropy_global", "entropy_monotonic", "learned"):
            raise ValueError("unsupported byte patch method")
        _integer(self.stride, "stride", minimum=1, maximum=_MAX_PATCH_BYTES)
        _integer(self.min_patch_bytes, "min_patch_bytes", minimum=1,
                 maximum=_MAX_PATCH_BYTES)
        _integer(self.max_patch_bytes, "max_patch_bytes", minimum=self.min_patch_bytes,
                 maximum=_MAX_PATCH_BYTES)
        _integer(self.entropy_threshold_millibits, "entropy_threshold_millibits",
                 maximum=16000)
        _integer(self.entropy_delta_millibits, "entropy_delta_millibits",
                 maximum=16000)
        _integer(self.learned_threshold_q20, "learned_threshold_q20", maximum=_Q20)
        _integer(self.learned_window, "learned_window", minimum=1, maximum=32)
        if self.stride > self.max_patch_bytes:
            raise ValueError("stride cannot exceed max_patch_bytes")

    def layout(self, byte_length: int, *,
               entropy_millibits: Sequence[int] | None = None,
               learned_scores_q20: Sequence[int] | None = None) -> PatchLayout:
        _integer(byte_length, "byte_length", minimum=1, maximum=_MAX_CONTEXT)
        if self.method.startswith("entropy"):
            if not isinstance(entropy_millibits, Sequence) \
                    or len(entropy_millibits) != byte_length:
                raise ValueError("entropy patching requires one causal score per byte")
            scores = tuple(entropy_millibits)
            for value in scores:
                _integer(value, "entropy score", maximum=16000)
        elif self.method == "learned":
            if not isinstance(learned_scores_q20, Sequence) \
                    or len(learned_scores_q20) != byte_length:
                raise ValueError("learned patching requires one causal score per byte")
            scores = tuple(learned_scores_q20)
            for value in scores:
                _integer(value, "learned patch score", maximum=_Q20)
        else:
            if entropy_millibits is not None or learned_scores_q20 is not None:
                raise ValueError("stride patching does not consume external scores")
            scores = ()

        starts = [0]
        last = 0
        for index in range(1, byte_length):
            distance = index - last
            forced = distance >= self.max_patch_bytes
            candidate = False
            if self.method == "stride":
                candidate = distance >= self.stride
            elif self.method == "entropy_global":
                candidate = scores[index] > self.entropy_threshold_millibits
            elif self.method == "entropy_monotonic":
                candidate = scores[index] - scores[index - 1] \
                    > self.entropy_delta_millibits
            elif self.method == "learned":
                candidate = scores[index] >= self.learned_threshold_q20
            if forced or distance >= self.min_patch_bytes and candidate:
                starts.append(index)
                last = index
        score_sha = hashlib.sha256(_canonical(scores).encode("utf-8")).hexdigest()
        result = PatchLayout(byte_length, tuple(starts), self.method, score_sha)
        if any(length < self.min_patch_bytes for length in result.lengths[:-1]) \
                or any(length > self.max_patch_bytes for length in result.lengths):
            raise AssertionError("patch layout violated configured bounds")
        return result

    def validate_layout(self, layout: PatchLayout, *,
                        byte_length: int | None = None) -> PatchLayout:
        """Validate an externally carried layout against this patch policy.

        Entropy/learned score streams are not recoverable from a layout, so their
        content digest remains provenance rather than a recomputed proof.  Shape and
        configured patch bounds are nevertheless mandatory.  Stride layouts have no
        external scores and therefore must equal the canonical layout exactly.
        """
        if not isinstance(layout, PatchLayout) \
                or layout.method != self.method \
                or byte_length is not None and layout.byte_length != byte_length:
            raise ValueError("patch layout is incompatible with the byte patch policy")
        lengths = layout.lengths
        if any(length < self.min_patch_bytes for length in lengths[:-1]) \
                or any(length > self.max_patch_bytes for length in lengths):
            raise ValueError("patch layout violates the configured patch bounds")
        if self.method == "stride" and layout != self.layout(layout.byte_length):
            raise ValueError("stride patch layout is not the canonical layout")
        return layout

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ByteDiffusionSpec:
    block_size: int = 8
    max_steps: int = 8
    unmask_strategy: str = "confidence"
    confidence_threshold_q20: int = int(0.7 * _Q20)
    entropy_budget_millibits: int = 1000
    loss_weight_q20: int = _Q20

    def __post_init__(self) -> None:
        _integer(self.block_size, "block_size", minimum=1, maximum=256)
        _integer(self.max_steps, "max_steps", minimum=1, maximum=self.block_size)
        if self.unmask_strategy not in ("confidence", "entropy_bounded", "all"):
            raise ValueError("unsupported diffusion unmask strategy")
        _integer(self.confidence_threshold_q20, "confidence_threshold_q20",
                 maximum=_Q20)
        _integer(self.entropy_budget_millibits, "entropy_budget_millibits",
                 maximum=16000 * self.block_size)
        _integer(self.loss_weight_q20, "loss_weight_q20", minimum=1, maximum=16 * _Q20)

    def corrupt(self, clean: Sequence[int], vocabulary: ByteVocabularySpec, *,
                mask_probability_q20: int, seed: int) -> tuple[int, ...]:
        if not isinstance(vocabulary, ByteVocabularySpec):
            raise ValueError("vocabulary must be ByteVocabularySpec")
        clean = tuple(clean)
        if not clean or len(clean) > self.block_size \
                or any(type(value) is not int or not 0 <= value < vocabulary.size
                       or value == vocabulary.mask_id for value in clean):
            raise ValueError("clean diffusion block is malformed")
        _integer(mask_probability_q20, "mask_probability_q20", minimum=1,
                 maximum=_Q20)
        _integer(seed, "seed", maximum=(1 << 63) - 1)
        result = list(clean)
        masked = []
        for index in range(len(result)):
            raw = hashlib.sha256(
                f"bcir-byte-diffusion-v1:{seed}:{index}".encode("ascii")).digest()
            draw = int.from_bytes(raw[:8], "little") % _Q20
            if draw < mask_probability_q20:
                result[index] = vocabulary.mask_id
                masked.append(index)
        if not masked:
            index = seed % len(result)
            result[index] = vocabulary.mask_id
        return tuple(result)

    def select_positions(self, masked: Sequence[int], *,
                         confidences_q20: Sequence[int],
                         entropies_millibits: Sequence[int]) -> tuple[int, ...]:
        masked = tuple(masked)
        if not masked or len(set(masked)) != len(masked) \
                or any(type(index) is not int or index < 0 for index in masked):
            raise ValueError("masked positions must be distinct non-negative integers")
        upper = max(masked)
        if len(confidences_q20) != len(entropies_millibits) \
                or not 1 <= len(confidences_q20) <= self.block_size \
                or len(confidences_q20) <= upper:
            raise ValueError("diffusion scores do not cover every masked position")
        for value in confidences_q20:
            _integer(value, "confidence", maximum=_Q20)
        for value in entropies_millibits:
            _integer(value, "entropy", maximum=16000)
        if self.unmask_strategy == "all":
            return masked
        if self.unmask_strategy == "confidence":
            chosen = tuple(index for index in masked
                           if confidences_q20[index] >= self.confidence_threshold_q20)
            if chosen:
                return chosen
            return (min(masked, key=lambda index: (-confidences_q20[index], index)),)
        ordered = sorted(masked, key=lambda index: (entropies_millibits[index], index))
        chosen = []
        total = 0
        for index in ordered:
            updated = total + entropies_millibits[index]
            if updated <= self.entropy_budget_millibits:
                chosen.append(index)
                total = updated
        return tuple(chosen or ordered[:1])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ByteSpeculationSpec:
    draft_bytes: int = 8
    verifier: str = "exact_greedy"

    def __post_init__(self) -> None:
        _integer(self.draft_bytes, "draft_bytes", minimum=1, maximum=256)
        if self.verifier != "exact_greedy":
            raise ValueError("the first byte-speculation rail admits only exact greedy verification")


@dataclass(frozen=True)
class DraftVerification:
    accepted: tuple[int, ...]
    emitted: tuple[int, ...]
    rejected: int
    matched_all: bool

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, tuple) or not isinstance(self.emitted, tuple) \
                or any(type(value) is not int or value < 0
                       for value in self.accepted + self.emitted):
            raise ValueError("draft verification ids must be non-negative integer tuples")
        _integer(self.rejected, "rejected", maximum=256)
        if type(self.matched_all) is not bool \
                or len(self.emitted) != len(self.accepted) + 1 \
                or self.emitted[:len(self.accepted)] != self.accepted:
            raise ValueError("draft verification shape is inconsistent")
        if self.matched_all != (self.rejected == 0):
            raise ValueError("draft verification match/rejection state is inconsistent")


def verify_byte_draft(draft: Sequence[int], verifier_predictions: Sequence[int],
                      *, free_prediction: int, vocabulary_size: int) -> DraftVerification:
    draft = tuple(draft)
    verifier_predictions = tuple(verifier_predictions)
    _integer(vocabulary_size, "vocabulary_size", minimum=2, maximum=1 << 24)
    if not draft or len(draft) != len(verifier_predictions):
        raise ValueError("draft and verifier predictions must have the same nonzero length")
    if len(draft) > 256:
        raise ValueError("byte draft exceeds the bounded verifier length")
    for value in draft + verifier_predictions + (free_prediction,):
        _integer(value, "byte-draft id", maximum=vocabulary_size - 1)
    accepted = []
    for candidate, prediction in zip(draft, verifier_predictions):
        if candidate != prediction:
            return DraftVerification(tuple(accepted), tuple(accepted) + (prediction,),
                                     len(draft) - len(accepted), False)
        accepted.append(candidate)
    return DraftVerification(tuple(accepted), tuple(accepted) + (free_prediction,), 0, True)


@dataclass(frozen=True)
class LearnedPatchPolicySpec:
    target_boundary_q20: int = _Q20 // 5
    discount_q20: int = int(0.99 * _Q20)
    logit_scale: int = 16
    policy_weight_q20: int = _Q20 // 100
    target_weight_q20: int = _Q20 // 100
    early_weight_q20: int = _Q20 // 10

    def __post_init__(self) -> None:
        _integer(self.target_boundary_q20, "target_boundary_q20", minimum=1,
                 maximum=_Q20 - 1)
        _integer(self.discount_q20, "discount_q20", minimum=1, maximum=_Q20)
        _integer(self.logit_scale, "logit_scale", minimum=1, maximum=1024)
        for field in ("policy_weight_q20", "target_weight_q20", "early_weight_q20"):
            _integer(getattr(self, field), field, minimum=1, maximum=16 * _Q20)

    @property
    def discount(self) -> float:
        return self.discount_q20 / _Q20


def discounted_batch_advantages(rewards: Sequence[Sequence[float]],
                                discount: float) -> tuple[tuple[float, ...], ...]:
    """Future discounted rewards centered across the batch at each byte position."""
    discount = _finite(discount, "discount", minimum=0.0, maximum=1.0)
    if not isinstance(rewards, Sequence) or not 1 <= len(rewards) <= 1 << 16:
        raise ValueError("rewards must be a bounded nonempty rectangular batch")
    width = None
    for row in rewards:
        if not isinstance(row, Sequence):
            raise ValueError("rewards must be a bounded nonempty rectangular batch")
        if width is None:
            width = len(row)
            if not 1 <= width <= _MAX_CONTEXT \
                    or len(rewards) > _MAX_POLICY_ELEMENTS // width:
                raise ValueError("reward batch exceeds the bounded element count")
        elif len(row) != width:
            raise ValueError("rewards must be a bounded nonempty rectangular batch")
    rows = tuple(tuple(_finite(value, "reward") for value in row) for row in rewards)
    if not rows or not rows[0]:
        raise ValueError("rewards must be a nonempty rectangular batch")
    returns = []
    for row in rows:
        running = 0.0
        current = [0.0] * len(row)
        for index in range(len(row) - 1, -1, -1):
            running = row[index] + discount * running
            current[index] = running
        returns.append(current)
    for index in range(len(rows[0])):
        mean = sum(row[index] for row in returns) / len(returns)
        for row in returns:
            row[index] -= mean
    return tuple(tuple(row) for row in returns)


def _transformer_elements(width: int, ff_width: int) -> int:
    return _checked_add(_checked_mul(4, width, width),
                        _checked_mul(3, width, ff_width),
                        _checked_mul(2, width))


def _ssm_elements(width: int, state_size: int, expansion: int,
                  conv_width: int) -> int:
    inner = _checked_mul(width, expansion)
    # norm; in/out projections; depthwise convolution; dt projection+bias;
    # B, C and diagonal A; learned direct path D.
    return _checked_add(width, _checked_mul(3, width, inner),
                        _checked_mul(inner, conv_width),
                        _checked_mul(inner, inner), _checked_mul(2, inner),
                        _checked_mul(3, inner, state_size))


@dataclass(frozen=True)
class MambaByteSpec:
    context_bytes: int
    width: int
    layers: int
    state_size: int = 16
    expansion: int = 2
    conv_width: int = 4
    tied_embeddings: bool = True
    vocabulary: ByteVocabularySpec = ByteVocabularySpec()
    rms_norm_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        _integer(self.context_bytes, "context_bytes", minimum=1, maximum=_MAX_CONTEXT)
        _integer(self.width, "width", minimum=4, maximum=_MAX_WIDTH)
        _integer(self.layers, "layers", minimum=1, maximum=_MAX_LAYERS)
        _integer(self.state_size, "state_size", minimum=1, maximum=_MAX_STATE)
        _integer(self.expansion, "expansion", minimum=1, maximum=16)
        _integer(self.conv_width, "conv_width", minimum=1, maximum=256)
        if type(self.tied_embeddings) is not bool \
                or not isinstance(self.vocabulary, ByteVocabularySpec):
            raise ValueError("MambaByte vocabulary/tied_embeddings are malformed")
        _finite(self.rms_norm_epsilon, "rms_norm_epsilon", minimum=1.0e-12,
                maximum=1.0)
        _checked_mul(self.context_bytes, self.width, self.layers,
                     self.state_size, self.expansion)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["schema"] = "bcir.mambabyte.v1"
        return result

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class MambaByteCostReport:
    spec_sha256: str
    parameter_elements: int
    recurrent_state_bytes: int
    convolution_state_bytes: int
    sequence_flops_lower_bound: int
    classification: str = "exact-size+analytic-lower-bound"

    def __post_init__(self) -> None:
        _sha256(self.spec_sha256, "spec_sha256")
        for field in ("parameter_elements", "recurrent_state_bytes",
                      "convolution_state_bytes", "sequence_flops_lower_bound"):
            _integer(getattr(self, field), field)
        if self.classification != "exact-size+analytic-lower-bound":
            raise ValueError("MambaByte cost classification is fixed")


def assess_mambabyte(spec: MambaByteSpec, *, sequence_bytes: int,
                     bytes_per_element: int = 2) -> MambaByteCostReport:
    if not isinstance(spec, MambaByteSpec):
        raise ValueError("spec must be MambaByteSpec")
    _integer(sequence_bytes, "sequence_bytes", minimum=1, maximum=spec.context_bytes)
    _integer(bytes_per_element, "bytes_per_element", minimum=1, maximum=16)
    vocab = spec.vocabulary.size
    parameters = _checked_add(_checked_mul(vocab, spec.width),
                              _checked_mul(spec.layers, _ssm_elements(
                                  spec.width, spec.state_size, spec.expansion,
                                  spec.conv_width)), spec.width)
    if not spec.tied_embeddings:
        parameters = _checked_add(parameters, _checked_mul(vocab, spec.width))
    inner = _checked_mul(spec.width, spec.expansion)
    state = _checked_mul(spec.layers, inner, spec.state_size, bytes_per_element)
    convolution = _checked_mul(spec.layers, inner, spec.conv_width - 1,
                               bytes_per_element)
    # Dominant projections and diagonal selective recurrence only.
    per_layer = _checked_add(_checked_mul(6, spec.width, inner),
                             _checked_mul(2, inner, inner),
                             _checked_mul(8, inner, spec.state_size))
    flops = _checked_add(_checked_mul(sequence_bytes, spec.layers, per_layer),
                         _checked_mul(2, sequence_bytes, spec.width, vocab))
    return MambaByteCostReport(spec.digest, parameters, state, convolution, flops)


@dataclass(frozen=True)
class ByteLatentSpec:
    context_bytes: int
    patcher: BytePatchSpec
    local_width: int
    global_width: int
    local_heads: int
    global_heads: int
    local_encoder_layers: int
    global_layers: int
    local_decoder_layers: int
    local_ff_width: int
    global_ff_width: int
    local_window_bytes: int = 128
    hash_ngram_sizes: tuple[int, ...] = ()
    hash_buckets: int = 0
    local_backbone: str = "transformer"
    ssm_state_size: int = 16
    ssm_expansion: int = 2
    ssm_conv_width: int = 4
    generation_mode: str = "autoregressive"
    diffusion: ByteDiffusionSpec | None = None
    speculation: ByteSpeculationSpec | None = None
    learned_policy: LearnedPatchPolicySpec | None = None
    tied_embeddings: bool = True
    vocabulary: ByteVocabularySpec = ByteVocabularySpec()
    rope_base: int = 10000
    rms_norm_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        _integer(self.context_bytes, "context_bytes", minimum=2, maximum=_MAX_CONTEXT)
        if not isinstance(self.patcher, BytePatchSpec):
            raise ValueError("patcher must be BytePatchSpec")
        for field in ("local_width", "global_width"):
            _integer(getattr(self, field), field, minimum=4, maximum=_MAX_WIDTH)
        for width, heads, label in ((self.local_width, self.local_heads, "local"),
                                    (self.global_width, self.global_heads, "global")):
            _integer(heads, f"{label}_heads", minimum=1, maximum=width)
            if width % heads or width // heads % 2:
                raise ValueError(f"{label} width needs an even integral head dimension")
        for field in ("local_encoder_layers", "global_layers", "local_decoder_layers"):
            _integer(getattr(self, field), field, minimum=1, maximum=_MAX_LAYERS)
        _integer(self.local_ff_width, "local_ff_width", minimum=self.local_width,
                 maximum=16 * _MAX_WIDTH)
        _integer(self.global_ff_width, "global_ff_width", minimum=self.global_width,
                 maximum=16 * _MAX_WIDTH)
        _integer(self.local_window_bytes, "local_window_bytes", minimum=1,
                 maximum=self.context_bytes)
        if not isinstance(self.hash_ngram_sizes, tuple) \
                or any(type(size) is not int or not 2 <= size <= 16
                       for size in self.hash_ngram_sizes) \
                or tuple(sorted(set(self.hash_ngram_sizes))) != self.hash_ngram_sizes:
            raise ValueError("hash_ngram_sizes must be sorted unique integers in [2, 16]")
        _integer(self.hash_buckets, "hash_buckets", maximum=1 << 24)
        if bool(self.hash_ngram_sizes) != bool(self.hash_buckets):
            raise ValueError("hash n-grams and buckets must be enabled together")
        if self.local_backbone not in ("transformer", "selective_ssm"):
            raise ValueError("local_backbone must be transformer or selective_ssm")
        _integer(self.ssm_state_size, "ssm_state_size", minimum=1, maximum=_MAX_STATE)
        _integer(self.ssm_expansion, "ssm_expansion", minimum=1, maximum=16)
        _integer(self.ssm_conv_width, "ssm_conv_width", minimum=1, maximum=256)
        if self.generation_mode not in (
                "autoregressive", "diffusion", "self_speculative", "diffusion_verify"):
            raise ValueError("unsupported byte generation mode")
        needs_diffusion = self.generation_mode in ("diffusion", "diffusion_verify")
        needs_speculation = self.generation_mode in ("self_speculative", "diffusion_verify")
        if needs_diffusion and not isinstance(self.diffusion, ByteDiffusionSpec):
            raise ValueError("diffusion generation modes require diffusion settings")
        if self.diffusion is not None and not isinstance(self.diffusion, ByteDiffusionSpec):
            raise ValueError("diffusion must be ByteDiffusionSpec or None")
        if needs_speculation != isinstance(self.speculation, ByteSpeculationSpec):
            raise ValueError("speculation settings must appear exactly for verified modes")
        if self.patcher.method == "learned" \
                and not isinstance(self.learned_policy, LearnedPatchPolicySpec):
            raise ValueError("learned patching requires LearnedPatchPolicySpec")
        if self.patcher.method != "learned" and self.learned_policy is not None:
            raise ValueError("learned_policy is only valid for learned patching")
        if type(self.tied_embeddings) is not bool \
                or not isinstance(self.vocabulary, ByteVocabularySpec):
            raise ValueError("byte model vocabulary/tied_embeddings are malformed")
        _integer(self.rope_base, "rope_base", minimum=2, maximum=1 << 30)
        _finite(self.rms_norm_epsilon, "rms_norm_epsilon", minimum=1.0e-12,
                maximum=1.0)
        _checked_mul(self.context_bytes, self.local_width, self.global_width,
                     self.local_encoder_layers + self.global_layers
                     + self.local_decoder_layers)

    def to_dict(self) -> dict:
        return {"schema": "bcir.byte_latent.v1", "context_bytes": self.context_bytes,
                "patcher": self.patcher.to_dict(), "local_width": self.local_width,
                "global_width": self.global_width, "local_heads": self.local_heads,
                "global_heads": self.global_heads,
                "local_encoder_layers": self.local_encoder_layers,
                "global_layers": self.global_layers,
                "local_decoder_layers": self.local_decoder_layers,
                "local_ff_width": self.local_ff_width,
                "global_ff_width": self.global_ff_width,
                "local_window_bytes": self.local_window_bytes,
                "hash_ngram_sizes": self.hash_ngram_sizes,
                "hash_buckets": self.hash_buckets,
                "local_backbone": self.local_backbone,
                "ssm_state_size": self.ssm_state_size,
                "ssm_expansion": self.ssm_expansion,
                "ssm_conv_width": self.ssm_conv_width,
                "generation_mode": self.generation_mode,
                "diffusion": None if self.diffusion is None else self.diffusion.to_dict(),
                "speculation": None if self.speculation is None else asdict(self.speculation),
                "learned_policy": None if self.learned_policy is None
                else asdict(self.learned_policy),
                "tied_embeddings": self.tied_embeddings,
                "vocabulary": self.vocabulary.to_dict(),
                "rope_base": self.rope_base,
                "rms_norm_epsilon": self.rms_norm_epsilon}

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


def _byte_latent_parameter_elements(spec: ByteLatentSpec) -> int:
    local, global_ = spec.local_width, spec.global_width
    vocab = spec.vocabulary.size
    parameters = _checked_add(_checked_mul(vocab, local), global_)
    if spec.hash_ngram_sizes:
        parameters = _checked_add(parameters, _checked_mul(
            len(spec.hash_ngram_sizes), spec.hash_buckets, local))
    if spec.local_backbone == "transformer":
        encoder = _transformer_elements(local, spec.local_ff_width)
    else:
        encoder = _ssm_elements(local, spec.ssm_state_size, spec.ssm_expansion,
                                spec.ssm_conv_width)
    parameters = _checked_add(
        parameters, _checked_mul(spec.local_encoder_layers, encoder),
        # Patch pool q/k/v local->global and o global->global.
        _checked_mul(3, local, global_), _checked_mul(global_, global_),
        _checked_mul(spec.global_layers,
                     _transformer_elements(global_, spec.global_ff_width)),
        # Decoder cross-attention q/o local->local and k/v global->local.
        _checked_mul(2, local, local), _checked_mul(2, global_, local),
        _checked_mul(spec.local_decoder_layers,
                     _transformer_elements(local, spec.local_ff_width)),
        local)
    if not spec.tied_embeddings:
        parameters = _checked_add(parameters, _checked_mul(vocab, local))
    if spec.patcher.method in ("entropy_global", "entropy_monotonic", "learned"):
        parameters = _checked_add(parameters, _checked_mul(vocab, local))
    if spec.patcher.method == "learned":
        parameters = _checked_add(parameters, _checked_mul(
            spec.patcher.learned_window + 1, local))
    return parameters


@dataclass(frozen=True)
class ByteLatentCostReport:
    spec_sha256: str
    byte_count: int
    patch_count: int
    parameter_elements: int
    patch_metadata_bytes: int
    local_cache_bytes: int
    global_kv_bytes: int
    decoder_kv_bytes: int
    recurrent_state_bytes: int
    diffusion_scratch_bytes: int
    local_flops_lower_bound: int
    patch_flops_lower_bound: int
    global_flops_lower_bound: int
    decoder_flops_lower_bound: int
    diffusion_flops_lower_bound: int
    total_flops_lower_bound: int
    classification: str = "exact-size+analytic-lower-bound"

    def __post_init__(self) -> None:
        _sha256(self.spec_sha256, "spec_sha256")
        for field in self.__dataclass_fields__:
            if field not in ("spec_sha256", "classification"):
                _integer(getattr(self, field), field)
        if self.classification != "exact-size+analytic-lower-bound":
            raise ValueError("byte latent cost classification is fixed")

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


def assess_byte_latent(spec: ByteLatentSpec, *, byte_count: int, patch_count: int,
                       batch: int = 1, bytes_per_element: int = 2) -> ByteLatentCostReport:
    if not isinstance(spec, ByteLatentSpec):
        raise ValueError("spec must be ByteLatentSpec")
    _integer(byte_count, "byte_count", minimum=1, maximum=spec.context_bytes)
    _integer(patch_count, "patch_count", minimum=1, maximum=byte_count)
    _integer(batch, "batch", minimum=1, maximum=1 << 16)
    _integer(bytes_per_element, "bytes_per_element", minimum=1, maximum=16)
    local, global_ = spec.local_width, spec.global_width
    window = min(byte_count, spec.local_window_bytes)
    local_pairs = window * (window + 1) // 2 + (byte_count - window) * window
    patch_pairs = patch_count * (patch_count + 1) // 2
    local_block = _checked_add(
        _checked_mul(2, byte_count,
                     _checked_add(_checked_mul(4, local, local),
                                  _checked_mul(3, local, spec.local_ff_width))),
        _checked_mul(4, local, local_pairs))
    if spec.local_backbone == "selective_ssm":
        inner = _checked_mul(local, spec.ssm_expansion)
        local_block = _checked_mul(byte_count, _checked_add(
            _checked_mul(6, local, inner), _checked_mul(2, inner, inner),
            _checked_mul(8, inner, spec.ssm_state_size)))
    local_flops = _checked_mul(batch, spec.local_encoder_layers, local_block)
    patch_flops = _checked_mul(batch, 2, patch_count,
                               _checked_add(_checked_mul(3, local, global_),
                                            _checked_mul(global_, global_)))
    global_block = _checked_add(
        _checked_mul(2, patch_count,
                     _checked_add(_checked_mul(4, global_, global_),
                                  _checked_mul(3, global_, spec.global_ff_width))),
        _checked_mul(4, global_, patch_pairs))
    global_flops = _checked_mul(batch, spec.global_layers, global_block)
    decoder_block = _checked_add(
        _checked_mul(2, byte_count,
                     _checked_add(_checked_mul(4, local, local),
                                  _checked_mul(3, local, spec.local_ff_width))),
        _checked_mul(4, local, byte_count * (byte_count + 1) // 2))
    decoder_flops = _checked_mul(batch, spec.local_decoder_layers, decoder_block)
    decoder_flops = _checked_add(
        decoder_flops, _checked_mul(batch, 2, byte_count,
                                    _checked_add(_checked_mul(2, local, local),
                                                 _checked_mul(2, global_, local))),
        _checked_mul(batch, 2, byte_count, local, spec.vocabulary.size))
    diffusion_flops = 0
    diffusion_scratch = 0
    if spec.diffusion is not None:
        block = spec.diffusion.block_size
        steps = spec.diffusion.max_steps
        diffusion_block_flops = _checked_add(
            _checked_mul(2, block,
                         _checked_add(_checked_mul(4, local, local),
                                      _checked_mul(3, local, spec.local_ff_width))),
            _checked_mul(4, local, block, block))
        diffusion_flops = _checked_mul(batch, steps, _checked_add(
            _checked_mul(spec.local_decoder_layers, diffusion_block_flops),
            _checked_mul(2, block, local, spec.vocabulary.size)))
        diffusion_scratch = _checked_mul(batch, block,
                                         _checked_add(local * bytes_per_element, 8))
    if spec.local_backbone == "transformer":
        local_cache = _checked_mul(batch, 2, spec.local_encoder_layers,
                                   min(byte_count, spec.local_window_bytes), local,
                                   bytes_per_element)
        recurrent = 0
    else:
        local_cache = 0
        inner = _checked_mul(local, spec.ssm_expansion)
        recurrent = _checked_mul(batch, spec.local_encoder_layers, inner,
                                 spec.ssm_state_size, bytes_per_element)
    global_kv = _checked_mul(batch, 2, spec.global_layers, patch_count, global_,
                             bytes_per_element)
    decoder_kv = _checked_mul(batch, 2, spec.local_decoder_layers, byte_count, local,
                              bytes_per_element)
    total = _checked_add(local_flops, patch_flops, global_flops,
                         decoder_flops, diffusion_flops)
    return ByteLatentCostReport(
        spec.digest, byte_count, patch_count, _byte_latent_parameter_elements(spec),
        _checked_mul(batch, patch_count + byte_count, 4), local_cache, global_kv,
        decoder_kv, recurrent, diffusion_scratch, local_flops, patch_flops,
        global_flops, decoder_flops, diffusion_flops, total)


@dataclass(frozen=True)
class TensorShape:
    name: str
    dimensions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name \
                or len(self.name.encode("utf-8")) > 4096:
            raise ValueError("tensor name must be a bounded nonempty string")
        if not isinstance(self.dimensions, tuple) or not self.dimensions:
            raise ValueError("tensor dimensions must be a nonempty tuple")
        for value in self.dimensions:
            _integer(value, "tensor dimension", minimum=1, maximum=1 << 30)

    @property
    def elements(self) -> int:
        return _checked_mul(*self.dimensions)


@dataclass(frozen=True)
class ByteifiedTransplantPlan:
    source_sha256: str
    target_spec_sha256: str
    mappings: tuple[tuple[str, str], ...]
    reused_elements: int
    initialized_elements: int
    global_learning_rate_q20: int
    freeze_global: bool
    classification: str = "partial-exact-global-transplant"

    def __post_init__(self) -> None:
        _sha256(self.source_sha256, "source_sha256")
        _sha256(self.target_spec_sha256, "target_spec_sha256")
        if not isinstance(self.mappings, tuple) or not self.mappings:
            raise ValueError("transplant mappings must be nonempty")
        targets = []
        sources = []
        for pair in self.mappings:
            if not isinstance(pair, tuple) or len(pair) != 2 \
                    or any(not isinstance(name, str) or not name for name in pair):
                raise ValueError("transplant mapping names are malformed")
            targets.append(pair[0]); sources.append(pair[1])
        if len(set(targets)) != len(targets) or len(set(sources)) != len(sources):
            raise ValueError("transplant mappings must be one-to-one")
        _integer(self.reused_elements, "reused_elements", minimum=1)
        _integer(self.initialized_elements, "initialized_elements")
        _integer(self.global_learning_rate_q20, "global_learning_rate_q20",
                 maximum=_Q20)
        if type(self.freeze_global) is not bool \
                or self.classification != "partial-exact-global-transplant":
            raise ValueError("transplant policy/classification is malformed")

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


def plan_byteified_transplant(spec: ByteLatentSpec, *, source_sha256: str,
                              source_shapes: Sequence[TensorShape],
                              target_shapes: Sequence[TensorShape],
                              mappings: Mapping[str, str],
                              global_learning_rate_q20: int = _Q20 // 10,
                              freeze_global: bool = False) -> ByteifiedTransplantPlan:
    """Admit only exact-shape global-block copies; local byte organs stay newly initialized."""
    if not isinstance(spec, ByteLatentSpec):
        raise ValueError("spec must be ByteLatentSpec")
    _sha256(source_sha256, "source_sha256")
    if not isinstance(mappings, Mapping) or not mappings \
            or len(mappings) > _MAX_TENSOR_INVENTORY:
        raise ValueError("mappings must be a nonempty mapping")
    try:
        source_items = tuple(itertools.islice(
            source_shapes, _MAX_TENSOR_INVENTORY + 1))
        target_items = tuple(itertools.islice(
            target_shapes, _MAX_TENSOR_INVENTORY + 1))
        mapping_items = tuple(itertools.islice(
            mappings.items(), _MAX_TENSOR_INVENTORY + 1))
    except TypeError as exc:
        raise ValueError("tensor inventories and mappings must be iterable") from exc
    if len(source_items) > _MAX_TENSOR_INVENTORY \
            or len(target_items) > _MAX_TENSOR_INVENTORY \
            or len(mapping_items) > _MAX_TENSOR_INVENTORY:
        raise ValueError("tensor inventory exceeds the bounded entry count")
    if any(not isinstance(item, TensorShape) for item in source_items + target_items):
        raise ValueError("tensor inventories must contain TensorShape values")
    source = {item.name: item for item in source_items}
    target = {item.name: item for item in target_items}
    if len(source) != len(source_items) or len(target) != len(target_items):
        raise ValueError("tensor inventories contain duplicate names")
    normalized = []
    reused = 0
    if any(not isinstance(item, tuple) or len(item) != 2
           or not isinstance(item[0], str) or not isinstance(item[1], str)
           or not item[0].startswith("global_blocks.")
           for item in mapping_items):
        raise ValueError("only global_blocks tensors may be transplanted")
    for target_name, source_name in sorted(mapping_items):
        if not target_name or not source_name:
            raise ValueError("only global_blocks tensors may be transplanted")
        if target_name not in target or source_name not in source:
            raise ValueError("transplant mapping names an unknown tensor")
        if target[target_name].dimensions != source[source_name].dimensions:
            raise ValueError("transplant tensor shapes do not match exactly")
        normalized.append((target_name, source_name))
        reused = _checked_add(reused, target[target_name].elements)
    global_targets = {name for name in target if name.startswith("global_blocks.")}
    if {name for name, _source in normalized} != global_targets:
        raise ValueError("transplant mappings must cover the provided global-block inventory")
    total = _checked_add(*(item.elements for item in target.values()))
    if reused > total:
        raise ValueError("reused element count exceeds target inventory")
    return ByteifiedTransplantPlan(
        source_sha256, spec.digest, tuple(normalized), reused, total - reused,
        _integer(global_learning_rate_q20, "global_learning_rate_q20", maximum=_Q20),
        freeze_global)


@dataclass(frozen=True)
class ByteIngestProfile:
    """Measured provider profile; zero device throughput means unavailable."""

    profile_sha256: str
    host_ns_per_byte_q20: int
    device_ns_per_byte_q20: int
    device_launch_ns: int
    device_pool_bytes: int
    max_chunk_bytes: int
    exact_roundtrip: bool

    def __post_init__(self) -> None:
        _sha256(self.profile_sha256, "profile_sha256")
        _integer(self.host_ns_per_byte_q20, "host_ns_per_byte_q20", minimum=1)
        _integer(self.device_ns_per_byte_q20, "device_ns_per_byte_q20")
        _integer(self.device_launch_ns, "device_launch_ns")
        _integer(self.device_pool_bytes, "device_pool_bytes")
        _integer(self.max_chunk_bytes, "max_chunk_bytes", minimum=1)
        if type(self.exact_roundtrip) is not bool:
            raise ValueError("exact_roundtrip must be bool")


@dataclass(frozen=True)
class ByteIngestDecision:
    provider: str
    byte_count: int
    predicted_ns: int
    profile_sha256: str
    classification: str = "measured-profile-selection"

    def __post_init__(self) -> None:
        if self.provider not in ("host", "device"):
            raise ValueError("ingest provider must be host or device")
        _integer(self.byte_count, "byte_count", minimum=1)
        _integer(self.predicted_ns, "predicted_ns", minimum=1)
        _sha256(self.profile_sha256, "profile_sha256")
        if self.classification != "measured-profile-selection":
            raise ValueError("ingest classification is fixed")


def choose_byte_ingest(profile: ByteIngestProfile, byte_count: int) -> ByteIngestDecision:
    if not isinstance(profile, ByteIngestProfile):
        raise ValueError("profile must be ByteIngestProfile")
    _integer(byte_count, "byte_count", minimum=1, maximum=_MAX_CONTEXT)
    host_product = _checked_mul(profile.host_ns_per_byte_q20, byte_count)
    host = max(1, host_product // _Q20 + int(bool(host_product % _Q20)))
    if not profile.exact_roundtrip or not profile.device_ns_per_byte_q20 \
            or byte_count > profile.device_pool_bytes:
        return ByteIngestDecision("host", byte_count, host, profile.profile_sha256)
    chunks = 1 + (byte_count - 1) // profile.max_chunk_bytes
    device_product = _checked_mul(profile.device_ns_per_byte_q20, byte_count)
    device = _checked_add(_checked_mul(chunks, profile.device_launch_ns),
                          device_product // _Q20 + int(bool(device_product % _Q20)))
    if device < host:
        return ByteIngestDecision("device", byte_count, max(1, device),
                                  profile.profile_sha256)
    return ByteIngestDecision("host", byte_count, host, profile.profile_sha256)


@dataclass(frozen=True)
class LoweredByteLatentPlan:
    report: ByteLatentCostReport
    module: Module
    realization: RealizationResult
    streampack: StreamPack
    streampack_data: bytes
    artifact_sha256: str


def lower_byte_latent(spec: ByteLatentSpec, layout: PatchLayout, *, batch: int = 1,
                      bytes_per_element: int = 2,
                      target: TargetProfile | None = None) -> LoweredByteLatentPlan:
    """Lower one bounded byte-model pass into R-law verified claims and StreamPack."""
    if not isinstance(spec, ByteLatentSpec) or not isinstance(layout, PatchLayout):
        raise ValueError("byte lowering requires ByteLatentSpec and PatchLayout")
    if layout.byte_length > spec.context_bytes:
        raise ValueError("patch layout is incompatible with the byte model")
    spec.patcher.validate_layout(layout, byte_length=layout.byte_length)
    _integer(batch, "batch", minimum=1, maximum=1 << 16)
    _integer(bytes_per_element, "bytes_per_element", minimum=1, maximum=16)
    report = assess_byte_latent(spec, byte_count=layout.byte_length,
                                patch_count=layout.patch_count, batch=batch,
                                bytes_per_element=bytes_per_element)
    rows = _checked_mul(batch, layout.byte_length)
    patches = _checked_mul(batch, layout.patch_count)
    module = Module(name=f"byte-latent:{spec.digest[:16]}")
    module.add_resource(Resource(1, Domain.RAM, 1, (rows,), name="raw_bytes"))
    module.add_resource(Resource(
        2, Domain.RAM, 4, (_checked_add(patches, rows),), name="patch_map"))
    module.add_resource(Resource(3, Domain.RAM, bytes_per_element,
                                 (rows, spec.local_width),
                                 name="byte_hidden"))
    module.add_resource(Resource(4, Domain.RAM, bytes_per_element,
                                 (patches, spec.global_width),
                                 name="patch_hidden"))
    module.add_resource(Resource(5, Domain.RAM, bytes_per_element,
                                 (report.parameter_elements,),
                                 name="weights"))
    state_bytes = max(1, _checked_add(
        report.local_cache_bytes, report.global_kv_bytes,
        report.decoder_kv_bytes, report.recurrent_state_bytes))
    module.add_resource(Resource(6, Domain.RAM, 1, (state_bytes,), name="model_state"))
    module.add_resource(Resource(7, Domain.RAM, bytes_per_element,
                                 (rows, spec.vocabulary.size), name="byte_logits"))
    if spec.diffusion is not None:
        module.add_resource(Resource(8, Domain.RAM, 1,
                                     (max(1, report.diffusion_scratch_bytes),),
                                     name="diffusion_block"))
    if spec.speculation is not None:
        module.add_resource(Resource(
            9, Domain.RAM, 1,
            (_checked_mul(batch, spec.speculation.draft_bytes + 1),),
            name="verified_bytes"))

    phase_id = 0
    claim_id = 0

    def emit(opcode: Opcode, lane: Lane, stride: StrideClass, *, count: int,
             reads: tuple[int, ...], writes: tuple[int, ...], op: str,
             cost_class: str) -> None:
        nonlocal phase_id, claim_id
        phase_id += 1; claim_id += 1
        claim = Claim(claim_id, opcode, lane, stride, count=max(1, count),
                      rd=reads, wr=writes, hazard="barriered", bounds="assumed_safe",
                      domain=Domain.RAM, op=op, cost_class=cost_class)
        module.add_phase(Phase(phase_id, (phase_id - 1,) if phase_id > 1 else (), [claim]))

    emit(Opcode.LOAD, Lane.U, StrideClass.UNIT, count=rows, reads=(1,), writes=(3,),
         op="model.byte.ingest.exact", cost_class="bandwidth")
    emit(Opcode.LOAD, Lane.GGG, StrideClass.RANDOM,
         count=_checked_mul(batch, layout.patch_count + layout.byte_length),
         reads=(1, 3, 5), writes=(2,), op=f"model.byte.patch.{layout.method}",
         cost_class="latency")
    if spec.local_backbone == "transformer":
        emit(Opcode.T_MACC, Lane.T, StrideClass.TILE,
             count=max(1, report.local_flops_lower_bound // 2),
             reads=(3, 5), writes=(3, 6), op="model.byte.encoder.local_attention",
             cost_class="compute")
    else:
        emit(Opcode.MUL, Lane.U, StrideClass.UNIT,
             count=max(1, report.local_flops_lower_bound // 2),
             reads=(3, 5, 6), writes=(3, 6), op="model.byte.encoder.selective_scan",
             cost_class="compute")
    emit(Opcode.T_MACC, Lane.T, StrideClass.TILE,
         count=max(1, report.patch_flops_lower_bound // 2),
         reads=(2, 3, 5), writes=(4,), op="model.byte.patch.cross_attention",
         cost_class="compute")
    emit(Opcode.T_MACC, Lane.T, StrideClass.TILE,
         count=max(1, report.global_flops_lower_bound // 2),
         reads=(4, 5, 6), writes=(4, 6), op="model.byte.global.causal_transformer",
         cost_class="compute")
    emit(Opcode.T_MACC, Lane.T, StrideClass.TILE,
         count=max(1, report.decoder_flops_lower_bound // 2),
         reads=(2, 3, 4, 5, 6), writes=(3, 6, 7),
         op="model.byte.decoder.causal_cross_attention", cost_class="compute")
    if spec.diffusion is not None:
        emit(Opcode.T_MACC, Lane.T, StrideClass.TILE,
             count=max(1, report.diffusion_flops_lower_bound // 2),
             reads=(4, 5, 8), writes=(7, 8),
             op="model.byte.decoder.block_diffusion", cost_class="compute")
    if spec.speculation is not None:
        emit(Opcode.BARRIER, Lane.H, StrideClass.SCALAR,
             count=_checked_mul(batch, spec.speculation.draft_bytes + 1),
             reads=(1, 7), writes=(9,),
             op="model.byte.verify.exact_greedy", cost_class="latency")

    chosen_target = target or TargetProfile.for_host()
    realization = optimize(module, chosen_target, Theta.cool())
    pack = hydrate_pipelined(module, realization,
                             plan=f"byte-latent:{spec.digest[:16]}", depth=2)
    diagnostics = verify_all(module, result=realization, pack=pack)
    diagnostics += verify_smart_lowering(module, pack=pack)
    if diagnostics:
        message = "; ".join(f"{row.law}: {row.message}" for row in diagnostics)
        raise ValueError("byte-model lowering failed verification: " + message)
    data = encode(pack)
    artifact = _digest({"spec": spec.digest, "layout": layout.digest,
                        "report": report.digest, "module": hash_module(module),
                        "streampack": hashlib.sha256(data).hexdigest()})
    return LoweredByteLatentPlan(report, module, realization, pack, data, artifact)


__all__ = [
    "ByteDiffusionSpec", "ByteIngestDecision", "ByteIngestProfile",
    "ByteLatentCostReport", "ByteLatentSpec", "BytePatchSpec",
    "ByteSpeculationSpec", "ByteVocabularySpec", "ByteifiedTransplantPlan",
    "DraftVerification", "LearnedPatchPolicySpec", "LoweredByteLatentPlan",
    "MambaByteCostReport", "MambaByteSpec", "PatchLayout", "RawByteSequence",
    "TensorShape", "UnicodeByteReference", "assess_byte_latent", "assess_mambabyte",
    "choose_byte_ingest", "decode_raw_bytes", "discounted_batch_advantages",
    "encode_raw_sequence", "lower_byte_latent", "plan_byteified_transplant",
    "verify_byte_draft",
]
