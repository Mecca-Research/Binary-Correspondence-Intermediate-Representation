"""Bounded PyTorch references for BCIR adaptive-transformer research contracts.

The classes here are opt-in hosted oracles.  They are intentionally small and readable;
they are not production kernels, BCIRQ8-compatible decoder replacements, or performance
evidence.  The dependency-free shapes, costs, and verified StreamPack lowering live in
``bcir.kbcir.adaptive_transformer``.
"""

from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
    raise ModuleNotFoundError(
        "adaptive transformer probes require PyTorch; install bcir[model-lab]"
    ) from exc

from ...kbcir.adaptive_transformer import (
    AdaptiveLanguageSpec,
    ExogenousAnchorSpec,
    MultiPatchSpec,
    assess_adaptive_language,
    assess_multi_patch,
)
from .contracts import StageRunReport, StageTrainSpec, canonical_json, sha256_text
from .stages import _deterministic_stage, _emit, _module_digest, _optimizer

_MAX_HOSTED_CONTEXT = 4096
_MAX_HOSTED_PARAMETER_ELEMENTS = 50_000_000
_MAX_HOSTED_ACTIVATION_ELEMENTS = 1 << 24


class _RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = float(epsilon)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        dtype = values.dtype
        stable = values.float()
        scale = torch.rsqrt(stable.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return (stable * scale).to(dtype) * self.weight


class _HeadRMSNorm(nn.Module):
    def __init__(self, heads: int, head_dim: int, epsilon: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, head_dim))
        self.epsilon = float(epsilon)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if (
            values.ndim != 4
            or values.shape[1] != self.weight.shape[0]
            or values.shape[-1] != self.weight.shape[1]
        ):
            raise ValueError("head RMSNorm received an incompatible tensor")
        dtype = values.dtype
        stable = values.float()
        scale = torch.rsqrt(stable.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return (stable * scale).to(dtype) * self.weight[None, :, None, :]


def _rope(length: int, head_dim: int, device, dtype):
    half = head_dim // 2
    positions = torch.arange(length, device=device, dtype=torch.float32)
    frequencies = 10000.0 ** (
        -2.0 * torch.arange(half, device=device, dtype=torch.float32) / head_dim
    )
    angles = positions[:, None] * frequencies[None, :]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _apply_rope(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = values.shape[-1] // 2
    left, right = values[..., :half], values[..., half:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


def _split_heads(values: torch.Tensor, heads: int) -> torch.Tensor:
    batch, length, width = values.shape
    return values.view(batch, length, heads, width // heads).transpose(1, 2)


def _merge_heads(values: torch.Tensor) -> torch.Tensor:
    batch, heads, length, head_dim = values.shape
    return values.transpose(1, 2).contiguous().view(batch, length, heads * head_dim)


class _AnchorBank(nn.Module):
    """Four dedicated projections computed once from H0 or H1."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        head_dim = width // heads
        self.projections = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(4))
        self.norms = nn.ModuleList(_HeadRMSNorm(heads, head_dim) for _ in range(4))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            norm(_split_heads(projection(values), self.heads))
            for projection, norm in zip(self.projections, self.norms)
        )


class _ProjectionMix(nn.Module):
    """Static or context-modulated ExoFormer coefficients for one physical block."""

    def __init__(self, width: int, heads: int, spec: ExogenousAnchorSpec):
        super().__init__()
        self.width = width
        self.heads = heads
        self.spec = spec
        channels = spec.coefficient_channels(width, heads)
        self.base = nn.Parameter(torch.full((4, 2, channels), spec.initial_base_coefficient))
        if spec.mode == "dynamic":
            self.dynamic_in = nn.Linear(width, spec.dynamic_hidden, bias=False)
            self.dynamic_out = nn.Linear(spec.dynamic_hidden, 8, bias=True)
            nn.init.zeros_(self.dynamic_out.weight)
            nn.init.zeros_(self.dynamic_out.bias)
        else:
            self.dynamic_in = None
            self.dynamic_out = None

    def _base(self, component: int, source: int) -> torch.Tensor:
        value = self.base[component, source]
        if self.spec.granularity == "scalar":
            return value.view(1, 1, 1, 1)
        if self.spec.granularity == "headwise":
            return value.view(1, self.heads, 1, 1)
        return value.view(1, self.heads, 1, self.width // self.heads)

    def forward(
        self, component: int, anchor: torch.Tensor, current: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        if not 0 <= component < 4 or anchor.shape != current.shape:
            raise ValueError("projection mixing received incompatible sources")
        left = self._base(component, 0)
        right = self._base(component, 1)
        if self.dynamic_in is not None:
            gamma = torch.sigmoid(self.dynamic_out(F.gelu(self.dynamic_in(context))))
            left = left * gamma[..., 2 * component].unsqueeze(1).unsqueeze(-1)
            right = right * gamma[..., 2 * component + 1].unsqueeze(1).unsqueeze(-1)
        return left * anchor + right * current


class _AdaptiveAttention(nn.Module):
    def __init__(self, width: int, heads: int, *, gated: bool):
        super().__init__()
        self.width = width
        self.heads = heads
        head_dim = width // heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.g_proj = nn.Linear(width, width, bias=False) if gated else None
        self.o_proj = nn.Linear(width, width, bias=False)
        self.q_norm = _HeadRMSNorm(heads, head_dim)
        self.k_norm = _HeadRMSNorm(heads, head_dim)

    def forward(
        self,
        values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
        anchors: tuple[torch.Tensor, ...] | None,
        mixing: _ProjectionMix | None,
    ) -> torch.Tensor:
        current = [
            _split_heads(self.q_proj(values), self.heads),
            _split_heads(self.k_proj(values), self.heads),
            _split_heads(self.v_proj(values), self.heads),
        ]
        if self.g_proj is not None:
            current.append(_split_heads(self.g_proj(values), self.heads))
        if anchors is not None:
            if mixing is None or len(current) != 4:
                raise ValueError("anchor attention requires gated projection mixing")
            current = [
                mixing(index, anchors[index], tensor, values)
                for index, tensor in enumerate(current)
            ]
        q = _apply_rope(self.q_norm(current[0]), cos, sin)
        k = _apply_rope(self.k_norm(current[1]), cos, sin)
        context = F.scaled_dot_product_attention(
            q, k, current[2], attn_mask=mask, dropout_p=0.0, is_causal=mask is None
        )
        merged = _merge_heads(context)
        if self.g_proj is not None:
            merged = merged * torch.sigmoid(_merge_heads(current[3]))
        return self.o_proj(merged)


class _SwiGLU(nn.Module):
    def __init__(self, width: int, ff_width: int):
        super().__init__()
        self.gate = nn.Linear(width, ff_width, bias=False)
        self.up = nn.Linear(width, ff_width, bias=False)
        self.down = nn.Linear(ff_width, width, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(values)) * self.up(values))


class _AdaptiveBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_width: int, architecture: AdaptiveLanguageSpec):
        super().__init__()
        self.residual_mode = architecture.residual_mode
        self.alpha = architecture.loop_alpha
        self.attention = _AdaptiveAttention(width, heads, gated=architecture.gated_attention)
        self.mlp = _SwiGLU(width, ff_width)
        self.attention_inner = _RMSNorm(width)
        self.mlp_inner = _RMSNorm(width)
        if self.residual_mode == "loop_deepnorm":
            self.attention_post = _RMSNorm(width)
            self.mlp_post = _RMSNorm(width)
        else:
            self.attention_post = None
            self.mlp_post = None

    def forward(
        self,
        values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None,
        anchors: tuple[torch.Tensor, ...] | None,
        mixing: _ProjectionMix | None,
    ) -> torch.Tensor:
        attention = self.attention(self.attention_inner(values), cos, sin, mask, anchors, mixing)
        if self.residual_mode == "pre_norm":
            values = values + attention
            return values + self.mlp(self.mlp_inner(values))
        values = self.attention_post(self.alpha * values + attention)
        return self.mlp_post(self.alpha * values + self.mlp(self.mlp_inner(values)))


class HostedAdaptiveLanguageModel(nn.Module):
    """Tiny tied-depth/variable-width/R-SWA/ExoFormer research reference."""

    def __init__(self, spec: AdaptiveLanguageSpec):
        super().__init__()
        if not isinstance(spec, AdaptiveLanguageSpec):
            raise ValueError("spec must be AdaptiveLanguageSpec")
        if spec.context_length > _MAX_HOSTED_CONTEXT:
            raise ValueError("hosted adaptive context exceeds the bounded reference limit")
        if (
            assess_adaptive_language(spec, prefill_tokens=1).parameter_elements
            > _MAX_HOSTED_PARAMETER_ELEMENTS
        ):
            raise ValueError("hosted adaptive model exceeds the bounded parameter limit")
        self.spec = spec
        schedule = spec.schedule
        self.residual_width = max(schedule.widths)
        self.embed_tokens = nn.Embedding(spec.vocab_size, schedule.widths[0])
        self.blocks = nn.ModuleList(
            _AdaptiveBlock(width, heads, ff_width, spec)
            for width, heads, ff_width in zip(schedule.widths, schedule.heads, schedule.ff_widths)
        )
        if spec.anchor is not None:
            width, heads = schedule.widths[0], schedule.heads[0]
            self.anchor_bank = _AnchorBank(width, heads)
            self.mixers = nn.ModuleList(
                _ProjectionMix(width, heads, spec.anchor) for _ in range(schedule.physical_layers)
            )
        else:
            self.anchor_bank = None
            self.mixers = None
        self.final_norm = _RMSNorm(schedule.widths[-1])
        self.lm_head = nn.Linear(schedule.widths[-1], spec.vocab_size, bias=False)
        self.apply(self._initialize)
        if spec.anchor is not None and spec.anchor.mode == "dynamic":
            # ``apply`` initialized every Linear, so restore the paper's identity start.
            for mixer in self.mixers:
                nn.init.zeros_(mixer.dynamic_out.weight)
                nn.init.zeros_(mixer.dynamic_out.bias)
        if spec.residual_mode == "loop_deepnorm":
            with torch.no_grad():
                for block in self.blocks:
                    block.attention.v_proj.weight.mul_(spec.loop_beta)
                    block.attention.o_proj.weight.mul_(spec.loop_beta)
                    block.mlp.gate.weight.mul_(spec.loop_beta)
                    block.mlp.up.weight.mul_(spec.loop_beta)
                    block.mlp.down.weight.mul_(spec.loop_beta)
        if spec.tied_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _mask(self, length: int, device, dtype) -> torch.Tensor | None:
        if self.spec.reference_window is None:
            return None
        return torch.tensor(
            self.spec.reference_window.additive_mask(length), device=device, dtype=dtype
        ).view(length, length)

    def _validate_ids(self, input_ids: torch.Tensor) -> None:
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.dtype != torch.long
            or input_ids.shape[0] < 1
            or input_ids.shape[1] < 1
            or input_ids.shape[1] > self.spec.context_length
        ):
            raise ValueError("input_ids must be nonempty rank-2 torch.long within context")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.spec.vocab_size):
            raise ValueError("input token id is outside the vocabulary")
        if input_ids.numel() * self.residual_width > _MAX_HOSTED_ACTIVATION_ELEMENTS:
            raise ValueError("hosted adaptive activation exceeds the bounded reference limit")

    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_ids(input_ids)
        values = self.embed_tokens(input_ids)
        if values.shape[-1] < self.residual_width:
            values = F.pad(values, (0, self.residual_width - values.shape[-1]))
        mask = self._mask(input_ids.shape[1], values.device, values.dtype)
        anchors = (
            self.anchor_bank(values[..., : self.spec.schedule.widths[0]])
            if self.spec.anchor is not None and self.spec.anchor.source_depth == 0
            else None
        )
        rope_cache = {}
        effective = 0
        for _repeat in range(self.spec.repeats):
            for physical, (block, width, heads) in enumerate(
                zip(self.blocks, self.spec.schedule.widths, self.spec.schedule.heads)
            ):
                head_dim = width // heads
                if head_dim not in rope_cache:
                    rope_cache[head_dim] = _rope(
                        input_ids.shape[1], head_dim, values.device, values.dtype
                    )
                active = values[..., :width]
                use_anchor = (
                    anchors
                    if self.spec.anchor is not None and effective >= self.spec.anchor.source_depth
                    else None
                )
                mixing = self.mixers[physical] if use_anchor is not None else None
                active = block(active, *rope_cache[head_dim], mask, use_anchor, mixing)
                values = torch.cat((active, values[..., width:]), dim=-1)
                effective += 1
                if self.spec.anchor is not None and self.spec.anchor.source_depth == effective:
                    anchors = self.anchor_bank(values[..., : self.spec.schedule.widths[0]])
        return self.final_norm(values[..., : self.spec.schedule.widths[-1]])

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if targets is not None and (
            not isinstance(targets, torch.Tensor)
            or targets.shape != input_ids.shape
            or targets.dtype != torch.long
            or targets.device != input_ids.device
        ):
            raise ValueError("targets must be torch.long with the input shape and device")
        if targets is not None and (
            torch.any(targets < 0) or torch.any(targets >= self.spec.vocab_size)
        ):
            raise ValueError("target token id is outside the vocabulary")
        logits = F.linear(self.hidden_states(input_ids), self.lm_head.weight)
        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.float().reshape(-1, self.spec.vocab_size), targets.reshape(-1)
            )
        )
        return logits, loss

    def unique_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@torch.no_grad()
def _language_loss(
    model: HostedAdaptiveLanguageModel, input_ids: torch.Tensor, targets: torch.Tensor
) -> float:
    was_training = model.training
    model.eval()
    try:
        _logits, loss = model(input_ids, targets)
        value = float(loss.item())
    finally:
        model.train(was_training)
    if not math.isfinite(value):
        raise RuntimeError("adaptive language evaluation produced non-finite loss")
    return value


def train_adaptive_pretrain(
    model: HostedAdaptiveLanguageModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    spec: StageTrainSpec,
    telemetry_sink=None,
) -> StageRunReport:
    """Run a bounded repeated-batch pretraining confirmation."""
    if (
        not isinstance(model, HostedAdaptiveLanguageModel)
        or not isinstance(spec, StageTrainSpec)
        or spec.stage != "pretrain"
    ):
        raise ValueError("adaptive pretraining requires its model and stage='pretrain'")
    if (
        not isinstance(input_ids, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or input_ids.shape != targets.shape
        or input_ids.ndim != 2
        or input_ids.shape[0] < 1
        or input_ids.device != targets.device
        or input_ids.device != next(model.parameters()).device
    ):
        raise ValueError("adaptive pretraining tensors are malformed or on the wrong device")
    model._validate_ids(input_ids)
    if (
        targets.dtype != torch.long
        or torch.any(targets < 0)
        or torch.any(targets >= model.spec.vocab_size)
    ):
        raise ValueError("adaptive targets must be in-vocabulary torch.long")
    if telemetry_sink is not None and not callable(telemetry_sink):
        raise ValueError("telemetry_sink must be callable or None")
    input_sha = sha256_text(
        canonical_json(
            {
                "inputs": input_ids.detach().cpu().tolist(),
                "targets": targets.detach().cpu().tolist(),
            }
        )
    )
    with _deterministic_stage(spec.seed):
        initial = _language_loss(model, input_ids, targets)
        optimizer = _optimizer(spec, model.parameters())
        model.train()
        for step in range(spec.steps):
            optimizer.zero_grad(set_to_none=True)
            _logits, loss = model(input_ids, targets)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("adaptive language model produced non-finite gradients")
            optimizer.step()
            _emit(
                telemetry_sink,
                "pretrain",
                step + 1,
                float(loss.detach().item()),
                float(norm.item()),
                (step + 1) * input_ids.shape[0],
            )
        final = _language_loss(model, input_ids, targets)
        return StageRunReport(
            "pretrain",
            spec.digest,
            input_sha,
            initial,
            final,
            spec.steps,
            input_ids.shape[0],
            _module_digest(model),
            model.spec.digest,
        )


class _PatchBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1.0e-6)
        self.norm2 = nn.LayerNorm(width, eps=1.0e-6)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"), nn.Linear(4 * width, width)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(values)
        attention = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        values = values + attention
        return values + self.mlp(self.norm2(values))


def _position_2d(side: int, width: int) -> torch.Tensor:
    quarter = width // 4
    positions = torch.arange(side, dtype=torch.float32)
    frequencies = 10000.0 ** (-torch.arange(quarter, dtype=torch.float32) / quarter)
    x, y = torch.meshgrid(positions, positions, indexing="ij")
    x_angles = x.reshape(-1, 1) * frequencies.reshape(1, -1)
    y_angles = y.reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat(
        (x_angles.sin(), x_angles.cos(), y_angles.sin(), y_angles.cos()), dim=1
    ).unsqueeze(0)


class HostedMultiPatchTransformer(nn.Module):
    """Tiny coarse-to-fine image transformer with carried condition tokens."""

    def __init__(self, spec: MultiPatchSpec):
        super().__init__()
        if not isinstance(spec, MultiPatchSpec):
            raise ValueError("spec must be MultiPatchSpec")
        if assess_multi_patch(spec).parameter_elements > _MAX_HOSTED_PARAMETER_ELEMENTS:
            raise ValueError("hosted multi-patch model exceeds the bounded parameter limit")
        self.spec = spec
        width = spec.hidden_width
        self.coarse_embed = nn.Conv2d(
            spec.input_channels, width, spec.coarse_patch, spec.coarse_patch
        )
        self.fine_skip = nn.Conv2d(spec.input_channels, width, spec.fine_patch, spec.fine_patch)
        self.condition = nn.Parameter(torch.zeros(1, spec.condition_tokens, width))
        self.coarse_blocks = nn.ModuleList(
            _PatchBlock(width, spec.heads) for _ in range(spec.coarse_layers)
        )
        self.upsample = nn.Linear(width, width * spec.upsample_factor**2)
        self.upsample_norm = nn.LayerNorm(width, eps=1.0e-6)
        self.upsample_refine = nn.Linear(width, width)
        self.fine_blocks = nn.ModuleList(
            _PatchBlock(width, spec.heads) for _ in range(spec.fine_layers)
        )
        self.output = nn.Linear(width, spec.fine_patch**2 * spec.input_channels)
        self.register_buffer(
            "coarse_position",
            _position_2d(spec.image_size // spec.coarse_patch, width),
            persistent=False,
        )
        self.register_buffer(
            "fine_position",
            _position_2d(spec.image_size // spec.fine_patch, width),
            persistent=False,
        )

    def _upsample(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = values.shape
        side = int(math.isqrt(tokens))
        if side * side != tokens:
            raise ValueError("coarse spatial token count must be a square")
        factor = self.spec.upsample_factor
        values = self.upsample(values).view(batch, side, side, width, factor, factor)
        values = (
            values.permute(0, 1, 4, 2, 5, 3)
            .contiguous()
            .view(batch, side * factor * side * factor, width)
        )
        return self.upsample_refine(self.upsample_norm(F.gelu(values)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        spec = self.spec
        if (
            not isinstance(images, torch.Tensor)
            or images.ndim != 4
            or images.shape[0] < 1
            or images.shape[1:] != (spec.input_channels, spec.image_size, spec.image_size)
            or not images.is_floating_point()
            or not torch.isfinite(images).all()
        ):
            raise ValueError("multi-patch images must be finite [batch,C,H,W] floats")
        if (
            images.numel() > _MAX_HOSTED_ACTIVATION_ELEMENTS
            or images.shape[0] * spec.fine_tokens * spec.hidden_width
            > _MAX_HOSTED_ACTIVATION_ELEMENTS
        ):
            raise ValueError("hosted multi-patch activation exceeds the bounded reference limit")
        coarse = self.coarse_embed(images).flatten(2).transpose(1, 2)
        coarse = coarse + self.coarse_position.to(coarse.dtype)
        condition = self.condition.expand(images.shape[0], -1, -1)
        values = torch.cat((condition, coarse), dim=1)
        for block in self.coarse_blocks:
            values = block(values)
        condition, coarse = values[:, : spec.condition_tokens], values[:, spec.condition_tokens :]
        fine = self._upsample(coarse)
        skip = self.fine_skip(images).flatten(2).transpose(1, 2)
        fine = fine + skip + self.fine_position.to(fine.dtype)
        values = torch.cat((condition, fine), dim=1)
        for block in self.fine_blocks:
            values = block(values)
        patches = self.output(values[:, spec.condition_tokens :])
        side = spec.image_size // spec.fine_patch
        patch = spec.fine_patch
        patches = patches.view(images.shape[0], side, side, patch, patch, spec.input_channels)
        return patches.permute(0, 5, 1, 3, 2, 4).contiguous().view_as(images)


@torch.no_grad()
def _reconstruction_loss(model: HostedMultiPatchTransformer, images: torch.Tensor) -> float:
    was_training = model.training
    model.eval()
    try:
        value = float(F.mse_loss(model(images).float(), images.float()).item())
    finally:
        model.train(was_training)
    if not math.isfinite(value):
        raise RuntimeError("multi-patch evaluation produced non-finite loss")
    return value


def train_multipatch_reconstruction(
    model: HostedMultiPatchTransformer,
    images: torch.Tensor,
    spec: StageTrainSpec,
    telemetry_sink=None,
) -> StageRunReport:
    if (
        not isinstance(model, HostedMultiPatchTransformer)
        or not isinstance(spec, StageTrainSpec)
        or spec.stage != "supervised"
    ):
        raise ValueError("multi-patch training requires its model and stage='supervised'")
    if not isinstance(images, torch.Tensor) or images.device != next(model.parameters()).device:
        raise ValueError("multi-patch images must share the model device")
    if telemetry_sink is not None and not callable(telemetry_sink):
        raise ValueError("telemetry_sink must be callable or None")
    # Forward validation precedes optimizer construction and mutation.
    initial = _reconstruction_loss(model, images)
    input_sha = sha256_text(canonical_json(images.detach().cpu().tolist()))
    with _deterministic_stage(spec.seed):
        optimizer = _optimizer(spec, model.parameters())
        model.train()
        for step in range(spec.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(images).float(), images.float())
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("multi-patch model produced non-finite gradients")
            optimizer.step()
            _emit(
                telemetry_sink,
                "supervised",
                step + 1,
                float(loss.detach().item()),
                float(norm.item()),
                (step + 1) * images.shape[0],
            )
        final = _reconstruction_loss(model, images)
        return StageRunReport(
            "supervised",
            spec.digest,
            input_sha,
            initial,
            final,
            spec.steps,
            images.shape[0],
            _module_digest(model),
            model.spec.digest,
        )


__all__ = [
    "HostedAdaptiveLanguageModel",
    "HostedMultiPatchTransformer",
    "train_adaptive_pretrain",
    "train_multipatch_reconstruction",
]
