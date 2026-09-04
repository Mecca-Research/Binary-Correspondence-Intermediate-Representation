"""Bounded PyTorch references for BCIR byte-native model contracts.

This opt-in module is an architectural and training oracle, not a production inference
engine.  It operates on direct byte/control ids, implements a causal local/global/local
Byte Latent Transformer, shared block-denoising for BLT-D, exact greedy verification for
BLT-S/BLT-DV, a selective diagonal SSM for MambaByte experiments, score-function patch
training, and shape-checked global-weight transplantation.

The dependency-free contracts, exact size accounting, and verified StreamPack lowering
live in :mod:`bcir.kbcir.byte_latent`.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
    raise ModuleNotFoundError(
        "byte-native model probes require PyTorch; install bcir[model-lab]"
    ) from exc

from ...kbcir.byte_latent import (
    ByteLatentSpec,
    ByteifiedTransplantPlan,
    MambaByteSpec,
    PatchLayout,
    TensorShape,
    assess_byte_latent,
    assess_mambabyte,
    discounted_batch_advantages,
    verify_byte_draft,
)
from ..models.checkpoint import tensor_mapping_digest
from .contracts import StageRunReport, StageTrainSpec, canonical_json, sha256_text
from .stages import _deterministic_stage, _emit, _module_digest, _optimizer

_MAX_HOSTED_CONTEXT = 4096
_MAX_HOSTED_BATCH = 256
_MAX_HOSTED_PARAMETER_ELEMENTS = 50_000_000
_MAX_HOSTED_ACTIVATION_ELEMENTS = 1 << 24
_MAX_HOSTED_TENSOR_INVENTORY = 65536
_Q20 = 1 << 20


class _RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = float(epsilon)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        stable = values.float()
        scale = torch.rsqrt(stable.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return (stable * scale).to(values.dtype) * self.weight


def _split_heads(values: torch.Tensor, heads: int) -> torch.Tensor:
    batch, length, width = values.shape
    return values.view(batch, length, heads, width // heads).transpose(1, 2)


def _merge_heads(values: torch.Tensor) -> torch.Tensor:
    batch, heads, length, width = values.shape
    return values.transpose(1, 2).contiguous().view(batch, length, heads * width)


class _SelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, rope_base: int):
        super().__init__()
        self.heads = heads
        self.rope_base = rope_base
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def _rope(self, values: torch.Tensor) -> torch.Tensor:
        half = values.shape[-1] // 2
        positions = torch.arange(values.shape[-2], dtype=torch.float32, device=values.device)
        frequency = torch.arange(half, dtype=torch.float32, device=values.device)
        frequency = torch.pow(float(self.rope_base), -frequency / half)
        angle = positions[:, None] * frequency[None, :]
        cosine = torch.cos(angle)[None, None].to(values.dtype)
        sine = torch.sin(angle)[None, None].to(values.dtype)
        first, second = values[..., :half], values[..., half:]
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)

    def forward(
        self, values: torch.Tensor, *, causal: bool, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        q = self._rope(_split_heads(self.q_proj(values), self.heads))
        k = self._rope(_split_heads(self.k_proj(values), self.heads))
        v = _split_heads(self.v_proj(values), self.heads)
        if mask is not None:
            if mask.ndim != 2 or tuple(mask.shape) != (values.shape[1], values.shape[1]):
                raise ValueError("attention mask has the wrong shape")
            mask = mask[None, None]
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=causal and mask is None
        )
        return self.o_proj(_merge_heads(result))


class _SwiGLU(nn.Module):
    def __init__(self, width: int, ff_width: int):
        super().__init__()
        self.gate = nn.Linear(width, ff_width, bias=False)
        self.up = nn.Linear(width, ff_width, bias=False)
        self.down = nn.Linear(ff_width, width, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(values)) * self.up(values))


class _TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_width: int, *, rope_base: int, epsilon: float):
        super().__init__()
        self.attention_norm = _RMSNorm(width, epsilon)
        self.attention = _SelfAttention(width, heads, rope_base)
        self.mlp_norm = _RMSNorm(width, epsilon)
        self.mlp = _SwiGLU(width, ff_width)

    def forward(
        self, values: torch.Tensor, *, causal: bool, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = values + self.attention(self.attention_norm(values), causal=causal, mask=mask)
        return values + self.mlp(self.mlp_norm(values))


class _SelectiveSSMBlock(nn.Module):
    """Readable diagonal selective recurrence; production needs a fused scan kernel."""

    def __init__(
        self, width: int, state_size: int, expansion: int, conv_width: int, *, epsilon: float
    ):
        super().__init__()
        inner = width * expansion
        self.inner = inner
        self.state_size = state_size
        self.conv_width = conv_width
        self.norm = _RMSNorm(width, epsilon)
        self.in_proj = nn.Linear(width, 2 * inner, bias=False)
        self.conv = nn.Conv1d(
            inner, inner, conv_width, groups=inner, bias=False, padding=conv_width - 1
        )
        self.dt_proj = nn.Linear(inner, inner, bias=True)
        self.b_proj = nn.Linear(inner, state_size, bias=False)
        self.c_proj = nn.Linear(inner, state_size, bias=False)
        self.a_log = nn.Parameter(torch.zeros(inner, state_size))
        self.direct = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, width, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected, gate = self.in_proj(self.norm(values)).chunk(2, dim=-1)
        convolved = self.conv(projected.transpose(1, 2))[..., : values.shape[1]]
        convolved = F.silu(convolved.transpose(1, 2))
        dt = F.softplus(self.dt_proj(convolved))
        b_values = torch.tanh(self.b_proj(convolved))
        c_values = torch.tanh(self.c_proj(convolved))
        a = -torch.exp(self.a_log.float()).to(values.dtype)
        state = torch.zeros(
            values.shape[0], self.inner, self.state_size, dtype=values.dtype, device=values.device
        )
        outputs = []
        for index in range(values.shape[1]):
            step = dt[:, index, :, None]
            state = (
                torch.exp(a[None] * step) * state
                + step * convolved[:, index, :, None] * b_values[:, index, None, :]
            )
            output = (state * c_values[:, index, None, :]).sum(dim=-1)
            output = output + self.direct * convolved[:, index]
            outputs.append(output * F.silu(gate[:, index]))
        return values + self.out_proj(torch.stack(outputs, dim=1))


def _initialize(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


@dataclass(frozen=True)
class ByteLatentLossReport:
    total: torch.Tensor
    autoregressive: torch.Tensor
    diffusion: torch.Tensor
    policy: torch.Tensor
    target_rate: torch.Tensor
    early_exit: torch.Tensor
    layouts: tuple[PatchLayout, ...]


class HostedByteLatentModel(nn.Module):
    """Small local-byte/global-patch/local-byte Transformer reference."""

    def __init__(self, spec: ByteLatentSpec):
        super().__init__()
        if not isinstance(spec, ByteLatentSpec):
            raise ValueError("spec must be ByteLatentSpec")
        if spec.context_bytes > _MAX_HOSTED_CONTEXT:
            raise ValueError("hosted byte context exceeds the bounded reference limit")
        probe = assess_byte_latent(spec, byte_count=1, patch_count=1)
        if probe.parameter_elements > _MAX_HOSTED_PARAMETER_ELEMENTS:
            raise ValueError("hosted byte model exceeds the bounded parameter limit")
        self.spec = spec
        local, global_ = spec.local_width, spec.global_width
        self.embed_bytes = nn.Embedding(spec.vocabulary.size, local)
        self.ngram_embeddings = nn.ModuleList(
            nn.Embedding(spec.hash_buckets, local) for _ in spec.hash_ngram_sizes
        )
        if spec.local_backbone == "transformer":
            self.local_encoder = nn.ModuleList(
                _TransformerBlock(
                    local,
                    spec.local_heads,
                    spec.local_ff_width,
                    rope_base=spec.rope_base,
                    epsilon=spec.rms_norm_epsilon,
                )
                for _ in range(spec.local_encoder_layers)
            )
        else:
            self.local_encoder = nn.ModuleList(
                _SelectiveSSMBlock(
                    local,
                    spec.ssm_state_size,
                    spec.ssm_expansion,
                    spec.ssm_conv_width,
                    epsilon=spec.rms_norm_epsilon,
                )
                for _ in range(spec.local_encoder_layers)
            )
        self.pool_q = nn.Linear(local, global_, bias=False)
        self.pool_k = nn.Linear(local, global_, bias=False)
        self.pool_v = nn.Linear(local, global_, bias=False)
        self.pool_o = nn.Linear(global_, global_, bias=False)
        self.global_blocks = nn.ModuleList(
            _TransformerBlock(
                global_,
                spec.global_heads,
                spec.global_ff_width,
                rope_base=spec.rope_base,
                epsilon=spec.rms_norm_epsilon,
            )
            for _ in range(spec.global_layers)
        )
        self.bos_patch = nn.Parameter(torch.zeros(global_))
        self.decoder_q = nn.Linear(local, local, bias=False)
        self.decoder_k = nn.Linear(global_, local, bias=False)
        self.decoder_v = nn.Linear(global_, local, bias=False)
        self.decoder_o = nn.Linear(local, local, bias=False)
        self.local_decoder = nn.ModuleList(
            _TransformerBlock(
                local,
                spec.local_heads,
                spec.local_ff_width,
                rope_base=spec.rope_base,
                epsilon=spec.rms_norm_epsilon,
            )
            for _ in range(spec.local_decoder_layers)
        )
        self.final_norm = _RMSNorm(local, spec.rms_norm_epsilon)
        self.lm_head = nn.Linear(local, spec.vocabulary.size, bias=False)
        if spec.patcher.method in ("entropy_global", "entropy_monotonic", "learned"):
            self.early_head = nn.Linear(local, spec.vocabulary.size, bias=False)
        else:
            self.early_head = None
        if spec.patcher.method == "learned":
            self.boundary_weight = nn.Parameter(torch.empty(spec.patcher.learned_window + 1, local))
        else:
            self.boundary_weight = None
        self.apply(_initialize)
        if self.boundary_weight is not None:
            nn.init.normal_(self.boundary_weight, mean=0.0, std=0.02)
        if spec.tied_embeddings:
            self.lm_head.weight = self.embed_bytes.weight
        if self.early_head is not None:
            with torch.no_grad():
                self.early_head.weight.copy_(self.lm_head.weight)

    def unique_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_ids(self, input_ids: torch.Tensor) -> None:
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.dtype != torch.long
            or input_ids.shape[0] < 1
            or input_ids.shape[0] > _MAX_HOSTED_BATCH
            or input_ids.shape[1] < 1
            or input_ids.shape[1] > self.spec.context_bytes
        ):
            raise ValueError("byte ids must be nonempty rank-2 torch.long within context")
        if (
            input_ids.numel() * max(self.spec.local_width, self.spec.global_width)
            > _MAX_HOSTED_ACTIVATION_ELEMENTS
        ):
            raise ValueError("hosted byte activation exceeds the bounded reference limit")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.spec.vocabulary.size):
            raise ValueError("byte/control id is outside the vocabulary")

    def _local_mask(self, length: int, device, dtype) -> torch.Tensor:
        row = torch.arange(length, device=device)[:, None]
        column = torch.arange(length, device=device)[None, :]
        visible = (column <= row) & (column >= row - self.spec.local_window_bytes + 1)
        result = torch.full((length, length), float("-inf"), device=device, dtype=dtype)
        return result.masked_fill(visible, 0.0)

    @staticmethod
    def _hash_ngram(values: Sequence[int], buckets: int) -> int:
        result = 2166136261
        for value in values:
            result = ((result ^ value) * 16777619) & 0xFFFFFFFF
        return result % buckets

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        values = self.embed_bytes(input_ids)
        if not self.ngram_embeddings:
            return values
        batch, length = input_ids.shape
        ids = input_ids.detach().cpu().tolist()
        count = torch.ones(batch, length, 1, dtype=values.dtype, device=values.device)
        for size, embedding in zip(self.spec.hash_ngram_sizes, self.ngram_embeddings):
            indices = []
            active = []
            for row in ids:
                row_indices = []
                row_active = []
                for index in range(length):
                    if index + 1 < size:
                        row_indices.append(0)
                        row_active.append(0)
                    else:
                        row_indices.append(
                            self._hash_ngram(
                                row[index + 1 - size : index + 1], self.spec.hash_buckets
                            )
                        )
                        row_active.append(1)
                indices.append(row_indices)
                active.append(row_active)
            index_tensor = torch.tensor(indices, dtype=torch.long, device=values.device)
            active_tensor = torch.tensor(active, dtype=values.dtype, device=values.device)[
                ..., None
            ]
            values = values + embedding(index_tensor) * active_tensor
            count = count + active_tensor
        return values / count

    def local_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_ids(input_ids)
        values = self._embed(input_ids)
        if self.spec.local_backbone == "transformer":
            mask = self._local_mask(values.shape[1], values.device, values.dtype)
            for block in self.local_encoder:
                values = block(values, causal=False, mask=mask)
        else:
            for block in self.local_encoder:
                values = block(values)
        return values

    def _entropy_layouts(self, hidden: torch.Tensor) -> tuple[PatchLayout, ...]:
        if self.early_head is None:
            raise ValueError("entropy patching requires an early byte head")
        with torch.no_grad():
            probabilities = F.softmax(self.early_head(hidden).float(), dim=-1)
            entropies = -(probabilities * torch.log2(probabilities.clamp_min(1e-30))).sum(-1)
            # A start at byte i must depend only on bytes <i.  The early head at row
            # i-1 predicts the distribution for byte i; byte zero is already a forced start.
            entropies = torch.cat((torch.zeros_like(entropies[:, :1]), entropies[:, :-1]), dim=1)
        return tuple(
            self.spec.patcher.layout(
                hidden.shape[1],
                entropy_millibits=tuple(int(value * 1000.0 + 0.5) for value in row.tolist()),
            )
            for row in entropies.cpu()
        )

    def _learned_layouts(
        self, hidden: torch.Tensor, *, sample: bool, seed: int
    ) -> tuple[tuple[PatchLayout, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.boundary_weight is None or self.spec.learned_policy is None:
            raise ValueError("learned layouts require a boundary policy")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        target = self.spec.learned_policy.target_boundary_q20 / _Q20
        bias = math.log(target / (1.0 - target))
        layouts = []
        all_logits = []
        all_samples = []
        all_masks = []
        for batch in range(hidden.shape[0]):
            decisions = [1]
            scores = [_Q20]
            logits = [hidden[batch, 0].sum() * 0.0]
            masks = [0.0]
            last = 0
            for index in range(1, hidden.shape[1]):
                # The boundary precedes byte index, so its feature is the preceding
                # causal state rather than a state that has already observed that byte.
                boundary_hidden = hidden[batch, index - 1]
                raw = torch.dot(boundary_hidden, self.boundary_weight[0])
                for distance in range(1, self.spec.patcher.learned_window + 1):
                    if index >= distance and decisions[index - distance]:
                        raw = raw + torch.dot(boundary_hidden, self.boundary_weight[distance])
                scaled = raw / self.spec.learned_policy.logit_scale + bias
                if sample:
                    scaled = 10.0 * torch.tanh(scaled / 10.0)
                probability = torch.sigmoid(scaled)
                distance = index - last
                forced = distance >= self.spec.patcher.max_patch_bytes
                eligible = distance >= self.spec.patcher.min_patch_bytes
                if forced:
                    decision = 1
                    decision_mask = 0.0
                elif not eligible:
                    decision = 0
                    decision_mask = 0.0
                elif sample:
                    draw = float(torch.rand((), generator=generator).item())
                    decision = int(draw < float(probability.detach().cpu().item()))
                    decision_mask = 1.0
                else:
                    decision = int(
                        float(probability.detach().cpu().item())
                        >= self.spec.patcher.learned_threshold_q20 / _Q20
                    )
                    decision_mask = 1.0
                decisions.append(decision)
                scores.append(_Q20 if decision else 0)
                logits.append(scaled)
                masks.append(decision_mask)
                if decision:
                    last = index
            layouts.append(
                self.spec.patcher.layout(hidden.shape[1], learned_scores_q20=tuple(scores))
            )
            all_logits.append(torch.stack(logits))
            all_samples.append(torch.tensor(decisions, dtype=hidden.dtype, device=hidden.device))
            all_masks.append(torch.tensor(masks, dtype=hidden.dtype, device=hidden.device))
        return (
            tuple(layouts),
            torch.stack(all_logits),
            torch.stack(all_samples),
            torch.stack(all_masks),
        )

    def _resolve_layouts(
        self, hidden: torch.Tensor, layouts: Sequence[PatchLayout] | None
    ) -> tuple[PatchLayout, ...]:
        if layouts is not None:
            result = tuple(layouts)
            if len(result) != hidden.shape[0]:
                raise ValueError("explicit patch layouts are incompatible with the batch")
            for item in result:
                self.spec.patcher.validate_layout(item, byte_length=hidden.shape[1])
            return result
        if self.spec.patcher.method == "stride":
            layout = self.spec.patcher.layout(hidden.shape[1])
            return (layout,) * hidden.shape[0]
        if self.spec.patcher.method.startswith("entropy"):
            return self._entropy_layouts(hidden)
        return self._learned_layouts(hidden, sample=False, seed=0)[0]

    def _pool_one(self, hidden: torch.Tensor, layout: PatchLayout) -> torch.Tensor:
        patches = []
        for start, length in zip(layout.starts, layout.lengths):
            byte_values = hidden[start : start + length]
            query = self.pool_q(byte_values.mean(dim=0, keepdim=True))
            keys = self.pool_k(byte_values)
            values = self.pool_v(byte_values)
            scores = (query @ keys.transpose(0, 1)) / math.sqrt(self.spec.global_width)
            patches.append(self.pool_o(torch.softmax(scores, dim=-1) @ values).squeeze(0))
        result = torch.stack(patches)[None]
        for block in self.global_blocks:
            result = block(result, causal=True)
        return result.squeeze(0)

    def global_hidden(
        self, hidden: torch.Tensor, layouts: Sequence[PatchLayout]
    ) -> tuple[torch.Tensor, ...]:
        return tuple(self._pool_one(hidden[index], layout) for index, layout in enumerate(layouts))

    def _decode_one(
        self, hidden: torch.Tensor, global_values: torch.Tensor, layout: PatchLayout
    ) -> torch.Tensor:
        latent = torch.cat((self.bos_patch[None], global_values), dim=0)
        keys = self.decoder_k(latent)
        values = self.decoder_v(latent)
        queries = self.decoder_q(hidden)
        patch_for_byte = layout.byte_to_patch()
        conditioned = []
        scale = math.sqrt(self.spec.local_width)
        for index, patch in enumerate(patch_for_byte):
            # BOS plus strictly previous completed patches; never read the current patch.
            count = patch + 1
            scores = (queries[index : index + 1] @ keys[:count].transpose(0, 1)) / scale
            conditioned.append(torch.softmax(scores, dim=-1) @ values[:count])
        decoded = hidden + self.decoder_o(torch.cat(conditioned, dim=0))
        decoded = decoded[None]
        for block in self.local_decoder:
            decoded = block(decoded, causal=True)
        return self.final_norm(decoded).squeeze(0)

    def _decode_batch(
        self,
        hidden: torch.Tensor,
        global_values: Sequence[torch.Tensor],
        layouts: Sequence[PatchLayout],
    ) -> torch.Tensor:
        decoded = torch.stack(
            [
                self._decode_one(hidden[index], global_values[index], layouts[index])
                for index in range(hidden.shape[0])
            ]
        )
        return F.linear(decoded, self.lm_head.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        layouts: Sequence[PatchLayout] | None = None,
    ):
        self._validate_ids(input_ids)
        if targets is not None and (
            not isinstance(targets, torch.Tensor)
            or targets.shape != input_ids.shape
            or targets.dtype != torch.long
            or targets.device != input_ids.device
        ):
            raise ValueError("byte targets must match input shape, type, and device")
        if targets is not None and (
            torch.any(targets < 0) or torch.any(targets >= self.spec.vocabulary.size)
        ):
            raise ValueError("byte target is outside the vocabulary")
        hidden = self.local_hidden(input_ids)
        resolved = self._resolve_layouts(hidden, layouts)
        global_values = self.global_hidden(hidden, resolved)
        logits = self._decode_batch(hidden, global_values, resolved)
        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.float().reshape(-1, self.spec.vocabulary.size), targets.reshape(-1)
            )
        )
        return logits, loss

    def _diffusion_logits_from_latent(
        self, corrupted: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        if self.spec.diffusion is None:
            raise ValueError("model has no diffusion decoder contract")
        if (
            not isinstance(corrupted, torch.Tensor)
            or not isinstance(latent, torch.Tensor)
            or corrupted.ndim != 1
            or corrupted.dtype != torch.long
            or corrupted.numel() < 1
            or corrupted.numel() > self.spec.diffusion.block_size
            or torch.any(corrupted < 0)
            or torch.any(corrupted >= self.spec.vocabulary.size)
        ):
            raise ValueError("corrupted diffusion block is malformed")
        if (
            latent.ndim != 1
            or latent.shape[0] != self.spec.global_width
            or latent.device != corrupted.device
        ):
            raise ValueError("diffusion latent is malformed")
        values = self.embed_bytes(corrupted)[None]
        condition = self.decoder_o(self.decoder_v(latent[None]))[None]
        values = values + condition
        for block in self.local_decoder:
            values = block(values, causal=False)
        return F.linear(self.final_norm(values), self.lm_head.weight).squeeze(0)

    def diffusion_logits(
        self, prefix_ids: torch.Tensor, corrupted_block: torch.Tensor
    ) -> torch.Tensor:
        if self.spec.diffusion is None:
            raise ValueError("model has no diffusion decoder contract")
        if (
            not isinstance(prefix_ids, torch.Tensor)
            or not isinstance(corrupted_block, torch.Tensor)
            or prefix_ids.ndim != 1
            or prefix_ids.dtype != torch.long
            or prefix_ids.numel() < 1
        ):
            raise ValueError("diffusion prefix must be a nonempty byte-id vector")
        if (
            prefix_ids.device != corrupted_block.device
            or prefix_ids.device != next(self.parameters()).device
        ):
            raise ValueError("diffusion tensors must share the model device")
        if prefix_ids.numel() + corrupted_block.numel() > self.spec.context_bytes:
            raise ValueError("diffusion prefix and block exceed model context")
        hidden = self.local_hidden(prefix_ids[None])
        layouts = self._resolve_layouts(hidden, None)
        latent = self.global_hidden(hidden, layouts)[0][-1]
        return self._diffusion_logits_from_latent(corrupted_block, latent)

    @torch.no_grad()
    def local_draft(self, prefix: Sequence[int], count: int) -> tuple[int, ...]:
        if type(count) is not int or count < 1:
            raise ValueError("local draft requires a prompt and positive count")
        prefix = _validate_prompt(prefix, self.spec.vocabulary, self.spec.context_bytes)
        if len(prefix) + count > self.spec.context_bytes:
            raise ValueError("local draft exceeds model context")
        device = next(self.parameters()).device
        prefix_tensor = torch.tensor([tuple(prefix)], dtype=torch.long, device=device)
        hidden = self.local_hidden(prefix_tensor)
        layouts = self._resolve_layouts(hidden, None)
        fixed_latent = self.global_hidden(hidden, layouts)[0][-1]
        generated = list(prefix)
        with torch.no_grad():
            for _ in range(count):
                ids = torch.tensor([generated], dtype=torch.long, device=device)
                local = self.local_hidden(ids)
                conditioned = local + self.decoder_o(self.decoder_v(fixed_latent))[None, None]
                for block in self.local_decoder:
                    conditioned = block(conditioned, causal=True)
                logits = F.linear(self.final_norm(conditioned), self.lm_head.weight)
                generated.append(_generation_argmax(logits[0, -1], self.spec.vocabulary))
        return tuple(generated[len(prefix) :])

    def loss_components(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        *,
        seed: int,
        mask_probability_q20: int | None = None,
    ) -> ByteLatentLossReport:
        self._validate_ids(input_ids)
        if type(seed) is not int or not 0 <= seed < 1 << 63:
            raise ValueError("byte training seed must be a non-negative bounded integer")
        if mask_probability_q20 is not None and (
            type(mask_probability_q20) is not int or not 1 <= mask_probability_q20 <= _Q20
        ):
            raise ValueError("mask_probability_q20 must be in [1, 2^20] or None")
        if (
            not isinstance(targets, torch.Tensor)
            or targets.shape != input_ids.shape
            or targets.dtype != torch.long
            or targets.device != input_ids.device
            or torch.any(targets < 0)
            or torch.any(targets >= self.spec.vocabulary.size)
        ):
            raise ValueError("byte training targets are malformed")
        hidden = self.local_hidden(input_ids)
        boundary_logits = boundary_samples = boundary_mask = None
        if self.spec.patcher.method == "learned":
            layouts, boundary_logits, boundary_samples, boundary_mask = self._learned_layouts(
                hidden, sample=True, seed=seed
            )
        else:
            layouts = self._resolve_layouts(hidden, None)
        globals_ = self.global_hidden(hidden, layouts)
        logits = self._decode_batch(hidden, globals_, layouts)
        token_nll = F.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none")
        autoregressive = token_nll.mean()
        zero = autoregressive * 0.0
        early_loss = zero
        policy_loss = zero
        target_loss = zero
        total = autoregressive
        if self.early_head is not None:
            early_logits = self.early_head(hidden)
            early_nll = F.cross_entropy(
                early_logits.float().transpose(1, 2), targets, reduction="none"
            )
            early_loss = early_nll.mean()
            if self.spec.patcher.method == "learned":
                rewards = (-token_nll + early_nll).detach()
                advantages = discounted_batch_advantages(
                    rewards.cpu().tolist(), self.spec.learned_policy.discount
                )
                advantage = torch.tensor(advantages, dtype=hidden.dtype, device=hidden.device)
                log_probability = -F.binary_cross_entropy_with_logits(
                    boundary_logits, boundary_samples, reduction="none"
                )
                denominator = boundary_mask.sum().clamp_min(1.0)
                policy_loss = -(log_probability * advantage * boundary_mask).sum() / denominator
                probabilities = torch.sigmoid(boundary_logits)
                target = self.spec.learned_policy.target_boundary_q20 / _Q20
                mean_logit = (boundary_logits * boundary_mask).sum() / denominator
                mean_probability = (probabilities * boundary_mask).sum() / denominator
                target_loss = mean_logit * (mean_probability - target).detach()
                policy = self.spec.learned_policy
                total = (
                    total
                    + policy.policy_weight_q20 / _Q20 * policy_loss
                    + policy.target_weight_q20 / _Q20 * target_loss
                    + policy.early_weight_q20 / _Q20 * early_loss
                )
            else:
                # The entropy head is the tiny causal byte LM used by the patcher.
                total = total + 0.1 * early_loss

        diffusion_loss = zero
        if self.spec.diffusion is not None:
            losses = []
            for batch, layout in enumerate(layouts):
                byte_to_patch = layout.byte_to_patch()
                for ordinal, start in enumerate(layout.starts):
                    clean = input_ids[batch, start : start + self.spec.diffusion.block_size]
                    if not clean.numel():
                        continue
                    probability = mask_probability_q20
                    if probability is None:
                        raw = hashlib.sha256(
                            f"bcir-bltd-time-v1:{seed}:{batch}:{ordinal}".encode("ascii")
                        ).digest()
                        probability = 1 + int.from_bytes(raw[:8], "little") % _Q20
                    corrupted_tuple = self.spec.diffusion.corrupt(
                        tuple(int(value) for value in clean.detach().cpu().tolist()),
                        self.spec.vocabulary,
                        mask_probability_q20=probability,
                        seed=seed + batch * 65537 + ordinal,
                    )
                    corrupted = torch.tensor(
                        corrupted_tuple, dtype=torch.long, device=input_ids.device
                    )
                    patch_index = byte_to_patch[start]
                    conditioning = (
                        self.bos_patch if patch_index == 0 else globals_[batch][patch_index - 1]
                    )
                    block_logits = self._diffusion_logits_from_latent(corrupted, conditioning)
                    masked = corrupted == self.spec.vocabulary.mask_id
                    losses.append(
                        F.cross_entropy(
                            block_logits[masked].float(), clean[masked], reduction="mean"
                        )
                        * (_Q20 / probability)
                    )
            if not losses:
                raise RuntimeError("diffusion objective produced no masked training bytes")
            diffusion_loss = torch.stack(losses).mean()
            total = total + self.spec.diffusion.loss_weight_q20 / _Q20 * diffusion_loss
        return ByteLatentLossReport(
            total, autoregressive, diffusion_loss, policy_loss, target_loss, early_loss, layouts
        )


class HostedMambaByteModel(nn.Module):
    """Token-free selective-SSM reference with fixed recurrent state complexity."""

    def __init__(self, spec: MambaByteSpec):
        super().__init__()
        if not isinstance(spec, MambaByteSpec):
            raise ValueError("spec must be MambaByteSpec")
        if (
            spec.context_bytes > _MAX_HOSTED_CONTEXT
            or assess_mambabyte(spec, sequence_bytes=1).parameter_elements
            > _MAX_HOSTED_PARAMETER_ELEMENTS
        ):
            raise ValueError("hosted MambaByte request exceeds bounded limits")
        self.spec = spec
        self.embed_bytes = nn.Embedding(spec.vocabulary.size, spec.width)
        self.blocks = nn.ModuleList(
            _SelectiveSSMBlock(
                spec.width,
                spec.state_size,
                spec.expansion,
                spec.conv_width,
                epsilon=spec.rms_norm_epsilon,
            )
            for _ in range(spec.layers)
        )
        self.final_norm = _RMSNorm(spec.width, spec.rms_norm_epsilon)
        self.lm_head = nn.Linear(spec.width, spec.vocabulary.size, bias=False)
        self.apply(_initialize)
        if spec.tied_embeddings:
            self.lm_head.weight = self.embed_bytes.weight

    def _validate_ids(self, input_ids: torch.Tensor) -> None:
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.dtype != torch.long
            or input_ids.shape[0] < 1
            or input_ids.shape[0] > _MAX_HOSTED_BATCH
            or input_ids.shape[1] < 1
            or input_ids.shape[1] > self.spec.context_bytes
        ):
            raise ValueError("MambaByte ids are malformed")
        if input_ids.numel() * self.spec.width > _MAX_HOSTED_ACTIVATION_ELEMENTS:
            raise ValueError("MambaByte activation exceeds the bounded reference limit")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.spec.vocabulary.size):
            raise ValueError("MambaByte ids are outside the vocabulary")

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        self._validate_ids(input_ids)
        if targets is not None and (
            not isinstance(targets, torch.Tensor)
            or targets.shape != input_ids.shape
            or targets.dtype != torch.long
            or targets.device != input_ids.device
            or torch.any(targets < 0)
            or torch.any(targets >= self.spec.vocabulary.size)
        ):
            raise ValueError("MambaByte targets are malformed")
        values = self.embed_bytes(input_ids)
        for block in self.blocks:
            values = block(values)
        logits = F.linear(self.final_norm(values), self.lm_head.weight)
        loss = (
            None
            if targets is None
            else F.cross_entropy(
                logits.float().reshape(-1, self.spec.vocabulary.size), targets.reshape(-1)
            )
        )
        return logits, loss

    def unique_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _generation_argmax(logits: torch.Tensor, vocabulary) -> int:
    """Select a raw byte or EOS; never emit prompt/padding/corruption sentinels."""
    if logits.ndim != 1 or logits.shape[0] != vocabulary.size:
        raise ValueError("generation logits are malformed")
    scores = logits.detach().clone()
    scores[vocabulary.bos_id] = float("-inf")
    scores[vocabulary.pad_id] = float("-inf")
    scores[vocabulary.mask_id] = float("-inf")
    return int(torch.argmax(scores).item())


def _validate_prompt(prompt: Sequence[int], vocabulary, context: int) -> tuple[int, ...]:
    try:
        result = tuple(itertools.islice(prompt, context + 1))
    except TypeError as exc:
        raise ValueError("prompt must be an iterable of byte/control ids") from exc
    if (
        not result
        or len(result) > context
        or any(type(value) is not int or not 0 <= value < vocabulary.size for value in result)
    ):
        raise ValueError("prompt is empty, over context, or outside the vocabulary")
    if vocabulary.pad_id in result or vocabulary.mask_id in result:
        raise ValueError("prompt cannot contain padding or diffusion-mask sentinels")
    return result


@torch.no_grad()
def greedy_generate_byte_latent(
    model: HostedByteLatentModel, prompt: Sequence[int], max_new_bytes: int
) -> tuple[int, ...]:
    if (
        not isinstance(model, HostedByteLatentModel)
        or type(max_new_bytes) is not int
        or max_new_bytes < 0
    ):
        raise ValueError("byte generation arguments are malformed")
    generated = list(_validate_prompt(prompt, model.spec.vocabulary, model.spec.context_bytes))
    if len(generated) + max_new_bytes > model.spec.context_bytes:
        raise ValueError("byte generation exceeds model context")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_bytes):
            ids = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(ids)
            generated.append(_generation_argmax(logits[0, -1], model.spec.vocabulary))
    finally:
        model.train(was_training)
    return tuple(generated)


@torch.no_grad()
def greedy_generate_mambabyte(
    model: HostedMambaByteModel, prompt: Sequence[int], max_new_bytes: int
) -> tuple[int, ...]:
    if (
        not isinstance(model, HostedMambaByteModel)
        or type(max_new_bytes) is not int
        or max_new_bytes < 0
    ):
        raise ValueError("MambaByte generation arguments are malformed")
    generated = list(_validate_prompt(prompt, model.spec.vocabulary, model.spec.context_bytes))
    if len(generated) + max_new_bytes > model.spec.context_bytes:
        raise ValueError("MambaByte generation exceeds context")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_bytes):
            ids = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(ids)
            generated.append(_generation_argmax(logits[0, -1], model.spec.vocabulary))
    finally:
        model.train(was_training)
    return tuple(generated)


def _verify_model_draft(model: HostedByteLatentModel, prefix: Sequence[int], draft: Sequence[int]):
    device = next(model.parameters()).device
    candidate = tuple(prefix) + tuple(draft)
    ids = torch.tensor([candidate], dtype=torch.long, device=device)
    logits, _ = model(ids)
    start = len(prefix) - 1
    predictions = tuple(
        _generation_argmax(logits[0, start + index], model.spec.vocabulary)
        for index in range(len(draft))
    )
    free = _generation_argmax(logits[0, -1], model.spec.vocabulary)
    return verify_byte_draft(
        draft, predictions, free_prediction=free, vocabulary_size=model.spec.vocabulary.size
    )


@torch.no_grad()
def self_speculative_generate(
    model: HostedByteLatentModel, prompt: Sequence[int], max_new_bytes: int
) -> tuple[int, ...]:
    if (
        not isinstance(model, HostedByteLatentModel)
        or model.spec.generation_mode != "self_speculative"
        or model.spec.speculation is None
    ):
        raise ValueError("self-speculation requires generation_mode='self_speculative'")
    generated = list(_validate_prompt(prompt, model.spec.vocabulary, model.spec.context_bytes))
    if (
        type(max_new_bytes) is not int
        or max_new_bytes < 0
        or len(generated) + max_new_bytes > model.spec.context_bytes
    ):
        raise ValueError("self-speculation length is invalid")
    target = len(generated) + max_new_bytes
    was_training = model.training
    model.eval()
    try:
        while len(generated) < target:
            draft_count = min(model.spec.speculation.draft_bytes, target - len(generated))
            draft = model.local_draft(generated, draft_count)
            verified = _verify_model_draft(model, generated, draft)
            generated.extend(verified.emitted[: target - len(generated)])
    finally:
        model.train(was_training)
    return tuple(generated)


@torch.no_grad()
def diffusion_draft(
    model: HostedByteLatentModel, prefix: Sequence[int], block_size: int | None = None
) -> tuple[int, ...]:
    if not isinstance(model, HostedByteLatentModel) or model.spec.diffusion is None:
        raise ValueError("diffusion drafting requires a configured byte model")
    prefix = _validate_prompt(prefix, model.spec.vocabulary, model.spec.context_bytes)
    diffusion = model.spec.diffusion
    size = diffusion.block_size if block_size is None else block_size
    if (
        type(size) is not int
        or not 1 <= size <= diffusion.block_size
        or len(prefix) + size > model.spec.context_bytes
    ):
        raise ValueError("diffusion block size is invalid")
    device = next(model.parameters()).device
    block = [model.spec.vocabulary.mask_id] * size
    prefix_tensor = torch.tensor(prefix, dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        for step in range(diffusion.max_steps):
            block_tensor = torch.tensor(block, dtype=torch.long, device=device)
            logits = model.diffusion_logits(prefix_tensor, block_tensor)
            # BOS/PAD/MASK are sentinels, never denoised byte predictions.
            logits = logits.clone()
            logits[
                :,
                [
                    model.spec.vocabulary.bos_id,
                    model.spec.vocabulary.pad_id,
                    model.spec.vocabulary.mask_id,
                ],
            ] = float("-inf")
            probabilities = F.softmax(logits.float(), dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            entropy = -(probabilities * torch.log2(probabilities.clamp_min(1e-30))).sum(-1)
            masked = tuple(
                index for index, value in enumerate(block) if value == model.spec.vocabulary.mask_id
            )
            if not masked:
                break
            if step + 1 == diffusion.max_steps:
                chosen = masked
            else:
                chosen = diffusion.select_positions(
                    masked,
                    confidences_q20=tuple(
                        int(value * _Q20 + 0.5) for value in confidence.cpu().tolist()
                    ),
                    entropies_millibits=tuple(
                        int(value * 1000.0 + 0.5) for value in entropy.cpu().tolist()
                    ),
                )
            for index in chosen:
                block[index] = int(prediction[index].item())
    finally:
        model.train(was_training)
    if any(value == model.spec.vocabulary.mask_id for value in block):
        raise RuntimeError("diffusion drafting exhausted its bounded progress budget")
    return tuple(block)


@torch.no_grad()
def diffusion_generate(
    model: HostedByteLatentModel, prompt: Sequence[int], max_new_bytes: int
) -> tuple[int, ...]:
    if not isinstance(model, HostedByteLatentModel) or model.spec.generation_mode != "diffusion":
        raise ValueError("pure diffusion generation requires generation_mode='diffusion'")
    generated = list(_validate_prompt(prompt, model.spec.vocabulary, model.spec.context_bytes))
    if (
        type(max_new_bytes) is not int
        or max_new_bytes < 0
        or len(generated) + max_new_bytes > model.spec.context_bytes
    ):
        raise ValueError("diffusion generation length is invalid")
    target = len(generated) + max_new_bytes
    was_training = model.training
    model.eval()
    try:
        while len(generated) < target:
            size = min(model.spec.diffusion.block_size, target - len(generated))
            generated.extend(diffusion_draft(model, generated, size))
    finally:
        model.train(was_training)
    return tuple(generated)


@torch.no_grad()
def diffusion_verify_generate(
    model: HostedByteLatentModel, prompt: Sequence[int], max_new_bytes: int
) -> tuple[int, ...]:
    if (
        not isinstance(model, HostedByteLatentModel)
        or model.spec.generation_mode != "diffusion_verify"
        or model.spec.speculation is None
    ):
        raise ValueError("BLT-DV requires diffusion_verify mode")
    generated = list(_validate_prompt(prompt, model.spec.vocabulary, model.spec.context_bytes))
    if (
        type(max_new_bytes) is not int
        or max_new_bytes < 0
        or len(generated) + max_new_bytes > model.spec.context_bytes
    ):
        raise ValueError("BLT-DV generation length is invalid")
    target = len(generated) + max_new_bytes
    was_training = model.training
    model.eval()
    try:
        while len(generated) < target:
            size = min(
                model.spec.diffusion.block_size,
                model.spec.speculation.draft_bytes,
                target - len(generated),
            )
            draft = diffusion_draft(model, generated, size)
            verified = _verify_model_draft(model, generated, draft)
            generated.extend(verified.emitted[: target - len(generated)])
    finally:
        model.train(was_training)
    return tuple(generated)


@torch.no_grad()
def byte_latent_tensor_shapes(model: HostedByteLatentModel) -> tuple[TensorShape, ...]:
    if not isinstance(model, HostedByteLatentModel):
        raise ValueError("model must be HostedByteLatentModel")
    return tuple(
        TensorShape(name, tuple(value.shape))
        for name, value in sorted(model.named_parameters())
        if value.ndim
    )


def _require_complete_global_plan(
    model: HostedByteLatentModel, plan: ByteifiedTransplantPlan
) -> None:
    parameters = dict(model.named_parameters())
    expected = {name for name in parameters if name.startswith("global_blocks.")}
    mapped = {target for target, _source in plan.mappings}
    if mapped != expected:
        raise ValueError("byteified plan must map every global-block parameter exactly once")
    reused = sum(parameters[name].numel() for name in expected)
    total = sum(parameter.numel() for parameter in parameters.values())
    if plan.reused_elements != reused or plan.initialized_elements != total - reused:
        raise ValueError("byteified plan element counts do not match the hosted model")


@torch.no_grad()
def apply_byteified_transplant(
    source_state: Mapping[str, torch.Tensor],
    model: HostedByteLatentModel,
    plan: ByteifiedTransplantPlan,
) -> ByteifiedTransplantPlan:
    """Validate the entire copy before mutating a target parameter."""
    if (
        not isinstance(source_state, Mapping)
        or not isinstance(model, HostedByteLatentModel)
        or not isinstance(plan, ByteifiedTransplantPlan)
    ):
        raise ValueError("byteified transplant arguments are malformed")
    if len(source_state) > _MAX_HOSTED_TENSOR_INVENTORY:
        raise ValueError("byteified source tensor inventory exceeds the hosted bound")
    try:
        source_snapshot = dict(source_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("byteified source tensor mapping cannot be snapshotted") from exc
    if any(
        not isinstance(name, str)
        or not name
        or len(name.encode("utf-8")) > 4096
        or not isinstance(value, torch.Tensor)
        for name, value in source_snapshot.items()
    ):
        raise ValueError("byteified source tensor inventory is malformed")
    try:
        source_digest = tensor_mapping_digest(source_snapshot)
    except (OverflowError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("byteified source tensor inventory cannot be hashed") from exc
    if plan.target_spec_sha256 != model.spec.digest or source_digest != plan.source_sha256:
        raise ValueError("byteified transplant provenance does not match")
    _require_complete_global_plan(model, plan)
    target_state = model.state_dict()
    staged = []
    for target_name, source_name in plan.mappings:
        if target_name not in target_state or source_name not in source_snapshot:
            raise ValueError("byteified transplant tensor is missing")
        source = source_snapshot[source_name]
        target = target_state[target_name]
        if (
            not isinstance(source, torch.Tensor)
            or source.shape != target.shape
            or source.dtype != target.dtype
            or not torch.isfinite(source).all()
        ):
            raise ValueError("byteified transplant tensor is incompatible or non-finite")
        staged.append((target, source.detach().clone()))
    for target, source in staged:
        target.copy_(source)
    if plan.freeze_global:
        for name, parameter in model.named_parameters():
            if name.startswith("global_blocks."):
                parameter.requires_grad_(False)
    return plan


def byteified_parameter_groups(
    model: HostedByteLatentModel, plan: ByteifiedTransplantPlan, *, base_learning_rate: float
):
    if (
        not isinstance(model, HostedByteLatentModel)
        or not isinstance(plan, ByteifiedTransplantPlan)
        or isinstance(base_learning_rate, bool)
        or not isinstance(base_learning_rate, (int, float))
        or not math.isfinite(float(base_learning_rate))
        or base_learning_rate <= 0
    ):
        raise ValueError("byteified optimizer group arguments are malformed")
    if plan.target_spec_sha256 != model.spec.digest:
        raise ValueError("byteified optimizer plan does not describe this model")
    _require_complete_global_plan(model, plan)
    global_parameters = []
    local_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("global_blocks."):
            if not plan.freeze_global:
                global_parameters.append(parameter)
        else:
            local_parameters.append(parameter)
    groups = []
    if local_parameters:
        groups.append(
            {"params": local_parameters, "lr": float(base_learning_rate), "name": "byte-organs"}
        )
    if global_parameters:
        groups.append(
            {
                "params": global_parameters,
                "lr": float(base_learning_rate) * plan.global_learning_rate_q20 / _Q20,
                "name": "transplanted-global",
            }
        )
    return groups


@torch.no_grad()
def _byte_ar_loss(model, input_ids, targets) -> float:
    was_training = model.training
    model.eval()
    try:
        _logits, loss = model(input_ids, targets)
        value = float(loss.item())
    finally:
        model.train(was_training)
    if not math.isfinite(value):
        raise RuntimeError("byte model evaluation produced non-finite loss")
    return value


def _train_byte_model(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    spec: StageTrainSpec,
    telemetry_sink,
    *,
    latent: bool,
) -> StageRunReport:
    if not isinstance(spec, StageTrainSpec) or spec.stage != "pretrain":
        raise ValueError("byte pretraining requires stage='pretrain'")
    if (
        not isinstance(input_ids, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or input_ids.shape != targets.shape
        or input_ids.ndim != 2
        or input_ids.device != targets.device
        or input_ids.device != next(model.parameters()).device
    ):
        raise ValueError("byte pretraining tensors are malformed or on the wrong device")
    model._validate_ids(input_ids)
    if (
        targets.dtype != torch.long
        or torch.any(targets < 0)
        or torch.any(targets >= model.spec.vocabulary.size)
    ):
        raise ValueError("byte targets must be in-vocabulary torch.long")
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
    architecture_sha = model.spec.digest
    with _deterministic_stage(spec.seed):
        initial = _byte_ar_loss(model, input_ids, targets)
        optimizer = _optimizer(spec, model.parameters())
        model.train()
        for step in range(spec.steps):
            optimizer.zero_grad(set_to_none=True)
            if latent:
                loss = model.loss_components(input_ids, targets, seed=spec.seed + step).total
            else:
                _logits, loss = model(input_ids, targets)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
            if not torch.isfinite(norm) or not torch.isfinite(loss):
                raise RuntimeError("byte model produced non-finite training values")
            optimizer.step()
            _emit(
                telemetry_sink,
                "pretrain",
                step + 1,
                float(loss.detach().item()),
                float(norm.item()),
                (step + 1) * input_ids.shape[0],
            )
        final = _byte_ar_loss(model, input_ids, targets)
        return StageRunReport(
            "pretrain",
            spec.digest,
            input_sha,
            initial,
            final,
            spec.steps,
            input_ids.shape[0],
            _module_digest(model),
            architecture_sha,
        )


def train_byte_latent_pretrain(
    model: HostedByteLatentModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    spec: StageTrainSpec,
    telemetry_sink=None,
) -> StageRunReport:
    if not isinstance(model, HostedByteLatentModel):
        raise ValueError("model must be HostedByteLatentModel")
    return _train_byte_model(model, input_ids, targets, spec, telemetry_sink, latent=True)


def train_mambabyte_pretrain(
    model: HostedMambaByteModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    spec: StageTrainSpec,
    telemetry_sink=None,
) -> StageRunReport:
    if not isinstance(model, HostedMambaByteModel):
        raise ValueError("model must be HostedMambaByteModel")
    return _train_byte_model(model, input_ids, targets, spec, telemetry_sink, latent=False)


__all__ = [
    "ByteLatentLossReport",
    "HostedByteLatentModel",
    "HostedMambaByteModel",
    "apply_byteified_transplant",
    "byte_latent_tensor_shapes",
    "byteified_parameter_groups",
    "diffusion_draft",
    "diffusion_generate",
    "diffusion_verify_generate",
    "greedy_generate_byte_latent",
    "greedy_generate_mambabyte",
    "self_speculative_generate",
    "train_byte_latent_pretrain",
    "train_mambabyte_pretrain",
]
