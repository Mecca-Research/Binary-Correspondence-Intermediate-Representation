"""Independent PyTorch Llama reference for the opt-in hosted model laboratory."""
from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by dependency gate
    raise ModuleNotFoundError(
        "bcir.hosted.models requires the 'model-lab' optional dependencies") from exc

from ...frontends.models.decode import DecoderSpec


class HostedRMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = float(epsilon)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        dtype = values.dtype
        stable = values.float()
        scale = torch.rsqrt(stable.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return (stable * scale).to(dtype) * self.weight


def _rope_cache(length: int, head_dim: int, base: float,
                device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    positions = torch.arange(length, device=device, dtype=torch.float32)
    frequency = base ** (-2.0 * torch.arange(half, device=device,
                                            dtype=torch.float32) / head_dim)
    angles = positions[:, None] * frequency[None, :]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _apply_half_rope(values: torch.Tensor, cos: torch.Tensor,
                     sin: torch.Tensor) -> torch.Tensor:
    """Standard Llama half-split RoPE; BCIR's strict ingest performs its permutation."""
    half = values.shape[-1] // 2
    left, right = values[..., :half], values[..., half:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


class HostedAttention(nn.Module):
    def __init__(self, spec: DecoderSpec):
        super().__init__()
        d, kvd = spec.d_model, spec.kv_dim
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, kvd, bias=False)
        self.v_proj = nn.Linear(d, kvd, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.n_heads = spec.n_heads
        self.n_kv_heads = spec.kv_heads
        self.head_dim = spec.d_k

    def forward(self, values: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        batch, length, width = values.shape
        q = self.q_proj(values).view(batch, length, self.n_heads,
                                    self.head_dim).transpose(1, 2)
        k = self.k_proj(values).view(batch, length, self.n_kv_heads,
                                    self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(batch, length, self.n_kv_heads,
                                    self.head_dim).transpose(1, 2)
        q = _apply_half_rope(q, cos, sin)
        k = _apply_half_rope(k, cos, sin)
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        context = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=True)
        context = context.transpose(1, 2).contiguous().view(batch, length, width)
        return self.o_proj(context)


class HostedMLP(nn.Module):
    def __init__(self, spec: DecoderSpec):
        super().__init__()
        self.gate_proj = nn.Linear(spec.d_model, spec.d_ff, bias=False)
        self.up_proj = nn.Linear(spec.d_model, spec.d_ff, bias=False)
        self.down_proj = nn.Linear(spec.d_ff, spec.d_model, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(values)) * self.up_proj(values))


class HostedDecoderLayer(nn.Module):
    def __init__(self, spec: DecoderSpec):
        super().__init__()
        self.input_layernorm = HostedRMSNorm(spec.d_model, spec.rms_norm_eps)
        self.self_attn = HostedAttention(spec)
        self.post_attention_layernorm = HostedRMSNorm(spec.d_model, spec.rms_norm_eps)
        self.mlp = HostedMLP(spec)

    def forward(self, values: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        values = values + self.self_attn(self.input_layernorm(values), cos, sin)
        return values + self.mlp(self.post_attention_layernorm(values))


class HostedBackbone(nn.Module):
    def __init__(self, spec: DecoderSpec):
        super().__init__()
        self.embed_tokens = nn.Embedding(spec.vocab_size, spec.d_model)
        self.layers = nn.ModuleList([HostedDecoderLayer(spec) for _ in range(spec.n_layers)])
        self.norm = HostedRMSNorm(spec.d_model, spec.rms_norm_eps)


class HostedLlama(nn.Module):
    """Small Llama-family model whose state keys match the strict HF ingest contract."""

    def __init__(self, spec: DecoderSpec, *, activation_checkpointing: bool = False,
                 context_length: int | None = None):
        super().__init__()
        if not isinstance(spec, DecoderSpec) or spec.activation != "silu_gate":
            raise ValueError("HostedLlama requires a Llama/SwiGLU DecoderSpec")
        if type(activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be bool")
        if context_length is not None \
                and (type(context_length) is not int or context_length < 1):
            raise ValueError("context_length must be None or a positive integer")
        self.spec = spec
        self.activation_checkpointing = activation_checkpointing
        self.context_length = context_length
        self.model = HostedBackbone(spec)
        self.lm_head = nn.Linear(spec.d_model, spec.vocab_size, bias=False)
        self.apply(self._initialize)
        if spec.tied_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        if targets is not None and (targets.shape != input_ids.shape or targets.dtype != torch.long):
            raise ValueError("targets must be torch.long with the input shape")
        if input_ids.shape[1] < 1:
            raise ValueError("input sequence must not be empty")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.spec.vocab_size):
            raise ValueError("input token id is outside the vocabulary")
        if targets is not None \
                and (torch.any(targets < 0) or torch.any(targets >= self.spec.vocab_size)):
            raise ValueError("target token id is outside the vocabulary")
        values = self.model.embed_tokens(input_ids)
        cos, sin = _rope_cache(input_ids.shape[1], self.spec.d_k, self.spec.rope_base,
                               values.device, values.dtype)
        use_checkpoint = self.activation_checkpointing and self.training
        for layer in self.model.layers:
            if use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                values = checkpoint(layer, values, cos, sin, use_reentrant=False)
            else:
                values = layer(values, cos, sin)
        values = self.model.norm(values)
        logits = F.linear(values, self.lm_head.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, self.spec.vocab_size),
                                   targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def greedy(self, prompt_ids: list[int], max_new: int = 1) -> list[int]:
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError("prompt_ids must be a nonempty list")
        if any(type(token) is not int or not 0 <= token < self.spec.vocab_size
               for token in prompt_ids):
            raise ValueError("prompt_ids must contain only in-vocabulary integer IDs")
        if type(max_new) is not int or max_new < 0:
            raise ValueError("max_new must be an integer >= 0")
        device = next(self.parameters()).device
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = []
        self.eval()
        for _ in range(max_new):
            logits, _ = self(ids)
            token = int(torch.argmax(logits[0, -1]).item())
            out.append(token)
            ids = torch.cat((ids, torch.tensor([[token]], device=device)), dim=1)
        return out

    def unique_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
