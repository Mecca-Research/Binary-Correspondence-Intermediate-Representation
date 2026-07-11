"""Rung 4, second half (ML/AI roadmap §7.4): REAL-WEIGHT INGESTION -- an HF-layout Llama
checkpoint (config.json + a safetensors shard) becomes the rung-3 decoder's DecoderWeights.

The three layout facts a released checkpoint carries that the reference decoder does not:

  1. **Linear orientation.** An HF `nn.Linear` stores its weight [out, in] and computes
     `x @ W^T`; `matmul_reference` computes `x @ W` with W [in, out] -- every projection
     TRANSPOSES on ingest. (The embedding and lm_head are [vocab, d] row-per-id -- the
     layout `_tied_logits`/`embedding_lookup` already read -- so they pass through.)
  2. **RoPE channel order.** HF Llama rotates HALF-SPLIT pairs (c, c + d_k/2) per head
     ("rotate_half", the GPT-NeoX layout); `rope_reference` rotates INTERLEAVED pairs
     (2k, 2k+1) (the GPT-J layout). The k-th pair's angle is the same in both, so
     permuting the OUT channels of w_q/w_k per head (`rope_interleave_order`) makes the
     interleaved rotation compute exactly the half-split rotation -- the llama.cpp
     conversion identity.
  3. **The MLP/head shape.** Llama's MLP is gated SiLU (gate/up/down, NO biases --
     `activation="silu_gate"`, w_gate) and its logits head is UNTIED (lm_head).

`spec_from_config` maps config.json to a DecoderSpec (num_key_value_heads -> the wave-8
GQA knob); `weights_from_tensors` maps the standard `model.layers.N.*` tensor names,
validating every shape against the spec -- a checkpoint that lies about a shape refuses
loudly. Dep-free stdlib; cost-side (imports no verifier)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from ...kbcir.unsupervised import EmbeddingTable
from .decode import DecoderSpec, DecoderWeights, LayerWeights
from .safetensors_io import load_tensors


def spec_from_config(config: dict) -> DecoderSpec:
    """The DecoderSpec a Llama-family config.json declares. `hidden_act` must be silu
    (the gated MLP is what the weights contain); rope_theta defaults to the Llama 10000."""
    act = str(config.get("hidden_act", "silu"))
    if act != "silu":
        raise ValueError(f"hidden_act {act!r} is not the Llama gated-SiLU MLP")
    n_heads = int(config["num_attention_heads"])
    return DecoderSpec(
        vocab_size=int(config["vocab_size"]),
        d_model=int(config["hidden_size"]),
        n_heads=n_heads,
        n_layers=int(config["num_hidden_layers"]),
        d_ff=int(config["intermediate_size"]),
        rope_base=float(config.get("rope_theta", 10000.0)),
        activation="silu_gate",
        n_kv_heads=int(config.get("num_key_value_heads", n_heads)),
        tied_embeddings=bool(config.get("tie_word_embeddings", False)),
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)))


def rope_interleave_order(n_heads: int, d_k: int) -> list[int]:
    """The per-head OUT-channel permutation pi with ours[o] = hf[pi[o]]: channel 2k reads
    the half-split channel k, channel 2k+1 reads k + d_k/2 (per head)."""
    pi: list[int] = []
    for h in range(n_heads):
        base = h * d_k
        for k in range(d_k // 2):
            pi.append(base + k)
            pi.append(base + k + d_k // 2)
    return pi


def _transpose(flat: list, rows: int, cols: int) -> list:
    """[rows x cols] -> [cols x rows] (the Linear orientation flip)."""
    return [flat[r * cols + c] for c in range(cols) for r in range(rows)]


def _proj(tensors: dict, name: str, out_dim: int, in_dim: int, perm: list | None = None) -> tuple:
    """One projection: shape-check the HF [out, in] tensor, apply the optional per-head
    RoPE out-channel permutation, transpose to the oracle's [in, out]."""
    if name not in tensors:
        raise ValueError(f"missing tensor {name!r}")
    _dt, shape, flat = tensors[name]
    if tuple(shape) != (out_dim, in_dim):
        raise ValueError(f"{name}: shape {tuple(shape)} != ({out_dim}, {in_dim})")
    if perm is not None:                               # reorder the OUT rows (HF layout)
        flat = [flat[o * in_dim + c] for o in perm for c in range(in_dim)]
    return tuple(_transpose(flat, out_dim, in_dim))


def _vec(tensors: dict, name: str, n: int) -> tuple:
    if name not in tensors:
        raise ValueError(f"missing tensor {name!r}")
    _dt, shape, flat = tensors[name]
    if tuple(shape) != (n,):
        raise ValueError(f"{name}: shape {tuple(shape)} != ({n},)")
    return tuple(flat)


def weights_from_tensors(spec: DecoderSpec, tensors: dict) -> DecoderWeights:
    """The standard HF Llama tensor names -> DecoderWeights, every shape validated."""
    d, f, kvd, dk = spec.d_model, spec.d_ff, spec.kv_dim, spec.d_k
    q_perm = rope_interleave_order(spec.n_heads, dk)
    k_perm = rope_interleave_order(spec.kv_heads, dk)
    name = "model.embed_tokens.weight"
    if name not in tensors:
        raise ValueError(f"missing tensor {name!r}")
    _dt, eshape, eflat = tensors[name]
    if tuple(eshape) != (spec.vocab_size, d):
        raise ValueError(f"{name}: shape {tuple(eshape)} != ({spec.vocab_size}, {d})")
    layers = []
    for li in range(spec.n_layers):
        p = f"model.layers.{li}."
        layers.append(LayerWeights(
            g_attn=_vec(tensors, p + "input_layernorm.weight", d),
            w_q=_proj(tensors, p + "self_attn.q_proj.weight", d, d, q_perm),
            w_k=_proj(tensors, p + "self_attn.k_proj.weight", kvd, d, k_perm),
            w_v=_proj(tensors, p + "self_attn.v_proj.weight", kvd, d),
            w_o=_proj(tensors, p + "self_attn.o_proj.weight", d, d),
            g_ff=_vec(tensors, p + "post_attention_layernorm.weight", d),
            w1=_proj(tensors, p + "mlp.up_proj.weight", f, d),
            b1=(),
            w2=_proj(tensors, p + "mlp.down_proj.weight", d, f),
            b2=(),
            w_gate=_proj(tensors, p + "mlp.gate_proj.weight", f, d)))
    lm_head: tuple = ()
    if not spec.tied_embeddings:
        _dt, hshape, hflat = tensors.get("lm_head.weight", (None, (), []))
        if tuple(hshape) != (spec.vocab_size, d):
            raise ValueError(f"lm_head.weight: shape {tuple(hshape)} != "
                             f"({spec.vocab_size}, {d})")
        lm_head = tuple(hflat)                         # [vocab, d] row-per-id: pass-through
    return DecoderWeights(
        embedding=EmbeddingTable(table=tuple(eflat), n_vocab=spec.vocab_size, dim=d),
        layers=tuple(layers),
        g_final=_vec(tensors, "model.norm.weight", d),
        lm_head=lm_head)


@dataclass(frozen=True)
class IngestReport:
    """Exact checkpoint census split into decoder parameters and validated auxiliaries."""

    consumed_tensors: tuple[str, ...]
    auxiliary_tensors: tuple[str, ...]
    decoder_element_count: int
    auxiliary_element_count: int
    shard_element_count: int


def _consumed_tensor_names(spec: DecoderSpec) -> tuple[str, ...]:
    names = ["model.embed_tokens.weight"]
    for li in range(spec.n_layers):
        p = f"model.layers.{li}."
        names.extend((p + "input_layernorm.weight",
                      p + "self_attn.q_proj.weight",
                      p + "self_attn.k_proj.weight",
                      p + "self_attn.v_proj.weight",
                      p + "self_attn.o_proj.weight",
                      p + "post_attention_layernorm.weight",
                      p + "mlp.gate_proj.weight",
                      p + "mlp.up_proj.weight",
                      p + "mlp.down_proj.weight"))
    names.append("model.norm.weight")
    if not spec.tied_embeddings:
        names.append("lm_head.weight")
    return tuple(names)


def _rotary_aux_layer(name: str, spec: DecoderSpec) -> int | None:
    prefix = "model.layers."
    suffix = ".self_attn.rotary_emb.inv_freq"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    middle = name[len(prefix):-len(suffix)]
    if not middle.isdigit():
        return None
    layer = int(middle)
    return layer if 0 <= layer < spec.n_layers else None


def _validate_rotary_aux(name: str, tensor: tuple, spec: DecoderSpec) -> None:
    dtype, shape, values = tensor
    want_n = spec.d_k // 2
    if tuple(shape) != (want_n,):
        raise ValueError(f"{name}: shape {tuple(shape)} != ({want_n},)")
    expected = [spec.rope_base ** (-(2.0 * i) / spec.d_k) for i in range(want_n)]
    # F16/BF16 storage rounds the analytical frequencies more coarsely than F32/F64.
    rel_tol = 5e-3 if dtype in ("F16", "BF16") else 1e-6
    for i, (actual, want) in enumerate(zip(values, expected)):
        if not math.isfinite(actual) or not math.isclose(actual, want, rel_tol=rel_tol,
                                                         abs_tol=1e-8):
            raise ValueError(f"{name}[{i}]={actual!r} does not match rope_base "
                             f"{spec.rope_base!r} (expected {want!r})")


def _build_ingest_report(spec: DecoderSpec, tensors: dict) -> IngestReport:
    consumed = _consumed_tensor_names(spec)
    consumed_set = set(consumed)
    # weights_from_tensors has already required all consumed tensors; retain a defensive
    # report-side check so this helper remains total if reused independently.
    missing = [name for name in consumed if name not in tensors]
    if missing:
        raise ValueError(f"missing consumed tensors: {missing}")
    auxiliary: list[str] = []
    unknown: list[str] = []
    for name in sorted(set(tensors) - consumed_set):
        if _rotary_aux_layer(name, spec) is not None:
            _validate_rotary_aux(name, tensors[name], spec)
            auxiliary.append(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(f"unconsumed checkpoint tensors are not allowlisted: {unknown}")

    def count(names) -> int:
        return sum(len(tensors[name][2]) for name in names)

    return IngestReport(
        consumed_tensors=consumed,
        auxiliary_tensors=tuple(auxiliary),
        decoder_element_count=count(consumed),
        auxiliary_element_count=count(auxiliary),
        shard_element_count=sum(len(tensor[2]) for tensor in tensors.values()))


def ingest_checkpoint_with_report(model_dir: str) \
        -> tuple[DecoderSpec, DecoderWeights, IngestReport]:
    """Ingest one shard and return a strict, auditable tensor/element census.

    Llama's per-layer ``rotary_emb.inv_freq`` buffers are the sole permitted non-parameter
    tensors: their shape and analytical values are checked against ``rope_base``.  Unknown
    unused tensors fail in the strict API instead of disappearing from the census.
    """
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        spec = spec_from_config(json.load(f))
    tensors = load_tensors(os.path.join(model_dir, "model.safetensors"))
    weights = weights_from_tensors(spec, tensors)
    return spec, weights, _build_ingest_report(spec, tensors)


def ingest_checkpoint(model_dir: str) -> tuple[DecoderSpec, DecoderWeights]:
    """Compatibility tuple API with the same strict auxiliary-tensor validation."""
    spec, weights, _report = ingest_checkpoint_with_report(model_dir)
    return spec, weights
