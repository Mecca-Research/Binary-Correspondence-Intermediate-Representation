"""Rung 4 of the open-weight ladder (ML/AI roadmap §7.4): the QUANTIZED inference artifact.

The Q8<->float32 bridge wrapped around the TRUSTED rung-3 decoder (integrate, don't
reinvent -- the `attention_via_bridge` / B5 / G1 pattern): every weight tensor round-trips
through per-group quantization (`kbcir.quantize`), and the quantized artifact IS the
reference decoder run over the round-tripped weights. So the oracle for the quantized path
needs no new kernel: the SOLE certified error vs the float decode is the WEIGHT
quantization, R17-bounded at <=1 ULP per value at each group's realized grid
(`precision.quantization_error_bound`).

What rung 4 adds on top of the bridge is the DRIFT DISCIPLINE: a `DriftRecord` per
deterministic prompt fixture -- the greedy ids of both paths, the max absolute logit drift
along the float trajectory (teacher-forced, so the two paths are compared on identical
contexts), and the mean NLL of the float continuation under both models (the perplexity
proxy). Records are deterministic (same weights, same prompt -> same record), the E3/R17
posture: measure the drift, never hide it. Cost-side: imports no verifier (two-truth)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ...kbcir.precision import quantization_error_bound
from ...kbcir.quantize import dequantize, quantize_per_group
from ...kbcir.unsupervised import EmbeddingTable
from .decode import (DecoderSpec, DecoderWeights, LayerWeights, _tied_logits,
                     decoder_forward_reference, reference_decode)


def _roundtrip(values, group_size: int, bits: int) -> tuple:
    return tuple(dequantize(quantize_per_group(list(values), group_size, bits)))


def quantize_decoder_weights(w: DecoderWeights, group_size: int, bits: int) -> DecoderWeights:
    """Round-trip EVERY weight tensor through the per-group Q bridge. The returned weights
    are the dequantized values -- shapes unchanged, so `check_decoder_weights` still holds
    and `reference_decode` over them IS the quantized inference artifact."""
    layers = tuple(
        LayerWeights(g_attn=_roundtrip(lw.g_attn, group_size, bits),
                     w_q=_roundtrip(lw.w_q, group_size, bits),
                     w_k=_roundtrip(lw.w_k, group_size, bits),
                     w_v=_roundtrip(lw.w_v, group_size, bits),
                     w_o=_roundtrip(lw.w_o, group_size, bits),
                     g_ff=_roundtrip(lw.g_ff, group_size, bits),
                     w1=_roundtrip(lw.w1, group_size, bits),
                     b1=_roundtrip(lw.b1, group_size, bits),
                     w2=_roundtrip(lw.w2, group_size, bits),
                     b2=_roundtrip(lw.b2, group_size, bits))
        for lw in w.layers)
    emb = EmbeddingTable(table=_roundtrip(w.embedding.table, group_size, bits),
                         n_vocab=w.embedding.n_vocab, dim=w.embedding.dim)
    return DecoderWeights(embedding=emb, layers=layers,
                          g_final=_roundtrip(w.g_final, group_size, bits))


@dataclass(frozen=True)
class DriftRecord:
    """One deterministic prompt fixture's R17-disciplined drift measurement: the two greedy
    continuations, the max |logit| drift along the FLOAT trajectory (teacher-forced), and
    the float continuation's mean NLL under both models (the perplexity proxy;
    `ppl = exp(nll)`). `ulp_bound` is the certified per-value weight-quantization bound."""

    prompt: tuple
    ids_f32: tuple
    ids_q: tuple
    ids_match: bool
    max_logit_drift: float
    nll_f32: float
    nll_q: float
    bits: int
    group_size: int
    ulp_bound: int


def _step_logits(ids: list, spec: DecoderSpec, w: DecoderWeights) -> list:
    hfin = decoder_forward_reference(ids, spec, w)
    d = spec.d_model
    return _tied_logits(hfin[(len(ids) - 1) * d:len(ids) * d], w.embedding)


def _nll_of(logits: list, chosen: int) -> float:
    """-log softmax(logits)[chosen], the numerically-stable log-sum-exp form."""
    m = max(logits)
    lse = m + math.log(sum(math.exp(v - m) for v in logits))
    return lse - logits[chosen]


def decode_drift_record(prompt_ids: list, spec: DecoderSpec, w: DecoderWeights, *,
                        group_size: int, bits: int, max_new: int) -> DriftRecord:
    """Run the float decoder and its quantized artifact over one deterministic prompt and
    RECORD the drift. Greedy ids come from each path independently (what each artifact
    would actually emit); logit drift and NLL are measured teacher-forced along the float
    trajectory so every step compares the two models on the identical context."""
    wq = quantize_decoder_weights(w, group_size, bits)
    ids_f32 = reference_decode(prompt_ids, spec, w, max_new)
    ids_q = reference_decode(prompt_ids, spec, wq, max_new)
    ctx = list(prompt_ids)
    drift = 0.0
    nll_f = nll_q = 0.0
    for tok in ids_f32:                                  # the float trajectory, teacher-forced
        lf = _step_logits(ctx, spec, w)
        lq = _step_logits(ctx, spec, wq)
        drift = max(drift, max(abs(a - b) for a, b in zip(lf, lq)))
        nll_f += _nll_of(lf, tok)
        nll_q += _nll_of(lq, tok)
        ctx.append(tok)
    n = max(1, len(ids_f32))
    return DriftRecord(prompt=tuple(prompt_ids), ids_f32=tuple(ids_f32), ids_q=tuple(ids_q),
                       ids_match=ids_f32 == ids_q, max_logit_drift=drift,
                       nll_f32=nll_f / n, nll_q=nll_q / n, bits=bits, group_size=group_size,
                       ulp_bound=quantization_error_bound())
