"""Rung 3 of the open-weight ladder (ML/AI roadmap §7.4): the REFERENCE DECODER.

A slow, dependency-light Python reference for a small DENSE decoder (the Gemma/Llama-family
pre-norm shape), COMPOSED from the existing oracle references -- not reinvented:

    embedding_lookup (unsupervised)  ->  per layer [ rmsnorm_reference (transformer_grads)
    -> Q/K/V projections (matmul_reference) -> rope_reference on Q,K per head
    -> causal scaled-dot-product attention (attention.scores_reference + causal_mask
       + softmax_reference + matmul_reference) -> W_o + residual
    -> rmsnorm_reference -> feedforward_reference (transformer) + residual ]
    ->  final rmsnorm_reference  ->  tied or untied head logits  ->  greedy argmax.

Two decode paths, one truth: `reference_decode` recomputes the full context every step (the
obviously-correct naive reference), and `decode_with_kv_cache` is the incremental KV-cache
twin (the rung's KV primitive). Every stage is row-independent and both paths run the SAME
row arithmetic in the SAME order (a masked score contributes exp(-inf)=0.0, a trailing exact
zero), so the two paths agree BIT-FOR-BIT -- pinned by `test_model_decode.py`, the E3
reference-vs-realization pattern. The only transcendentals are exp/cos/sin/sqrt on the
trusted libm edge (the quarantine posture of every oracle reference).

Not imported by the package `__init__` on purpose: rungs 1-2 (manifest, tokenizer) stay
dep-free stdlib; this module pulls in the kbcir reference kernels. Cost-side: imports no
verifier (two-truth)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ...kbcir.activation import softmax_reference
from ...kbcir.attention import AttentionSpec, scores_reference
from ...kbcir.matmul import matmul_reference
from ...kbcir.transformer import causal_mask, feedforward_reference, swiglu_reference
from ...kbcir.transformer_grads import rmsnorm_reference, rope_reference
from ...kbcir.unsupervised import EmbeddingTable, embedding_lookup


# This is a dependency-free reference implementation, not an unbounded serving engine.
# The ceiling is deliberately above current long-context model gates while refusing the
# accidental billion-token requests that otherwise turn a malformed input into an
# effectively unbounded allocation/CPU job.
_MAX_REFERENCE_CONTEXT = 1 << 20


@dataclass(frozen=True)
class DecoderSpec:
    """The shape of the small dense decoder: `d_model` splits into `n_heads` RoPE'd heads
    (`d_k` must be even -- RoPE rotates channel pairs); the FF activation rides the same
    relu/gelu contract as `feedforward_reference`. `n_kv_heads` (GQA, rung 5): the number of
    SHARED key/value heads -- query head h reads kv head h // (n_heads // n_kv_heads); the
    default 0 means n_kv_heads == n_heads (plain MHA, every existing spec undisturbed)."""

    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    rope_base: float = 10000.0
    activation: str = "gelu"
    n_kv_heads: int = 0  # 0 == n_heads (MHA); the Llama/Gemma GQA knob
    tied_embeddings: bool = True  # False == a separate lm_head (the Llama choice); the
    #   logits then read lm_head, not the embedding table
    rms_norm_eps: float = 1e-6  # checkpoint-declared RMSNorm epsilon

    def __post_init__(self) -> None:
        dims = (self.vocab_size, self.d_model, self.n_heads, self.n_layers, self.d_ff)
        if any(type(v) is not int for v in dims):
            raise ValueError("decoder dims must be integers (bool is not a dimension)")
        if type(self.n_kv_heads) is not int:
            raise ValueError("n_kv_heads must be an integer")
        if type(self.tied_embeddings) is not bool:
            raise ValueError("tied_embeddings must be bool")
        if min(self.vocab_size, self.d_model, self.n_heads, self.n_layers, self.d_ff) < 1:
            raise ValueError("decoder dims must all be >= 1")
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("d_k must be even (RoPE rotates channel pairs)")
        if self.activation not in ("relu", "gelu", "silu_gate"):
            raise ValueError(f"activation must be relu|gelu|silu_gate; got {self.activation!r}")
        if not math.isfinite(self.rope_base) or self.rope_base <= 0.0:
            raise ValueError(f"rope_base must be finite and > 0; got {self.rope_base!r}")
        if self.n_kv_heads < 0 or self.n_kv_heads > self.n_heads:
            raise ValueError(f"n_kv_heads {self.n_kv_heads} must be in [0, n_heads]")
        if self.n_kv_heads and self.n_heads % self.n_kv_heads:
            raise ValueError(
                f"n_heads {self.n_heads} not divisible by "
                f"n_kv_heads {self.n_kv_heads} (GQA shares whole head groups)"
            )
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0.0:
            raise ValueError(f"rms_norm_eps must be finite and > 0; got {self.rms_norm_eps!r}")

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        """The effective K/V head count (n_kv_heads, or n_heads when unset -- MHA)."""
        return self.n_kv_heads or self.n_heads

    @property
    def kv_dim(self) -> int:
        """The K/V projection width: kv_heads * d_k (== d_model for MHA)."""
        return self.kv_heads * self.d_k


@dataclass(frozen=True)
class LayerWeights:
    """One decoder layer's parameters (all flat row-major): the two pre-norm RMSNorm gammas,
    the attention projections (w_k/w_v are d_model x kv_dim -- narrower under GQA), and the
    FF weights/biases."""

    g_attn: tuple  # d_model            (attention pre-norm gamma)
    w_q: tuple  # d_model x d_model
    w_k: tuple  # d_model x kv_dim   (== d_model for MHA)
    w_v: tuple  # d_model x kv_dim
    w_o: tuple  # d_model x d_model
    g_ff: tuple  # d_model            (FF pre-norm gamma)
    w1: tuple  # d_model x d_ff
    b1: tuple  # d_ff
    w2: tuple  # d_ff x d_model
    b2: tuple  # d_model
    w_gate: tuple = ()  # d_model x d_ff -- the gated-SiLU MLP's gate projection
    #   (silu_gate only; empty for relu/gelu, every existing layer)


@dataclass(frozen=True)
class DecoderWeights:
    """The whole decoder: the embedding table (ALSO the tied logits head, the Gemma choice),
    the per-layer weights, and the final-norm gamma."""

    embedding: EmbeddingTable
    layers: tuple  # tuple[LayerWeights, ...]
    g_final: tuple  # d_model
    lm_head: tuple = ()  # vocab x d_model -- the UNTIED logits head (empty == tied to the
    #   embedding table, the Gemma choice; every existing model)


def check_decoder_weights(spec: DecoderSpec, w: DecoderWeights) -> list[str]:
    """Op-level well-formedness (mirrors `check_attention`/`check_classical`; NOT a new
    global R-law): every weight's length against the spec. Returns messages, [] when clean."""
    msgs: list[str] = []
    if not isinstance(w, DecoderWeights):
        return ["weights are not a DecoderWeights value"]
    if not isinstance(w.embedding, EmbeddingTable):
        return ["embedding is not an EmbeddingTable"]
    if not isinstance(w.layers, (tuple, list)):
        return ["layers are not a tuple/list"]
    if not isinstance(w.g_final, (tuple, list)):
        return ["g_final is not a tuple/list"]
    if not isinstance(w.lm_head, (tuple, list)):
        return ["lm_head is not a tuple/list"]
    d, f, kvd = spec.d_model, spec.d_ff, spec.kv_dim
    if (w.embedding.n_vocab, w.embedding.dim) != (spec.vocab_size, d):
        msgs.append(
            f"embedding is {w.embedding.n_vocab}x{w.embedding.dim}, spec says {spec.vocab_size}x{d}"
        )
    if len(w.layers) != spec.n_layers:
        msgs.append(f"{len(w.layers)} layers, spec says {spec.n_layers}")
    if len(w.g_final) != d:
        msgs.append(f"g_final has {len(w.g_final)} entries, want {d}")
    gated = spec.activation == "silu_gate"
    want = {
        "g_attn": d,
        "w_q": d * d,
        "w_k": d * kvd,
        "w_v": d * kvd,
        "w_o": d * d,
        "g_ff": d,
        "w1": d * f,
        "w2": f * d,
        # the Llama MLP has NO biases and a gate projection; relu/gelu is the inverse
        "b1": 0 if gated else f,
        "b2": 0 if gated else d,
        "w_gate": d * f if gated else 0,
    }
    for li, lw in enumerate(w.layers):
        if not isinstance(lw, LayerWeights):
            msgs.append(f"layer {li}: not a LayerWeights value")
            continue
        for name, n in want.items():
            values = getattr(lw, name)
            if not isinstance(values, (tuple, list)):
                msgs.append(f"layer {li}: {name} is not a tuple/list")
            elif len(values) != n:
                msgs.append(f"layer {li}: {name} has {len(values)} entries, want {n}")
    head = len(getattr(w, "lm_head", ()))
    if spec.tied_embeddings and head:
        msgs.append(f"lm_head has {head} entries but the spec ties the embeddings")
    if not spec.tied_embeddings and head != spec.vocab_size * d:
        msgs.append(f"lm_head has {head} entries, want vocab*d = {spec.vocab_size * d}")
    return msgs


def _validate_decode_request(
    prompt_ids, spec: DecoderSpec, w: DecoderWeights, max_new: int, eos_id: int | None = None
) -> list[int]:
    """Validate an externally supplied decode request before cache/model mutation.

    Public reference/serving entries share this one strict boundary: token IDs are
    integers in-vocabulary (no lossy ``int()`` coercion), the context is bounded, and
    all weight shapes are checked before the first layer appends to a KV cache.
    """
    if not isinstance(spec, DecoderSpec):
        raise ValueError("decode spec is not a DecoderSpec")
    if type(max_new) is not int or max_new < 0:
        raise ValueError("max_new must be an integer >= 0")
    if not isinstance(prompt_ids, (list, tuple)) or not prompt_ids:
        raise ValueError("decode needs a non-empty list/tuple prompt")
    if len(prompt_ids) > _MAX_REFERENCE_CONTEXT or max_new > _MAX_REFERENCE_CONTEXT - len(
        prompt_ids
    ):
        raise ValueError(f"decode context exceeds {_MAX_REFERENCE_CONTEXT} tokens")
    prompt = list(prompt_ids)
    for i, tid in enumerate(prompt):
        if type(tid) is not int or not 0 <= tid < spec.vocab_size:
            raise ValueError(
                f"prompt token {i} must be an integer in [0, {spec.vocab_size}); got {tid!r}"
            )
    if eos_id is not None and (type(eos_id) is not int or not 0 <= eos_id < spec.vocab_size):
        raise ValueError(f"eos_id must be an integer in [0, {spec.vocab_size})")
    bad = check_decoder_weights(spec, w)
    if bad:
        raise ValueError("decoder weights rejected: " + "; ".join(bad))
    return prompt


def decoder_param_count(spec: DecoderSpec) -> int:
    """The parameter count the spec implies -- the manifest tie (rung 1's `param_count` over
    a shard census of these tensors must equal it)."""
    d, f, kvd = spec.d_model, spec.d_ff, spec.kv_dim
    if spec.activation == "silu_gate":  # gate+up+down, NO biases
        mlp = 3 * d * f
    else:  # up+down with biases
        mlp = d * f + f + f * d + d
    per_layer = 2 * d + 2 * d * d + 2 * d * kvd + mlp
    head = 0 if spec.tied_embeddings else spec.vocab_size * d
    return spec.vocab_size * d + spec.n_layers * per_layer + d + head


# --- the shared row arithmetic (BOTH decode paths call these on identical values) -----------


def _ff(h2: list, rows: int, spec: DecoderSpec, lw: LayerWeights) -> list:
    """The layer's MLP: the gated-SiLU form (silu_gate -- gate/up/down, no biases) or the
    classic two-matmul feed-forward (relu/gelu). One dispatch point, both decode paths."""
    if spec.activation == "silu_gate":
        return swiglu_reference(
            h2, rows, spec.d_model, spec.d_ff, list(lw.w_gate), list(lw.w1), list(lw.w2)
        )
    return feedforward_reference(
        h2,
        rows,
        spec.d_model,
        spec.d_ff,
        list(lw.w1),
        list(lw.b1),
        list(lw.w2),
        list(lw.b2),
        activation=spec.activation,
    )


def head_logits(h_row: list, w: DecoderWeights) -> list:
    """The next-token logits row: the TIED embedding table, or the untied lm_head (same
    [vocab x d] row-major layout, so both read through `_tied_logits`)."""
    if w.lm_head:
        emb = w.embedding
        return _tied_logits(
            h_row, EmbeddingTable(table=tuple(w.lm_head), n_vocab=emb.n_vocab, dim=emb.dim)
        )
    return _tied_logits(h_row, w.embedding)


# Private compatibility alias retained for serve/paged-KV callers from earlier rungs.  New
# code imports the public function so tied-vs-untied selection has one implementation point.
_head_logits = head_logits


def _split_head(rows: list, n: int, h: int, n_heads: int, d_k: int) -> list:
    """Head h's n x d_k slice of an n x (n_heads*d_k) projection (a pure gather)."""
    return [rows[i * n_heads * d_k + h * d_k + t] for i in range(n) for t in range(d_k)]


def _concat_heads(heads: list, n: int, d_k: int) -> list:
    """The inverse gather: h per-head n x d_k buffers back into one n x (h*d_k) buffer."""
    nh = len(heads)
    out = [0.0] * (n * nh * d_k)
    for h, rows in enumerate(heads):
        for i in range(n):
            out[i * nh * d_k + h * d_k : i * nh * d_k + (h + 1) * d_k] = rows[
                i * d_k : (i + 1) * d_k
            ]
    return out


def gqa_attention_reference(
    q: list, k: list, v: list, seq: int, n_heads: int, n_kv_heads: int, d_k: int
) -> list:
    """Causal GROUPED-QUERY attention over PRE-ROPED projections (rung 5's GQA primitive --
    and the C twin's differential target): `q` is seq x (n_heads*d_k); `k`/`v` are
    seq x (n_kv_heads*d_k); query head h reads kv head h // (n_heads // n_kv_heads). Per head:
    (Q_h @ K_g^T)/sqrt(d_k) + causal mask -> softmax -> @ V_g, ascending-index accumulation
    everywhere (the shared arithmetic order both decode paths and the C twin reproduce).
    n_kv_heads == n_heads is exactly multi-head attention. Returns seq x (n_heads*d_k)."""
    if n_kv_heads < 1 or n_kv_heads > n_heads or n_heads % n_kv_heads:
        raise ValueError(
            f"n_heads {n_heads} not divisible by n_kv_heads {n_kv_heads} "
            f"(GQA shares whole head groups)"
        )
    group = n_heads // n_kv_heads
    d = n_heads * d_k
    mask = causal_mask(seq)
    aspec = AttentionSpec(seq, d_k)
    concat = [0.0] * (seq * d)
    for hd in range(n_heads):
        qh = _split_head(q, seq, hd, n_heads, d_k)
        g = hd // group  # the shared kv head
        kh = _split_head(k, seq, g, n_kv_heads, d_k)
        vh = _split_head(v, seq, g, n_kv_heads, d_k)
        s = scores_reference(qh, kh, aspec)  # (Q_h @ K_g^T) / sqrt(d_k)
        s = [s[i] + mask[i] for i in range(seq * seq)]  # causal: exp(-inf) = 0
        a = softmax_reference(s, axis_len=seq)
        ctx = matmul_reference(a, vh, seq, d_k, seq)  # A @ V_g
        for i in range(seq):
            concat[i * d + hd * d_k : i * d + (hd + 1) * d_k] = ctx[i * d_k : (i + 1) * d_k]
    return concat


def _tied_logits(h_row: list, emb: EmbeddingTable) -> list:
    """logits[v] = <h, E_v> -- the TIED-embedding readout, ascending-k accumulation exactly
    like `matmul_reference` (so both decode paths share the arithmetic order)."""
    d = emb.dim
    out = [0.0] * emb.n_vocab
    for v in range(emb.n_vocab):
        s = 0.0
        o = v * d
        for k in range(d):
            s += h_row[k] * emb.table[o + k]
        out[v] = s
    return out


def _argmax(logits: list) -> int:
    """Greedy pick; Python `max` keeps the FIRST maximum, so ties break to the lowest id
    (deterministic decode)."""
    best = 0
    for v in range(1, len(logits)):
        if logits[v] > logits[best]:
            best = v
    return best


# --- the naive full-sequence path (the obviously-correct reference) -------------------------


def decoder_layer_reference(x: list, seq: int, spec: DecoderSpec, lw: LayerWeights) -> list:
    """One pre-norm decoder layer over the FULL seq x d_model input: RMSNorm -> RoPE'd causal
    grouped-query attention (`gqa_attention_reference`; n_kv_heads == n_heads is plain MHA)
    -> residual -> RMSNorm -> FF -> residual. Composed from the existing references (see the
    module docstring); returns a fresh seq x d_model buffer."""
    d, nh, dk, kvh, kvd = spec.d_model, spec.n_heads, spec.d_k, spec.kv_heads, spec.kv_dim
    h = rmsnorm_reference(x, seq, d, list(lw.g_attn), eps=spec.rms_norm_eps)
    q = matmul_reference(h, lw.w_q, seq, d, d)
    k = matmul_reference(h, lw.w_k, seq, kvd, d)  # seq x kv_dim (GQA: narrower)
    v = matmul_reference(h, lw.w_v, seq, kvd, d)
    q_r = _concat_heads(
        [
            rope_reference(_split_head(q, seq, hd, nh, dk), seq, dk, spec.rope_base)
            for hd in range(nh)
        ],
        seq,
        dk,
    )
    k_r = _concat_heads(
        [
            rope_reference(_split_head(k, seq, g, kvh, dk), seq, dk, spec.rope_base)
            for g in range(kvh)
        ],
        seq,
        dk,
    )
    concat = gqa_attention_reference(q_r, k_r, v, seq, nh, kvh, dk)
    attn = matmul_reference(concat, lw.w_o, seq, d, d)
    x = [x[i] + attn[i] for i in range(seq * d)]  # residual
    h2 = rmsnorm_reference(x, seq, d, list(lw.g_ff), eps=spec.rms_norm_eps)
    ff = _ff(h2, seq, spec, lw)
    return [x[i] + ff[i] for i in range(seq * d)]  # residual


def decoder_forward_reference(ids: list, spec: DecoderSpec, w: DecoderWeights) -> list:
    """The full forward: embed -> n_layers -> final RMSNorm. Returns seq x d_model."""
    seq = len(ids)
    x = embedding_lookup(w.embedding, ids, spec.d_model)
    for lw in w.layers:
        x = decoder_layer_reference(x, seq, spec, lw)
    return rmsnorm_reference(x, seq, spec.d_model, list(w.g_final), eps=spec.rms_norm_eps)


def next_token_logits(ids: list, spec: DecoderSpec, w: DecoderWeights) -> list:
    """The selected head's logits at the LAST position (the next-token distribution's scores)."""
    hfin = decoder_forward_reference(ids, spec, w)
    d = spec.d_model
    return head_logits(hfin[(len(ids) - 1) * d : len(ids) * d], w)


def reference_decode(
    prompt_ids: list, spec: DecoderSpec, w: DecoderWeights, max_new: int, eos_id: int | None = None
) -> list:
    """GREEDY decode by naive full-context recompute -- the slow, obviously-correct rung-3
    reference. Returns the NEW ids (up to `max_new`; stops after emitting `eos_id`)."""
    ids = _validate_decode_request(prompt_ids, spec, w, max_new, eos_id)
    out: list = []
    for _ in range(max_new):
        nxt = _argmax(next_token_logits(ids, spec, w))
        ids.append(nxt)
        out.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
    return out


# --- the incremental KV-cache twin (the rung's KV primitive) --------------------------------


class KVCache:
    """Per-layer, per-KV-HEAD cached K/V rows (K stored POST-RoPE at each token's absolute
    position, so an appended row never needs re-rotation). Grows one row per decoded token.
    Under GQA the cache holds `spec.kv_heads` head lanes -- the memory saving IS the point
    of grouped-query attention -- and every query head in a group reads the same lane."""

    def __init__(self, spec: DecoderSpec) -> None:
        self.k = [[[] for _ in range(spec.kv_heads)] for _ in range(spec.n_layers)]
        self.v = [[[] for _ in range(spec.kv_heads)] for _ in range(spec.n_layers)]
        self.pos = 0  # rows cached so far

    def _step_row(self, x_row: list, spec: DecoderSpec, w: DecoderWeights) -> list:
        """Advance the cache by ONE token (position `self.pos`): run the row through every
        layer, appending each layer's roped K row + V row ONCE PER KV HEAD, and return the
        final-norm row."""
        d, nh, dk, kvh, kvd = spec.d_model, spec.n_heads, spec.d_k, spec.kv_heads, spec.kv_dim
        group = nh // kvh
        sc = AttentionSpec(self.pos + 1, dk).scale  # 1/sqrt(d_k), like the naive path
        for li, lw in enumerate(w.layers):
            h = rmsnorm_reference(x_row, 1, d, list(lw.g_attn), eps=spec.rms_norm_eps)
            q = matmul_reference(h, lw.w_q, 1, d, d)
            k = matmul_reference(h, lw.w_k, 1, kvd, d)
            v = matmul_reference(h, lw.w_v, 1, kvd, d)
            for g in range(kvh):  # append ONCE per kv head
                kh = rope_reference(
                    _split_head(k, 1, g, kvh, dk), 1, dk, spec.rope_base, pos_offset=self.pos
                )
                self.k[li][g].append(kh)
                self.v[li][g].append(_split_head(v, 1, g, kvh, dk))
            concat = [0.0] * d
            for hd in range(nh):
                qh = rope_reference(
                    _split_head(q, 1, hd, nh, dk), 1, dk, spec.rope_base, pos_offset=self.pos
                )
                g = hd // group  # the shared kv lane
                t_len = len(self.k[li][g])
                s = [0.0] * t_len  # this row of (Q@K^T)/sqrt(d_k)
                for j in range(t_len):
                    acc = 0.0
                    kr = self.k[li][g][j]
                    for c in range(dk):
                        acc += qh[c] * kr[c]
                    s[j] = acc * sc
                a = softmax_reference(s, axis_len=t_len)
                ctx = [0.0] * dk  # A @ V_g, ascending j like matmul
                for c in range(dk):
                    acc = 0.0
                    for j in range(t_len):
                        acc += a[j] * self.v[li][g][j][c]
                    ctx[c] = acc
                concat[hd * dk : (hd + 1) * dk] = ctx
            attn = matmul_reference(concat, lw.w_o, 1, d, d)
            x_row = [x_row[i] + attn[i] for i in range(d)]
            h2 = rmsnorm_reference(x_row, 1, d, list(lw.g_ff), eps=spec.rms_norm_eps)
            ff = _ff(h2, 1, spec, lw)
            x_row = [x_row[i] + ff[i] for i in range(d)]
        self.pos += 1
        return rmsnorm_reference(x_row, 1, d, list(w.g_final), eps=spec.rms_norm_eps)


def decode_with_kv_cache(
    prompt_ids: list, spec: DecoderSpec, w: DecoderWeights, max_new: int, eos_id: int | None = None
) -> list:
    """GREEDY decode with an incremental KV cache: prefill the prompt one row at a time, then
    one `_step_row` per generated token. Every row runs the SAME arithmetic as the naive path
    (row-independent stages; a masked naive score is an exact trailing zero), so the emitted
    ids match `reference_decode` BIT-FOR-BIT -- the twin gate."""
    prompt = _validate_decode_request(prompt_ids, spec, w, max_new, eos_id)
    cache = KVCache(spec)
    hrow: list = []
    for tid in prompt:  # prefill
        hrow = cache._step_row(w.embedding.row(tid), spec, w)
    out: list = []
    for _ in range(max_new):
        nxt = _argmax(head_logits(hrow, w))
        out.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
        hrow = cache._step_row(w.embedding.row(nxt), spec, w)
    return out
