"""Rung 3 of the open-weight ladder (ML/AI roadmap §7.4): the REFERENCE DECODER.

A slow, dependency-light Python reference for a small DENSE decoder (the Gemma/Llama-family
pre-norm shape), COMPOSED from the existing oracle references -- not reinvented:

    embedding_lookup (unsupervised)  ->  per layer [ rmsnorm_reference (transformer_grads)
    -> Q/K/V projections (matmul_reference) -> rope_reference on Q,K per head
    -> causal scaled-dot-product attention (attention.scores_reference + causal_mask
       + softmax_reference + matmul_reference) -> W_o + residual
    -> rmsnorm_reference -> feedforward_reference (transformer) + residual ]
    ->  final rmsnorm_reference  ->  TIED-embedding logits  ->  greedy argmax.

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

from dataclasses import dataclass

from ...kbcir.activation import softmax_reference
from ...kbcir.attention import AttentionSpec, scores_reference
from ...kbcir.matmul import matmul_reference
from ...kbcir.transformer import causal_mask, feedforward_reference
from ...kbcir.transformer_grads import rmsnorm_reference, rope_reference
from ...kbcir.unsupervised import EmbeddingTable, embedding_lookup


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
    n_kv_heads: int = 0                 # 0 == n_heads (MHA); the Llama/Gemma GQA knob

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.d_model, self.n_heads, self.n_layers, self.d_ff) < 1:
            raise ValueError("decoder dims must all be >= 1")
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("d_k must be even (RoPE rotates channel pairs)")
        if self.activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be relu|gelu; got {self.activation!r}")
        if self.n_kv_heads < 0 or self.n_kv_heads > self.n_heads:
            raise ValueError(f"n_kv_heads {self.n_kv_heads} must be in [0, n_heads]")
        if self.n_kv_heads and self.n_heads % self.n_kv_heads:
            raise ValueError(f"n_heads {self.n_heads} not divisible by "
                             f"n_kv_heads {self.n_kv_heads} (GQA shares whole head groups)")

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

    g_attn: tuple          # d_model            (attention pre-norm gamma)
    w_q: tuple             # d_model x d_model
    w_k: tuple             # d_model x kv_dim   (== d_model for MHA)
    w_v: tuple             # d_model x kv_dim
    w_o: tuple             # d_model x d_model
    g_ff: tuple            # d_model            (FF pre-norm gamma)
    w1: tuple              # d_model x d_ff
    b1: tuple              # d_ff
    w2: tuple              # d_ff x d_model
    b2: tuple              # d_model


@dataclass(frozen=True)
class DecoderWeights:
    """The whole decoder: the embedding table (ALSO the tied logits head, the Gemma choice),
    the per-layer weights, and the final-norm gamma."""

    embedding: EmbeddingTable
    layers: tuple            # tuple[LayerWeights, ...]
    g_final: tuple           # d_model


def check_decoder_weights(spec: DecoderSpec, w: DecoderWeights) -> list[str]:
    """Op-level well-formedness (mirrors `check_attention`/`check_classical`; NOT a new
    global R-law): every weight's length against the spec. Returns messages, [] when clean."""
    d, f, kvd = spec.d_model, spec.d_ff, spec.kv_dim
    msgs: list[str] = []
    if (w.embedding.n_vocab, w.embedding.dim) != (spec.vocab_size, d):
        msgs.append(f"embedding is {w.embedding.n_vocab}x{w.embedding.dim}, "
                    f"spec says {spec.vocab_size}x{d}")
    if len(w.layers) != spec.n_layers:
        msgs.append(f"{len(w.layers)} layers, spec says {spec.n_layers}")
    if len(w.g_final) != d:
        msgs.append(f"g_final has {len(w.g_final)} entries, want {d}")
    want = {"g_attn": d, "w_q": d * d, "w_k": d * kvd, "w_v": d * kvd, "w_o": d * d,
            "g_ff": d, "w1": d * f, "b1": f, "w2": f * d, "b2": d}
    for li, lw in enumerate(w.layers):
        for name, n in want.items():
            if len(getattr(lw, name)) != n:
                msgs.append(f"layer {li}: {name} has {len(getattr(lw, name))} entries, want {n}")
    return msgs


def decoder_param_count(spec: DecoderSpec) -> int:
    """The parameter count the spec implies -- the manifest tie (rung 1's `param_count` over
    a shard census of these tensors must equal it)."""
    d, f, kvd = spec.d_model, spec.d_ff, spec.kv_dim
    per_layer = 2 * d + 2 * d * d + 2 * d * kvd + d * f + f + f * d + d
    return spec.vocab_size * d + spec.n_layers * per_layer + d


# --- the shared row arithmetic (BOTH decode paths call these on identical values) -----------

def _split_head(rows: list, n: int, h: int, n_heads: int, d_k: int) -> list:
    """Head h's n x d_k slice of an n x (n_heads*d_k) projection (a pure gather)."""
    return [rows[i * n_heads * d_k + h * d_k + t] for i in range(n) for t in range(d_k)]


def _concat_heads(heads: list, n: int, d_k: int) -> list:
    """The inverse gather: h per-head n x d_k buffers back into one n x (h*d_k) buffer."""
    nh = len(heads)
    out = [0.0] * (n * nh * d_k)
    for h, rows in enumerate(heads):
        for i in range(n):
            out[i * nh * d_k + h * d_k:i * nh * d_k + (h + 1) * d_k] = rows[i * d_k:(i + 1) * d_k]
    return out


def gqa_attention_reference(q: list, k: list, v: list, seq: int,
                            n_heads: int, n_kv_heads: int, d_k: int) -> list:
    """Causal GROUPED-QUERY attention over PRE-ROPED projections (rung 5's GQA primitive --
    and the C twin's differential target): `q` is seq x (n_heads*d_k); `k`/`v` are
    seq x (n_kv_heads*d_k); query head h reads kv head h // (n_heads // n_kv_heads). Per head:
    (Q_h @ K_g^T)/sqrt(d_k) + causal mask -> softmax -> @ V_g, ascending-index accumulation
    everywhere (the shared arithmetic order both decode paths and the C twin reproduce).
    n_kv_heads == n_heads is exactly multi-head attention. Returns seq x (n_heads*d_k)."""
    if n_kv_heads < 1 or n_kv_heads > n_heads or n_heads % n_kv_heads:
        raise ValueError(f"n_heads {n_heads} not divisible by n_kv_heads {n_kv_heads} "
                         f"(GQA shares whole head groups)")
    group = n_heads // n_kv_heads
    d = n_heads * d_k
    mask = causal_mask(seq)
    aspec = AttentionSpec(seq, d_k)
    concat = [0.0] * (seq * d)
    for hd in range(n_heads):
        qh = _split_head(q, seq, hd, n_heads, d_k)
        g = hd // group                                           # the shared kv head
        kh = _split_head(k, seq, g, n_kv_heads, d_k)
        vh = _split_head(v, seq, g, n_kv_heads, d_k)
        s = scores_reference(qh, kh, aspec)                       # (Q_h @ K_g^T) / sqrt(d_k)
        s = [s[i] + mask[i] for i in range(seq * seq)]            # causal: exp(-inf) = 0
        a = softmax_reference(s, axis_len=seq)
        ctx = matmul_reference(a, vh, seq, d_k, seq)              # A @ V_g
        for i in range(seq):
            concat[i * d + hd * d_k:i * d + (hd + 1) * d_k] = ctx[i * d_k:(i + 1) * d_k]
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
    h = rmsnorm_reference(x, seq, d, list(lw.g_attn))
    q = matmul_reference(h, lw.w_q, seq, d, d)
    k = matmul_reference(h, lw.w_k, seq, kvd, d)                  # seq x kv_dim (GQA: narrower)
    v = matmul_reference(h, lw.w_v, seq, kvd, d)
    q_r = _concat_heads([rope_reference(_split_head(q, seq, hd, nh, dk), seq, dk,
                                        spec.rope_base) for hd in range(nh)], seq, dk)
    k_r = _concat_heads([rope_reference(_split_head(k, seq, g, kvh, dk), seq, dk,
                                        spec.rope_base) for g in range(kvh)], seq, dk)
    concat = gqa_attention_reference(q_r, k_r, v, seq, nh, kvh, dk)
    attn = matmul_reference(concat, lw.w_o, seq, d, d)
    x = [x[i] + attn[i] for i in range(seq * d)]                  # residual
    h2 = rmsnorm_reference(x, seq, d, list(lw.g_ff))
    ff = feedforward_reference(h2, seq, d, spec.d_ff, list(lw.w1), list(lw.b1),
                               list(lw.w2), list(lw.b2), activation=spec.activation)
    return [x[i] + ff[i] for i in range(seq * d)]                 # residual


def decoder_forward_reference(ids: list, spec: DecoderSpec, w: DecoderWeights) -> list:
    """The full forward: embed -> n_layers -> final RMSNorm. Returns seq x d_model."""
    seq = len(ids)
    x = embedding_lookup(w.embedding, ids, spec.d_model)
    for lw in w.layers:
        x = decoder_layer_reference(x, seq, spec, lw)
    return rmsnorm_reference(x, seq, spec.d_model, list(w.g_final))


def next_token_logits(ids: list, spec: DecoderSpec, w: DecoderWeights) -> list:
    """The tied-embedding logits of the LAST position (the next-token distribution's scores)."""
    hfin = decoder_forward_reference(ids, spec, w)
    d = spec.d_model
    return _tied_logits(hfin[(len(ids) - 1) * d:len(ids) * d], w.embedding)


def reference_decode(prompt_ids: list, spec: DecoderSpec, w: DecoderWeights, max_new: int,
                     eos_id: int | None = None) -> list:
    """GREEDY decode by naive full-context recompute -- the slow, obviously-correct rung-3
    reference. Returns the NEW ids (up to `max_new`; stops after emitting `eos_id`)."""
    if not prompt_ids or max_new < 0:
        raise ValueError("reference_decode needs a non-empty prompt and max_new >= 0")
    ids = list(prompt_ids)
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
        self.pos = 0                                             # rows cached so far

    def _step_row(self, x_row: list, spec: DecoderSpec, w: DecoderWeights) -> list:
        """Advance the cache by ONE token (position `self.pos`): run the row through every
        layer, appending each layer's roped K row + V row ONCE PER KV HEAD, and return the
        final-norm row."""
        d, nh, dk, kvh, kvd = spec.d_model, spec.n_heads, spec.d_k, spec.kv_heads, spec.kv_dim
        group = nh // kvh
        sc = AttentionSpec(self.pos + 1, dk).scale               # 1/sqrt(d_k), like the naive path
        for li, lw in enumerate(w.layers):
            h = rmsnorm_reference(x_row, 1, d, list(lw.g_attn))
            q = matmul_reference(h, lw.w_q, 1, d, d)
            k = matmul_reference(h, lw.w_k, 1, kvd, d)
            v = matmul_reference(h, lw.w_v, 1, kvd, d)
            for g in range(kvh):                                 # append ONCE per kv head
                kh = rope_reference(_split_head(k, 1, g, kvh, dk), 1, dk, spec.rope_base,
                                    pos_offset=self.pos)
                self.k[li][g].append(kh)
                self.v[li][g].append(_split_head(v, 1, g, kvh, dk))
            concat = [0.0] * d
            for hd in range(nh):
                qh = rope_reference(_split_head(q, 1, hd, nh, dk), 1, dk, spec.rope_base,
                                    pos_offset=self.pos)
                g = hd // group                                  # the shared kv lane
                t_len = len(self.k[li][g])
                s = [0.0] * t_len                                # this row of (Q@K^T)/sqrt(d_k)
                for j in range(t_len):
                    acc = 0.0
                    kr = self.k[li][g][j]
                    for c in range(dk):
                        acc += qh[c] * kr[c]
                    s[j] = acc * sc
                a = softmax_reference(s, axis_len=t_len)
                ctx = [0.0] * dk                                 # A @ V_g, ascending j like matmul
                for c in range(dk):
                    acc = 0.0
                    for j in range(t_len):
                        acc += a[j] * self.v[li][g][j][c]
                    ctx[c] = acc
                concat[hd * dk:(hd + 1) * dk] = ctx
            attn = matmul_reference(concat, lw.w_o, 1, d, d)
            x_row = [x_row[i] + attn[i] for i in range(d)]
            h2 = rmsnorm_reference(x_row, 1, d, list(lw.g_ff))
            ff = feedforward_reference(h2, 1, d, spec.d_ff, list(lw.w1), list(lw.b1),
                                       list(lw.w2), list(lw.b2), activation=spec.activation)
            x_row = [x_row[i] + ff[i] for i in range(d)]
        self.pos += 1
        return rmsnorm_reference(x_row, 1, d, list(w.g_final))


def decode_with_kv_cache(prompt_ids: list, spec: DecoderSpec, w: DecoderWeights, max_new: int,
                         eos_id: int | None = None) -> list:
    """GREEDY decode with an incremental KV cache: prefill the prompt one row at a time, then
    one `_step_row` per generated token. Every row runs the SAME arithmetic as the naive path
    (row-independent stages; a masked naive score is an exact trailing zero), so the emitted
    ids match `reference_decode` BIT-FOR-BIT -- the twin gate."""
    if not prompt_ids or max_new < 0:
        raise ValueError("decode_with_kv_cache needs a non-empty prompt and max_new >= 0")
    cache = KVCache(spec)
    hrow: list = []
    for tid in prompt_ids:                                       # prefill
        hrow = cache._step_row(w.embedding.row(int(tid)), spec, w)
    out: list = []
    for _ in range(max_new):
        nxt = _argmax(_tied_logits(hrow, w.embedding))
        out.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
        hrow = cache._step_row(w.embedding.row(nxt), spec, w)
    return out
