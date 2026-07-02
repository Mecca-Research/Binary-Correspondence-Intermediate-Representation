"""E3 (ML-breadth) -- the full TRANSFORMER ENCODER BLOCK: an ORACLE-SIDE COMPOSITION of the ops BCIR already
has (matmul + softmax + a new layernorm), the THIRD slice of the ML-breadth ladder after E1 (OLS) and E2 (PCA).

WHY THIS IS A COMPOSITION, NOT A LIBRARY WRAP (the load-bearing framing): E1 and E2 each WRAPPED a single
external LAPACK kernel through the trusted ``c.call.libm:`` edge (``sgels`` / ``ssyev`` -- "integrate, don't
reinvent" one opaque routine). E3 is DIFFERENT: a transformer block is not one external kernel, it is a
COMPOSITION of ops BCIR already ships -- two matmuls + a scale + the EXISTING softmax (per head) + a residual
add + a new layernorm + a feed-forward (two matmuls + an activation). So this module mirrors
``bcir.kbcir.attention`` (the G7 single-head attention -- itself a composition), NOT the LAPACK-wrap pattern.
attention.py's own docstring named the deferred follow-up: *"Multi-head, batched, masked/causal attention ...
are the deferred follow-up."* THIS module IS that follow-up. So it REUSES, never reinvents:

  * ``attention.scores_reference``     -- the scaled ``Q@K^T / sqrt(d_k)`` per head (reused verbatim);
  * ``activation.softmax_reference``   -- the numerically-stable last-axis softmax (reused verbatim);
  * ``matmul.matmul_reference``        -- every projection / ``A@V`` / feed-forward GEMM (reused verbatim).

THE ONE NEW NUMERIC PRIMITIVE is LAYERNORM. Every other component already exists and is already tested. So the
ONLY net-new numeric kernel (Python reference here + the C twin ``lower.c_kernel.emit_layernorm_c``) is the
per-row layernorm; the matmul / softmax / single-head-attention C twins (``emit_attention_c``, the matmul
emitters) already exist and the rest of the block composes those already-tested twins.

THE QUARANTINE BOUNDARY (the load-bearing constraint -- see also kbcir.attention / activation / precision). Two
transcendentals appear and BOTH ride the trusted ``c.call.libm:`` edge, fully OFF the deterministic legality
rail (libm is opaque/trusted, like any external edge):
  * the softmax's ``exp`` (expf) -- already quarantined by activation.softmax;
  * layernorm's ``1/sqrt(var+eps)`` (sqrtf) -- the ONE new libm edge here (``c.call.libm:sqrtf`` -> ``-lm``,
    which the link rail ALREADY maps; no linkflags change).
Every other op (the matmuls, the scale, the residual adds, the centering/variance sums) is exact/deterministic.
So:
  * The DENSE ``transformer_block_reference`` matches a naive from-scratch composition to FLOAT ROUND-OFF (the
    softmax exp + the layernorm rsqrt ride the libm edge per the quarantine; the same float story as
    activation.softmax / attention / B5).
  * The ONLY certified error on a QUANTIZED block is the Q8<->f32<->Q8 input round-trip (R17), exactly as B5 /
    G1 / G7: ``transformer_block_via_bridge`` round-trips the input (and optionally the weights) through the
    bridge and runs the trusted block, so the bridge is the SOLE error source
    (``precision.quantization_error_bound``); the matmuls are exact and softmax/sqrt ride the trusted libm edge.

POST-LN (the canonical choice). This is the original "Attention Is All You Need" POST-LN encoder block: the
layernorm is applied AFTER each sublayer's residual add --
    a  = MultiHeadAttention(x)        ;  x1  = LayerNorm(x  + a, gamma1, beta1)
    f  = FeedForward(x1)              ;  out = LayerNorm(x1 + f, gamma2, beta2)
The one-line PRE-LN variant (GPT-2 / modern stacks; better deep-stack gradient flow) normalizes BEFORE each
sublayer and keeps the residual around the RAW input:  x1 = x + Attn(LN(x)) ; out = x1 + FF(LN(x1)). Post-LN is
the default here; ``pre_ln=True`` selects the variant (documented + tested).

LAYOUT (row-major throughout). ``x`` is the (batch*seq) x d_model token matrix row-major: ``x[(b*seq + i)*d_model
+ c]`` is batch b, token i, channel c. A single sequence is batch == 1. Each projection weight ``W_*`` is
d_model x d_model row-major; ``W1`` is d_model x d_ff, ``W2`` is d_ff x d_model; the biases are length d_ff
(``b1``) / d_model (``b2``); ``gamma*``/``beta*`` are length d_model. Batch is a SIMPLE OUTER LOOP over the B
independent sequences (documented; no fused batched matmul). All functions return FRESH buffers and do NOT mutate
their inputs (side-effect-free oracle).

This module is OFF the legality path: it imports NO verifier, emits NO Diagnostic, and returns plain floats (no
graded/confidence wrapper) -- exactly like attention.py / pca.py / ols.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import attention, matmul
from .activation import gelu_reference, relu_reference, softmax_reference
from .precision import quantization_error_bound
from .quantize import dequantize, quantize_per_group

# The default layernorm numerical-stability floor (the standard PyTorch / TF default). Added inside the sqrt so
# a zero-variance row never divides by zero -- the layernorm analog of softmax subtracting the row max.
_DEFAULT_EPS = 1e-5

# A large finite stand-in for -inf in an additive mask. math.exp(-inf)=0.0 in Python (no error) so a true
# float("-inf") works through softmax_reference; -1e30 is the finite alternative a caller may prefer (it makes
# expf underflow to ~0 just the same). The causal mask uses float("-inf") by default.
NEG_INF = float("-inf")


# ============================================================================================================
# 1. LayerNorm -- the ONE new numeric primitive (the only net-new reference; rides the libm sqrtf edge).
# ============================================================================================================

def layernorm_reference(x: list[float], rows: int, dim: int,
                        gamma: list[float], beta: list[float], eps: float = _DEFAULT_EPS) -> list[float]:
    """Per-ROW layer normalization over the ``dim`` feature axis -- the single source of truth, the net-new
    numeric primitive of E3. ``x`` is ``rows x dim`` row-major (``x[r*dim + c]``); for each row r:

        mean = (1/dim) * sum_c x[r,c]
        var  = (1/dim) * sum_c (x[r,c] - mean)^2          (POPULATION variance: the /dim divisor, NOT /(dim-1))
        out[r,c] = gamma[c] * (x[r,c] - mean) / sqrt(var + eps) + beta[c]

    ``gamma``/``beta`` are length ``dim`` (the learned per-channel scale/shift; gamma=1, beta=0 is the pure
    normalization). The ``sqrt`` is the ONLY transcendental -- it rides the trusted ``c.call.libm:sqrtf`` edge
    in the C twin (``-lm``, already mapped). The mean/variance sums and the scale/shift are exact. Returns a
    FRESH ``rows x dim`` buffer; ``x`` is NOT mutated (side-effect-free). Validates dims (rows>=1, dim>=1,
    len(x)==rows*dim, len(gamma)==len(beta)==dim, eps>0).

    Population (/dim) variance is the layernorm convention (PyTorch ``nn.LayerNorm`` / TF use the biased /dim
    estimator, NOT the sample /(dim-1) one -- a normalized row should have variance exactly 1, which /dim gives
    and /(dim-1) would not). ``layernorm_stats`` recomputes (mean, var) independently so a test can assert a
    gamma=1/beta=0 row has mean~0, var~1."""
    if rows < 1:
        raise ValueError(f"layernorm rows must be >= 1; got {rows}")
    if dim < 1:
        raise ValueError(f"layernorm dim must be >= 1; got {dim}")
    if len(x) != rows * dim:
        raise ValueError(f"layernorm input must have rows*dim = {rows * dim} entries; got {len(x)}")
    if len(gamma) != dim:
        raise ValueError(f"layernorm gamma must be length dim = {dim}; got {len(gamma)}")
    if len(beta) != dim:
        raise ValueError(f"layernorm beta must be length dim = {dim}; got {len(beta)}")
    if eps <= 0.0:
        raise ValueError(f"layernorm eps must be > 0; got {eps}")
    out = [0.0] * (rows * dim)
    for r in range(rows):
        base = r * dim
        mean = 0.0
        for c in range(dim):
            mean += float(x[base + c])
        mean /= dim
        var = 0.0
        for c in range(dim):
            d = float(x[base + c]) - mean
            var += d * d
        var /= dim                                          # population variance (/dim, biased -- the LN rule)
        inv = 1.0 / math.sqrt(var + eps)                    # the rsqrt -- the ONLY transcendental (libm sqrtf)
        for c in range(dim):
            out[base + c] = float(gamma[c]) * (float(x[base + c]) - mean) * inv + float(beta[c])
    return out


def layernorm_stats(y: list[float], rows: int, dim: int) -> list[tuple[float, float]]:
    """INDEPENDENT verifier: recompute each row's (mean, population-variance) DIRECTLY from ``y`` (not via the
    layernorm code path) -- the analog of OLS's ``normal_equation_residual`` / PCA's ``eigen_residual``. A row
    normalized with gamma=1, beta=0 must have mean ~ 0 and variance ~ 1; a test asserts exactly that against
    this. ``y`` is ``rows x dim`` row-major; returns a length-``rows`` list of (mean, var) pairs."""
    if rows < 1 or dim < 1 or len(y) != rows * dim:
        raise ValueError(f"layernorm_stats: need rows>=1, dim>=1, len(y)==rows*dim; "
                         f"got rows={rows} dim={dim} len(y)={len(y)}")
    stats: list[tuple[float, float]] = []
    for r in range(rows):
        base = r * dim
        mean = sum(float(y[base + c]) for c in range(dim)) / dim
        var = sum((float(y[base + c]) - mean) ** 2 for c in range(dim)) / dim
        stats.append((mean, var))
    return stats


# ============================================================================================================
# 2. Masked multi-head attention -- REUSES attention.scores_reference + softmax_reference + matmul_reference.
# ============================================================================================================

def causal_mask(seq_len: int, neg: float = NEG_INF) -> list[float]:
    """The additive CAUSAL (look-ahead) mask: a flat ``seq_len x seq_len`` row-major matrix with ``0.0`` for
    j <= i (token i may attend to token j at or before it) and ``neg`` (``-inf`` by default) for j > i (the
    future is masked out). ADDED to the scaled scores before softmax: ``exp(-inf)=0`` (Python: no error), so the
    masked logits contribute zero weight. CRITICAL: the diagonal (j == i) is ALWAYS unmasked, so every row keeps
    >= 1 finite entry -- the softmax of an all-``-inf`` row would be NaN, but a causal mask can never produce
    one. A caller may instead pass any custom additive mask of this shape to the attention functions."""
    if seq_len < 1:
        raise ValueError(f"causal_mask seq_len must be >= 1; got {seq_len}")
    m = [0.0] * (seq_len * seq_len)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            m[i * seq_len + j] = neg                        # mask the strictly-upper triangle (the future)
    return m


def _project(x: list[float], w: list[float], rows: int, d_model: int) -> list[float]:
    """A token-wise linear projection ``x @ W`` (rows x d_model times d_model x d_model -> rows x d_model), a
    plain ``matmul.matmul_reference`` (reused). ``w`` is d_model x d_model row-major."""
    return matmul.matmul_reference(x, w, rows, d_model, d_model)


def _split_head(proj: list[float], seq: int, h: int, n_heads: int, d_k: int) -> list[float]:
    """Slice head ``h``'s ``seq x d_k`` block out of a ``seq x d_model`` projection (row-major). Head h owns
    channels ``[h*d_k : (h+1)*d_k]`` of every token row (the standard contiguous head split)."""
    d_model = n_heads * d_k
    return [proj[i * d_model + h * d_k + t] for i in range(seq) for t in range(d_k)]


def _single_head(qh: list[float], kh: list[float], vh: list[float], seq: int, d_k: int,
                 mask: list[float] | None) -> list[float]:
    """One head's masked scaled-dot-product attention, REUSING the existing pieces: ``attention.scores_reference``
    for the scaled ``Q_h @ K_h^T / sqrt(d_k)``, ADD the additive mask (if any), then ``softmax_reference`` over
    each row, then ``matmul.matmul_reference`` for ``A @ V_h``. Returns the ``seq x d_k`` head context. This is
    exactly the single-head attention path -- NOT reinvented, composed from attention.py + activation + matmul."""
    spec = attention.AttentionSpec(seq, d_k)
    scores = attention.scores_reference(qh, kh, spec)       # (Q_h @ K_h^T) / sqrt(d_k)  -- REUSED
    if mask is not None:
        scores = [scores[idx] + float(mask[idx]) for idx in range(seq * seq)]   # additive mask -> -inf -> exp=0
    weights = softmax_reference(scores, axis_len=seq)       # the EXISTING softmax over each row -- REUSED
    return matmul.matmul_reference(weights, vh, seq, d_k, seq)   # A @ V_h -- REUSED


@dataclass(frozen=True)
class _MHAParams:
    """The four projection weights of one multi-head attention sublayer (each d_model x d_model row-major)."""
    w_q: list[float]
    w_k: list[float]
    w_v: list[float]
    w_o: list[float]


def multihead_attention_reference(x: list[float], spec: "TransformerBlockSpec", params: "_MHAParams",
                                  mask: list[float] | None = None) -> list[float]:
    """Masked MULTI-HEAD attention over a BATCH of sequences, composing the existing ops (no reinvention).
    ``x`` is (batch*seq) x d_model row-major. Per sequence in the batch (a SIMPLE OUTER LOOP -- the documented
    batch handling): project ``x`` by ``W_q/W_k/W_v`` (each d_model x d_model) into Q,K,V; split into
    ``n_heads`` heads of ``d_k = d_model/n_heads``; run each head's masked scaled-dot-product attention via
    ``_single_head`` (which REUSES attention.scores_reference + the EXISTING softmax + matmul); concatenate the
    head contexts back to ``seq x d_model``; project by ``W_o`` (d_model x d_model). Returns the (batch*seq) x
    d_model output. The same additive ``mask`` (a ``seq x seq`` row-major matrix, e.g. ``causal_mask(seq)``) is
    applied to every head of every sequence; pass ``None`` for full (unmasked) attention. Side-effect-free."""
    d_model, n_heads, d_k, seq, batch = spec.d_model, spec.n_heads, spec.d_k, spec.seq_len, spec.batch
    if len(x) != batch * seq * d_model:
        raise ValueError(f"mha input must have batch*seq*d_model = {batch * seq * d_model} entries; got {len(x)}")
    if mask is not None and len(mask) != seq * seq:
        raise ValueError(f"mha mask must be seq*seq = {seq * seq} entries; got {len(mask)}")
    out = [0.0] * (batch * seq * d_model)
    for b in range(batch):                                  # the batch outer loop (independent sequences)
        xs = x[b * seq * d_model:(b + 1) * seq * d_model]   # this sequence: seq x d_model
        q = _project(xs, params.w_q, seq, d_model)          # Q,K,V projections (each seq x d_model) -- REUSED
        k = _project(xs, params.w_k, seq, d_model)
        v = _project(xs, params.w_v, seq, d_model)
        concat = [0.0] * (seq * d_model)                    # the concatenated per-head contexts (seq x d_model)
        for h in range(n_heads):
            qh = _split_head(q, seq, h, n_heads, d_k)        # head h's seq x d_k slices of Q,K,V
            kh = _split_head(k, seq, h, n_heads, d_k)
            vh = _split_head(v, seq, h, n_heads, d_k)
            ctx = _single_head(qh, kh, vh, seq, d_k, mask)   # masked single-head attention -- REUSED pieces
            for i in range(seq):                             # write head h back into channels [h*d_k:(h+1)*d_k]
                for t in range(d_k):
                    concat[i * d_model + h * d_k + t] = ctx[i * d_k + t]
        proj = _project(concat, params.w_o, seq, d_model)   # the output projection W_o (seq x d_model) -- REUSED
        for idx in range(seq * d_model):
            out[b * seq * d_model + idx] = proj[idx]
    return out


# ============================================================================================================
# 3. Feed-forward -- REUSES matmul.matmul_reference + the existing activation references.
# ============================================================================================================

def feedforward_reference(x: list[float], rows: int, d_model: int, d_ff: int,
                          W1: list[float], b1: list[float], W2: list[float], b2: list[float],
                          activation: str = "relu") -> list[float]:
    """The position-wise FEED-FORWARD network ``out = act(x @ W1 + b1) @ W2 + b2`` -- two matmuls + a bias add +
    an activation, all REUSING ``matmul.matmul_reference`` and the existing activation references. ``x`` is
    ``rows x d_model`` row-major; ``W1`` is d_model x d_ff, ``b1`` length d_ff; ``W2`` is d_ff x d_model, ``b2``
    length d_model. ``activation`` is "relu" (the exact clean-rail max, the original Transformer choice) or
    "gelu" (the GPT/BERT tanh-approx, which rides the libm tanhf edge). Returns a FRESH ``rows x d_model``
    buffer. Side-effect-free. Validates the weight/bias shapes."""
    if rows < 1 or d_model < 1 or d_ff < 1:
        raise ValueError(f"feedforward dims must be >= 1; got rows={rows} d_model={d_model} d_ff={d_ff}")
    if len(x) != rows * d_model:
        raise ValueError(f"feedforward input must have rows*d_model = {rows * d_model} entries; got {len(x)}")
    if len(W1) != d_model * d_ff or len(b1) != d_ff:
        raise ValueError(f"feedforward W1 must be d_model*d_ff = {d_model * d_ff}, b1 length d_ff = {d_ff}")
    if len(W2) != d_ff * d_model or len(b2) != d_model:
        raise ValueError(f"feedforward W2 must be d_ff*d_model = {d_ff * d_model}, b2 length d_model = {d_model}")
    if activation not in ("relu", "gelu"):
        raise ValueError(f"feedforward activation must be 'relu' or 'gelu'; got {activation!r}")
    h = matmul.matmul_reference(x, W1, rows, d_ff, d_model)          # x @ W1 : rows x d_ff -- REUSED
    h = [h[i * d_ff + j] + float(b1[j]) for i in range(rows) for j in range(d_ff)]   # + b1 (broadcast per row)
    h = relu_reference(h) if activation == "relu" else gelu_reference(h)             # the activation -- REUSED
    o = matmul.matmul_reference(h, W2, rows, d_model, d_ff)          # . @ W2 : rows x d_model -- REUSED
    return [o[i * d_model + j] + float(b2[j]) for i in range(rows) for j in range(d_model)]   # + b2


# ============================================================================================================
# 4./5. The block + the shape/param carriers.
# ============================================================================================================

@dataclass(frozen=True)
class TransformerBlockSpec:
    """A gem.transformer_block as a FIRST-CLASS planned claim: the block's shape/dtype metadata, mirroring
    ``attention.AttentionSpec``. ``d_model`` is the model width, ``n_heads`` the head count (``d_k = d_model //
    n_heads`` is derived), ``d_ff`` the feed-forward inner width, ``seq_len`` the sequence length, ``batch`` the
    number of independent sequences (default 1), ``dtype`` the contracted element type. ``d_model`` must be
    divisible by ``n_heads`` (each head owns d_k contiguous channels)."""

    d_model: int
    n_heads: int
    d_ff: int
    seq_len: int
    batch: int = 1
    dtype: str = "f32"

    @property
    def d_k(self) -> int:
        """The per-head key/query dimension d_model // n_heads (the channels each head owns)."""
        return self.d_model // self.n_heads

    @property
    def rows(self) -> int:
        """The total token count batch*seq_len (the row count of the flattened token matrix)."""
        return self.batch * self.seq_len

    @property
    def token_count(self) -> int:
        return self.batch * self.seq_len * self.d_model


@dataclass(frozen=True)
class TransformerBlockParams:
    """The full weight set of one POST-LN Transformer encoder block (all row-major). The attention projections
    ``w_q``/``w_k``/``w_v``/``w_o`` are each d_model x d_model; the feed-forward ``W1`` is d_model x d_ff with
    bias ``b1`` (length d_ff), ``W2`` is d_ff x d_model with bias ``b2`` (length d_model); the two layernorms
    carry ``gamma1``/``beta1`` (post-attention) and ``gamma2``/``beta2`` (post-feed-forward), each length
    d_model. Mirrors AttentionSpec's role as the params carrier."""

    w_q: list[float]
    w_k: list[float]
    w_v: list[float]
    w_o: list[float]
    W1: list[float]
    b1: list[float]
    W2: list[float]
    b2: list[float]
    gamma1: list[float]
    beta1: list[float]
    gamma2: list[float]
    beta2: list[float]

    @property
    def mha(self) -> "_MHAParams":
        """The four attention projections as the ``_MHAParams`` the multi-head path consumes."""
        return _MHAParams(self.w_q, self.w_k, self.w_v, self.w_o)


def transformer_block_reference(x: list[float], spec: TransformerBlockSpec, params: TransformerBlockParams,
                                mask: list[float] | None = None, activation: str = "relu",
                                eps: float = _DEFAULT_EPS, pre_ln: bool = False) -> list[float]:
    """The canonical POST-LN Transformer encoder block (the original "Attention Is All You Need" block), a pure
    COMPOSITION of the components above (each itself REUSING matmul / softmax / the existing activations). ``x``
    is (batch*seq) x d_model row-major. POST-LN (default):

        a   = MultiHeadAttention(x, mask)
        x1  = LayerNorm(x  + a, gamma1, beta1)              # residual add THEN normalize
        f   = FeedForward(x1)
        out = LayerNorm(x1 + f, gamma2, beta2)

    The one-line PRE-LN variant (``pre_ln=True``; GPT-2 / modern stacks) normalizes BEFORE each sublayer and
    keeps the residual around the RAW input:

        x1  = x  + MultiHeadAttention(LayerNorm(x,  gamma1, beta1), mask)
        out = x1 + FeedForward(LayerNorm(x1, gamma2, beta2))

    ``mask`` is an optional additive ``seq x seq`` mask (e.g. ``causal_mask(seq)``); ``activation`` is the
    feed-forward activation ("relu" / "gelu"). Returns a FRESH (batch*seq) x d_model buffer; ``x`` is NOT
    mutated (side-effect-free). The two transcendentals (softmax exp, layernorm rsqrt) ride the trusted libm
    edge -- the quarantine boundary, so the emitted C reproduces this to float round-off."""
    rows, d_model = spec.rows, spec.d_model
    if len(x) != rows * d_model:
        raise ValueError(f"block input must have rows*d_model = {rows * d_model} entries; got {len(x)}")
    if pre_ln:
        # PRE-LN: normalize, sublayer, residual around the RAW input (the modern deep-stack variant).
        a = multihead_attention_reference(
            layernorm_reference(x, rows, d_model, params.gamma1, params.beta1, eps), spec, params.mha, mask)
        x1 = [float(x[i]) + a[i] for i in range(rows * d_model)]
        f = feedforward_reference(
            layernorm_reference(x1, rows, d_model, params.gamma2, params.beta2, eps),
            rows, d_model, spec.d_ff, params.W1, params.b1, params.W2, params.b2, activation)
        return [x1[i] + f[i] for i in range(rows * d_model)]
    # POST-LN (default): sublayer, residual add, THEN normalize.
    a = multihead_attention_reference(x, spec, params.mha, mask)            # MHA sublayer
    res1 = [float(x[i]) + a[i] for i in range(rows * d_model)]              # residual add (x + a)
    x1 = layernorm_reference(res1, rows, d_model, params.gamma1, params.beta1, eps)   # LN1
    f = feedforward_reference(x1, rows, d_model, spec.d_ff,                 # feed-forward sublayer
                              params.W1, params.b1, params.W2, params.b2, activation)
    res2 = [x1[i] + f[i] for i in range(rows * d_model)]                    # residual add (x1 + f)
    return layernorm_reference(res2, rows, d_model, params.gamma2, params.beta2, eps)  # LN2


# ============================================================================================================
# 6. The Q8 bridge -- round-trip the input (and optionally the weights) then run the trusted block.
# ============================================================================================================

def _roundtrip(values: list[float], group_size: int, bits: int) -> list[float]:
    """Q8<->float32<->Q8 round-trip of a flat vector (the bridge step)."""
    return dequantize(quantize_per_group(values, group_size, bits))


def transformer_block_via_bridge(x: list[float], spec: TransformerBlockSpec, params: TransformerBlockParams,
                                 group_size: int, bits: int, mask: list[float] | None = None,
                                 activation: str = "relu", eps: float = _DEFAULT_EPS, pre_ln: bool = False,
                                 quantize_weights: bool = False) -> list[float]:
    """E3: the Q8<->float32<->Q8 bridge wrapped around the TRUSTED transformer block (integrate, don't
    reinvent) -- the block analog of ``attention_via_bridge`` / ``pca_via_bridge`` / ``ols_via_bridge``. The
    INPUT ``x`` arrives as per-group quantized storage; the bridge dequantizes it to float32 and the trusted
    ``transformer_block_reference`` runs on the round-tripped value. With ``quantize_weights=True`` the projection/
    feed-forward weights are round-tripped too (the realistic weights-also-quantized inference path); biases,
    gamma, beta are kept full-precision (the standard inference convention). So the oracle reference for the
    quantized path is the block of the *round-tripped* tensors -- the SOLE certified error is the INPUT
    round-trip (R17-certified by ``quantization_error_bound``); the matmuls are exact and the softmax/sqrt ride
    the trusted libm edge (treated as exact, like B5's sgemm). Mirrors B5 / G1 / G7 exactly."""
    dx = _roundtrip(x, group_size, bits)
    p = params
    if quantize_weights:
        p = TransformerBlockParams(
            w_q=_roundtrip(params.w_q, group_size, bits), w_k=_roundtrip(params.w_k, group_size, bits),
            w_v=_roundtrip(params.w_v, group_size, bits), w_o=_roundtrip(params.w_o, group_size, bits),
            W1=_roundtrip(params.W1, group_size, bits), b1=list(params.b1),
            W2=_roundtrip(params.W2, group_size, bits), b2=list(params.b2),
            gamma1=list(params.gamma1), beta1=list(params.beta1),
            gamma2=list(params.gamma2), beta2=list(params.beta2))
    return transformer_block_reference(dx, spec, p, mask, activation, eps, pre_ln)


# ============================================================================================================
# 7. Op-level well-formedness (mirrors check_attention). D2: promoted to the R22/R23 law surface
# via verify.verify_ml_spec (the CALLER passes these messages -- no verifier import here, the
# two-truth quarantine holds).
# ============================================================================================================

_DTYPES = ("f32", "i32")


def check_transformer(spec: TransformerBlockSpec, x_shape: tuple[int, ...], in_dtype: str,
                      out_shape: tuple[int, ...], out_dtype: str) -> list[str]:
    """Op-level well-formedness for a gem.transformer_block claim, returned as a list of error strings (empty ==
    well-formed) -- the shape/dtype-law shape gem.matmul/gem.attention use inline, lifted into one named checker.
    NOT a globally-numbered R-law (matches check_attention / check_activation, which add none). The laws:

      1. extents are POSITIVE: d_model, n_heads, d_ff, seq_len, batch all >= 1;
      2. d_model is divisible by n_heads (each head owns d_k = d_model/n_heads contiguous channels);
      3. dtype is a KNOWN dtype and PRESERVED: out_dtype == in_dtype == the spec's declared dtype;
      4. the input/output are each the (batch*seq, d_model) token shape -- the claim's metadata is consistent;
      5. THE QUARANTINE DTYPE RULE: the block contains a softmax AND a layernorm (transcendentals whose libm
         edges return float), so it needs an f32 result -- never i32 (mirrors attention's quarantine rule)."""
    errs: list[str] = []
    # (1) positive extents.
    for nm, v in (("d_model", spec.d_model), ("n_heads", spec.n_heads), ("d_ff", spec.d_ff),
                  ("seq_len", spec.seq_len), ("batch", spec.batch)):
        if v < 1:
            errs.append(f"transformer: {nm} must be >= 1; got {v}")
    # (2) head divisibility.
    if spec.n_heads >= 1 and spec.d_model % spec.n_heads != 0:
        errs.append(f"transformer: d_model {spec.d_model} not divisible by n_heads {spec.n_heads}")
    # (3) dtype known + preserved.
    for nm, dt in (("in", in_dtype), ("out", out_dtype), ("spec", spec.dtype)):
        if dt not in _DTYPES:
            errs.append(f"transformer: {nm} dtype {dt!r} is not a known dtype {_DTYPES}")
    if in_dtype != out_dtype:
        errs.append(f"transformer: dtype not preserved (in {in_dtype!r} != out {out_dtype!r})")
    if spec.dtype != in_dtype:
        errs.append(f"transformer: spec dtype {spec.dtype!r} != input dtype {in_dtype!r}")
    # (4) shapes consistent with the spec (the flattened (batch*seq, d_model) token matrix).
    want = (spec.batch * spec.seq_len, spec.d_model)
    for nm, sh in (("x", x_shape), ("out", out_shape)):
        if tuple(sh) != want:
            errs.append(f"transformer: {nm} shape {tuple(sh)} != (batch*seq, d_model) {want}")
    # (5) the quarantine dtype rule: the softmax + layernorm transcendentals need an f32 result.
    if in_dtype != "f32":
        errs.append(f"transformer: contains a softmax + a layernorm (transcendentals; the libm edges return "
                    f"float), so it needs f32, got {in_dtype!r}")
    return errs


# ============================================================================================================
# 8. Independent verifiers (recomputed independently of the block code path -- the genuine checks).
# ============================================================================================================

def causal_weights_upper_triangle_max(qh: list[float], kh: list[float], seq: int, d_k: int,
                                       mask: list[float]) -> float:
    """INDEPENDENT causal-mask check: recompute the post-softmax attention weights for ONE head (directly, from
    Q_h/K_h + the additive mask) and return the LARGEST weight in the strictly-upper triangle (j > i). With a
    causal mask this must be EXACTLY 0.0 -- the mask truly prevents a token from attending to the future. (The
    diagonal/lower triangle keep finite weights, so each row's softmax is well-defined.) Recomputed here from
    scratch, NOT via the block path, so it is a genuine independent verifier."""
    spec = attention.AttentionSpec(seq, d_k)
    scores = attention.scores_reference(qh, kh, spec)
    scores = [scores[idx] + float(mask[idx]) for idx in range(seq * seq)]
    weights = softmax_reference(scores, axis_len=seq)
    worst = 0.0
    for i in range(seq):
        for j in range(i + 1, seq):                          # the strictly-upper (future) triangle
            worst = max(worst, abs(weights[i * seq + j]))
    return worst


def softmax_rowsum_residual(qh: list[float], kh: list[float], seq: int, d_k: int,
                            mask: list[float] | None = None) -> float:
    """INDEPENDENT softmax check: recompute the per-head attention weights directly and return ``max_i |sum_j
    A[i,j] - 1|`` -- every softmax row is a probability distribution that sums to 1 (even masked rows, since the
    causal mask keeps >= 1 finite entry). ~0 at a correct softmax."""
    spec = attention.AttentionSpec(seq, d_k)
    scores = attention.scores_reference(qh, kh, spec)
    if mask is not None:
        scores = [scores[idx] + float(mask[idx]) for idx in range(seq * seq)]
    weights = softmax_reference(scores, axis_len=seq)
    worst = 0.0
    for i in range(seq):
        worst = max(worst, abs(sum(weights[i * seq + j] for j in range(seq)) - 1.0))
    return worst


def multihead_concat_residual(x: list[float], spec: TransformerBlockSpec, params: TransformerBlockParams,
                              mask: list[float] | None = None) -> float:
    """INDEPENDENT multi-head check (single sequence, batch==1): recompute the multi-head output a SECOND,
    fully-independent way -- project, split, run EACH head's single-head attention by a SEPARATE direct path
    (a fresh softmax composition), concatenate, then project by W_o -- and return the max abs difference vs
    ``multihead_attention_reference``. ~0: multi-head == concat-of-independently-computed-heads then W_o.
    Recomputed without calling the block's _single_head helper (a genuinely separate path)."""
    d_model, n_heads, d_k, seq = spec.d_model, spec.n_heads, spec.d_k, spec.seq_len
    xs = x[:seq * d_model]                                   # batch==1 sequence
    q = matmul.matmul_reference(xs, params.w_q, seq, d_model, d_model)
    k = matmul.matmul_reference(xs, params.w_k, seq, d_model, d_model)
    v = matmul.matmul_reference(xs, params.w_v, seq, d_model, d_model)
    concat = [0.0] * (seq * d_model)
    scale = 1.0 / math.sqrt(d_k)
    for h in range(n_heads):
        for i in range(seq):                                 # a from-scratch single-head attention for head h
            logits = []
            for j in range(seq):
                s = sum(q[i * d_model + h * d_k + t] * k[j * d_model + h * d_k + t] for t in range(d_k)) * scale
                if mask is not None:
                    s += float(mask[i * seq + j])
                logits.append(s)
            mx = max(logits)
            ex = [math.exp(s - mx) for s in logits]
            z = sum(ex) or 1.0
            a = [e / z for e in ex]
            for t in range(d_k):
                concat[i * d_model + h * d_k + t] = sum(a[j] * v[j * d_model + h * d_k + t] for j in range(seq))
    direct = matmul.matmul_reference(concat, params.w_o, seq, d_model, d_model)
    got = multihead_attention_reference(x[:seq * d_model],
                                        TransformerBlockSpec(d_model, n_heads, spec.d_ff, seq, 1, spec.dtype),
                                        params.mha, mask)
    return max(abs(direct[idx] - got[idx]) for idx in range(seq * d_model))
