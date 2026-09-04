"""G7: gem.attention as a planned claim -- single-head scaled-dot-product attention, modeled EXACTLY like
gem.matmul (kbcir.matmul) and gem.activation (kbcir.activation).

The vision-alignment audit (docs/VISION_ALIGNMENT_AUDIT.md, Pillar 5c) flags that ``gem.matmul`` was the
only first-class tensor op -- no attention. This module adds single-head scaled-dot-product attention

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

the transformer primitive. Like a matmul, its RESULT is fixed by a single reference (this module verifies
the realization against the naive computation), while its REALIZATION is a free choice the K_BCIR cost
model plans.

ATTENTION DECOMPOSES INTO THE EXISTING OPS (the load-bearing design choice -- do NOT reinvent). Attention
is NOT a new primitive: it is a COMPOSITION of ops BCIR already has --

    S = Q @ K^T                 (a gem.matmul: seq_len x d_k  times  d_k x seq_len  ->  seq_len x seq_len)
    S' = S / sqrt(d_k)          (an elementwise scale -- exact, 0 ULP, on the deterministic rail)
    A = softmax(S', last axis)  (the EXISTING activation.softmax_reference -- reused verbatim, NOT redefined)
    out = A @ V                 (a second gem.matmul: seq_len x seq_len  times  seq_len x d_k)

So this module's reference COMPOSES ``matmul.matmul_reference`` and ``activation.softmax_reference`` (the
softmax is reused, never reinvented -- the task's explicit constraint), and its realization is PLANNED as
the two underlying matmuls through ``matmul.plan_matmul`` (the existing tile/loop search) -- no new cost
model, no new roofline term.

THE QUARANTINE BOUNDARY (the load-bearing constraint -- see also kbcir.activation, kbcir.precision). The
softmax's exp is a TRANSCENDENTAL: it rides the trusted ``c.call.libm:`` edge (expf), exactly as
activation.softmax does, fully OFF the deterministic legality rail (libm is opaque/trusted, like any
external edge). The two matmuls and the scale are exact/deterministic. So:

  * The DENSE reference matches a naive attention to FLOAT ROUND-OFF (the softmax transcendental rides the
    libm edge per the quarantine -- the same float story as activation.softmax / B5).
  * The ONLY certified error on a QUANTIZED attention is the Q8<->f32<->Q8 bridge step (R17), exactly as
    B5 / G1 / the conv bridge: ``attention_via_bridge`` round-trips Q/K/V through the bridge, runs the
    trusted attention on the dequantized values, so the bridge is the SOLE error source
    (``precision.quantization_error_bound``), the attention arithmetic itself being exact-or-trusted-libm.

Scope (honest). SINGLE-HEAD, single (unbatched) sequence -- the deliverable. Multi-head, batched, masked/
causal attention, and the MLIR ``gem.attention`` law op are the deferred follow-up (the MLIR port is kept
separate, exactly as matmul.py / activation.py kept theirs)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import matmul
from .activation import softmax_reference
from .cost import COMPUTE, MEMORY, CostVector, TargetProfile
from .precision import quantization_error_bound
from .quantize import dequantize, quantize_per_group


# ---- the planned-claim shape metadata (mirrors matmul.TilePlan / activation.ActivationSpec) -----


@dataclass(frozen=True)
class AttentionSpec:
    """A gem.attention as a FIRST-CLASS planned claim: single-head scaled-dot-product attention's shape/
    dtype metadata -- the attention analog of matmul.TilePlan's shape carrier. Q, K, V are each row-major
    ``seq_len x d_k`` (single head, so d_model == d_k). ``dtype`` is the contracted element type. The two
    underlying matmul plans (the K_BCIR-selected tile/loop choices for ``Q@K^T`` and ``A@V``) are carried
    as ``scores_tile`` / ``context_tile`` (None until planned)."""

    seq_len: int  # the sequence length (rows of Q/K/V)
    d_k: int  # the head / key dimension (cols of Q/K/V); d_model == d_k for one head
    dtype: str = "f32"
    scores_tile: "matmul.TilePlan | None" = None  # the Q@K^T matmul plan
    context_tile: "matmul.TilePlan | None" = None  # the A@V matmul plan

    @property
    def scale(self) -> float:
        """The scaled-dot-product normalizer 1/sqrt(d_k) (the variance-control factor that keeps the logits
        in softmax's well-conditioned range)."""
        return 1.0 / math.sqrt(self.d_k)

    @property
    def scores_dims(self) -> tuple[int, int, int]:
        """The (M,N,K) of the Q@K^T matmul: M=seq_len, N=seq_len, K=d_k -> the seq x seq score matrix."""
        return self.seq_len, self.seq_len, self.d_k

    @property
    def context_dims(self) -> tuple[int, int, int]:
        """The (M,N,K) of the A@V matmul: M=seq_len, N=d_k, K=seq_len -> the seq x d_k context."""
        return self.seq_len, self.d_k, self.seq_len

    @property
    def qkv_count(self) -> int:
        return self.seq_len * self.d_k

    @property
    def out_count(self) -> int:
        return self.seq_len * self.d_k


# ---- the math: a single reference (the realization must agree to float round-off) ----------------
# Attention COMPOSES the existing ops: matmul (Q@K^T) -> scale -> the EXISTING softmax -> matmul (@V). The
# softmax is activation.softmax_reference (reused, NOT reinvented). The transcendental (exp) rides the
# trusted libm edge, so the C kernel that calls the SAME libm reproduces this to float round-off.


def _transpose(b, rows: int, cols: int) -> list[float]:
    """Transpose a row-major ``rows x cols`` matrix to ``cols x rows`` (so ``K`` becomes ``K^T`` for the
    Q@K^T matmul -- a plain reshaping, no arithmetic)."""
    t = [0.0] * (rows * cols)
    for i in range(rows):
        for j in range(cols):
            t[j * rows + i] = b[i * cols + j]
    return t


def scores_reference(q, k, spec: AttentionSpec) -> list[float]:
    """The scaled scores ``S' = (Q @ K^T) / sqrt(d_k)`` -- a gem.matmul (Q@K^T) then the exact elementwise
    scale. ``q``/``k`` are row-major seq_len x d_k; the result is row-major seq_len x seq_len. The scale is
    a pure multiply (exact, 0 ULP, on the deterministic rail -- NO transcendental here)."""
    n, _, d = spec.scores_dims
    kt = _transpose(k, spec.seq_len, spec.d_k)  # K^T : d_k x seq_len
    s = matmul.matmul_reference(q, kt, n, n, d)  # Q @ K^T : seq_len x seq_len
    sc = spec.scale
    return [v * sc for v in s]


def attention_reference(q, k, v, spec: AttentionSpec) -> list[float]:
    """The naive single-head scaled-dot-product attention -- the single source of truth the realization is
    checked against. ``out = softmax(Q @ K^T / sqrt(d_k)) @ V``, composing matmul + the EXISTING
    activation.softmax_reference (reused over the last/key axis of length seq_len) + a second matmul.
    ``q``/``k``/``v`` are row-major seq_len x d_k; the result is row-major seq_len x d_k. The softmax exp is
    the only transcendental (it rides the trusted libm edge per the quarantine), so the C kernel reproduces
    this to float round-off."""
    s = scores_reference(q, k, spec)  # scaled scores: seq x seq
    a = softmax_reference(s, axis_len=spec.seq_len)  # the EXISTING softmax over each row
    m, n2, kk = spec.context_dims
    return matmul.matmul_reference(a, v, m, n2, kk)  # A @ V : seq x d_k


def attention_realize(q, k, v, spec: AttentionSpec) -> list[float]:
    """Run attention via the spec's K_BCIR-SELECTED realization (the two matmul tile/loop plans
    ``scores_tile`` / ``context_tile``). Tiling is a reassociation of the same sums, so for integer inputs
    the matmuls are bit-identical to the reference; the softmax rides the same libm edge -- so the
    realization reproduces ``attention_reference`` to float round-off (bit-exact on the integer/exact path
    up to the softmax). This is the single entry the verifier drives, the way test_matmul drives the chosen
    tile."""
    n, _, d = spec.scores_dims
    kt = _transpose(k, spec.seq_len, spec.d_k)
    s = (
        matmul.matmul_tiled(q, kt, n, n, d, spec.scores_tile)
        if spec.scores_tile is not None
        else matmul.matmul_reference(q, kt, n, n, d)
    )
    sc = spec.scale
    s = [val * sc for val in s]
    a = softmax_reference(s, axis_len=spec.seq_len)
    m, n2, kk = spec.context_dims
    return (
        matmul.matmul_tiled(a, v, m, n2, kk, spec.context_tile)
        if spec.context_tile is not None
        else matmul.matmul_reference(a, v, m, n2, kk)
    )


def attention_via_bridge(q, k, v, spec: AttentionSpec, group_size: int, bits: int) -> list[float]:
    """The Q8<->float32<->Q8 bridge wrapped around the TRUSTED attention (integrate, don't reinvent) -- the
    attention analog of matmul.gemm_via_bridge / activation.activation_via_bridge / conv.conv_via_bridge.
    Q, K, V arrive as per-group quantized storage; the bridge dequantizes them to float32, the trusted
    attention runs on the round-tripped values, and BCIR owns the boundary quantization. So the oracle
    reference for the quantized path is the attention of the *round-tripped* inputs -- the SOLE certified
    error vs the true attention is the INPUT quantization (R17-certified by quantization_error_bound); the
    two matmuls are exact and the softmax's exp is the trusted libm edge (treated as exact, like B5's
    sgemm). Mirrors B5 / G1 exactly."""
    dq = dequantize(quantize_per_group(q, group_size, bits))
    dk = dequantize(quantize_per_group(k, group_size, bits))
    dv = dequantize(quantize_per_group(v, group_size, bits))
    return attention_reference(dq, dk, dv, spec)


# ---- shape/dtype well-formedness (mirrors activation.check_activation; NOT a new global R-law) ----
# gem.matmul / gem.activation / gem.conv validate their op shape inline and add NO globally-numbered R-law.
# We do the SAME here: a small op-level check the oracle rail calls explicitly.

_DTYPES = ("f32", "i32")


def check_attention(
    spec: AttentionSpec,
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
    in_dtype: str,
    out_shape: tuple[int, ...],
    out_dtype: str,
) -> list[str]:
    """Op-level well-formedness for a gem.attention claim, returned as a list of error strings (empty ==
    well-formed) -- the shape/dtype-law shape gem.matmul/gem.activation/gem.conv use inline, lifted into one
    named checker. NOT a globally-numbered R-law (matches the precedents, which add none). The laws:

      1. extents are POSITIVE: seq_len >= 1 and d_k >= 1;
      2. dtype is a KNOWN dtype and PRESERVED: out_dtype == in_dtype == the spec's declared dtype;
      3. Q, K, V are each the (seq_len, d_k) single-head shape (single head: d_model == d_k), and the
         output is (seq_len, d_k) -- the claim's metadata is internally consistent;
      4. THE QUARANTINE DTYPE RULE: attention contains a softmax (a transcendental whose libm edge returns
         float), so it needs an f32 result -- never i32 (mirrors activation.softmax's quarantine rule)."""
    errs: list[str] = []
    # (1) positive extents.
    if spec.seq_len < 1:
        errs.append(f"attention: seq_len must be >= 1; got {spec.seq_len}")
    if spec.d_k < 1:
        errs.append(f"attention: d_k must be >= 1; got {spec.d_k}")
    # (2) dtype known + preserved.
    for nm, dt in (("in", in_dtype), ("out", out_dtype), ("spec", spec.dtype)):
        if dt not in _DTYPES:
            errs.append(f"attention: {nm} dtype {dt!r} is not a known dtype {_DTYPES}")
    if in_dtype != out_dtype:
        errs.append(f"attention: dtype not preserved (in {in_dtype!r} != out {out_dtype!r})")
    if spec.dtype != in_dtype:
        errs.append(f"attention: spec dtype {spec.dtype!r} != input dtype {in_dtype!r}")
    # (3) Q/K/V/out shapes consistent with the spec.
    want = (spec.seq_len, spec.d_k)
    for nm, sh in (("Q", q_shape), ("K", k_shape), ("V", v_shape), ("out", out_shape)):
        if tuple(sh) != want:
            errs.append(f"attention: {nm} shape {tuple(sh)} != (seq_len,d_k) {want}")
    # (4) the quarantine dtype rule: the softmax transcendental needs an f32 result.
    if in_dtype != "f32":
        errs.append(
            f"attention: contains a softmax (transcendental; the libm edge returns float), so it "
            f"needs f32, got {in_dtype!r}"
        )
    return errs


# ---- the cost / realization plug-in (reuses matmul.cost_of for BOTH matmuls) ---------------------


def cost_of(
    spec: AttentionSpec,
    target: TargetProfile,
    scores_tile: "matmul.TilePlan | None" = None,
    context_tile: "matmul.TilePlan | None" = None,
) -> tuple[int, int, bool]:
    """The analytic (compute, mem, fits_cache) cost of an attention realization, PRICED THROUGH THE EXISTING
    matmul roofline (matmul.cost_of) for BOTH matmuls -- attention is a composition of two gem.matmuls, so
    its cost is the SUM of the two priced matmul costs (no new roofline term; the scale and softmax passes
    are O(seq^2), dominated by and folded into the Q@K^T traffic). The (compute, mem) of each matmul comes
    from matmul.cost_of at its chosen tile; the conv ``fits_cache`` is the AND (both gemms must fit)."""
    sm, sn, sk = spec.scores_dims
    cm, cn, ck = spec.context_dims
    st = scores_tile or matmul.plan_matmul(sm, sn, sk, target)
    ct = context_tile or matmul.plan_matmul(cm, cn, ck, target)
    c1, m1, f1 = matmul.cost_of(sm, sn, sk, st.tile_m, st.tile_n, st.tile_k, target)
    c2, m2, f2 = matmul.cost_of(cm, cn, ck, ct.tile_m, ct.tile_n, ct.tile_k, target)
    return c1 + c2, m1 + m2, f1 and f2


def cost_vector(spec: AttentionSpec, *, quant_bits: int = 0) -> CostVector:
    """Express an attention plan's cost in the production 12-dim K_BCIR CostVector -- the production wiring
    that lets a planned gem.attention be SCORED by the same algebra as every other claim (exactly as
    matmul.cost_vector / activation.cost_vector / conv.cost_vector). The roofline terms (the summed
    two-matmul cost, via cost_of) land on compute/memory; a quantized realization carries the R17 Q8-bridge
    bound on the accuracy axis (the softmax transcendental adds none -- the libm kernel is trusted/exact,
    so the only certified error is the bridge step, exactly as the activation bridge)."""
    compute, mem, _ = cost_of(spec, TargetProfile.for_host(), spec.scores_tile, spec.context_tile)
    acc = quantization_error_bound() if quant_bits > 0 else 0
    return CostVector.of(compute=compute, memory=mem, accuracy=acc)


def bottleneck(cv: CostVector) -> int:
    """The max,+ roofline bottleneck of an attention cost vector: the binding resource among compute and
    memory (identical to matmul.bottleneck -- the SOTA-scan finding that the bottleneck is a max, not a
    min,+ sum). The O(seq^2 d) score matmul dominates a long sequence -> memory-bound; a wide head shifts
    compute up."""
    return max(cv.v[COMPUTE], cv.v[MEMORY])


def plan_attention(
    seq_len: int, d_k: int, dtype: str = "f32", target: TargetProfile | None = None
) -> AttentionSpec:
    """K_BCIR's deterministic attention realization -- the attention analog of matmul.plan_matmul. Attention
    decomposes into two matmuls, so the plan IS the two underlying matmul plans: ``Q@K^T`` (seq,seq,d_k) and
    ``A@V`` (seq,d_k,seq), each planned by the EXISTING matmul.plan_matmul tile/loop dual-semiring search
    (min,+ to pick the tile, max,+ for the roofline bottleneck). Deterministic + reproducible (plan_matmul
    is). Returns the chosen AttentionSpec carrying both tile plans."""
    target = target or TargetProfile.for_host()
    base = AttentionSpec(seq_len, d_k, dtype)
    sm, sn, sk = base.scores_dims
    cm, cn, ck = base.context_dims
    scores_tile = matmul.plan_matmul(sm, sn, sk, target)
    context_tile = matmul.plan_matmul(cm, cn, ck, target)
    return AttentionSpec(seq_len, d_k, dtype, scores_tile, context_tile)
