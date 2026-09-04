"""B1: gem.matmul as a planned claim -- the K_BCIR tile/loop-order search + its verification.

A matmul is a grid of dot products (the A1.2 ``quantize.quantized_dot`` primitive). Its REALIZATION --
tile sizes, loop order, layout -- does not change the result (this module verifies every realization
against a single reference) but DOES change the cost, and K_BCIR picks the realization by a deterministic
analytic search. Per the closure register (docs/machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md,
section 6), that search is
DUAL-SEMIRING: ``min,+`` composes per-stage costs ALONG a candidate realization and selects the cheapest
candidate, while ``max,+`` takes the binding ROOFLINE bottleneck (compute vs memory) WITHIN a candidate --
a pure min,+ shortest-path would miss that the bottleneck is a max over competing resources (the scan's
load-bearing finding; K_BCIR already carries both semirings -- min,+ optimize + max,+ overlap).

Scope: this is the oracle realization + planner + verification. Wiring the chosen plan into the MLIR
``gem.matmul`` law op and the production K_BCIR CostVector pass is the follow-on (kept separate, as A1's
frontend lowering was)."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

from .cost import COMPUTE, MEMORY, CostVector, TargetProfile
from .precision import quantization_error_bound
from .quantize import dequantize, quantize_per_group, scaled_dot

# ---- the math: a reference and the tiled/quantized realizations (all must agree) ----------------
# Row-major A is M x K, row-major B is K x N, C is M x N. The reference is the definition; every
# realization below must reproduce it (exactly for the integer/quantized path; up to fp reassociation
# for the plain float path).


def _validate_dimensions(M: int, N: int, K: int) -> None:
    for name, value in (("M", M), ("N", N), ("K", K)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _validate_inputs(a, b, M: int, N: int, K: int) -> None:
    _validate_dimensions(M, N, K)
    a_count, b_count, c_count = M * K, K * N, M * N
    if max(a_count, b_count, c_count) > sys.maxsize:
        raise ValueError("matmul shape exceeds the host addressable sequence size")
    try:
        a_len, b_len = len(a), len(b)
    except TypeError as exc:
        raise ValueError("matmul inputs must be sized sequences") from exc
    if a_len != a_count or b_len != b_count:
        raise ValueError(
            f"matmul input length mismatch: A has {a_len}, expected {a_count}; "
            f"B has {b_len}, expected {b_count}"
        )


def matmul_reference(a, b, M: int, N: int, K: int) -> list[float]:
    """C = A @ B, the naive definition -- the single source of truth every realization is checked against."""
    _validate_inputs(a, b, M, N, K)
    c = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += a[i * K + k] * b[k * N + j]
            c[i * N + j] = s
    return c


def matmul_tiled(a, b, M: int, N: int, K: int, plan: "TilePlan") -> list[float]:
    """Compute A @ B in the plan's tile sizes + loop order. Tiling/reordering is a REASSOCIATION of the
    same sums, so for integer inputs the result is bit-identical to the reference -- the property the
    verifier checks (it is what makes the realization a free choice the cost model may optimize)."""
    _validate_inputs(a, b, M, N, K)
    if not isinstance(plan, TilePlan):
        raise ValueError("matmul plan must be a TilePlan")
    for name, value in (("tile_m", plan.tile_m), ("tile_n", plan.tile_n), ("tile_k", plan.tile_k)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
    if plan.loop_order not in _LOOP_ORDERS:
        raise ValueError(f"unsupported matmul loop order {plan.loop_order!r}")
    c = [0.0] * (M * N)
    tm, tn, tk = plan.tile_m, plan.tile_n, plan.tile_k
    for i0, j0, k0 in tile_origins(M, N, K, plan):
        for i in range(i0, min(i0 + tm, M)):
            for j in range(j0, min(j0 + tn, N)):
                s = 0.0
                for k in range(k0, min(k0 + tk, K)):
                    s += a[i * K + k] * b[k * N + j]
                c[i * N + j] += s
    return c


def gemm_via_bridge(a, b, M: int, N: int, K: int, group_size: int, bits: int) -> list[float]:
    """B5: the Q8<->float32<->Q8 bridge wrapped around a TRUSTED float gemm (integrate, don't reinvent).
    Inputs arrive as per-group quantized storage; the bridge dequantizes them to float32, a trusted
    external sgemm (CBLAS, via the `c.call.libm:` edge) computes the product, and BCIR owns the calling
    side (row-major layout + the boundary quantization). The oracle reference for that path is the
    reference matmul of the *round-tripped* inputs -- so the only error vs the true product is the input
    quantization (R17-certified), the float gemm itself being exact/trusted."""
    _validate_inputs(a, b, M, N, K)
    da = dequantize(quantize_per_group(a, group_size, bits))
    db = dequantize(quantize_per_group(b, group_size, bits))
    return matmul_reference(da, db, M, N, K)


def quantized_matmul(a, b, M: int, N: int, K: int, group_size: int, bits: int) -> list[float]:
    """C[i,j] = quantized_dot(row i of A, col j of B) -- the matmul as a grid of A1.2 quantized dot
    products (exact integer code accumulation + per-group rescale). Error vs the reference is the input
    quantization alone."""
    _validate_inputs(a, b, M, N, K)
    c = [0.0] * (M * N)
    for i in range(M):
        row = a[i * K : (i + 1) * K]
        for j in range(N):
            col = [b[k * N + j] for k in range(K)]
            qa = quantize_per_group(row, group_size, bits)
            qb = quantize_per_group(col, group_size, bits)
            c[i * N + j] = scaled_dot(qa, qb)
    return c


# ---- the dual-semiring tile/loop-order plan -----------------------------------------------------


@dataclass(frozen=True)
class TilePlan:
    tile_m: int
    tile_n: int
    tile_k: int
    loop_order: str
    compute_cost: int  # MAC throughput term (min,+ stage cost, summed along the realization)
    mem_cost: int  # traffic / bandwidth term
    bottleneck: int  # max(compute, mem) -- the max,+ roofline binding resource
    fits_cache: bool


_LOOP_ORDERS = (
    "ijk",
    "ikj",
    "jik",
)  # the outer tile-loop permutations we cost (k always innermost-accum)


def tile_origins(M: int, N: int, K: int, plan: TilePlan):
    """Yield tiled-loop origins in the plan's declared outer-loop order."""
    _validate_dimensions(M, N, K)
    if not isinstance(plan, TilePlan) or plan.loop_order not in _LOOP_ORDERS:
        raise ValueError("tile origins require a valid TilePlan")
    steps = {"i": plan.tile_m, "j": plan.tile_n, "k": plan.tile_k}
    limits = {"i": M, "j": N, "k": K}
    if any(type(step) is not int or step <= 0 for step in steps.values()):
        raise ValueError("tile extents must be positive integers")
    axes = tuple(plan.loop_order)
    for first in range(0, limits[axes[0]], steps[axes[0]]):
        for second in range(0, limits[axes[1]], steps[axes[1]]):
            for third in range(0, limits[axes[2]], steps[axes[2]]):
                origin = {axes[0]: first, axes[1]: second, axes[2]: third}
                yield origin["i"], origin["j"], origin["k"]


def _divisor_tiles(dim: int) -> list[int]:
    """Candidate tile extents for one dimension: powers of two up to `dim`, plus `dim` itself (the
    whole-dimension / untiled choice). Deterministic + small."""
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"tile dimension must be a positive integer; got {dim!r}")
    cands, t = set(), 1
    while t < dim:
        cands.add(t)
        t *= 2
    cands.add(dim)
    return sorted(cands)


def _cache_budget_elems(target: TargetProfile) -> int:
    """A modeled per-core working-set budget in ELEMENTS (a tile of A, B, C must co-reside). Derived from
    the target's cacheline so it retargets with the profile -- 512 lines is an ~L1/L2-ish working set."""
    return (512 * target.cacheline) // max(1, target.elem_bytes)


def cost_of(
    M: int, N: int, K: int, tm: int, tn: int, tk: int, target: TargetProfile
) -> tuple[int, int, bool]:
    """The analytic (compute, mem, fits_cache) cost of one tiling, in target units.

    compute: M*N*K MACs / (vector_width * fma) -- the FLOP-throughput term.
    mem: bytes streamed / (mem_unit * mem_channels) bandwidth. Tiling reuses each loaded A/B tile across
         the third dimension, so traffic falls as the *other* tile extent grows: A is re-read once per
         N-tile, B once per M-tile (the classic blocked-GEMM reuse). Bigger tiles -> less traffic, but the
         working set (tm*tk + tk*tn + tm*tn) must fit the cache budget."""
    _validate_dimensions(M, N, K)
    for name, value in (("tm", tm), ("tn", tn), ("tk", tk)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
    if not isinstance(target, TargetProfile):
        raise ValueError("matmul target must be a TargetProfile")
    macs = M * N * K
    thr = max(1, target.vector_width * (2 if target.fma else 1))
    compute = (macs + thr - 1) // thr
    a_reads = M * K * math.ceil(N / max(1, tn))  # A reused across the columns in a tile
    b_reads = K * N * math.ceil(M / max(1, tm))  # B reused across the rows in a tile
    c_traffic = M * N * (math.ceil(K / max(1, tk)) + 1)
    bw = max(1, target.mem_unit * target.mem_channels)
    mem = (a_reads + b_reads + c_traffic + bw - 1) // bw
    working_set = tm * tk + tk * tn + tm * tn
    return compute, mem, working_set <= _cache_budget_elems(target)


def cost_vector(plan: "TilePlan", *, quant_bits: int = 0) -> CostVector:
    """Express a matmul plan's cost in the production 12-dim K_BCIR CostVector (cost.py) -- this is the
    "production wiring" that lets a planned gem.matmul be SCORED by the same algebra as every other claim.
    The analytic roofline terms land on the `compute`/`memory` axes; a quantized realization carries the
    R17 Q8-bridge bound on the `accuracy` axis. min,+ ranking is then ``cv.couple(factor).dot(weights)``
    (the existing path/CSE/fusion algebra applies unchanged); the roofline bottleneck is ``bottleneck(cv)``
    (max,+ over the binding resources -- the SOTA-scan finding that a min,+ sum alone would miss)."""
    acc = quantization_error_bound() if quant_bits > 0 else 0
    return CostVector.of(compute=plan.compute_cost, memory=plan.mem_cost, accuracy=acc)


def bottleneck(cv: CostVector) -> int:
    """The max,+ roofline bottleneck of a cost vector: the binding resource among compute and memory
    (cf. the ML roadmap closure register -- the bottleneck is a max over competing rooflines,
    not a min,+ sum)."""
    return max(cv.v[COMPUTE], cv.v[MEMORY])


def plan_matmul(M: int, N: int, K: int, target: TargetProfile | None = None) -> TilePlan:
    """K_BCIR's deterministic tile/loop-order choice. For every candidate (tile, loop-order): compute its
    roofline BOTTLENECK = max(compute, mem) (the max,+ step), prefer cache-fitting plans, and select the
    minimum-bottleneck candidate (the min,+ step). Ties break on a stable key so the plan is reproducible
    (a deliberate reproducibility property -- see the ML roadmap closure register)."""
    _validate_dimensions(M, N, K)
    target = target or TargetProfile.for_host()
    if not isinstance(target, TargetProfile):
        raise ValueError("matmul target must be a TargetProfile")
    best: TilePlan | None = None
    for lo in _LOOP_ORDERS:
        for tm in _divisor_tiles(M):
            for tn in _divisor_tiles(N):
                for tk in _divisor_tiles(K):
                    compute, mem, fits = cost_of(M, N, K, tm, tn, tk, target)
                    bottleneck = max(compute, mem)  # max,+ : the binding roofline resource
                    cand = TilePlan(tm, tn, tk, lo, compute, mem, bottleneck, fits)
                    # min,+ selection: cache-fitting first, then min bottleneck, then a stable tie-break
                    # (bigger tiles, then loop order) so the choice is fully deterministic.
                    key = (not cand.fits_cache, cand.bottleneck, -(tm * tn * tk), lo)
                    if best is None or key < (
                        not best.fits_cache,
                        best.bottleneck,
                        -(best.tile_m * best.tile_n * best.tile_k),
                        best.loop_order,
                    ):
                        best = cand
    assert best is not None
    return best
