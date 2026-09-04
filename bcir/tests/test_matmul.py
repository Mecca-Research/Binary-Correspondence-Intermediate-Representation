"""B1 -- gem.matmul as a planned claim: tiling/loop-order is a free, verified choice the dual-semiring
cost search optimizes.

Covers: every tiling + loop order reproduces the reference EXACTLY (the verification that makes the
realization a free choice); the quantized-dot matmul tracks the float reference within the quantization
error; the plan is deterministic; and the cost search needs BOTH semirings -- min,+ to pick the cheapest
candidate and max,+ for the roofline bottleneck (a memory-bound shape is decided by the mem term a
compute-only ranking would ignore). All deterministic + pure-Python."""

import random

from bcir.kbcir.cost import ACCURACY, COMPUTE, MEMORY, CostVector, TargetProfile
from bcir.kbcir.matmul import (
    TilePlan,
    bottleneck,
    cost_of,
    cost_vector,
    matmul_reference,
    matmul_tiled,
    plan_matmul,
    quantized_matmul,
)

_HOST = TargetProfile.for_host()


def _imatrix(rng, n, lo=-4, hi=4):
    return [float(rng.randint(lo, hi)) for _ in range(n)]  # integer-valued -> float sums are exact


# --- the verification: every realization reproduces the reference --------------------------------


def test_tiling_and_loop_order_are_exact_for_integer_inputs():
    rng = random.Random(0xB1)
    M, N, K = 6, 5, 7
    a, b = _imatrix(rng, M * K), _imatrix(rng, K * N)
    ref = matmul_reference(a, b, M, N, K)
    for tm in (1, 2, 3, M):
        for tn in (1, 2, N):
            for tk in (1, 3, K):
                for lo in ("ijk", "ikj", "jik"):
                    plan = TilePlan(tm, tn, tk, lo, 0, 0, 0, True)
                    assert matmul_tiled(a, b, M, N, K, plan) == ref, (tm, tn, tk, lo)


def test_the_planned_realization_matches_the_reference():
    rng = random.Random(2)
    M, N, K = 8, 8, 8
    a, b = _imatrix(rng, M * K), _imatrix(rng, K * N)
    plan = plan_matmul(M, N, K, _HOST)
    assert matmul_tiled(a, b, M, N, K, plan) == matmul_reference(a, b, M, N, K)


def test_fuzz_tiling_invariance_holds_over_random_shapes():
    rng = random.Random(0xB1F)
    for _ in range(300):
        M, N, K = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
        a, b = _imatrix(rng, M * K), _imatrix(rng, K * N)
        ref = matmul_reference(a, b, M, N, K)
        plan = plan_matmul(M, N, K, _HOST)
        assert 1 <= plan.tile_m <= M and 1 <= plan.tile_n <= N and 1 <= plan.tile_k <= K
        assert matmul_tiled(a, b, M, N, K, plan) == ref


# --- the quantized-dot matmul (built on A1.2) tracks the float reference --------------------------


def test_quantized_matmul_tracks_the_reference_within_quant_error():
    rng = random.Random(7)
    M, N, K = 4, 4, 12
    a = [rng.uniform(-2, 2) for _ in range(M * K)]
    b = [rng.uniform(-2, 2) for _ in range(K * N)]
    ref = matmul_reference(a, b, M, N, K)
    got = quantized_matmul(a, b, M, N, K, group_size=K, bits=8)
    assert all(abs(r - g) <= 0.5 * K + 1e-6 for r, g in zip(ref, got))


# --- the plan is deterministic + reproducible ----------------------------------------------------


def test_plan_is_deterministic():
    for M, N, K in [(8, 8, 8), (16, 4, 32), (3, 7, 5)]:
        assert plan_matmul(M, N, K, _HOST) == plan_matmul(M, N, K, _HOST)


def test_plan_prefers_a_cache_fitting_realization():
    plan = plan_matmul(64, 64, 64, _HOST)
    assert plan.fits_cache  # a fitting plan exists and is chosen
    assert plan.bottleneck == max(plan.compute_cost, plan.mem_cost)


# --- the dual-semiring necessity: the bottleneck is a max, not a sum -----------------------------


def test_bottleneck_is_max_plus_not_min_plus():
    # The chosen plan minimizes the max(compute, mem) ROOFLINE. Show that the binding term really is the
    # max: the untiled (whole-matrix) realization has the same compute but more memory traffic, so its
    # bottleneck is >= the planned one -- a fact a compute-only (min,+ without the max) ranking misses.
    M, N, K = 32, 32, 64
    plan = plan_matmul(M, N, K, _HOST)
    untiled_compute, untiled_mem, _ = cost_of(M, N, K, M, N, K, _HOST)
    assert plan.bottleneck <= max(untiled_compute, untiled_mem)
    # compute is tiling-invariant (same MAC count); the planner moved the bottleneck via the mem term.
    assert plan.compute_cost == untiled_compute


def test_memory_bound_shape_is_decided_by_the_mem_term():
    # A tall-skinny K-heavy shape is memory-bound: among candidates the mem term dominates the bottleneck,
    # so the planner's choice changes with mem -- exactly what a max,+ bottleneck (not a min,+ sum of
    # stage costs) captures. Verify the chosen plan's bottleneck equals its mem term (mem-bound) and is
    # not worse than the untiled traffic.
    M, N, K = 2, 2, 256
    plan = plan_matmul(M, N, K, _HOST)
    assert plan.mem_cost >= plan.compute_cost  # memory-bound regime
    assert plan.bottleneck == plan.mem_cost
    _, untiled_mem, _ = cost_of(M, N, K, M, N, K, _HOST)
    assert plan.mem_cost <= untiled_mem  # tiling did not increase the binding term


def test_cost_vector_wires_the_plan_into_the_production_12d_algebra():
    plan = plan_matmul(32, 32, 64, _HOST)
    cv = cost_vector(plan)
    assert isinstance(cv, CostVector)
    assert cv.v[COMPUTE] == plan.compute_cost and cv.v[MEMORY] == plan.mem_cost
    # the max,+ roofline bottleneck is recoverable from the vector and equals the planner's.
    assert bottleneck(cv) == plan.bottleneck == max(cv.v[COMPUTE], cv.v[MEMORY])
    # it composes with the existing cost algebra: couple by an identity factor + scalarize by weights.
    w = tuple(1 if i in (COMPUTE, MEMORY) else 0 for i in range(len(cv.v)))
    assert cv.couple((256,) * len(cv.v)).dot(w) == plan.compute_cost + plan.mem_cost


def test_cost_vector_carries_the_r17_bound_on_the_accuracy_axis_when_quantized():
    plan = plan_matmul(8, 8, 8, _HOST)
    assert cost_vector(plan, quant_bits=0).v[ACCURACY] == 0  # dense: no accuracy cost
    assert (
        cost_vector(plan, quant_bits=8).v[ACCURACY] == 1
    )  # quantized: the R17 bridge bound (1 ULP)


def test_cost_is_monotone_and_well_formed():
    # sanity: more work -> more compute; the cost helpers never crash on degenerate 1x1x1.
    c1, m1, _ = cost_of(8, 8, 8, 8, 8, 8, _HOST)
    c2, m2, _ = cost_of(16, 16, 16, 16, 16, 16, _HOST)
    assert c2 > c1 and m2 >= m1
    c, m, fits = cost_of(1, 1, 1, 1, 1, 1, _HOST)
    assert c >= 1 and m >= 1 and fits


def test_malformed_shapes_inputs_and_tiles_fail_before_execution():
    for dims in ((0, 1, 1), (1, -1, 1), (True, 1, 1), (1.0, 1, 1)):
        try:
            plan_matmul(*dims, _HOST)
            assert False, f"expected malformed shape {dims!r} to be rejected"
        except ValueError:
            pass
    for args in (([], [1.0], 1, 1, 1), ([1.0], [], 1, 1, 1)):
        try:
            matmul_reference(*args)
            assert False, "expected mismatched matrix storage to be rejected"
        except ValueError:
            pass
    bad = TilePlan(0, 1, 1, "ijk", 0, 0, 0, True)
    try:
        matmul_tiled([1.0], [1.0], 1, 1, 1, bad)
        assert False, "expected zero tile to be rejected before range(step=0)"
    except ValueError:
        pass
