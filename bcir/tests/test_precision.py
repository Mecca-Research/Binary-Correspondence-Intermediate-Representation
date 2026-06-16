"""Deterministic precision analysis (mined from MPAT/AEDACI -- the portable integer
subset). Tests the *measured* numerical win (compensated Q8 reduction == int64-exact,
naive drifts), the static interval error bound, and the two-truth stability signals
(graded, never legislating). All integer/Q8 -- no floats."""

import random

from bcir.kbcir import (
    Interval,
    accuracy_bound,
    cancellation,
    compensated_reduce_q8,
    condition_milli,
    exact_reduce_q8,
    iadd,
    imul_q8,
    isub,
    meets_tolerance,
    naive_reduce_q8,
    reduction_error_bound,
    ulp_distance,
)
from bcir.kbcir.twotruth import Graded, is_classical
from bcir.examples import gather_reduce, vector_add


# --- the measured numerical win: compensated == exact, naive drifts --------------

def test_compensated_reduction_is_bit_identical_to_int64_exact():
    rng = random.Random(1234)
    for _ in range(50):
        n = rng.randint(1, 400)
        vals = [rng.randint(-10_000, 10_000) for _ in range(n)]
        w = rng.randint(1, 511)                      # a Q8 weight with low bits set
        assert compensated_reduce_q8(vals, w) == exact_reduce_q8(vals, w)


def test_naive_reduction_drifts_and_compensation_removes_it():
    # each x*w has a nonzero low byte, so the naive per-term floor loses ~1 ULP/term.
    vals = [3] * 300                                 # 3*129 = 387 -> low byte 0x83
    w = 129
    exact = exact_reduce_q8(vals, w)
    naive_err = ulp_distance(naive_reduce_q8(vals, w), exact)
    comp_err = ulp_distance(compensated_reduce_q8(vals, w), exact)
    assert naive_err > 0                             # the naive accumulator drifts
    assert comp_err == 0                             # compensation removes it entirely
    assert naive_err < len(vals)                     # ... but stays within the N-ULP bound


def test_drift_is_bounded_by_the_static_estimate():
    rng = random.Random(99)
    for _ in range(30):
        n = rng.randint(1, 500)
        vals = [rng.randint(0, 5000) for _ in range(n)]
        w = rng.randint(1, 511)
        err = ulp_distance(naive_reduce_q8(vals, w), exact_reduce_q8(vals, w))
        assert err <= reduction_error_bound(n)       # the bound truly upper-bounds reality


# --- static interval error bounds (the accuracy-dim producer) --------------------

def test_interval_arithmetic_is_exact_integer():
    a, b = Interval(-2, 5), Interval(1, 3)
    assert iadd(a, b) == Interval(-1, 8)
    assert isub(a, b) == Interval(-5, 4)
    # Q8 multiply: [-2,5] x [1,3] in Q8 ticks, each product >> 8.
    m = imul_q8(a, b)
    assert m.lo <= 0 <= m.hi and m.width >= 0


def test_accuracy_bound_distinguishes_reduction_from_elementwise():
    red = gather_reduce(1024).phases[0].claims[0]          # reduce.gather, count 1024
    ew = vector_add(1024).phases[0].claims[0]              # vector.add
    assert accuracy_bound(red) == 1024                     # naive reduction: count ULP
    assert accuracy_bound(red, compensated=True) == 1      # compensated: 1 ULP
    assert accuracy_bound(ew) == 1                         # an elementwise op: <= 1 ULP


def test_meets_tolerance_is_a_checkable_contract():
    red = gather_reduce(1024).phases[0].claims[0]
    assert not meets_tolerance(red, tolerance_ulp=8)                    # naive bound 1024 > 8
    assert meets_tolerance(red, tolerance_ulp=8, compensated=True)      # compensated 1 <= 8


# --- two-truth stability diagnostics (inform, never legislate) -------------------

def test_cancellation_confidence_falls_as_leading_bits_cancel():
    near = cancellation(1000, 999)        # a - b with a ~ b: many bits cancel
    far = cancellation(1000, 1)           # well-separated: little cancellation
    assert isinstance(near, Graded) and isinstance(far, Graded)
    assert near.confidence_milli < far.confidence_milli
    assert near.value == 1 and far.value == 999          # the (graded) result value


def test_stability_signal_stays_graded_and_never_legislates():
    g = cancellation(1024, 1020)
    assert not is_classical(g)            # a graded proposition is NOT a classical verdict
    assert 0 <= g.confidence_milli <= 1000 and g.source.startswith("precision.")


def test_condition_number_flags_cancellation_prone_sums():
    assert condition_milli([1000, -999]) > condition_milli([1000, 1000])   # ill- vs well-conditioned
    assert condition_milli([5, 5, 5]) == 1000                              # all-same-sign: ideal


def test_ulp_distance_is_symmetric_integer():
    assert ulp_distance(300, 256) == 44 == ulp_distance(256, 300)
