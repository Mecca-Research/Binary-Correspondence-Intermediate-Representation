"""The measured-evidence rail: time BCIR's selected realization vs the scalar
baseline. Timings are environment-dependent, so these tests assert the rail
*runs* and yields valid measurements -- not a speedup threshold (which would be
flaky). The honest finding (elementwise kernels are bandwidth-bound, so width is
measured-neutral -- confirming the memory-dominated cost model) is documented in
the strategy roadmap, not pinned here.
"""

from bcir.bench import (
    Comparison,
    GatherComparison,
    Measurement,
    ReduceComparison,
    bench_available,
    compare,
    compare_gather,
    compare_reduce,
    compare_strided,
    measure,
)
from bcir.examples import gather_reduce, saxpy_strided, vector_add
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.kbcir.realize import candidates_for
from bcir.model import Lane


def test_rail_produces_valid_measurements():
    if not bench_available():
        return  # skip cleanly without a C compiler
    c = compare("vector_add", target="x86_avx512", theta="cool", opt="-O2",
                n=1 << 16, reps=50)
    assert isinstance(c, Comparison)
    assert c.bcir.ok and c.baseline.ok
    assert c.bcir.ns_per_call > 0 and c.baseline.ns_per_call > 0
    assert c.bcir.width == 16 and c.baseline.width == 1     # selected vs scalar
    assert c.speedup_milli > 0                              # a finite measured ratio


def test_measure_reports_the_selected_width():
    if not bench_available():
        return
    m = vector_add(1024)
    r = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    meas = measure(m, r, label="sel", opt="-O2", n=1 << 16, reps=20)
    assert isinstance(meas, Measurement) and meas.ok and meas.width == 16


def test_both_realizations_are_correct_under_timing():
    # the timed harness self-initializes and runs; a build/run failure surfaces as
    # ok=False with detail (the rail never silently reports a bogus speedup).
    if not bench_available():
        return
    c = compare("vector_add", opt="-O1", n=1 << 14, reps=20)
    assert c.bcir.ok, c.bcir.detail
    assert c.baseline.ok, c.baseline.detail


# --- gather avoidance: the cost-model decision + the measured penalty -------------

def test_cost_model_avoids_gather_on_a_strided_claim():
    # On a strided claim both a strided (U) and a gather (GGG) candidate exist; the
    # gather candidate must cost strictly more (gather_penalty), so it is never the
    # cheapest -- BCIR avoids GGG by construction.
    m = saxpy_strided(1024)
    claim = m.phases[0].claims[0]
    h = TARGETS["x86_avx512"]
    cands = candidates_for(claim, h, m.resource(claim.rd[0]))
    by_lane = {c.lane: c for c in cands}
    assert Lane.U in by_lane and Lane.GGG in by_lane          # both realizations offered
    from bcir.kbcir.weights import PERF, weights
    w = weights(h, Theta.cool(), 0, PERF)
    assert by_lane[Lane.GGG].base.dot(w) > by_lane[Lane.U].base.dot(w)  # gather is dearer


def test_gather_avoidance_is_measured_and_correct():
    if not bench_available():
        return
    # a working set that exceeds cache (4 MB) so the random gather pays real
    # misses -- the gather_penalty realized on silicon.
    c = compare_gather("vector_add", opt="-O2", n=1 << 20, reps=40, shuffle=True)
    assert isinstance(c, GatherComparison)
    assert c.direct.ok and c.gather.ok                        # both built + self-checked
    assert c.direct.ns_per_call > 0 and c.gather.ns_per_call > 0
    # avoiding GGG is measurably faster. We observe ~6.5x; assert a conservative
    # 1.5x floor to stay CI-robust across cache sizes.
    assert c.avoidance_wins and c.speedup_milli >= 1500


def test_identity_gather_isolates_indexed_load_overhead():
    if not bench_available():
        return
    c = compare_gather("vector_add", opt="-O2", n=1 << 16, reps=30, shuffle=False)
    assert c.direct.ok and c.gather.ok and c.speedup_milli > 0


# --- gather/blocked reduction: a real gather program lowered two ways -------------

def test_cost_model_selects_blocked_for_a_reducible_gather():
    # `reduce.gather` offers blocked (sequential) + gather (random); BCIR picks
    # blocked -- the gather avoided, by construction.
    m = gather_reduce(1024)
    claim = m.phases[0].claims[0]
    cands = candidates_for(claim, TARGETS["x86_avx512"], m.resource(claim.rd[0]))
    assert {c.name for c in cands} == {"blocked", "gather"}
    r = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    assert r.steps[0].candidate.name == "blocked"


def test_reduce_gather_avoidance_is_correct_and_measured():
    if not bench_available():
        return
    c = compare_reduce("gather_reduce", opt="-O2", n=1 << 20, reps=30)
    assert isinstance(c, ReduceComparison)
    assert c.selected == "blocked" and c.correct          # identical integer sum
    # observed ~17x; assert a conservative 2x floor (random-load reduction is
    # dominated by gather latency, so the win is large and robust).
    assert c.avoidance_wins and c.speedup_milli >= 2000


def test_strided_gather_avoidance_is_measured_non_reduction():
    # saxpy_strided: the cost model picks the direct strided realization over the
    # gather; a non-reduction gather avoidance (BCIR knows the stride).
    m = saxpy_strided(1024)
    r = optimize(m, TARGETS["x86_avx512"], Theta.cool())
    assert r.steps[0].candidate.name == "strided"     # gather avoided by selection
    if not bench_available():
        return
    c = compare_strided("saxpy_strided", opt="-O2", n=1 << 22, reps=30)
    assert c.selected == "strided" and c.correct       # identical Y
    # observed ~1.4x (the gather-instruction overhead); conservative 1.1x floor.
    assert c.avoidance_wins and c.speedup_milli >= 1100
