"""The measured-evidence rail: time BCIR's selected realization vs the scalar
baseline. Timings are environment-dependent, so these tests assert the rail
*runs* and yields valid measurements -- not a speedup threshold (which would be
flaky). The honest finding (elementwise kernels are bandwidth-bound, so width is
measured-neutral -- confirming the memory-dominated cost model) is documented in
the strategy roadmap, not pinned here.
"""

from bcir.bench import Comparison, Measurement, bench_available, compare, measure
from bcir.examples import vector_add
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta


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
