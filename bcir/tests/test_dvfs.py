"""Phase-aware DVFS: downclock memory-bound phases (power saved, throughput
unaffected), overclock compute-bound ones (within the thermal/power budget), hold
balanced/uncertain phases nominal. Deterministic, integer."""

from bcir.gem.dvfs import (
    DOWNCLOCK,
    NOMINAL,
    OVERCLOCK,
    classify,
    clock_for,
    plan_dvfs,
)
from bcir.examples import vector_add
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta

AVX = TARGETS["x86_avx512"]


# --- the classifier + clock policy (pure units) ----------------------------------

def test_classify_by_arithmetic_intensity():
    assert classify(compute=1000, memory=100) == "compute"   # compute >> memory
    assert classify(compute=10, memory=1000) == "memory"     # memory >> compute
    assert classify(compute=500, memory=1000) == "balanced"  # in between


def test_compute_bound_overclocks_only_within_thermal_budget():
    clk, _ = clock_for("compute", Theta.cool())
    assert clk == OVERCLOCK
    hot, reason = clock_for("compute", Theta.hot())          # thermal 80 >= cap
    assert hot == NOMINAL and "capped" in reason             # overclock suppressed


def test_memory_bound_downclocks_and_balanced_holds_nominal():
    assert clock_for("memory", Theta.cool())[0] == DOWNCLOCK
    assert clock_for("memory", Theta.hot())[0] == DOWNCLOCK  # safe regardless of Theta
    assert clock_for("balanced", Theta.cool())[0] == NOMINAL


# --- real module: an elementwise (bandwidth-bound) phase downclocks ---------------

def test_bandwidth_bound_phase_is_downclocked():
    m = vector_add(1 << 16)
    r = optimize(m, AVX, Theta.cool())
    plan = plan_dvfs(m, r, Theta.cool())
    pid = m.phases[0].phase_id
    assert plan.clock_of(pid) == DOWNCLOCK and pid in plan.downclocked


def test_downclock_saves_power_without_losing_throughput():
    m = vector_add(1 << 16)
    r = optimize(m, AVX, Theta.cool())
    d = plan_dvfs(m, r, Theta.cool()).decisions[0]
    assert d.klass == "memory"
    # power drops below nominal; throughput is bandwidth-capped, so it does NOT drop.
    assert d.power_milli < NOMINAL and d.throughput_milli == NOMINAL


# --- safety + determinism --------------------------------------------------------

def test_no_overclock_when_hot_is_the_gains_only_guard():
    # a synthetic compute-bound phase: cool overclocks, hot holds nominal.
    cool, _ = clock_for("compute", Theta.cool())
    hot, _ = clock_for("compute", Theta(thermal=85))
    powercap, _ = clock_for("compute", Theta(power=90))
    assert cool == OVERCLOCK and hot == NOMINAL and powercap == NOMINAL


def test_plan_is_deterministic():
    m = vector_add(4096)
    r = optimize(m, AVX, Theta.cool())
    a = plan_dvfs(m, r, Theta.cool())
    b = plan_dvfs(m, r, Theta.cool())
    assert a.decisions == b.decisions
