"""T3 — derived/aggregate telemetry metrics (.MEASURE) + plan-cost sensitivity (.SENS).

Asserts:
  * derived metrics (max/min/avg/rms/count) match an INDEPENDENT, stdlib-only,
    hand-computed result on a fixed batch; empty batch is well-defined; a closed-form
    RMS anchor;
  * `measure_trig_targ` returns the right span for a known TRIG→TARG crossing, None when
    TARG never happens, and respects rising vs falling;
  * the sensitivity finite-difference moves the plan score when the perturbed Theta
    pressure matters to the workload (thermal on a tile-heavy matmul), ranks a dim the
    workload does not exercise LOWER, is deterministic, and `sampling_budget` sums to
    `total` and respects the floor;
  * the two-truth boundary: a sensitivity run emits NO Diagnostic and the module still
    `verify()`s clean before AND after (it perturbs Theta/cost only, never legality).
"""

import math

from bcir.examples import matmul_tiled, vector_add
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.realize import optimize
from bcir.kbcir.telemetry_metrics import (
    METRIC_FIELDS,
    FieldMetrics,
    derive_all,
    derive_field_metrics,
    empty_metrics,
    measure_trig_targ,
    sampling_budget,
    signal_sensitivity,
)
from bcir.kbcir.weights import PERF
from bcir.signal_registry import default_registry
from bcir.telemetry import DataDNA
from bcir.verify import verify


def _rec(**kw) -> DataDNA:
    base = dict(segment_id="s", claim_id=1)
    base.update(kw)
    return DataDNA(**base)


# === derived metrics (.MEASURE) ======================================================


def test_derive_field_metrics_matches_hand_computed():
    # A fixed batch; thermal values [10, 30, 30, 90]. Hand-computed, stdlib only.
    batch = [_rec(thermal=10), _rec(thermal=30), _rec(thermal=30), _rec(thermal=90)]
    vals = [10, 30, 30, 90]
    m = derive_field_metrics(batch, "thermal")
    assert m.count == len(vals) == 4
    assert m.minimum == min(vals) == 10
    assert m.maximum == max(vals) == 90
    # avg: round-half-up integer mean. sum=160, n=4 -> (160 + 2)//4 = 40.
    assert m.avg == (sum(vals) + len(vals) // 2) // len(vals) == 40
    # rms: round(sqrt(mean(x^2))). mean(x^2) = (100+900+900+8100)/4 = 2500 -> sqrt = 50.
    assert m.rms == round(math.sqrt(sum(v * v for v in vals) / len(vals))) == 50


def test_avg_round_half_up_is_exact_integer():
    # sum=10 over n=4 -> 2.5 rounds half-up to 3 ((10 + 2)//4 = 3). Deterministic, float-free.
    batch = [_rec(misses=1), _rec(misses=2), _rec(misses=3), _rec(misses=4)]
    m = derive_field_metrics(batch, "misses")
    assert m.avg == (10 + 2) // 4 == 3


def test_rms_closed_form_anchor_3_4():
    # Classic closed-form anchor: RMS of [3, 4] = sqrt((9+16)/2) = sqrt(12.5) = 3.5355...
    batch = [_rec(utilization=3), _rec(utilization=4)]
    m = derive_field_metrics(batch, "utilization")
    true_rms = math.sqrt((3 * 3 + 4 * 4) / 2)
    assert abs(true_rms - 3.53553) < 1e-4               # the documented ~3.535 anchor
    assert m.rms == round(true_rms) == 4                # round-to-nearest


def test_rms_all_zero_is_zero():
    batch = [_rec(thermal=0), _rec(thermal=0), _rec(thermal=0)]
    m = derive_field_metrics(batch, "thermal")
    assert m.rms == 0 and m.avg == 0 and m.minimum == 0 and m.maximum == 0


def test_counter_field_metrics_exact_over_large_values():
    # cycles is a non-negative counter (not a 0..100 pressure): metrics must be exact ints.
    batch = [_rec(cycles=1000), _rec(cycles=3000), _rec(cycles=2000)]
    m = derive_field_metrics(batch, "cycles")
    assert m.minimum == 1000 and m.maximum == 3000
    assert m.avg == (6000 + 1) // 3 == 2000
    assert m.rms == round(math.sqrt((1000**2 + 3000**2 + 2000**2) / 3))


def test_empty_batch_is_well_defined_not_a_crash():
    m = derive_field_metrics([], "thermal")
    assert m == empty_metrics("thermal")
    assert m.count == 0
    assert m.minimum is None and m.maximum is None and m.avg is None and m.rms is None


def test_unknown_field_raises():
    raised = False
    try:
        derive_field_metrics([_rec(thermal=10)], "no_such_field")
    except KeyError:
        raised = True
    assert raised, "an unknown metric field must raise, not silently return a wrong answer"


def test_derive_all_covers_every_metric_field():
    batch = [_rec(thermal=50, misses=10, voltage=5, utilization=80, cycles=99, bytes=7)]
    allm = derive_all(batch)
    assert set(allm) == set(METRIC_FIELDS)
    for f in METRIC_FIELDS:
        assert isinstance(allm[f], FieldMetrics)
        assert allm[f].count == 1
    assert allm["thermal"].maximum == 50
    assert allm["utilization"].avg == 80


def test_derive_is_side_effect_free():
    batch = [_rec(thermal=10), _rec(thermal=20)]
    snapshot = [r.thermal for r in batch]
    derive_field_metrics(batch, "thermal")
    derive_all(batch)
    assert [r.thermal for r in batch] == snapshot       # records untouched


# === measure_trig_targ (the SPICE TRIG…TARG figure-of-merit) =========================


def test_trig_targ_rising_known_span():
    # rising: TRIG at first >=60 (index 2, value 65), TARG at first >=90 at/after (index 4).
    series = [10, 40, 65, 80, 95, 100]
    assert measure_trig_targ(series, trig=60, targ=90) == 4 - 2 == 2


def test_trig_targ_same_index_span_zero():
    # A sample that satisfies both TRIG and TARG on the same step -> span 0.
    series = [10, 20, 95]
    assert measure_trig_targ(series, trig=60, targ=90) == 0


def test_trig_targ_none_when_targ_never_reached():
    series = [10, 70, 75, 80]                            # crosses trig=60, never targ=90
    assert measure_trig_targ(series, trig=60, targ=90) is None


def test_trig_targ_none_when_trig_never_reached():
    series = [1, 2, 3]
    assert measure_trig_targ(series, trig=60, targ=90) is None


def test_trig_targ_falling_semantics():
    # falling: TRIG at first <=40 (index 1, value 40), TARG at first <=10 at/after (index 3).
    series = [80, 40, 30, 10, 5]
    assert measure_trig_targ(series, trig=40, targ=10, rising=False) == 3 - 1 == 2
    # the rising reading of the same series finds no >=40-then->=90 (no value reaches 90).
    assert measure_trig_targ(series, trig=40, targ=90, rising=True) is None


def test_trig_targ_targ_only_searched_at_or_after_trig():
    # An early sample that already satisfies targ but NOT trig must not anchor the span:
    # here trig=70, targ=90. idx 0 (value 92) is >=90 but does NOT cross trig=70? it does
    # (92>=70) -> so choose a value that satisfies targ-but-not-trig is impossible when
    # targ>trig; instead show TARG is searched only AT/AFTER the trig index: the second
    # targ crossing (idx 4) is the one paired with the trig at idx 2, not the later targ.
    series = [10, 40, 75, 80, 95]    # trig=70 first at idx 2 (75); targ=90 first at idx 4 (95)
    assert measure_trig_targ(series, trig=70, targ=90) == 4 - 2 == 2


# === sensitivity (.SENS) =============================================================

_H = TargetProfile.x86_avx512()
# A near-hot Theta: thermal 55 so a +10 bump crosses the >=60 thermal-coupling threshold
# that wide (width>=16) tile candidates pay (see realize._context_factor).
_THETA = Theta(thermal=55, power=50, mem_pressure=40, contention=30)


def test_thermal_sensitivity_moves_plan_score_on_tile_heavy_workload():
    # matmul_tiled's wide T-lane tiles pay the thermal coupling when hot, so perturbing
    # the thermal pressure must change the plan score (|Δscore| > 0).
    ranked = signal_sensitivity(matmul_tiled(), _H, _THETA)
    thermal_rows = [r for r in ranked if r.cost_dim == "thermal"]
    assert thermal_rows, "expected at least one thermal-mapped signal"
    for r in thermal_rows:
        assert r.theta_field == "thermal"
        assert r.abs_delta > 0, "perturbing thermal must move the tile-heavy plan score"


def test_thermal_ranks_above_a_dim_the_workload_does_not_exercise():
    ranked = signal_sensitivity(matmul_tiled(), _H, _THETA)
    by_dim_max = {}
    for r in ranked:
        by_dim_max[r.cost_dim] = max(by_dim_max.get(r.cost_dim, 0), r.abs_delta)
    # matmul_tiled does not drive mem_pressure through the optimizer's Theta coupling
    # (no memory-pressure-sensitive candidate in this single-phase tile plan), so the
    # memory-mapped signals rank below thermal.
    assert by_dim_max.get("thermal", 0) > by_dim_max.get("memory", 0), (
        f"thermal should outrank memory on a tile-heavy workload: {by_dim_max}")
    # The top-ranked row is thermal.
    assert ranked[0].cost_dim == "thermal"


def test_sensitivity_ranking_is_deterministic_across_runs():
    a = signal_sensitivity(matmul_tiled(), _H, _THETA)
    b = signal_sensitivity(matmul_tiled(), _H, _THETA)
    assert [(r.signal, r.delta) for r in a] == [(r.signal, r.delta) for r in b]
    # ranked by |Δscore| desc, then name asc.
    keys = [(-r.abs_delta, r.signal) for r in a]
    assert keys == sorted(keys)


def test_sensitivity_is_a_finite_difference_over_optimize():
    # Independently reproduce one row: re-run optimize with thermal bumped by delta.
    ranked = signal_sensitivity(matmul_tiled(), _H, _THETA, delta=10)
    baseline = optimize(matmul_tiled(), _H, _THETA, PERF).score
    bumped = optimize(matmul_tiled(), _H, Theta(thermal=65, power=50,
                                                mem_pressure=40, contention=30), PERF).score
    thermal_row = next(r for r in ranked if r.cost_dim == "thermal")
    assert thermal_row.baseline_score == baseline
    assert thermal_row.perturbed_score == bumped
    assert thermal_row.delta == bumped - baseline


def test_unmapped_cost_dim_signal_is_zero_delta():
    # An explicit signal whose cost_dim has no Theta channel (fabric) gets Δscore 0.
    ranked = signal_sensitivity(matmul_tiled(), _H, _THETA,
                                signals=["fabric.interconnect_bytes", "thermal.pressure"])
    fabric = next(r for r in ranked if r.signal == "fabric.interconnect_bytes")
    assert fabric.cost_dim == "fabric"
    assert fabric.theta_field is None
    assert fabric.delta == 0
    # and it ranks at/below the thermal signal.
    assert ranked[0].signal == "thermal.pressure"


def test_default_signals_are_only_perturbable_ones():
    ranked = signal_sensitivity(matmul_tiled(), _H, _THETA)
    reg = default_registry()
    for r in ranked:
        assert r.theta_field is not None, (
            "default signal set should only include Theta-perturbable signals")
        assert reg.get(r.signal) is not None


def test_zero_delta_perturbation_rejected():
    raised = False
    try:
        signal_sensitivity(matmul_tiled(), _H, _THETA, delta=0)
    except ValueError:
        raised = True
    assert raised


# === sampling_budget =================================================================


def test_sampling_budget_sums_to_total_and_respects_floor():
    ranked = signal_sensitivity(matmul_tiled(), _H, _THETA)
    total = 137
    budget = sampling_budget(ranked, total)
    assert sum(budget.values()) == total
    assert set(budget) == {r.signal for r in ranked}
    for v in budget.values():
        assert v >= 1                                   # the default floor
    # the highest-|Δscore| signal gets at least as much as the lowest.
    top = ranked[0].signal
    bottom = ranked[-1].signal
    assert budget[top] >= budget[bottom]


def test_sampling_budget_floor_keeps_zero_sensitivity_signal_alive():
    ranked = signal_sensitivity(matmul_tiled(), _H, _THETA)
    zero_rows = [r for r in ranked if r.abs_delta == 0]
    assert zero_rows, "expected at least one zero-sensitivity signal (e.g. memory/contention)"
    budget = sampling_budget(ranked, 100, floor=2)
    for r in zero_rows:
        assert budget[r.signal] >= 2                    # the floor protects it from going dark
    assert sum(budget.values()) == 100


def test_sampling_budget_proportional_to_abs_delta():
    # A hand-built ranked set: deltas 0, 30, 70. floor=1, total=101 -> remaining 98 split
    # 30:70 -> 29.4 : 68.6 -> base 29:68 (sum 97), 1 leftover to the larger remainder (70).
    class _Row:
        def __init__(self, signal, ad):
            self.signal, self._ad = signal, ad
        @property
        def abs_delta(self):
            return self._ad
    rows = [_Row("hi", 70), _Row("mid", 30), _Row("lo", 0)]
    budget = sampling_budget(rows, 101, floor=1)
    assert sum(budget.values()) == 101
    assert budget["lo"] == 1                             # floor only (zero sensitivity)
    assert budget["hi"] > budget["mid"] > budget["lo"]
    assert budget["hi"] == 1 + 69 and budget["mid"] == 1 + 29


def test_sampling_budget_all_zero_spreads_evenly():
    class _Row:
        def __init__(self, signal):
            self.signal, self.abs_delta = signal, 0
    rows = [_Row("a"), _Row("b"), _Row("c")]
    budget = sampling_budget(rows, 11, floor=1)          # remaining 8 over 3 -> 3,3,2 + floors
    assert sum(budget.values()) == 11
    assert budget["a"] >= budget["c"]                    # leftover to the earliest


def test_sampling_budget_rejects_total_below_floors():
    ranked = signal_sensitivity(matmul_tiled(), _H, _THETA)
    raised = False
    try:
        sampling_budget(ranked, 1)                       # 1 < floor(1) * len(ranked)
    except ValueError:
        raised = True
    assert raised


def test_sampling_budget_empty_is_empty():
    assert sampling_budget([], 100) == {}


# === two-truth boundary ==============================================================


def test_sensitivity_never_touches_legality():
    # The module verifies clean BEFORE the sensitivity run...
    m = matmul_tiled()
    assert verify(m) == [], "module must verify clean before"
    ranked = signal_sensitivity(m, _H, _THETA)
    # ...the result carries NO Diagnostic (it is a list of plain SignalSensitivity rows)...
    for r in ranked:
        assert not hasattr(r, "code"), "a sensitivity row must not be a Diagnostic"
    # ...and the module still verifies clean AFTER (Theta/cost perturbed, never legality).
    assert verify(m) == [], "module must verify clean after a sensitivity run"


def test_sensitivity_does_not_mutate_theta():
    before = Theta(thermal=55, power=50, mem_pressure=40, contention=30)
    signal_sensitivity(matmul_tiled(), _H, before)
    assert before == Theta(thermal=55, power=50, mem_pressure=40, contention=30)


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
    print(f"\n{len(fns) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys

    sys.exit(_run_all())
