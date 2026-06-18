"""The non-flaky perf-budget gate (`bcir.perf_budget`): strict on correctness + measurement
validity *everywhere*, perf floors enforced *only* on bare-metal, with trend tracking.

These use synthetic `Comparison` objects (the real `bcir.bench` dataclasses, no toolchain) so
the budget *logic* is pinned deterministically on any host — exactly the property the gate
gives the real rail.
"""

import os
import tempfile

from bcir.bench import Comparison, Measurement
from bcir.perf_budget import Budget, evaluate, read_trend, record_trend, trend_summary


def _meas(ns: int, *, width: int, ok: bool = True, detail: str = "") -> Measurement:
    return Measurement(label="x", width=width, opt="-O2", n=1 << 16, reps=20,
                       ns_per_call=ns, ok=ok, detail=detail)


def _cmp(bcir_ns: int, base_ns: int, *, width: int = 16, bcir_ok: bool = True,
         base_ok: bool = True, detail: str = "") -> Comparison:
    return Comparison(program="p", target="t", opt="-O2",
                      bcir=_meas(bcir_ns, width=width, ok=bcir_ok, detail=detail),
                      baseline=_meas(base_ns, width=1, ok=base_ok))


def test_valid_winning_measurement_passes_a_validity_budget():
    r = evaluate(_cmp(100, 250), Budget("gather", expect_width=16), baremetal=False)
    assert r.ok, r.reasons
    assert r.speedup_milli == 2500                          # 250/100 == 2.5x
    assert "correctness" in r.enforced and "measurement-validity" in r.enforced


def test_correctness_violation_fails_regardless_of_baremetal():
    bad = _cmp(100, 250, bcir_ok=False, detail="byte mismatch vs oracle")
    for bare in (False, True):
        r = evaluate(bad, Budget("gather"), baremetal=bare)
        assert not r.ok
        assert any("incorrect" in why for why in r.reasons)


def test_invalid_measurement_fails_regardless_of_baremetal():
    for bare in (False, True):
        r = evaluate(_cmp(0, 250), Budget("gather"), baremetal=bare)   # non-positive ns/call
        assert not r.ok
        assert any("invalid measurement" in why or "not finite" in why for why in r.reasons)


def test_wrong_selected_width_is_a_correctness_failure():
    r = evaluate(_cmp(100, 250, width=8), Budget("gather", expect_width=16), baremetal=False)
    assert not r.ok
    assert any("width" in why for why in r.reasons)


def test_speedup_floor_is_waived_off_baremetal_but_enforced_on_it():
    # measured 1.05x, floor demands 1.5x.
    slow = _cmp(950, 1000)
    assert slow.speedup_milli < 1500
    budget = Budget("reduce", min_speedup_milli=1500)

    off = evaluate(slow, budget, baremetal=False)
    assert off.ok, "a marginal speedup must NOT fail on a shared runner (non-flaky)"
    assert any("min_speedup_milli" in w for w in off.waived)

    on = evaluate(slow, budget, baremetal=True)
    assert not on.ok, "the floor must bite on a controlled bare-metal box"
    assert any("below floor" in why for why in on.reasons)


def test_latency_ceiling_is_baremetal_only():
    over = _cmp(900, 1000)                                  # correct + valid, but slow
    budget = Budget("latency", max_ns_per_call=500)
    assert evaluate(over, budget, baremetal=False).ok       # waived on shared CI
    assert not evaluate(over, budget, baremetal=True).ok     # enforced bare-metal


def test_trend_log_records_and_summarizes():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sub", "trend.jsonl")        # nested dir is created
        for ns in (120, 110, 100):
            record_trend(evaluate(_cmp(ns, 250), Budget("gather"), baremetal=False), path)
        rows = read_trend(path)
        assert len(rows) == 3
        assert all("host" in r and "machine" in r and "ts" in r for r in rows)
        s = trend_summary(path, "gather")
        assert s["count"] == 3 and s["max"] >= s["min"]
        assert s["last"] == rows[-1]["speedup_milli"]


def test_trend_summary_is_empty_for_unknown_budget():
    with tempfile.TemporaryDirectory() as d:
        assert trend_summary(os.path.join(d, "none.jsonl"), "missing") == {
            "name": "missing", "count": 0}


def test_dense_match_band_is_baremetal_only():
    # a dense kernel that's ~parity vs Clang (matches) must pass; a wide divergence must fail on
    # bare-metal but be waived on shared CI (non-flaky).
    near = _cmp(1000, 1020)                              # ~1.02x, inside the band
    far = _cmp(500, 1500)                               # 3.0x — a suspicious "win" on a dense kernel
    budget = Budget("dense", match_band=(700, 1300))
    assert evaluate(near, budget, baremetal=True).ok
    off = evaluate(far, budget, baremetal=False)
    assert off.ok and any("match_band" in w for w in off.waived)   # waived on shared CI
    on = evaluate(far, budget, baremetal=True)
    assert not on.ok and any("parity band" in why for why in on.reasons)


def test_reference_milli_is_tracked_not_asserted():
    # a measured speedup far below the documented reference must still PASS (CI isn't strict on the
    # exact number) — the reference only rides along into the result + trend for drift watching.
    budget = Budget("gather", min_speedup_milli=1500, reference_milli=6000)
    res = evaluate(_cmp(100, 250), budget, baremetal=True)   # 2.5x: above floor, below doc 6.0x
    assert res.ok and res.reference_milli == 6000
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.jsonl")
        record_trend(res, path)
        s = trend_summary(path, "gather")
        assert s["reference"] == 6000 and s["last_vs_reference_pct"] == round(100 * 2500 / 6000)
