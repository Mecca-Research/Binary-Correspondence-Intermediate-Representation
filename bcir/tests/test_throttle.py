"""The L1 cost throttle: amortization certificates keep ML throttled where it matters."""

from bcir.kbcir import (
    AmortizationCertificate,
    ThrottleReport,
    TIER_BUDGET,
    certify,
    measure_inference_cost,
)
from bcir.verify import verify_provenance


def _laws(diags):
    return {d.law for d in diags}


# --- the amortization inequality -------------------------------------------------


def test_l0_prohibition_zero_inference_is_admitted():
    # The hot path carries decisions, not models: an L0 component with zero
    # inference cost is admitted; any nonzero inference at L0 is not.
    assert certify("lane_promote", "L0", inference_cost=0, gain=0).admitted
    assert not certify("rogue", "L0", inference_cost=5, gain=1000).admitted


def test_component_must_pay_for_itself():
    # gain >= inference_cost: a component whose inference costs more than it
    # improves does not amortize.
    assert certify("gate", "L1", inference_cost=200, gain=5000).admitted
    assert not certify("wasteful", "L1", inference_cost=5000, gain=200).amortizes


def test_within_budget_is_the_throttle():
    # The L1 budget bounds plan-time inference cost regardless of gain.
    over = certify("heavy", "L1", inference_cost=TIER_BUDGET["L1"] + 1, gain=10**12)
    assert not over.within_budget and not over.admitted
    ok = certify("light", "L1", inference_cost=TIER_BUDGET["L1"], gain=10**9)
    assert ok.within_budget and ok.admitted


def test_accelerator_is_net_negative_overhead():
    # The accelerator's inference is cheaper than the search it saves -> gain
    # dominates trivially (the ideal learned component).
    assert certify("search_accel", "L1", inference_cost=50, gain=900).admitted


def test_unknown_tier_is_not_admitted():
    assert not certify("x", "L9", inference_cost=0, gain=0).admitted


# --- the throttle dashboard ------------------------------------------------------


def test_report_renders_and_flags_offenders():
    certs = [
        certify("calib", "L1", 10, 4000),
        certify("gate", "L1", 300, 6000),
        certify("rogue", "L0", 7, 9000),
    ]
    rep = ThrottleReport(certs=certs)
    assert not rep.admitted()
    assert [c.component for c in rep.offenders()] == ["rogue"]
    text = rep.render()
    assert "THROTTLE" in text and "calib" in text and "gate" in text


def test_measure_inference_cost_is_positive():
    # The offline measurement helper returns a positive median; we assert only
    # the invariant (timing values never enter the deterministic certificate).
    assert measure_inference_cost(lambda: sum(range(1000)), reps=3) >= 1


# --- R13: amortization provenance ------------------------------------------------


def test_admitted_certificates_satisfy_R13():
    certs = [
        certify("lane_promote", "L0", 0, 0),
        certify("gate", "L1", 300, 6000),
        certify("search_accel", "L1", 50, 900),
    ]
    assert verify_provenance(None, amortization_certificates=certs) == []


def test_l0_inference_is_R13():
    bad = certify("rogue", "L0", inference_cost=12, gain=9999)
    assert "R13" in _laws(verify_provenance(None, amortization_certificates=[bad]))


def test_nonamortizing_component_is_R13():
    bad = certify("wasteful", "L1", inference_cost=5000, gain=10)
    assert "R13" in _laws(verify_provenance(None, amortization_certificates=[bad]))


def test_over_budget_component_is_R13():
    bad = certify("heavy", "L1", inference_cost=TIER_BUDGET["L1"] + 1, gain=10**12)
    assert "R13" in _laws(verify_provenance(None, amortization_certificates=[bad]))
