"""Phase 4 — segment 4 (perf-regression): the deterministic frontend compile-cost guard.

Non-brittle, self-relative invariants -- claim count scales linearly (no super-linear lowering
blowup), the inter-procedural plan reuses a callee's summary across call sites (IPO not regressed),
and the cost is deterministic -- so the gate catches a real regression but passes ordinary changes.
"""

from __future__ import annotations

from bcir.frontends.cfront.cperf import chain_program, frontend_cost, repeated_call_program


def test_claim_count_scales_linearly():
    # doubling the statement count must at most ~double the claims (linear, not quadratic).
    c20 = frontend_cost(chain_program(20))["claims"]
    c40 = frontend_cost(chain_program(40))["claims"]
    c80 = frontend_cost(chain_program(80))["claims"]
    assert 0 < c20 < c40 < c80
    assert c40 <= c20 * 2.2 and c80 <= c40 * 2.2  # sub-quadratic growth


def test_ipo_summary_is_reused_across_every_call_site():
    # plan-once: the helper's cost is reused at each of the k call sites (the inter-procedural win).
    for k in (2, 5, 20):
        cost = frontend_cost(repeated_call_program(k))
        assert cost["reused"] == k  # a regression here = lost IPO reuse
        assert cost["leaves"] <= 2 * k  # leaves stay bounded (callee not re-planned)


def test_repeated_calls_do_not_replan_the_callee():
    # the body of the helper is planned once: going from 2 to 20 call sites adds the entry's own
    # call/add claims, not 18 extra copies of the helper's plan.
    l2 = frontend_cost(repeated_call_program(2))["leaves"]
    l20 = frontend_cost(repeated_call_program(20))["leaves"]
    assert (l20 - l2) <= 20  # bounded by the added call sites, not k*body


def test_cost_is_deterministic():
    src = chain_program(30)
    assert frontend_cost(src) == frontend_cost(src)
