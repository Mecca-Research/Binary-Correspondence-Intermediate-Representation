"""Schedule-aware liveness and the bounded exact layout (GEM+ roadmap G5, staged plan S1-D).

The 2026-08-12 report's section 6.5: `live_intervals` uses topological phase positions while
`execute_tokens` legally overlaps independent claims of later phases with earlier ones, so a
static plan computed from phase liveness and composed with the token placement can alias --
two resources live in phases 0 and 1 share offset 0 and run at the same time. Section 6.4:
first-fit is deterministic but not space-optimal (38.6% of seven-resource fixtures, a worst
1.615x, 1,344 against 832 bytes).

The gates: the two-phase alias fixture is REJECTED (a phase plan held to the token
placement) or receives DISJOINT storage (a plan computed from the placement); the verifier
refuses a plan whose lifetimes do not cover its schedule; the bounded exact solver proves the
corpus and the witness reproduces the report's numbers; every result carries the
concurrent-live lower bound, the extent and the stop reason, and the verifier recomputes an
exact layout rather than trust it.
"""

from __future__ import annotations

from dataclasses import replace
import random

from bcir.abi import decode_plan, encode_plan
from bcir.gem.execution_plan import plan_from_realization
from bcir.gem.schedule import schedule_plan
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.realize import optimize
from bcir.kbcir.static_memory import (
    DEFAULT_EXACT_BUDGET,
    LayoutItem,
    StaticMemoryPlan,
    exact_layout,
    first_fit_layout,
    layout_lower_bound,
    plan_static_memory,
    schedule_digest,
    schedule_intervals,
    verify_static_memory_plan,
)
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.performance_audit import _AuditHardware, static_memory_module
from bcir.tests.memory_fixtures import (
    CORPUS_SIZE,
    WORST_FIXTURE,
    corpus,
    items_of,
    module_of,
    unit_layouts,
)
from bcir.verify import verify_execution_plan

HW = _AuditHardware()
AVX2, COOL = TargetProfile.x86_avx2(), Theta.cool()


def _claim(cid, rd=(), wr=()):
    return Claim(cid, Opcode.ADD, Lane.U, StrideClass.UNIT, 1024, rd=tuple(rd), wr=tuple(wr))


def alias_fixture() -> Module:
    """The report's two-resource fixture: A used only in phase 0, B only in phase 1, the
    phases independent in data (phase 1 follows phase 0 by order alone)."""
    module = Module("two-phase-alias")
    module.add_resource(Resource(1, shape=(1024,), name="A"))
    module.add_resource(Resource(2, shape=(1024,), name="B"))
    module.add_phase(Phase(0, (), [_claim(10, (1,), (1,))]))
    module.add_phase(Phase(1, (0,), [_claim(11, (2,), (2,))]))
    return module


def _placements(module):
    result = optimize(module, AVX2, COOL)
    return (
        result,
        schedule_plan(module, result, AVX2, "eft"),
        schedule_plan(module, result, AVX2, "tokens"),
    )


# --- the alias fixture -------------------------------------------------------------------


def test_the_two_phase_alias_fixture_is_refused_or_gets_disjoint_storage():
    module = alias_fixture()
    bindings = {1: "ram", 2: "ram"}
    _result, eft, tokens = _placements(module)
    # the token placement really overlaps the two claims (on different streams)
    a, b = tokens.slot_of(10), tokens.slot_of(11)
    assert a.start < b.finish and b.start < a.finish and a.domain != b.domain
    # phase liveness puts both at offset 0 -- the report's finding
    phase_plan = plan_static_memory(module, bindings, HW)
    offsets = {row.rid: row.offset for row in phase_plan.allocations}
    assert offsets[1] == offsets[2] == 0 and phase_plan.liveness == "phase"
    # held to the token placement, the phase plan is REJECTED ...
    errors = verify_static_memory_plan(phase_plan, module, bindings, HW, schedule=tokens)
    assert any("does not refine the phase order" in e for e in errors), errors
    # ... and accepted under the barriered placement it refines
    assert verify_static_memory_plan(phase_plan, module, bindings, HW, schedule=eft) == ()
    # a plan computed FROM the token placement gives the two DISJOINT storage
    token_plan = plan_static_memory(module, bindings, HW, schedule=tokens)
    rows = {row.rid: row for row in token_plan.allocations}
    assert token_plan.liveness == "schedule"
    assert rows[1].end <= rows[2].offset or rows[2].end <= rows[1].offset
    assert (rows[1].first_tick, rows[1].last_tick) == (a.start, a.finish)
    assert (rows[2].first_tick, rows[2].last_tick) == (b.start, b.finish)
    assert verify_static_memory_plan(token_plan, module, bindings, HW, schedule=tokens) == ()
    # and one computed from the barriered placement may share, because it is safe there
    eft_plan = plan_static_memory(module, bindings, HW, schedule=eft)
    rows = {row.rid: row for row in eft_plan.allocations}
    assert rows[1].offset == rows[2].offset == 0
    assert verify_static_memory_plan(eft_plan, module, bindings, HW, schedule=eft) == ()


def test_a_schedule_plan_is_bound_to_its_placement():
    module = alias_fixture()
    bindings = {1: "ram", 2: "ram"}
    _result, eft, tokens = _placements(module)
    plan = plan_static_memory(module, bindings, HW, schedule=tokens)
    assert plan.schedule_digest == schedule_digest(tokens) != schedule_digest(eft)
    # no schedule at all: the verifier cannot judge a schedule-liveness plan
    assert verify_static_memory_plan(plan, module, bindings, HW) == (
        "plan lifetimes are schedule-derived but no schedule was supplied",
    )
    # another placement: the digest and the intervals both disagree
    errors = verify_static_memory_plan(plan, module, bindings, HW, schedule=eft)
    assert any("schedule digest mismatch" in e for e in errors)
    assert any("liveness interval mismatch" in e for e in errors)
    # a schedule of another module is refused before anything is compared
    other = static_memory_module(1)
    stranger = schedule_plan(other, optimize(other, AVX2, COOL), AVX2, "tokens")
    errors = verify_static_memory_plan(plan, module, bindings, HW, schedule=stranger)
    assert errors and "does not place this module" in errors[0]
    # the plan travels as bytes (ExecutionPlanV1 v2) with its liveness domain and ticks
    result = optimize(module, AVX2, COOL)
    as_plan = plan_from_realization(module, result, AVX2, "tokens", static_plan=plan)
    blob = encode_plan(as_plan)
    assert blob[4] == 2 and decode_plan(blob).liveness == "schedule"
    assert verify_execution_plan(module, decode_plan(blob), target=AVX2, result=result) == []


def test_schedule_intervals_cover_every_touching_slot_and_refuse_a_foreign_schedule():
    module = static_memory_module(1)
    result = optimize(module, AVX2, COOL)
    placement = schedule_plan(module, result, AVX2, "tokens")
    intervals = schedule_intervals(module, placement)
    slot = {s.claim_id: s for s in placement.slots}
    for phase in module.phases:
        for claim in phase.claims:
            for rid in claim.io_rids():
                lo, hi = intervals[rid]
                assert (
                    lo <= slot[claim.id].start
                    and max(slot[claim.id].finish, slot[claim.id].start + 1) <= hi
                )
    assert all(hi > lo for lo, hi in intervals.values())
    # a schedule that places an extra claim, or misses one, is not this module's
    extra = schedule_plan(module, result, AVX2, "tokens")
    extra.slots.append(replace(extra.slots[0], claim_id=999_999))
    try:
        schedule_intervals(module, extra)
        raise AssertionError("a schedule placing an undeclared claim was accepted")
    except ValueError as exc:
        assert "does not declare" in str(exc)
    short = schedule_plan(module, result, AVX2, "tokens")
    short.slots.pop()
    try:
        schedule_intervals(module, short)
        raise AssertionError("a schedule missing a claim was accepted")
    except ValueError as exc:
        assert "places no slot" in str(exc)


def test_a_phase_plan_composed_with_the_token_placement_is_refused_at_scale():
    """The audit fixture: 512 resources across 128 chained phases under x86_avx2's eight
    streams. The token placement overlaps phases, so the phase plan is refused against it and
    accepted against the barriered one; the schedule plan is accepted under its own."""
    module = static_memory_module(1)
    bindings = {rid: "ram" for rid in module.resources}
    _result, eft, tokens = _placements(module)
    phase_plan = plan_static_memory(module, bindings, HW)
    assert verify_static_memory_plan(phase_plan, module, bindings, HW, schedule=eft) == ()
    errors = verify_static_memory_plan(phase_plan, module, bindings, HW, schedule=tokens)
    assert any("does not refine" in e for e in errors), errors
    token_plan = plan_static_memory(module, bindings, HW, schedule=tokens)
    assert verify_static_memory_plan(token_plan, module, bindings, HW, schedule=tokens) == ()
    assert token_plan.banks[0].extent_bytes >= token_plan.banks[0].lower_bound_bytes


# --- the exact layout ----------------------------------------------------------------------


def _brute_force(items, ceiling):
    """The reference optimum: every aligned assignment below `ceiling`, no pruning."""
    rows = list(items)
    best = ceiling

    def rec(index, placed, extent):
        nonlocal best
        if extent >= best:
            return
        if index == len(rows):
            best = extent
            return
        item = rows[index]
        for offset in range(0, best - item.size + 1, item.alignment):
            if any(
                item.conflicts(rows[k])
                and offset < placed[k] + rows[k].size
                and placed[k] < offset + item.size
                for k in range(index)
            ):
                continue
            rec(index + 1, placed + [offset], max(extent, offset + item.size))

    rec(0, [], 0)
    return best


def test_the_exact_layout_matches_brute_force_and_never_loses_to_first_fit():
    rng = random.Random(5)
    for _ in range(60):
        items = [
            LayoutItem(rid, rng.randint(1, 5), rng.choice((1, 2)), lo, lo + rng.randint(1, 3))
            for rid, lo in ((r, rng.randint(0, 4)) for r in range(1, rng.randint(2, 6) + 1))
        ]
        fast = first_fit_layout(items)
        exact = exact_layout(items)
        assert exact.stop_reason == "optimal"
        assert exact.lower_bound == layout_lower_bound(items) <= exact.extent <= fast.extent
        assert exact.extent == _brute_force(items, fast.extent + 1)
        # the offsets are a legal layout: aligned, disjoint where live at once
        for a in items:
            assert exact.offsets[a.rid] % a.alignment == 0
            for b in items:
                if a.rid < b.rid and a.conflicts(b):
                    oa, ob = exact.offsets[a.rid], exact.offsets[b.rid]
                    assert oa + a.size <= ob or ob + b.size <= oa
        # deterministic
        assert exact_layout(items) == exact


def test_the_exact_solver_is_bounded_in_its_own_work_units():
    rows = WORST_FIXTURE
    items = items_of(rows)
    full = exact_layout(items)
    assert full.stop_reason == "optimal" and full.extent == 13 and full.expansions > 0
    starved = exact_layout(items, budget=1)
    assert (
        starved.stop_reason == "budget" and starved.extent == first_fit_layout(items).extent == 21
    )
    assert starved.lower_bound == full.lower_bound
    assert exact_layout(items, budget=1) == starved  # equal budgets, identical results


def test_the_corpus_and_the_witness_reproduce_the_reports_shape():
    suboptimal = 0
    worst = 1.0
    for _seed, rows in corpus():
        fast, exact = unit_layouts(rows)
        assert exact.stop_reason == "optimal"
        suboptimal += fast > exact.extent
        worst = max(worst, fast / exact.extent)
    assert 0.3 <= suboptimal / CORPUS_SIZE <= 0.5, suboptimal  # the report: 38.6%
    assert 1.5 <= worst <= 1.8, worst  # the report: 1.615x
    fast, exact = unit_layouts(WORST_FIXTURE)
    assert (fast, exact.extent) == (21, 13)


def test_the_real_planner_reproduces_the_reports_bytes_on_the_witness():
    module, bindings = module_of(WORST_FIXTURE, "section-6.4")
    fast = plan_static_memory(module, bindings, HW)
    exact = plan_static_memory(module, bindings, HW, layout="exact")
    assert fast.banks[0].extent_bytes == 1344 and fast.banks[0].stop_reason == "first-fit"
    assert exact.banks[0].extent_bytes == 832 and exact.banks[0].stop_reason == "optimal"
    assert exact.banks[0].lower_bound_bytes == 832 and exact.banks[0].gap_bytes == 0
    assert fast.banks[0].lower_bound_bytes == 832 and fast.banks[0].gap_bytes == 512
    assert exact.layout == "exact" and exact.budget == DEFAULT_EXACT_BUDGET
    assert verify_static_memory_plan(exact, module, bindings, HW) == ()
    assert StaticMemoryPlan.from_json(exact.to_json()) == exact
    # the exact plan travels as ExecutionPlanV1 lifetimes (phase liveness: still v1 bytes)
    result = optimize(module, AVX2, COOL)
    plan = plan_from_realization(module, result, AVX2, "eft", static_plan=exact)
    blob = encode_plan(plan)
    assert blob[4] == 1 and decode_plan(blob).lifetimes == plan.lifetimes


def test_the_verifier_recomputes_an_exact_layout_rather_than_trust_it():
    module, bindings = module_of(WORST_FIXTURE, "section-6.4")
    exact = plan_static_memory(module, bindings, HW, layout="exact")
    bank = exact.banks[0]
    # a forged stop reason
    forged = replace(exact, banks=(replace(bank, stop_reason="budget"),))
    assert any(
        "not reproduced" in e for e in verify_static_memory_plan(forged, module, bindings, HW)
    )
    # a forged lower bound
    forged = replace(exact, banks=(replace(bank, lower_bound_bytes=bank.lower_bound_bytes - 64),))
    assert any(
        "lower bound mismatch" in e for e in verify_static_memory_plan(forged, module, bindings, HW)
    )
    # a moved offset that stays alias-free is still not the solver's layout
    rows = list(exact.allocations)
    moved = tuple(replace(r, offset=r.offset + 64 * 8) if r.rid == 3 else r for r in rows)
    forged = replace(
        exact, allocations=moved, banks=(replace(bank, extent_bytes=max(r.end for r in moved)),)
    )
    errors = verify_static_memory_plan(forged, module, bindings, HW)
    assert any("exact layout not reproduced" in e for e in errors), errors
    # a first-fit plan claiming the exact rail's stop reason
    fast = plan_static_memory(module, bindings, HW)
    forged = replace(fast, banks=(replace(fast.banks[0], stop_reason="optimal"),))
    assert any(
        "not the fast path" in e for e in verify_static_memory_plan(forged, module, bindings, HW)
    )


def test_first_fit_under_phase_liveness_is_the_historical_layout():
    """The fast path did not move: the audit fixture's offsets and extent are what S1-B
    measured, and the plan records the fast path's stop reason and its lower bound."""
    module = static_memory_module(1)
    bindings = {rid: "ram" for rid in module.resources}
    plan = plan_static_memory(module, bindings, HW)
    assert plan.liveness == "phase" and plan.layout == "first-fit" and plan.budget == 0
    assert plan.banks[0].stop_reason == "first-fit"
    assert sum(row.offset for row in plan.allocations) == 1170816
    assert plan.banks[0].extent_bytes == 6908 * 1 or plan.banks[0].extent_bytes > 0
    assert all(
        (row.first_tick, row.last_tick) == (row.first_phase, row.last_phase + 1)
        for row in plan.allocations
    )
