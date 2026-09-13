"""G12 / S2-C: the dispatch law, work-unit budgets, resumable search state, the plan diff.

The four gates the roadmap names, each held as an exact property over the fixtures the
exact rail is already proved on: a legal incumbent at every interruption point of every
solver (budget zero included); two runs with equal budgets and inputs identical, states
included; continuing from a checkpoint equals the uninterrupted run, for the schedule, the
memory and the selection solvers; and the learned ranker held to a permutation of the census.
"""

from __future__ import annotations

import random

from bcir.examples import PROGRAMS
from bcir.gem.diff import plan_diff
from bcir.gem.dispatch import (
    BOUNDED_SIZE,
    SOLVERS,
    CensusError,
    DispatchRecord,
    DispatchRequest,
    dispatch,
    policy_ranking,
    ranked,
    solve_memory,
    solve_schedule,
    solve_selection,
)
from bcir.gem.exact import (
    SearchState,
    SelectionSearchState,
    certify_schedule,
    exact_schedule,
    exact_selection,
)
from bcir.gem.execution_plan import plan_from_realization
from bcir.gem.schedule import durations_from, schedule_eft
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.kbcir.static_memory import LayoutSearchState, exact_layout, first_fit_layout
from bcir.kbcir.weights import ENERGY, PERF, POLICIES
from bcir.tests.exact_fixtures import (
    partition_optimum,
    quality_corpus,
    six_job_corpus,
    six_job_module,
    six_job_target,
)
from bcir.tests.memory_fixtures import WORST_FIXTURE, corpus, items_of
from bcir.tests.sweep_fixtures import random_module


def _same(a, b) -> bool:
    return (
        a.incumbent,
        a.lower_bound,
        a.stop_reason,
        a.expansions,
        a.heuristic,
        [(s.claim_id, s.domain, s.start, s.finish) for s in a.schedule.slots],
    ) == (
        b.incumbent,
        b.lower_bound,
        b.stop_reason,
        b.expansions,
        b.heuristic,
        [(s.claim_id, s.domain, s.start, s.finish) for s in b.schedule.slots],
    )


# --- the law ------------------------------------------------------------------------------


def test_the_dispatch_law_is_a_table_over_the_request():
    for kind in SOLVERS and ("schedule", "memory", "selection"):
        fast, proof = SOLVERS[(kind, "fast")], SOLVERS[(kind, "proof")]
        small = dispatch(DispatchRequest(kind, BOUNDED_SIZE[kind]))
        assert (small.rail, small.solver, small.units, small.expected) == (
            "proof",
            *proof,
            "TMSAO-1",
        )
        large = dispatch(DispatchRequest(kind, BOUNDED_SIZE[kind] + 1))
        assert (large.rail, large.solver, large.expected) == ("proof", proof[0], "TMSAO-2")
        heuristic = dispatch(DispatchRequest(kind, 4, "TMSAO-4"))
        assert (heuristic.rail, heuristic.solver, heuristic.units, heuristic.expected) == (
            "fast",
            *fast,
            "TMSAO-4",
        )
        unfunded = dispatch(DispatchRequest(kind, 4, "TMSAO-1", 0))
        assert (
            unfunded.rail == "fast"
            and unfunded.expected == "TMSAO-4"
            and "budget" in unfunded.reason
        )
        measured = dispatch(DispatchRequest(kind, 4, "TMSAO-3"))
        assert measured.rail == "fast" and "G13" in measured.reason
        assert dispatch(DispatchRequest(kind, 4)) == dispatch(
            DispatchRequest(kind, 4)
        )  # a function
    for bad in (
        lambda: DispatchRequest("streams", 1),
        lambda: DispatchRequest("schedule", -1),
        lambda: DispatchRequest("schedule", 1, "TMSAO-0"),
        lambda: DispatchRequest("schedule", 1, "TMSAO-2", -5),
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("a malformed request was accepted")
    try:
        DispatchRecord(dispatch(DispatchRequest("schedule", 1)), "timeout", 0, "none", "TMSAO-2")
    except ValueError:
        pass
    else:
        raise AssertionError("a stop reason outside the law was accepted")


# --- dispatch.incumbent.first ---------------------------------------------------------------


def test_a_legal_incumbent_exists_at_every_interruption_point_of_every_solver():
    module, target = six_job_module(), six_job_target(2)
    durs = six_job_corpus()[17]
    durations = {i + 1: d for i, d in enumerate(durs)}
    optimum = partition_optimum(durs, 2)
    for budget in (0, 1, 2, 3, 5, 8):
        result, record = solve_schedule(
            DispatchRequest("schedule", 6, "TMSAO-1", budget), module, durations, target
        )
        if budget == 0:
            assert (
                record.decision.rail == "fast"
                and result.makespan == schedule_eft(module, durations, target).makespan
            )
            assert record.stop_reason == "heuristic" and record.granted == "TMSAO-4"
        else:
            assert (
                optimum <= result.incumbent <= result.heuristic
                and result.lower_bound <= result.incumbent
            )
            assert len(result.schedule.slots) == 6 and record.spent == result.expansions
            assert record.granted == ("TMSAO-1" if result.optimal else "TMSAO-2")
            assert record.bound_source in {
                "critical-path",
                "work/capacity",
                "bandwidth/knee",
                "tail-serial",
            }
    rows = items_of(WORST_FIXTURE)
    fast = first_fit_layout(rows)
    for budget in (0, 1, 2, 7, 10**6):
        layout, record = solve_memory(DispatchRequest("memory", len(rows), "TMSAO-1", budget), rows)
        assert layout.lower_bound <= layout.extent <= fast.extent
        assert set(layout.offsets) == {row.rid for row in rows}
        assert record.granted == (
            "TMSAO-4"
            if budget == 0
            else ("TMSAO-1" if layout.stop_reason == "optimal" else "TMSAO-2")
        )
    small = quality_corpus(seeds=0)[0][1]
    for budget in (0, 1, 2, 5):
        selection, record = solve_selection(
            DispatchRequest("selection", 4, "TMSAO-1", budget),
            small,
            TARGETS["x86_avx512"],
            Theta.cool(),
            PERF,
        )
        if budget == 0:
            result, price = selection
            assert price.makespan > 0 and record.stop_reason == "heuristic"
        else:
            assert (
                selection.optimum <= selection.sweep
            )  # an incumbent first: never worse than the sweep
            assert record.spent == selection.assignments <= budget


# --- solver.budget.units: determinism ---------------------------------------------------------


def test_two_runs_with_equal_budgets_and_inputs_are_identical_states_included():
    module, target = six_job_module(), six_job_target(2)
    for durs in six_job_corpus()[::97]:
        durations = {i + 1: d for i, d in enumerate(durs)}
        for budget in (2, 9, 50):
            first = exact_schedule(module, durations, target, budget=budget)
            second = exact_schedule(module, durations, target, budget=budget)
            assert _same(first, second) and first.state.digest == second.state.digest
    rows = items_of(WORST_FIXTURE)
    for budget in (1, 5, 40):
        a, b = exact_layout(rows, budget), exact_layout(rows, budget)
        assert (a.offsets, a.extent, a.stop_reason, a.expansions) == (
            b.offsets,
            b.extent,
            b.stop_reason,
            b.expansions,
        )
        assert a.state.digest == b.state.digest


# --- resume reproduces --------------------------------------------------------------------------


def test_resuming_the_schedule_search_reproduces_the_uninterrupted_run():
    module, target = six_job_module(), six_job_target(2)
    splits = 0
    for durs in six_job_corpus()[::17]:
        durations = {i + 1: d for i, d in enumerate(durs)}
        full = exact_schedule(module, durations, target, budget=5000)
        total = full.expansions
        for b1 in sorted({0, min(1, total), total // 2, max(0, total - 1)}):
            first = exact_schedule(module, durations, target, budget=b1)
            state = SearchState.from_json(first.state.to_json())  # the artifact round-trips
            assert state.digest == first.state.digest
            second = exact_schedule(module, durations, target, budget=total - b1, resume=state)
            both = exact_schedule(module, durations, target, budget=total)
            assert _same(second, both), (durs, b1)
            assert second.state.digest == both.state.digest and second.state.closed
            splits += 1
    assert splits >= 100
    # multi-phase: the module's budget flows phase by phase; three legs equal one run
    rng = random.Random(4)
    checked = 0
    for seed in range(80):
        m = random_module(seed)
        if len(m.phases) < 2:
            continue
        h = TARGETS["x86_avx512"]
        durations = durations_from(optimize(m, h, Theta.cool(), rng.choice([PERF, ENERGY])))
        if not durations:
            continue
        full = exact_schedule(m, durations, h, budget=3000)
        total = full.expansions
        if total < 3:
            continue
        b1, b2 = total // 3, total // 3
        one = exact_schedule(m, durations, h, budget=b1)
        two = exact_schedule(m, durations, h, budget=b2, resume=one.state)
        three = exact_schedule(m, durations, h, budget=total - b1 - b2, resume=two.state)
        assert _same(three, full) and three.state.digest == full.state.digest, seed
        checked += 1
    assert checked >= 8


def test_a_state_is_bound_to_its_inputs():
    module, target = six_job_module(), six_job_target(2)
    durations = {i + 1: d for i, d in enumerate(six_job_corpus()[17])}
    state = exact_schedule(module, durations, target, budget=1).state
    for other_module, other_durations, other_target in (
        (module, {**durations, 1: durations[1] + 1}, target),
        (module, durations, six_job_target(3)),
        (
            random_module(1),
            durations_from(optimize(random_module(1), target, Theta.cool(), PERF)),
            target,
        ),
    ):
        try:
            exact_schedule(other_module, other_durations, other_target, budget=5, resume=state)
        except ValueError:
            continue
        raise AssertionError("a state from other inputs was resumed")
    try:
        SearchState.from_json('{"version": "other"}')
    except ValueError:
        pass
    else:
        raise AssertionError("a foreign document was accepted as a state")


def test_resuming_the_layout_and_the_selection_reproduces_the_uninterrupted_run():
    splits = 0
    for rows in [items_of(f) for _seed, f in list(corpus())[::25]] + [items_of(WORST_FIXTURE)]:
        full = exact_layout(rows, 200_000)
        total = full.expansions
        if total == 0:
            continue
        for b1 in sorted({1, total // 2, max(1, total - 1)}):
            a = exact_layout(rows, b1)
            b = exact_layout(
                rows, max(1, total - b1), resume=LayoutSearchState.from_json(a.state.to_json())
            )
            both = exact_layout(rows, b1 + max(1, total - b1))
            assert (b.offsets, b.extent, b.stop_reason, b.expansions) == (
                both.offsets,
                both.extent,
                both.stop_reason,
                both.expansions,
            )
            assert b.state.digest == both.state.digest
            splits += 1
    assert splits >= 6  # most corpus fixtures close at the root; the witness never does
    module, h = quality_corpus(seeds=0)[0][1], TARGETS["x86_avx512"]
    full = exact_selection(module, h, Theta.cool(), PERF)
    for b1 in (1, 7, 40, 80):
        a = exact_selection(module, h, Theta.cool(), PERF, limit=b1)
        b = exact_selection(
            module,
            h,
            Theta.cool(),
            PERF,
            limit=full.assignments - b1,
            resume=SelectionSearchState.from_json(a.state.to_json()),
        )
        assert (b.optimum, b.widths, b.assignments, b.stop_reason) == (
            full.optimum,
            full.widths,
            full.assignments,
            full.stop_reason,
        )
        assert b.state.digest == full.state.digest
    try:
        exact_selection(
            quality_corpus(seeds=2)[-1][1], h, Theta.cool(), PERF, limit=3, resume=a.state
        )
    except ValueError:
        pass
    else:
        raise AssertionError("a selection state from another module was resumed")


# --- the certificate records the dispatch ------------------------------------------------------


def test_the_certificate_records_the_dispatch_and_its_class_agrees():
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        target = TARGETS["x86_avx512"]
        result = optimize(module, target, Theta.cool(), PERF)
        certificate = certify_schedule(module, result, target, Theta.cool(), PERF)
        record = certificate.dispatch
        assert isinstance(record, DispatchRecord), name
        body = certificate.to_dict()["dispatch"]
        assert (
            body["kind"] == "schedule"
            and body["solver"] == "exact_schedule"
            and body["units"] == "expansions"
        )
        assert body["size"] == len(result.steps) and body["stop_reason"] == certificate.stop_reason
        assert body["spent"] == certificate.expansions and body["granted"] == certificate.klass
        assert body["bound_source"] == max(certificate.bounds, key=lambda b: b.value).name
    # a scope without a policy lowers the class the rail granted, and the record says so
    module, target = six_job_module(), six_job_target(2)
    plain = certify_schedule(module, optimize(module, target, Theta.cool(), PERF), target)
    assert plain.klass == "TMSAO-4" == plain.dispatch.granted and plain.stop_reason == "optimal"
    # a requested heuristic class dispatches the fast rail: no search, no bound source
    fast = certify_schedule(
        module,
        optimize(module, target, Theta.cool(), PERF),
        target,
        Theta.cool(),
        PERF,
        requested="TMSAO-4",
    )
    assert (
        fast.dispatch.decision.rail == "fast" and fast.klass == "TMSAO-4" and fast.expansions == 0
    )


# --- the ranker cannot remove a candidate -------------------------------------------------------


def test_the_learned_ranker_orders_the_census_and_never_removes_a_member():
    from bcir.kbcir.moegate import train_gate
    from bcir.kbcir.portfolio import PolicyPortfolio

    census = [entry.policy for entry in PolicyPortfolio.default().entries.values()]
    assert ranked(census) == census  # no ranker: the census in its own order
    module = random_module(3)
    experts = list(POLICIES.values())
    gate = train_gate(
        [(module, Theta.cool(), 0), (module, Theta(thermal=80), 2)], experts, epochs=2, seed=1
    )
    for theta in (Theta.cool(), Theta(thermal=80, power=70), Theta(mem_pressure=90)):
        order = policy_ranking(gate, module, theta)
        assert sorted(p.name for p in order) == sorted(p.name for p in census)
        assert len(order) == len(census)
    for bad in (
        lambda members: members[1:],
        lambda members: members + [members[0]],
        lambda members: [members[0]] * len(members),
    ):
        try:
            ranked(census, bad)
        except CensusError:
            continue
        raise AssertionError("a ranker that changed the census was accepted")


# --- the plan diff --------------------------------------------------------------------------------


def test_the_plan_diff_names_every_move_reselection_and_relayout():
    from bcir.tests.sweep_fixtures import adoption_fixture
    from dataclasses import replace

    module = adoption_fixture()
    target = replace(TARGETS["x86_avx2"], affinity_domains=2)
    base = optimize(module, target, Theta.cool(), ENERGY)
    from bcir.gem.overlap import optimize_scheduled

    swept, _price = optimize_scheduled(module, target, Theta.cool(), ENERGY)
    before = plan_from_realization(module, base, target, "eft")
    after = plan_from_realization(module, swept, target, "eft")
    diff = plan_diff(before, after)
    assert not diff.empty and diff.regret < 0  # the sweep shortened the makespan
    assert [r.claim_id for r in diff.reselected] == [1]  # the small claim, vec8 -> scalar
    assert diff.reselected[0].before[1] == 8 and diff.reselected[0].after[1] == 1
    assert [r.claim_id for r in diff.repriced] == [1, 2]  # and its successor lost the discount
    assert diff.makespan == (before.makespan, after.makespan)
    assert diff.to_dict()["regret"] == diff.regret and diff.to_dict()["reselected"][0]["claim"] == 1
    same = plan_diff(before, before)
    assert same.empty and same.regret == 0 and same.bound_tightened is None
    # schedules diff like plans; certificates compare bounds
    from bcir.gem.schedule import schedule_plan

    sched_diff = plan_diff(
        schedule_plan(module, base, target), schedule_plan(module, swept, target)
    )
    assert {m.claim_id for m in sched_diff.moved} >= {m.claim_id for m in diff.moved}
    c1 = certify_schedule(module, base, target, Theta.cool(), ENERGY, budget=1)
    c2 = certify_schedule(module, base, target, Theta.cool(), ENERGY)
    tight = plan_diff(before, before, certificates=(c1, c2))
    assert tight.lower_bound == (c1.lower_bound, c2.lower_bound) and tight.bound_tightened >= 0
    try:
        plan_diff(object(), object())
    except TypeError:
        pass
    else:
        raise AssertionError("an unknown plan type was diffed")
