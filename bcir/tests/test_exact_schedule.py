"""G4 / S2-B: the bounded exact scheduler, the lower-bound stack and the first TMSAO-2.

Every claim the exact rail makes is held to an oracle that shares no code with it: the
section 6.1 corpus to the partition optimum, hazard-bearing tiny modules to the enumeration
of every active schedule, the one-sweep re-selection to the exhaustive candidate enumeration.
The heuristic artifact is never replaced -- it is certified: `U`, `L`, both gaps, the stop
reason and the budget, bound to the scope they range over.
"""

from __future__ import annotations

import random
from dataclasses import replace

from bcir.examples import PROGRAMS
from bcir.gem.exact import (
    DEFAULT_EXACT_BUDGET,
    Bound,
    ExactSchedule,
    certify_schedule,
    exact_schedule,
    exact_selection,
)
from bcir.gem.schedule import durations_from, schedule_eft, schedule_plan
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.kbcir.weights import ENERGY, PERF
from bcir.tests.exact_fixtures import (
    active_schedule_optimum,
    partition_optimum,
    quality_corpus,
    six_job_corpus,
    six_job_module,
    six_job_target,
)
from bcir.tests.sweep_fixtures import general_fixture, random_module


def test_the_six_job_corpus_is_proved_and_the_heuristic_reproduces_the_report():
    """The report's numbers exactly (190 / 1.0078 / 17:15 on two domains; 18 / 1.0013 / 7:6
    on three), the exact rail at the optimum on every instance, the bound closed."""
    module = six_job_module()
    corpus = six_job_corpus()
    assert len(corpus) == 1716
    expected = {2: (190, 17 / 15, 1.0078), 3: (18, 7 / 6, 1.0013)}
    for domains in (2, 3):
        target = six_job_target(domains)
        suboptimal = 0
        worst = 0.0
        mean = 0.0
        expansions = 0
        for index, durs in enumerate(corpus):
            durations = {i + 1: d for i, d in enumerate(durs)}
            optimum = partition_optimum(durs, domains)
            heuristic = schedule_eft(module, durations, target).makespan
            ratio = heuristic / optimum
            suboptimal += ratio > 1
            worst = max(worst, ratio)
            mean += ratio
            if index % 3:  # the exact rail on a third of the corpus keeps the quick tier quick
                continue
            certified = exact_schedule(module, durations, target)
            assert certified.heuristic == heuristic
            assert certified.incumbent == optimum == certified.lower_bound, (durs, domains)
            assert certified.optimal and certified.schedule.makespan == optimum
            assert certified.incumbent_gap["relative"] == 0.0
            expansions = max(expansions, certified.expansions)
        assert suboptimal == expected[domains][0], domains
        assert worst == expected[domains][1], domains
        assert abs(mean / len(corpus) - expected[domains][2]) < 5e-4
        assert expansions < DEFAULT_EXACT_BUDGET // 10


def test_tiny_hazard_modules_match_the_enumeration_of_every_active_schedule():
    rng = random.Random(5)
    checked = 0
    for seed in range(300):
        module = random_module(seed)
        if any(len(phase.claims) > 4 for phase in module.phases):
            continue
        for tname in ("x86_avx2", "nvidia_ptx"):
            target = replace(TARGETS[tname], affinity_domains=rng.choice([1, 2, 3]))
            for policy in (PERF, ENERGY):
                durations = durations_from(optimize(module, target, Theta.cool(), policy))
                if not durations:
                    continue
                certified = exact_schedule(module, durations, target)
                want = active_schedule_optimum(module, durations, target)
                assert certified.incumbent == want == certified.lower_bound, (seed, tname)
                assert certified.optimal and certified.incumbent <= certified.heuristic
                assert certified.schedule.makespan == want
                checked += 1
    assert checked >= 60


def test_the_incumbent_is_a_legal_placement_of_the_module():
    """The returned schedule honors every hazard edge, the eligibility rules and the
    durations: a certificate names a plan that can run."""
    from bcir.gem.schedule import phase_hazards

    for name, build in sorted(PROGRAMS.items()):
        module = build()
        target = TARGETS["x86_avx512"]
        durations = durations_from(optimize(module, target, Theta.cool(), PERF))
        certified = exact_schedule(module, durations, target, budget=5_000)
        slots = {slot.claim_id: slot for slot in certified.schedule.slots}
        assert set(slots) == set(durations), name
        hazards = phase_hazards(module)
        for pid, preds in hazards.items():
            for cid, before in preds.items():
                for p in before:
                    assert slots[p].finish <= slots[cid].start, (name, cid, p)
        for cid, slot in slots.items():
            assert slot.finish - slot.start == max(0, durations[cid]), name
        by_stream: dict[int, list] = {}
        for slot in slots.values():
            by_stream.setdefault(slot.domain, []).append(slot)
        for stream, placed in by_stream.items():
            placed.sort(key=lambda s: s.start)
            for earlier, later in zip(placed, placed[1:]):
                assert earlier.finish <= later.start, (name, stream)
        assert certified.schedule.makespan == max(s.finish for s in slots.values())


def _first_suboptimal_instance():
    """The first six-job instance (two domains) the heuristic gets wrong: the root bound is
    below the heuristic there, so a search with a budget of three expansions cannot close."""
    module, target = six_job_module(), six_job_target(2)
    for durs in six_job_corpus():
        durations = {i + 1: d for i, d in enumerate(durs)}
        if schedule_eft(module, durations, target).makespan > partition_optimum(durs, 2):
            return module, target, durations
    raise AssertionError("the corpus has no suboptimal instance")


def test_a_budget_stop_keeps_the_heuristic_and_a_valid_bound():
    module, target, durations = _first_suboptimal_instance()
    certified = exact_schedule(module, durations, target, budget=3)
    closed = exact_schedule(module, durations, target, budget=DEFAULT_EXACT_BUDGET)
    assert closed.optimal and closed.incumbent < closed.heuristic
    assert certified.stop_reason == "budget" and certified.expansions == 3
    assert certified.incumbent == certified.heuristic  # nothing improved yet: the artifact stands
    assert certified.lower_bound <= closed.lower_bound <= closed.incumbent <= certified.incumbent
    assert 0.0 <= certified.incumbent_gap["relative"] <= 1.0
    assert certified.heuristic_gap["absolute"] == certified.heuristic - certified.lower_bound
    root = max(bound.value for bound in certified.bounds)
    assert certified.lower_bound >= root  # never below the root stack
    try:
        exact_schedule(module, durations, target, budget=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("a negative budget was accepted")


def test_the_bound_stack_is_named_valid_and_summed_over_phases():
    module = general_fixture(3, 3)
    target = TARGETS["x86_avx2"]
    durations = durations_from(optimize(module, target, Theta.cool(), ENERGY))
    certified = exact_schedule(module, durations, target)
    names = [bound.name for bound in certified.bounds]
    assert names == ["critical-path", "work/capacity", "bandwidth/knee", "tail-serial"]
    for bound in certified.bounds:
        assert isinstance(bound, Bound) and 0 <= bound.value <= certified.incumbent
    for name in names:
        per_phase = sum(
            bound.value
            for phase in certified.phases
            for bound in phase.bounds
            if bound.name == name
        )
        assert per_phase == next(b.value for b in certified.bounds if b.name == name)
    assert len(certified.phases) == 3
    assert sum(phase.incumbent for phase in certified.phases) == certified.incumbent


def test_the_certificate_carries_l_u_both_gaps_and_refuses_to_over_claim():
    module, target = six_job_module(), six_job_target(2)
    result = optimize(module, target, Theta.cool(), PERF)
    closed = certify_schedule(module, result, target, Theta.cool(), PERF)
    assert closed.klass == "TMSAO-1" and closed.stop_reason == "optimal"
    assert closed.lower_bound == closed.incumbent <= closed.heuristic
    assert closed.incumbent_gap["relative"] == 0.0
    assert closed.heuristic_gap["absolute"] == closed.heuristic - closed.lower_bound
    body = closed.to_dict()
    assert body["objective"] == "makespan" and body["class"] == "TMSAO-1"
    assert body["bounds"] and all(len(item) == 2 for item in body["bounds"])
    assert len(closed.scope) == 64
    # a plan under a budget that cannot close: an explicit gap, TMSAO-2
    from bcir.gem.exact import ScheduleCertificate

    module, target, durations = _first_suboptimal_instance()
    bounded = exact_schedule(module, durations, target, budget=3)
    assert bounded.stop_reason == "budget" and bounded.lower_bound < bounded.incumbent
    from bcir.kbcir.scope import certificate_class_allowed, scope_for

    scope = scope_for(module, target, Theta.cool(), PERF, objective={"name": "makespan"})
    klass, statement = certificate_class_allowed(
        scope, {"incumbent": True, "lower_bound": True, "proof": False}
    )
    assert klass == "TMSAO-2" and "gap" in statement
    assert isinstance(
        certify_schedule(
            module,
            optimize(module, target, Theta.cool(), PERF),
            target,
            Theta.cool(),
            PERF,
            budget=3,
        ),
        ScheduleCertificate,
    )
    # an undeclared admitted set (no policy) blocks every optimality rung, however strong the
    # evidence: the certificate says so instead of over-claiming
    undeclared = certify_schedule(
        six_job_module(), optimize(six_job_module(), target, Theta.cool(), PERF), target
    )
    assert undeclared.klass == "TMSAO-4" and "declared" in undeclared.statement
    assert undeclared.scope != closed.scope


def test_the_one_sweep_selection_matches_the_exhaustive_enumeration_on_the_corpus():
    worst = 1.0
    cases = 0
    for name, module in quality_corpus(seeds=24):
        for policy in (PERF, ENERGY):
            selection = exact_selection(module, TARGETS["x86_avx512"], Theta.cool(), policy)
            assert selection.stop_reason == "optimal" and selection.assignments >= 1, name
            assert selection.sweep >= selection.optimum, name
            worst = max(worst, selection.ratio)
            cases += 1
    assert cases >= 80
    assert worst == 1.0  # the canonical artifact removed the retired pricer's anomaly


def test_the_selection_budget_stops_the_enumeration_with_the_best_seen():
    module = quality_corpus(seeds=0)[0][1]
    selection = exact_selection(module, TARGETS["x86_avx512"], Theta.cool(), PERF, limit=2)
    assert selection.stop_reason == "budget" and selection.assignments == 2
    assert (
        selection.optimum
        >= exact_selection(module, TARGETS["x86_avx512"], Theta.cool(), PERF).optimum
    )


def test_the_exact_rail_does_not_move_the_artifact():
    """Certifying never changes what the executors and the twins read."""
    module = six_job_module()
    target = six_job_target(2)
    result = optimize(module, target, Theta.cool(), PERF)
    before = schedule_plan(module, result, target, "eft")
    certify_schedule(module, result, target, Theta.cool(), PERF)
    exact_schedule(module, durations_from(result), target)
    assert schedule_plan(module, result, target, "eft") == before
    assert isinstance(exact_schedule(module, durations_from(result), target), ExactSchedule)
