"""G2 / S2-A: incremental delta pricing is an EXACT refactor of the re-selection sweep.

The roadmap's law for G2: "a faster sweep that picks a different plan has not been made
faster; it has been changed" -- so every test here holds the delta search to the full
re-placement reference (`optimize_scheduled(..., delta=False)`, the S1-A sweep) claim by
claim, price by price, artifact by artifact, and holds `schedule.EftPlacer` -- the base
placement recorded once, a trial replayed from a checkpoint, later phases skipped unless
they touch a rid the replay moved -- to `schedule_eft` on every fixture, trial and adoption.
"""

from __future__ import annotations

import random
from dataclasses import replace

from bcir.examples import PROGRAMS
from bcir.gem.overlap import optimize_scheduled, price_scheduled
from bcir.gem.schedule import EftPlacer, durations_from, schedule_eft, schedule_plan
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.kbcir.weights import ENERGY, PERF, POLICIES
from bcir.model import Claim, Lane, Opcode, Phase, Resource, StrideClass
from bcir.tests.sweep_fixtures import (
    adoption_fixture,
    general_fixture,
    random_module,
    sweep_fixture,
)
from bcir.verify import verify_plan

_TARGETS = ("x86_avx2", "x86_avx512", "nvidia_ptx")


def _fixtures():
    out = [(name, build()) for name, build in sorted(PROGRAMS.items())]
    out += [
        ("sweep64", sweep_fixture(64)),
        ("general1x16", general_fixture(1, 16)),
        ("general4x4", general_fixture(4, 4)),
        ("adoption", adoption_fixture()),
    ]
    out += [(f"rand{seed}", random_module(seed)) for seed in range(40)]
    return out


def _placements():
    for name, module in _fixtures():
        for tname in _TARGETS:
            h = TARGETS[tname]
            for policy in (PERF, ENERGY):
                durations = durations_from(optimize(module, h, Theta.cool(), policy))
                if durations:
                    yield name, tname, policy, module, h, durations


def _assignment(result):
    return [(step.claim_id, step.candidate, step.cost) for step in result.steps]


# --- the placer is schedule_eft --------------------------------------------------------


def test_the_placer_is_schedule_eft_on_every_fixture():
    count = 0
    for name, tname, _policy, module, h, durations in _placements():
        ref = schedule_eft(module, durations, h)
        placer = EftPlacer(module, durations, h)
        assert placer.makespan == ref.makespan, (name, tname)
        assert placer.schedule() == ref, (name, tname)
        count += 1
    assert count >= 300


def test_trials_and_adoptions_price_exactly_and_a_trial_adopts_nothing():
    rng = random.Random(2026)
    trials = adoptions = 0
    for name, tname, _policy, module, h, durations in _placements():
        ref = schedule_eft(module, durations, h)
        placer = EftPlacer(module, durations, h)
        ids = list(durations)
        for _ in range(8):
            if rng.random() < 0.4 and len(ids) > 1:  # the sweep's shape: two adjacent steps
                i = rng.randrange(len(ids) - 1)
                changes = {ids[i]: rng.randrange(0, 3000), ids[i + 1]: rng.randrange(0, 3000)}
            else:
                chosen = rng.sample(ids, min(rng.choice([1, 1, 2, 3]), len(ids)))
                changes = {
                    c: max(0, durations[c] + rng.choice([-durations[c], -1, 1, 5000]))
                    for c in chosen
                }
            want = schedule_eft(module, {**durations, **changes}, h).makespan
            assert placer.trial(changes) == want, (name, tname, changes)
            assert placer.makespan == ref.makespan and placer.schedule() == ref, (name, tname)
            trials += 1
            if rng.random() < 0.25:
                placer.adopt(changes)
                durations = {**durations, **changes}
                ref = schedule_eft(module, durations, h)
                assert placer.makespan == ref.makespan, (name, tname, "adopt")
                assert placer.schedule() == ref, (name, tname, "adopt")
                adoptions += 1
    assert trials >= 2000 and adoptions >= 300


def test_a_chain_replays_from_the_checkpoint_before_the_changed_claim():
    """On a serial hazard chain the pop order is the chain: a change deep in it resumes
    from the last checkpoint before that claim, shortened or lengthened."""
    module, h = sweep_fixture(64), TARGETS["x86_avx2"]
    durations = durations_from(optimize(module, h, Theta.cool(), PERF))
    placer = EftPlacer(module, durations, h)
    for cid, dur in ((60, 1), (60, 10**6), (33, 0), (2, 5)):
        placer.pops = 0
        want = schedule_eft(module, {**durations, cid: dur}, h).makespan
        assert placer.trial({cid: dur}) == want
        replayed = 64 - cid + 8  # the suffix from the checkpoint below (stride 8)
        assert 0 < placer.pops <= replayed, (cid, dur, placer.pops)
    placer.pops = 0
    assert placer.trial({1: 10**6}) == schedule_eft(module, {**durations, 1: 10**6}, h).makespan
    assert placer.pops == 64  # the root: nothing precedes it, the phase is re-run


def test_later_phases_are_skipped_unless_they_touch_a_moved_rid():
    h = TARGETS["x86_avx2"]
    module = general_fixture(4, 4)  # 4 phases x 8 claims on disjoint resources
    durations = durations_from(optimize(module, h, Theta.cool(), ENERGY))
    placer = EftPlacer(module, durations, h)
    for cid in (1, 2, 5):  # phase 0
        placer.pops = 0
        want = schedule_eft(module, {**durations, cid: 1}, h).makespan
        assert placer.trial({cid: 1}) == want
        assert placer.pops <= 8, placer.pops  # phase 0 alone; phases 1..3 touch nothing moved
    # a phase that reads phase 0's operand is re-dispatched when that operand moved
    touching = general_fixture(2, 4)
    touching.add_resource(Resource(rid=99, shape=(4096,)))
    touching.add_phase(
        Phase(
            phase_id=2,
            claims=[
                Claim(
                    id=100,
                    opcode=Opcode.ADD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=4096,
                    rd=(1, 99),
                    wr=(99,),
                    op="vector.add",
                )
            ],
        )
    )
    durations = durations_from(optimize(touching, h, Theta.cool(), ENERGY))
    placer = EftPlacer(touching, durations, h)
    for changes in ({1: 1}, {1: 10**6, 2: 3}, {3: 0}):
        placer.pops = 0
        want = schedule_eft(touching, {**durations, **changes}, h).makespan
        assert placer.trial(changes) == want, changes
        assert placer.pops <= 8 + 1, placer.pops  # phase 1 (disjoint) is never re-placed


def test_no_op_changes_are_free_and_unknown_claims_are_refused():
    module, h = general_fixture(1, 4), TARGETS["x86_avx2"]
    durations = durations_from(optimize(module, h, Theta.cool(), ENERGY))
    placer = EftPlacer(module, durations, h)
    placer.pops = 0
    assert placer.trial({1: durations[1], 2: durations[2]}) == placer.makespan
    assert placer.adopt({1: durations[1]}) == placer.makespan
    assert placer.pops == 0
    try:
        placer.trial({999: 1})
    except KeyError:
        pass
    else:
        raise AssertionError("a claim the placer does not place was priced")


# --- the sweep: identical assignment ---------------------------------------------------


def test_the_delta_sweep_returns_the_reference_assignment_claim_by_claim():
    thetas = (Theta.cool(), Theta(thermal=80, power=70))
    cases = 0
    for name, module in _fixtures():
        for tname in _TARGETS:
            h = TARGETS[tname]
            for theta in thetas:
                for policy in POLICIES.values():
                    r_ref, p_ref = optimize_scheduled(module, h, theta, policy, delta=False)
                    r, p = optimize_scheduled(module, h, theta, policy)
                    assert _assignment(r) == _assignment(r_ref), (name, tname, policy.name)
                    assert r.score == r_ref.score
                    assert (p.makespan, p.serial) == (p_ref.makespan, p_ref.serial), (name, tname)
                    assert p.schedule == p_ref.schedule == schedule_plan(module, r, h, "eft")
                    assert p == price_scheduled(module, r, h, theta, policy)
                    assert verify_plan(module, r) == [], (name, tname)
                    cases += 1
    assert cases >= 1300  # 56 fixtures x 3 targets x 2 thetas x 4 policies


def test_the_general_case_places_one_trial_per_pair_and_re_places_only_its_phase():
    module, h = general_fixture(8, 8), TARGETS["x86_avx2"]
    delta: dict = {}
    full: dict = {}
    r, p = optimize_scheduled(module, h, Theta.cool(), ENERGY, stats=delta)
    r_ref, p_ref = optimize_scheduled(module, h, Theta.cool(), ENERGY, delta=False, stats=full)
    assert _assignment(r) == _assignment(r_ref) and p.schedule == p_ref.schedule
    assert delta["claims"] == full["claims"] == 128
    assert delta["trials"] == full["trials"] == 64  # one step-shortening alternative per pair
    assert delta["adopted"] == full["adopted"] == 0  # the serial optimum stands
    assert full["pops"] == 64 * 128  # RED: every trial re-placed the whole module
    assert delta["pops"] == 64 * 16  # GREEN: the affected phase (16 claims) and nothing else
    serial = optimize(module, h, Theta.cool(), ENERGY)
    assert _assignment(r) == _assignment(serial)


def test_an_adopted_trial_is_the_same_adoption_in_both_searches():
    module = adoption_fixture()
    for domains in (2, 8):
        h = replace(TARGETS["x86_avx2"], affinity_domains=domains)
        base = optimize(module, h, Theta.cool(), ENERGY)
        assert [s.candidate.width for s in base.steps] == [8, 8, 8]
        stats: dict = {}
        r, p = optimize_scheduled(module, h, Theta.cool(), ENERGY, stats=stats)
        r_ref, p_ref = optimize_scheduled(module, h, Theta.cool(), ENERGY, delta=False)
        assert [s.candidate.width for s in r.steps] == [1, 8, 8]  # the scalar small claim
        assert _assignment(r) == _assignment(r_ref)
        assert p.makespan == p_ref.makespan < schedule_plan(module, base, h).makespan
        assert p.serial == p_ref.serial == r.score > base.score  # a longer chain, shorter
        assert p.schedule == p_ref.schedule == schedule_plan(module, r, h, "eft")
        assert p == price_scheduled(module, r, h, Theta.cool(), ENERGY)
        assert stats["trials"] == stats["adopted"] == 1
        assert verify_plan(module, r) == []


def test_the_sweep_builds_the_candidate_map_once():
    import bcir.kbcir.realize as realize

    calls = {"n": 0}
    original = realize.fused_candidates

    def counting(module, h):
        calls["n"] += 1
        return original(module, h)

    realize.fused_candidates = counting
    try:
        module, h = sweep_fixture(32), TARGETS["x86_avx512"]
        optimize_scheduled(module, h, Theta.cool(), PERF)
    finally:
        realize.fused_candidates = original
    assert calls["n"] == 1  # inside optimize(); the sweep reads result.cand_map
