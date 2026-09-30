"""G2's residual: the EFT dispatch reads each resource's hazard frontier, not every conflict.

`concurrency.hazard_predecessors` lists every earlier RAW/WAR/WAW conflict, so the §6.3 sweep's
serial chain (512 claims reading and writing one resource) handed the dispatch n(n - 1) / 2
edges -- 130,816 -- to build successors from and to walk again on every trial placement: the
residual the roadmap named after S2-A. `hazard_frontier` keeps, per resource, the last writer
and the readers since it: the same transitive closure in O(resource touches) edges.

The law is the G2 law ("a faster sweep that picks a different plan has not been made faster;
it has been changed"): every placement on the frontier is the full DAG's slot for slot, every
trial price is the same, and the sweep adopts the same plan. The full DAG stays the published
one -- the token plan's awaits and the exact solver's symmetry classes read it literally.
"""

from __future__ import annotations

import random

from bcir.examples import PROGRAMS
from bcir.gem.concurrency import hazard_frontier, hazard_predecessors
from bcir.gem.schedule import (
    EftPlacer,
    durations_from,
    phase_frontiers,
    phase_hazards,
    schedule_eft,
)
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.kbcir.weights import ENERGY, PERF
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.tests.sweep_fixtures import (
    adoption_fixture,
    general_fixture,
    random_module,
    sweep_fixture,
)


def _c(cid: int, rd=(), wr=(), *, hazard: str = "unique", volatile: bool = False) -> Claim:
    return Claim(
        cid,
        Opcode.ADD,
        Lane.U,
        StrideClass.UNIT,
        count=8,
        rd=tuple(rd),
        wr=tuple(wr),
        op="vector.add",
        hazard=hazard,
        volatile=volatile,
    )


def _closure(claims: list[Claim], preds: dict[int, list[int]]) -> dict[int, int]:
    """Every claim's ancestor set as a bitset over positions (claims are in position order)."""
    position = {claim.id: index for index, claim in enumerate(claims)}
    ancestors: dict[int, int] = {}
    for claim in claims:
        bits = 0
        for p in preds[claim.id]:
            bits |= ancestors[p] | (1 << position[p])
        ancestors[claim.id] = bits
    return ancestors


def _random_claims(seed: int) -> list[Claim]:
    """A phase's worth of claims over a few resources: reads, writes, read-modify-writes and
    the odd fence, so every per-resource history shape and the fence layering are exercised."""
    rng = random.Random(seed)
    rids = list(range(1, rng.randint(2, 7)))
    claims = []
    for cid in range(1, rng.randint(2, 40)):
        rd = tuple(rng.sample(rids, rng.randint(0, min(3, len(rids)))))
        wr = tuple(rng.sample(rids, rng.randint(0, min(2, len(rids)))))
        fence = rng.random() < 0.06
        claims.append(
            _c(cid, rd, wr, hazard="barriered" if fence else "unique", volatile=rng.random() < 0.02)
        )
    return claims


def test_the_frontier_is_a_subset_of_the_hazard_dag_with_the_same_closure():
    """Dropping an edge is only sound when its endpoints stay ordered: on every random phase
    the frontier's edges are hazard edges, and the two DAGs order exactly the same pairs."""
    cases = [_random_claims(seed) for seed in range(400)]
    for seed in range(60):  # the sweep fixtures' random modules, phase by phase
        for phase in random_module(seed).phases:
            cases.append(sorted(phase.claims, key=lambda c: c.id))
    dropped = 0
    for claims in cases:
        full, frontier = hazard_predecessors(claims), hazard_frontier(claims)
        for claim in claims:
            assert set(frontier[claim.id]) <= set(full[claim.id]), claim.id
            dropped += len(full[claim.id]) - len(frontier[claim.id])
        assert _closure(claims, frontier) == _closure(claims, full)
    assert dropped > 0  # the corpus reaches the reduction (L2): some edge was redundant


def test_a_resources_frontier_is_its_last_writer_and_the_readers_since():
    """The pinned shape: a reader waits for the last writer only; a writer waits for the last
    writer and the readers since it, not for the readers before that write."""
    w1, r1, r2, w2, r3, w3 = (
        _c(1, wr=(10,)),
        _c(2, rd=(10,)),
        _c(3, rd=(10,)),
        _c(4, wr=(10,)),
        _c(5, rd=(10,)),
        _c(6, rd=(10,), wr=(10,)),
    )
    claims = [w1, r1, r2, w2, r3, w3]
    assert hazard_frontier(claims) == {1: [], 2: [1], 3: [1], 4: [1, 2, 3], 5: [4], 6: [4, 5]}
    assert hazard_predecessors(claims) == {
        1: [],
        2: [1],
        3: [1],
        4: [1, 2, 3],
        5: [1, 4],
        6: [1, 2, 3, 4, 5],
    }


def test_a_serial_chain_keeps_n_minus_one_edges():
    """The §6.3 fixture's shape: n claims that each read and write one resource. The full DAG
    has every pair; the frontier has each claim's predecessor -- the Hasse diagram of a total
    order, the fewest edges any DAG with that closure can have."""
    n = 512
    chain = [_c(cid, (1,), (1,)) for cid in range(1, n + 1)]
    full, frontier = hazard_predecessors(chain), hazard_frontier(chain)
    assert sum(len(v) for v in full.values()) == n * (n - 1) // 2
    assert frontier == {cid: [cid - 1] if cid > 1 else [] for cid in range(1, n + 1)}


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


def test_every_placement_on_the_frontier_is_the_full_dags_slot_for_slot():
    count = 0
    for name, module in _fixtures():
        full, frontier = phase_hazards(module), phase_frontiers(module)
        for tname in ("x86_avx2", "x86_avx512", "nvidia_ptx"):
            h = TARGETS[tname]
            for policy in (PERF, ENERGY):
                durations = durations_from(optimize(module, h, Theta.cool(), policy))
                if not durations:
                    continue
                ref = schedule_eft(module, durations, h, hazards=full)
                assert schedule_eft(module, durations, h, hazards=frontier) == ref, (name, tname)
                assert schedule_eft(module, durations, h) == ref, (name, tname)  # the default
                count += 1
    assert count >= 300


def test_every_trial_prices_the_same_on_the_frontier():
    """The sweep's placer, trial by trial and adoption by adoption, on either DAG."""
    rng = random.Random(2026)
    trials = 0
    for name, module in _fixtures():
        h = TARGETS["x86_avx2"]
        durations = durations_from(optimize(module, h, Theta.cool(), PERF))
        if not durations:
            continue
        on_full = EftPlacer(module, durations, h, hazards=phase_hazards(module))
        on_frontier = EftPlacer(module, durations, h, hazards=phase_frontiers(module))
        assert on_frontier.makespan == on_full.makespan, name
        ids = sorted(durations)
        for _ in range(6):
            changes = {cid: rng.randint(0, 40) for cid in rng.sample(ids, min(2, len(ids)))}
            assert on_frontier.trial(changes) == on_full.trial(changes), name
            trials += 1
            if rng.random() < 0.3:
                on_full.adopt(changes)
                on_frontier.adopt(changes)
        assert on_frontier.schedule() == on_full.schedule(), name
    assert trials >= 200


def _chain(count: int) -> Module:
    module = Module(name=f"sweep{count}")
    module.add_resource(Resource(rid=1, shape=(64,)))
    module.add_phase(Phase(phase_id=0, claims=[_c(i + 1, (1,), (1,)) for i in range(count)]))
    return module


def test_the_sweep_adopts_the_same_plan_and_reads_the_frontier():
    """`optimize_scheduled` on the frontier against the same sweep on the full DAG (the parent's
    reading): the same assignment claim by claim and the same price, and on the §6.3 chain it
    hands the dispatch n - 1 edges where the parent handed it n(n - 1) / 2."""
    import bcir.gem.overlap as overlap

    def run(module, h, policy, dag):
        saved = overlap.phase_frontiers
        overlap.phase_frontiers = dag
        try:
            stats: dict = {}
            result, price = overlap.optimize_scheduled(module, h, Theta.cool(), policy, stats=stats)
        finally:
            overlap.phase_frontiers = saved
        steps = [(step.claim_id, step.candidate, step.cost) for step in result.steps]
        return steps, price, stats

    cases = [(name, module) for name, module in _fixtures()[:12]] + [("chain128", _chain(128))]
    for name, module in cases:
        for tname in ("x86_avx2", "nvidia_ptx"):
            h = TARGETS[tname]
            for policy in (PERF, ENERGY):
                new = run(module, h, policy, phase_frontiers)
                old = run(module, h, policy, phase_hazards)
                assert new[:2] == old[:2], (name, tname)
    n = 512
    _, _, stats = run(_chain(n), TARGETS[sorted(TARGETS)[0]], PERF, phase_frontiers)
    assert stats["edges"] == n - 1
    _, _, parent = run(_chain(n), TARGETS[sorted(TARGETS)[0]], PERF, phase_hazards)
    assert parent["edges"] == n * (n - 1) // 2
