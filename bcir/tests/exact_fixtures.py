"""Fixtures of the exact rail (G4 / S2-B), shared by the tests and the harness.

`six_job_module` / `six_job_corpus` is the 2026-08-12 report's section 6.1 corpus, pinned:
six independent unit-stride claims on disjoint resources, every nondecreasing six-job
duration multiset with values 1..8 (1,716 instances), placed on 2 and 3 affinity domains.
`schedule_eft` reproduces the report on it exactly -- 190 suboptimal (11.07%), mean 1.0078,
worst 17/15 on two domains; 18 (1.05%), 1.0013, 7/6 on three -- and `partition_optimum` is
the INDEPENDENT oracle the exact scheduler is held to there (independent jobs on identical
machines: the optimum is a partition, found by a symmetry-broken search over loads that
shares no code with `gem.exact`).

`active_schedule_optimum` enumerates EVERY active schedule of a tiny module (no bound, no
symmetry breaking) -- the oracle for the exact scheduler on hazard-bearing modules.

`quality_corpus` is the section 6.3 shape -- four claims, up to three candidates each --
over which `exact_selection` holds the one-sweep re-selection to the exhaustive enumeration:
the shaped modules (independent, a chain, a shared operand, pairs) and seeded random ones.
The report's own 4-claim fixture (one-sweep 55,552 against exact 55,168) was measured under
the retired wave pricer and is not reconstructible under the canonical artifact; the corpus
here is measured, not quoted.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import replace

from bcir.gem.concurrency import _topo_phase_ids
from bcir.gem.schedule import _PhaseDispatch, _streams, phase_hazards
from bcir.kbcir import TARGETS
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass

DURATION_VALUES = range(1, 9)
JOBS = 6


def six_job_module() -> Module:
    module = Module(name="sixjob")
    for rid in range(1, 2 * JOBS + 1):
        module.add_resource(Resource(rid=rid, shape=(64,)))
    module.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(
                    id=index + 1,
                    opcode=Opcode.MUL,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=64,
                    rd=(2 * index + 1,),
                    wr=(2 * index + 2,),
                    op="vector.mul",
                )
                for index in range(JOBS)
            ],
        )
    )
    return module


def six_job_corpus() -> list[tuple[int, ...]]:
    """Every nondecreasing six-job duration multiset with values 1..8: 1,716 instances."""
    return list(itertools.combinations_with_replacement(DURATION_VALUES, JOBS))


def six_job_target(domains: int):
    return replace(TARGETS["x86_avx2"], affinity_domains=domains)


def partition_optimum(durations, machines: int) -> int:
    """The optimal makespan of independent jobs on identical machines: a symmetry-broken
    search over the machines' loads (equal loads are interchangeable)."""
    jobs = sorted(durations, reverse=True)
    best = sum(jobs)
    loads = [0] * machines

    def place(index: int) -> None:
        nonlocal best
        if max(loads) >= best:
            return
        if index == len(jobs):
            best = max(loads)
            return
        seen: set[int] = set()
        for machine in range(machines):
            if loads[machine] in seen:
                continue
            seen.add(loads[machine])
            loads[machine] += jobs[index]
            place(index + 1)
            loads[machine] -= jobs[index]

    place(0)
    return best


def _phase_active_optimum(dispatch: _PhaseDispatch, durations: dict[int, int]) -> int:
    """The minimum makespan over every active schedule of one phase (no bound, no symmetry
    breaking): a plain recursion over (ready claim, eligible stream) choices."""
    ids = [claim.id for claim in dispatch.claims]
    dur = {cid: max(0, durations.get(cid, 1)) for cid in ids}
    preds = {cid: [p for p in dispatch.preds_of[cid] if p in dispatch.claim_by_id] for cid in ids}
    best: int | None = None

    def walk(free: list[int], finish: dict[int, int]) -> None:
        nonlocal best
        if len(finish) == len(ids):
            makespan = max(free)
            if best is None or makespan < best:
                best = makespan
            return
        for cid in ids:
            if cid in finish or any(p not in finish for p in preds[cid]):
                continue
            release = max((finish[p] for p in preds[cid]), default=0)
            for stream in dispatch.eligible[cid]:
                start = max(free[stream], release)
                nxt = list(free)
                nxt[stream] = start + dur[cid]
                walk(nxt, {**finish, cid: start + dur[cid]})

    walk([0] * (dispatch.domains + 1), {})
    return best or 0


def active_schedule_optimum(module: Module, durations: dict[int, int], target) -> int:
    """The minimum makespan over EVERY active schedule of `module` under the artifact's own
    eligibility rules -- phase by phase, no bound, no symmetry breaking. Tiny modules only."""
    domains, knee = _streams(target)
    hazards = phase_hazards(module)
    pmap = module.phase_map()
    total = 0
    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        total += _phase_active_optimum(
            _PhaseDispatch(claims, hazards[pid], domains, knee, True), durations
        )
    return total


# --- the section 6.3 corpus -----------------------------------------------------------------


def _quality_module(name, counts, reads, writes, rids) -> Module:
    module = Module(name=name)
    for rid in rids:
        module.add_resource(Resource(rid=rid, shape=(max(counts),)))
    module.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(
                    id=index + 1,
                    opcode=Opcode.ADD,
                    lane=Lane.U,
                    stride_class=StrideClass.UNIT,
                    count=counts[index],
                    rd=reads[index],
                    wr=writes[index],
                    op="vector.add",
                )
                for index in range(4)
            ],
        )
    )
    return module


def quality_corpus(seeds: int = 40) -> list[tuple[str, Module]]:
    """Four-claim modules with up to three candidates each: the shaped ones at four sizes
    and `seeds` random ones (random operands, hazards and strides, one or two phases)."""
    out: list[tuple[str, Module]] = []
    for n in (1024, 2048, 4096, 8192):
        counts = (n, n, n, n)
        out.append(
            (
                f"indep{n}",
                _quality_module(
                    "indep", counts, [(1,), (2,), (3,), (4,)], [(5,), (6,), (7,), (8,)], range(1, 9)
                ),
            )
        )
        out.append(
            (
                f"chain{n}",
                _quality_module(
                    "chain", counts, [(1,), (2,), (3,), (4,)], [(2,), (3,), (4,), (5,)], range(1, 6)
                ),
            )
        )
        out.append(
            (
                f"shared{n}",
                _quality_module(
                    "shared",
                    counts,
                    [(1, 2), (1, 3), (1, 4), (1, 5)],
                    [(6,), (7,), (8,), (9,)],
                    range(1, 10),
                ),
            )
        )
        out.append(
            (
                f"pairs{n}",
                _quality_module(
                    "pairs", counts, [(1,), (1,), (2,), (2,)], [(3,), (4,), (5,), (6,)], range(1, 7)
                ),
            )
        )
    for seed in range(seeds):
        rng = random.Random(seed)
        rids = list(range(1, 9))
        module = Module(name=f"quality{seed}")
        for rid in rids:
            module.add_resource(Resource(rid=rid, shape=(8192,)))
        phases = rng.choice([1, 1, 2])
        per_phase = [4] if phases == 1 else [2, 2]
        cid = 1
        for phase_id in range(phases):
            claims = []
            for _ in range(per_phase[phase_id]):
                stride = rng.choice(
                    [StrideClass.UNIT] * 4 + [StrideClass.STRIDED, StrideClass.CACHELINE]
                )
                claims.append(
                    Claim(
                        id=cid,
                        opcode=rng.choice([Opcode.ADD, Opcode.MUL]),
                        lane=Lane.U,
                        stride_class=stride,
                        count=rng.choice([64, 256, 1024, 4096]),
                        rd=tuple(rng.sample(rids, rng.randint(1, 2))),
                        wr=(rng.choice(rids),),
                        op="vector.add",
                    )
                )
                cid += 1
            module.add_phase(Phase(phase_id=phase_id, claims=claims))
        out.append((f"quality{seed}", module))
    return out


__all__ = [
    "DURATION_VALUES",
    "JOBS",
    "active_schedule_optimum",
    "partition_optimum",
    "quality_corpus",
    "six_job_corpus",
    "six_job_module",
    "six_job_target",
]
