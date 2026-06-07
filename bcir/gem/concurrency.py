"""CT2: mixed-stride concurrent graph execution + affinity.

Turns the phase DAG into a concurrent task graph. Within a phase, independent
claims (no read/write hazard between them) co-execute in the same *wave*;
conflicting claims serialize into successive waves. Sparse GGG/random claims are
decoupled into a tail stream so the irreducible random work does not stall the
sequential (U/UX/T) waves. Each wave's claims are pinned round-robin to the
target's affinity domains (thread->cache); a wave wider than the available
domains is oversubscribed, modeled as `contention` (cache thrash).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Claim, Lane, Module, StrideClass


@dataclass
class ConcurrentSchedule:
    waves: list[list[int]] = field(default_factory=list)   # claim ids per wave, in order
    ggg_tail: list[int] = field(default_factory=list)      # decoupled random/gather claims
    affinity: dict[int, int] = field(default_factory=dict)  # claim id -> affinity domain
    contention: int = 0                                     # cache-thrash oversubscription

    def max_parallelism(self) -> int:
        return max((len(w) for w in self.waves), default=0)


def _conflict(a: Claim, b: Claim) -> bool:
    """A read/write hazard between two claims (RAW / WAR / WAW)."""
    aw, ar = set(a.wr), set(a.rd)
    bw, br = set(b.wr), set(b.rd)
    return bool(aw & (br | bw)) or bool(bw & ar)


def _is_sparse(c: Claim) -> bool:
    return c.lane == Lane.GGG or c.stride_class == StrideClass.RANDOM


def _topo_phase_ids(module: Module) -> list[int]:
    pmap = module.phase_map()
    color: dict[int, int] = {}
    order: list[int] = []

    def visit(pid: int) -> None:
        color[pid] = 1
        for d in pmap[pid].deps:
            if d in pmap and color.get(d, 0) == 0:
                visit(d)
        color[pid] = 2
        order.append(pid)

    for p in module.phases:
        if color.get(p.phase_id, 0) == 0:
            visit(p.phase_id)
    return order


def schedule_concurrent(module: Module, target=None) -> ConcurrentSchedule:
    """Compute a concurrent wave schedule with a decoupled GGG tail + affinity."""
    domains = max(1, getattr(target, "affinity_domains", 1)) if target is not None else 1
    sched = ConcurrentSchedule()
    pmap = module.phase_map()

    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        main = [c for c in claims if not _is_sparse(c)]
        sched.ggg_tail.extend(c.id for c in claims if _is_sparse(c))

        # Greedy wave assignment within the phase (phase boundary = implicit barrier).
        wave_of: dict[int, int] = {}
        for i, c in enumerate(main):
            w = 0
            for prev in main[:i]:
                if _conflict(prev, c):
                    w = max(w, wave_of[prev.id] + 1)
            wave_of[c.id] = w

        nwaves = (max(wave_of.values()) + 1) if wave_of else 0
        for w in range(nwaves):
            members = [c.id for c in main if wave_of[c.id] == w]
            for slot, cid in enumerate(members):
                sched.affinity[cid] = slot % domains
            if len(members) > domains:
                sched.contention += len(members) - domains
            sched.waves.append(members)

    return sched
