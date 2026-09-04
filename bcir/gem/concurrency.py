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

from ..model import Claim, Lane, Module, StrideClass, topological_phase_ids


@dataclass
class ConcurrentSchedule:
    waves: list[list[int]] = field(default_factory=list)  # claim ids per wave, in order
    ggg_tail: list[int] = field(default_factory=list)  # decoupled random/gather claims
    affinity: dict[int, int] = field(default_factory=dict)  # claim id -> affinity domain
    contention: int = 0  # cache-thrash oversubscription

    def max_parallelism(self) -> int:
        return max((len(w) for w in self.waves), default=0)


def _conflict(a: Claim, b: Claim) -> bool:
    """A read/write hazard between two claims (RAW / WAR / WAW)."""
    aw, ar = set(a.wr), set(a.rd)
    bw, br = set(b.wr), set(b.rd)
    return bool(aw & (br | bw)) or bool(bw & ar)


def _is_sparse(c: Claim) -> bool:
    return c.lane == Lane.GGG or c.stride_class == StrideClass.RANDOM


def _claim_accesses(claim: Claim) -> tuple[set[int], set[int]]:
    """Deduplicated read/write resources for one claim."""
    return set(claim.rd), set(claim.wr)


def _conflict_predecessors(claims: list[Claim]) -> dict[int, list[int]]:
    """Build every earlier RAW/WAR/WAW predecessor from per-resource histories.

    The historical implementation compared every pair of claims, allocating four
    temporary sets per comparison.  This produces the same predecessor order while
    doing work proportional to resource touches plus the dependency edges that must
    actually be represented.  A dense single-resource chain still has O(n^2) output;
    an independent workload is linear instead of quadratic.
    """
    if len({claim.id for claim in claims}) != len(claims):
        raise ValueError("GEM dependency planning requires unique claim ids")
    readers: dict[int, list[int]] = {}
    writers: dict[int, list[int]] = {}
    position: dict[int, int] = {}
    out: dict[int, list[int]] = {}
    for index, claim in enumerate(claims):
        rd, wr = _claim_accesses(claim)
        predecessors: set[int] = set()
        for rid in rd:
            predecessors.update(writers.get(rid, ()))
        for rid in wr:
            predecessors.update(writers.get(rid, ()))
            predecessors.update(readers.get(rid, ()))
        out[claim.id] = sorted(predecessors, key=position.__getitem__)
        position[claim.id] = index
        for rid in rd:
            readers.setdefault(rid, []).append(claim.id)
        for rid in wr:
            writers.setdefault(rid, []).append(claim.id)
    return out


def _wave_indices(claims: list[Claim]) -> tuple[dict[int, int], dict[int, list[Claim]]]:
    """Return the exact greedy conflict wave for each claim in linear touch work.

    A claim's wave is one greater than the largest wave of a conflicting prior
    access.  Per-resource maxima are therefore a sufficient summary; enumerating all
    prior claims is unnecessary.
    """
    if len({claim.id for claim in claims}) != len(claims):
        raise ValueError("GEM wave planning requires unique claim ids")
    reader_wave: dict[int, int] = {}
    writer_wave: dict[int, int] = {}
    wave_of: dict[int, int] = {}
    members: dict[int, list[Claim]] = {}
    for claim in claims:
        rd, wr = _claim_accesses(claim)
        wave = 0
        for rid in rd:
            if rid in writer_wave:
                wave = max(wave, writer_wave[rid] + 1)
        for rid in wr:
            if rid in writer_wave:
                wave = max(wave, writer_wave[rid] + 1)
            if rid in reader_wave:
                wave = max(wave, reader_wave[rid] + 1)
        wave_of[claim.id] = wave
        members.setdefault(wave, []).append(claim)
        for rid in rd:
            reader_wave[rid] = max(reader_wave.get(rid, -1), wave)
        for rid in wr:
            writer_wave[rid] = max(writer_wave.get(rid, -1), wave)
    return wave_of, members


def _topo_phase_ids(module: Module) -> list[int]:
    """Compatibility alias for the shared, iterative phase traversal."""
    return topological_phase_ids(module)


def schedule_concurrent(module: Module, target=None) -> ConcurrentSchedule:
    """Compute a concurrent wave schedule with a decoupled GGG tail + affinity."""
    domains = max(1, getattr(target, "affinity_domains", 1)) if target is not None else 1
    sched = ConcurrentSchedule()
    pmap = module.phase_map()
    seen_claim_ids: set[int] = set()

    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        claim_ids = [claim.id for claim in claims]
        if len(set(claim_ids)) != len(claim_ids) or seen_claim_ids & set(claim_ids):
            raise ValueError("GEM scheduling requires module-wide unique claim ids")
        seen_claim_ids.update(claim_ids)
        main = [c for c in claims if not _is_sparse(c)]
        sched.ggg_tail.extend(c.id for c in claims if _is_sparse(c))

        # Greedy wave assignment within the phase (phase boundary = implicit barrier).
        wave_of, wave_members = _wave_indices(main)

        nwaves = (max(wave_of.values()) + 1) if wave_of else 0
        for w in range(nwaves):
            members = [c.id for c in wave_members.get(w, ())]
            for slot, cid in enumerate(members):
                sched.affinity[cid] = slot % domains
            if len(members) > domains:
                sched.contention += len(members) - domains
            sched.waves.append(members)

    return sched
