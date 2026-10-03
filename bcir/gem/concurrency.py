"""CT2: mixed-stride concurrent graph execution + affinity.

Turns the phase DAG into a concurrent task graph. Within a phase, independent
claims (no read/write hazard between them) co-execute in the same *wave*;
conflicting claims serialize into successive waves. Sparse GGG/random claims are
decoupled into a tail stream so the irreducible random work does not stall the
sequential (U/UX/T) waves. Each wave's claims are pinned round-robin to the
target's affinity domains (thread->cache); a wave wider than the available
domains is oversubscribed, modeled as `contention` (cache thrash).

The ONE hazard predicate of the GEM rails lives here (G1 / S1-A): `hazard_conflict`
is a data hazard (RAW / WAR / WAW) OR an ordering fence -- a `barriered` or
`volatile` claim conflicts with every other claim -- and `hazard_predecessors`
builds the dependency DAG the duration-aware scheduler (`gem.schedule`), the token
plan (`gem.async_tokens`) and the bundle reorderer (`kbcir.bundle`) all honor. The
edges are built over EVERY claim of a phase, sparse tail included, BEFORE the
stream split, so a gather that reads what a wave claim writes waits for it.
`hazard_frontier` is the same DAG cut to each resource's frontier -- the same transitive
closure in O(resource touches) edges -- which the EFT dispatch reads (G2 residual).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import attrgetter

from ..model import Claim, Lane, Module, StrideClass, topological_phase_ids

_claim_id = attrgetter("id")  # the sort key of a phase's claims: ascending claim id


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


def is_fence(c: Claim) -> bool:
    """An ordering fence: a `barriered`-hazard or `volatile` claim (ASM3b / §5.14). It is
    never reordered, fused, bundled or overlapped with any other claim -- on either side."""
    return c.hazard == "barriered" or bool(c.volatile)


def hazard_conflict(a: Claim, b: Claim) -> bool:
    """The ONE scheduling-hazard predicate (G1): a data hazard or an ordering fence on
    either side. `kbcir.bundle._conflict` is this predicate; the schedulers build their
    dependency DAGs from it (`hazard_predecessors`)."""
    return is_fence(a) or is_fence(b) or _conflict(a, b)


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

    The histories hold positions, so a claim's predecessors sort as integers and are spelled
    as ids once; a claim whose conflicts all lie in one history (one resource read, say) takes
    that history as it is -- appended in position order, each position once -- and only a
    claim that merges histories pays the set.
    """
    ids = [claim.id for claim in claims]
    if len(set(ids)) != len(ids):
        raise ValueError("GEM dependency planning requires unique claim ids")
    readers: dict[int, list[int]] = {}  # per resource: the positions of its readers so far
    writers: dict[int, list[int]] = {}  # per resource: the positions of its writers so far
    out: dict[int, list[int]] = {}
    spell = ids.__getitem__
    for index, claim in enumerate(claims):
        rd, wr = _claim_accesses(claim)
        first: list[int] | None = None  # the one history the claim conflicts with, or
        merged: set[int] | None = None  # the positions of several, merged
        for rid in rd:
            history = writers.get(rid)
            if history:
                if first is None:
                    first = history
                elif merged is None:
                    merged = set(first)
                    merged.update(history)
                else:
                    merged.update(history)
        for rid in wr:
            history = writers.get(rid)
            if history:
                if first is None:
                    first = history
                elif merged is None:
                    merged = set(first)
                    merged.update(history)
                else:
                    merged.update(history)
            history = readers.get(rid)
            if history:
                if first is None:
                    first = history
                elif merged is None:
                    merged = set(first)
                    merged.update(history)
                else:
                    merged.update(history)
        if merged is not None:
            out[claim.id] = list(map(spell, sorted(merged)))
        elif first is not None:
            out[claim.id] = list(map(spell, first))
        else:
            out[claim.id] = []
        for rid in rd:
            history = readers.get(rid)
            if history is None:
                readers[rid] = [index]
            else:
                history.append(index)
        for rid in wr:
            history = writers.get(rid)
            if history is None:
                writers[rid] = [index]
            else:
                history.append(index)
    return out


def _frontier_predecessors(claims: list[Claim]) -> dict[int, list[int]]:
    """`_conflict_predecessors` reduced to each resource's frontier: the same transitive
    closure in edges proportional to resource touches.

    Of a claim's earlier RAW/WAR/WAW conflicts, the ones not already ordered before another
    of them are, per resource: the LAST writer (every earlier writer of the resource is its
    WAW predecessor) and, for a resource the claim writes, the readers SINCE that write
    (every earlier reader is the last writer's WAR predecessor). Every other conflict edge
    is implied by a path through those, so the closure -- which pairs are ordered -- is
    exactly the full DAG's. A dense single-resource chain of n claims keeps n - 1 edges
    instead of n(n - 1) / 2 (G2's residual: the dispatch read them all once per placement).
    The frontier holds positions, spelled as ids once they are sorted.
    """
    ids = [claim.id for claim in claims]
    if len(set(ids)) != len(ids):
        raise ValueError("GEM dependency planning requires unique claim ids")
    last_writer: dict[int, int] = {}  # per resource: the position of its last writer
    readers_since: dict[int, list[int]] = {}  # per resource: readers since its last write
    out: dict[int, list[int]] = {}
    spell = ids.__getitem__
    for index, claim in enumerate(claims):
        rd, wr = _claim_accesses(claim)
        only = -1  # the one position the claim waits for, or
        positions: set[int] | None = None  # the several, as a set
        for rid in rd:
            writer = last_writer.get(rid)
            if writer is not None:
                if positions is not None:
                    positions.add(writer)
                elif only < 0:
                    only = writer
                elif writer != only:
                    positions = {only, writer}
        for rid in wr:
            writer = last_writer.get(rid)
            if writer is not None:
                if positions is not None:
                    positions.add(writer)
                elif only < 0:
                    only = writer
                elif writer != only:
                    positions = {only, writer}
            since = readers_since.get(rid)
            if since:
                if positions is None:
                    positions = set(since)
                    if only >= 0:
                        positions.add(only)
                else:
                    positions.update(since)
        if positions is not None:
            out[claim.id] = list(map(spell, sorted(positions)))
        elif only >= 0:
            out[claim.id] = [ids[only]]
        else:
            out[claim.id] = []
        for rid in rd:
            if rid not in wr:  # a claim that also writes the resource is its new last writer
                since = readers_since.get(rid)
                if since is None:
                    readers_since[rid] = [index]
                else:
                    since.append(index)
        for rid in wr:
            last_writer[rid] = index
            readers_since[rid] = []
    return out


def hazard_predecessors(claims: list[Claim]) -> dict[int, list[int]]:
    """Every earlier claim a claim must wait for: its data hazards (`_conflict_predecessors`)
    plus the ordering fences (`is_fence`).

    A fence waits for every claim since the previous fence and for that fence; every
    later claim waits for the most recent fence. Transitively that is exactly "a fence
    conflicts with everything" (`hazard_conflict`) in O(claims) fence edges instead of
    O(claims^2), and a module with no fence gets byte-identical edges to the data-only
    DAG. Predecessors are listed in position order. Built over EVERY claim handed in --
    the caller passes a whole phase (or the whole module), sparse tail included, so the
    cross-stream RAW/WAR/WAW edges exist before any stream split (G1 / S1-A).
    """
    return _with_fences(claims, _conflict_predecessors(claims))


def hazard_frontier(claims: list[Claim]) -> dict[int, list[int]]:
    """`hazard_predecessors` with its data edges cut to each resource's frontier
    (`_frontier_predecessors`), the fences layered on the same way: the same transitive
    closure, so the same admissible orders, in O(resource touches) edges.

    A list scheduler that pushes a claim when its last predecessor is popped and releases
    it at the latest predecessor finish decides identically on either DAG: every dropped
    predecessor is an ancestor of a kept one, so it is popped earlier and (durations being
    non-negative) finishes no later. The EFT dispatch reads this (`schedule.phase_frontiers`);
    `hazard_predecessors` stays the published DAG -- the token plan's awaits, and the exact
    solver, whose symmetry classes compare predecessor sets literally.
    """
    return _with_fences(claims, _frontier_predecessors(claims))


def _with_fences(claims: list[Claim], out: dict[int, list[int]]) -> dict[int, list[int]]:
    """Layer the ordering fences (`hazard_predecessors`) onto a data DAG, in place. A phase
    without a fence is left as it is: the pass below would add no edge to it."""
    for claim in claims:
        if is_fence(claim):
            break
    else:
        return out
    position = {claim.id: index for index, claim in enumerate(claims)}
    last_fence: int | None = None
    segment: list[int] = []  # claims since the last fence (non-fences)
    for claim in claims:
        predecessors = out[claim.id]
        if is_fence(claim):
            extra = segment if last_fence is None else [last_fence, *segment]
            if extra:
                out[claim.id] = sorted(set(predecessors) | set(extra), key=position.__getitem__)
            last_fence = claim.id
            segment = []
        else:
            if last_fence is not None and last_fence not in predecessors:
                out[claim.id] = sorted(set(predecessors) | {last_fence}, key=position.__getitem__)
            segment.append(claim.id)
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
            prior = writer_wave.get(rid)
            if prior is not None and prior >= wave:
                wave = prior + 1
        for rid in wr:
            prior = writer_wave.get(rid)
            if prior is not None and prior >= wave:
                wave = prior + 1
            prior = reader_wave.get(rid)
            if prior is not None and prior >= wave:
                wave = prior + 1
        wave_of[claim.id] = wave
        row = members.get(wave)
        if row is None:
            members[wave] = [claim]
        else:
            row.append(claim)
        for rid in rd:
            if reader_wave.get(rid, -1) < wave:
                reader_wave[rid] = wave
        for rid in wr:
            if writer_wave.get(rid, -1) < wave:
                writer_wave[rid] = wave
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

    waves, tail, affinity = sched.waves, sched.ggg_tail, sched.affinity

    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=_claim_id)
        claim_ids = [claim.id for claim in claims]
        fresh = set(claim_ids)
        if len(fresh) != len(claim_ids) or not seen_claim_ids.isdisjoint(fresh):
            raise ValueError("GEM scheduling requires module-wide unique claim ids")
        seen_claim_ids |= fresh
        main: list[Claim] = []
        for claim in claims:  # sparsity read once per claim: the tail, in claim-id order
            if _is_sparse(claim):
                tail.append(claim.id)
            else:
                main.append(claim)

        # Greedy wave assignment within the phase (phase boundary = implicit barrier).
        wave_of, wave_members = _wave_indices(main)

        nwaves = (max(wave_of.values()) + 1) if wave_of else 0
        for w in range(nwaves):
            members = [c.id for c in wave_members.get(w, ())]
            for slot, cid in enumerate(members):
                affinity[cid] = slot % domains
            if len(members) > domains:
                sched.contention += len(members) - domains
            waves.append(members)

    return sched
