"""Duration-aware GEM scheduling: the ONE canonical schedule artifact (G1 / S1-A).

`schedule_plan` places a selected K_BCIR plan once, deterministically and
integer-only, and everything that needs a schedule reads that placement: the
scheduled price M(pi, Theta) (`gem.overlap.price_scheduled`), the phase-barriered
executor (`schedule_eft`), the token-DAG executor (`execute_tokens`) and the power
rail. Before this slice the objective priced fixed greedy waves with round-robin
bins while the executor ran LPT/EFT placement -- two prices for one plan (the
2026-08-12 report's P0.1, 51,200 against 25,700 on four independent claims).

Four properties over the unit-time wave scheduler (`concurrency.schedule_concurrent`):

  1. **Duration-aware waves (HEFT-lite).** Claims carry durations (the plan's
     step costs, `durations_from`); an event-driven list scheduler dispatches the
     ready claim with the longest duration first (LPT priority -- the degenerate
     upward rank of HEFT inside one phase) onto the stream with the earliest
     finish time.
  2. **One hazard DAG, built before the stream split.** The dependency edges are
     `concurrency.hazard_predecessors` over EVERY claim of the phase (or, under
     tokens, of the module): RAW/WAR/WAW data hazards plus the ordering fences
     (`barriered` / `volatile` claims). The sparse GGG/random tail is dispatched
     on its own stream (`TAIL_STREAM`) inside the same event loop, so a gather
     that reads what a wave claim writes waits for it, and a fence is overlapped
     by nothing. The old split -- main claims first, the tail as a serial chain
     from the phase start ignoring hazards -- let a tail claim start before its
     producer finished.
  3. **Token-DAG execution.** `execute_tokens` consumes the `!bcir.token`
     fork/await plan instead of phase barriers: a claim starts when the claims it
     awaits finish, so independent claims of a *later phase* overlap an earlier
     phase -- software pipelining falls out of the dependency structure. The
     phase-barriered schedule is its degenerate case; both modes are the same
     placement over different edge sets.
  4. **Locality + the bandwidth knee.** Among earliest-finish ties a claim
     prefers the domain already holding the most of its operand RIDs; bandwidth-
     class claims contend for at most `bandwidth_knee(H)` concurrent domains
     (the roofline knee from the target's `mem_channels`), compute-class claims
     for the full domain set.

Durations are exactly the plan's step costs (a zero-cost step is a point on the
timeline), so the sum of the durations is the serial bound and the placement's
makespan never exceeds it: the R9 invariant `makespan + overlap_gain == serial`
holds by construction of the artifact, not by a separate pricer's discipline.

The unit-time wave scheduler remains the law for CT2 wave *formation*; this
module is the duration-aware placement that prices and places real work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Claim, Module
from .async_tokens import async_plan
from .concurrency import _is_sparse, _topo_phase_ids, hazard_predecessors

TAIL_STREAM = -1  # the decoupled GGG/random tail executes on its own stream


@dataclass(frozen=True)
class Slot:
    claim_id: int
    domain: int  # affinity domain, or TAIL_STREAM for the decoupled tail
    start: int
    finish: int


@dataclass
class GemSchedule:
    """A placed, timed schedule (mode: "eft" phase-barriered, "tokens" pipelined)."""

    mode: str
    slots: list[Slot] = field(default_factory=list)
    makespan: int = 0
    knee: int = 1
    affinity: dict[int, int] = field(default_factory=dict)

    def slot_of(self, claim_id: int) -> Slot:
        for s in self.slots:
            if s.claim_id == claim_id:
                return s
        raise KeyError(claim_id)


def bandwidth_knee(h) -> int:
    """The roofline knee: concurrent bandwidth-bound streams the target sustains."""
    domains = max(1, getattr(h, "affinity_domains", 1))
    channels = max(1, getattr(h, "mem_channels", 4))
    return max(1, min(domains, channels))


def durations_from(result) -> dict[int, int]:
    """Claim durations from a K_BCIR realization: exactly the scalarized step costs.

    Their sum is the plan score, which is the serial bound the scheduled price is
    measured against (R9: makespan + overlap_gain == serial). A zero-cost step (a
    weight-free fence under a policy that prices only compute and memory) is a
    zero-length slot, not a unit one -- a floor would let a serialized chain of
    such steps price above its own serial bound."""
    return {s.claim_id: max(0, s.cost) for s in result.steps}


def _rids(c: Claim) -> set[int]:
    return set(c.rd) | set(c.wr)


def _pick_domain(
    rids: set[int],
    ready_t: int,
    dur: int,
    domain_free: list[int],
    resident: list[set[int]],
    eligible: range,
    locality: bool,
) -> tuple[int, int]:
    """Earliest finish first; ties prefer the stream holding the claim's operands, then the
    lowest index -- the key (finish, -score, index), with the locality score computed only
    for the streams that tie on finish."""
    best_d = eligible[0]
    best_start = max(domain_free[best_d], ready_t)
    for d in eligible:
        start = max(domain_free[d], ready_t)
        if start < best_start:
            best_d, best_start = d, start
    if locality:
        best_score = len(rids & resident[best_d])
        for d in eligible:
            if d != best_d and max(domain_free[d], ready_t) == best_start:
                score = len(rids & resident[d])
                if score > best_score or (score == best_score and d < best_d):
                    best_d, best_score = d, score
    return best_d, best_start


class _PhaseDispatch:
    """The static structure of one dispatch -- everything the event-driven LPT list
    scheduling derives from the claims and their predecessor edges before it looks at a
    duration -- built once and run many times (`run`). `_dispatch` is its one-shot form;
    `EftPlacer` (G2 / S2-A) keeps one per phase and replays it from a checkpoint.

    `run` is exactly the historical dispatch: a ready heap keyed by the ``(-duration,
    claim_id)`` LPT priority (the ordinal breaks nothing -- ids are unique -- and is kept
    for the historical key), the earliest-finish domain among the claim's eligible streams
    with the locality tie-break, the sparse GGG/random tail on its own stream inside the
    same loop so its hazard edges hold across the streams. The pop order is a function of
    the edges and the durations alone (a successor is pushed when its last predecessor is
    POPPED, not when it finishes), which is what makes a checkpointed replay exact: until
    a changed claim enters the heap, nothing differs."""

    __slots__ = (
        "claims",
        "claim_by_id",
        "ordinal",
        "preds_of",
        "successors",
        "indegree0",
        "roots",
        "eligible",
        "rids",
        "domains",
        "locality",
    )

    def __init__(
        self,
        claims: list[Claim],
        preds: dict[int, list[int]],
        domains: int,
        knee: int,
        locality: bool,
    ) -> None:
        claim_by_id = {claim.id: claim for claim in claims}
        if len(claim_by_id) != len(claims):
            raise ValueError("GEM dispatch requires unique claim ids")
        self.claims = claims
        self.claim_by_id = claim_by_id
        self.ordinal = {claim.id: index for index, claim in enumerate(claims)}
        self.preds_of = {claim.id: preds.get(claim.id, ()) for claim in claims}  # no copy
        self.indegree0 = {claim.id: 0 for claim in claims}
        self.successors: dict[int, list[int]] = {claim.id: [] for claim in claims}
        for claim in claims:
            for predecessor in self.preds_of[claim.id]:
                if predecessor in claim_by_id:
                    self.indegree0[claim.id] += 1
                    self.successors[predecessor].append(claim.id)
        self.roots = [claim.id for claim in claims if self.indegree0[claim.id] == 0]
        tail = range(domains, domains + 1)  # the decoupled tail: its own stream, the same edges
        wave = {True: range(knee), False: range(domains)}  # bandwidth: the knee; compute: all
        self.eligible = {
            claim.id: tail if _is_sparse(claim) else wave[claim.cost_class == "bandwidth"]
            for claim in claims
        }
        self.rids = {claim.id: _rids(claim) for claim in claims}
        self.domains = domains
        self.locality = locality

    def run(
        self,
        durations: dict[int, int],
        t0: int,
        domain_free: list[int],
        resident: list[set[int]],
        finish_of: dict[int, int],
        sched: GemSchedule | None,
        resume: "_Checkpoint | None" = None,
        record: "_Record | None" = None,
    ) -> int:
        """Dispatch every claim (or, from `resume`, every claim it had not yet popped) and
        return the pop count. `sched` collects the slots (None: the makespan reader needs
        only `finish_of` and the residency); `record` takes the push indices and, when it
        asks for them, checkpoints every `record.stride` pops."""
        import heapq

        if len(domain_free) != self.domains + 1 or len(resident) != self.domains + 1:
            raise ValueError("GEM dispatch needs one stream per affinity domain plus the tail")
        for claim in self.claims:
            for predecessor in self.preds_of[claim.id]:
                if predecessor not in self.claim_by_id and predecessor not in finish_of:
                    raise ValueError(
                        f"GEM dispatch predecessor {predecessor} for claim {claim.id} "
                        "is unavailable"
                    )

        def duration_of(claim_id: int) -> int:
            return max(0, durations.get(claim_id, 1))  # an unpriced claim takes a unit

        ordinal, successors, preds_of = self.ordinal, self.successors, self.preds_of
        eligible, rids_of, domains, locality = self.eligible, self.rids, self.domains, self.locality
        if resume is None:
            indegree = dict(self.indegree0)
            ready = [(-duration_of(cid), cid, ordinal[cid]) for cid in self.roots]
            heapq.heapify(ready)  # the key is total (ids are unique): the pop order is the same
            pops = 0
        else:
            indegree = dict(resume.indegree)
            ready = list(resume.ready)
            pops = resume.pops
        stride = record.stride if record is not None and record.checkpoints is not None else 0
        while ready:
            if stride and pops and pops % stride == 0:
                record.checkpoints.append(
                    _Checkpoint(
                        pops,
                        list(ready),
                        dict(indegree),
                        list(domain_free),
                        [set(s) for s in resident],
                        dict(finish_of),
                    )
                )
            key = heapq.heappop(ready)
            claim_id = key[1]
            if record is not None:
                record.pop_index[claim_id] = pops
                record.pop_keys.append(key)
            dur = duration_of(claim_id)
            ready_t = t0
            for p in preds_of[claim_id]:
                if finish_of[p] > ready_t:
                    ready_t = finish_of[p]
            rids = rids_of[claim_id]
            d, start = _pick_domain(
                rids, ready_t, dur, domain_free, resident, eligible[claim_id], locality
            )
            finish = start + dur
            domain_free[d] = finish
            resident[d] |= rids
            finish_of[claim_id] = finish
            if sched is not None:
                stream = TAIL_STREAM if d == domains else d
                sched.slots.append(Slot(claim_id, stream, start, finish))
                sched.affinity[claim_id] = stream
            pops += 1
            for successor in successors[claim_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    if record is not None:
                        record.push_index[successor] = pops
                    heapq.heappush(ready, (-duration_of(successor), successor, ordinal[successor]))
        if pops != len(self.claims):
            raise ValueError("GEM dispatch dependency graph is cyclic")
        return pops


@dataclass
class _Checkpoint:
    """The dispatch state at the top of pop `pops`: the ready heap, the indegrees, the
    stream frontier, the residency and the finishes -- everything `run` continues from."""

    pops: int
    ready: list[tuple[int, int, int]]
    indegree: dict[int, int]
    domain_free: list[int]
    resident: list[set[int]]
    finish_of: dict[int, int]


@dataclass
class _Record:
    """What a recorded run leaves behind: the pop count at which every claim entered the
    ready heap (roots: 0), the index at which it was popped, the popped keys in order,
    and, when `checkpoints` is a list, a checkpoint every `stride` pops."""

    push_index: dict[int, int] = field(default_factory=dict)
    pop_index: dict[int, int] = field(default_factory=dict)
    pop_keys: list[tuple[int, int, int]] = field(default_factory=list)
    checkpoints: list[_Checkpoint] | None = None
    stride: int = 0


def _dispatch(
    claims: list[Claim],
    preds: dict[int, list[int]],
    durations: dict[int, int],
    t0: int,
    domain_free: list[int],
    resident: list[set[int]],
    domains: int,
    knee: int,
    locality: bool,
    finish_of: dict[int, int],
    sched: GemSchedule,
) -> None:
    """Event-driven LPT list scheduling of `claims` honoring `preds` edges: the one-shot
    form of `_PhaseDispatch.run` (one structure, one placement, one tie-break).

    `domain_free` / `resident` carry one entry per affinity domain plus one for the
    tail stream (index `domains`): a sparse GGG/random claim is dispatched on the
    tail inside this same loop, so its hazard edges hold across the streams and its
    slot is reported on `TAIL_STREAM`."""
    _PhaseDispatch(claims, preds, domains, knee, locality).run(
        durations, t0, domain_free, resident, finish_of, sched
    )


def stream_geometry(target) -> tuple[int, int]:
    """(affinity domains, bandwidth knee) of a target; one domain and no knee without one.
    The two header fields of the plan's byte form (`gem.execution_plan`, G11)."""
    domains = max(1, getattr(target, "affinity_domains", 1)) if target is not None else 1
    knee = bandwidth_knee(target) if target is not None else 1
    return domains, knee


_streams = stream_geometry


def phase_hazards(module: Module) -> dict[int, dict[int, list[int]]]:
    """The intra-phase hazard DAG of every phase -- over main AND tail claims, in claim-id
    order, before any stream split -- keyed by phase id. A pure function of the module, so
    a caller placing many duration vectors of one module (the re-selection sweep) builds it
    once and hands it to `schedule_eft`."""
    pmap = module.phase_map()
    out: dict[int, dict[int, list[int]]] = {}
    seen_claim_ids: set[int] = set()
    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        claim_ids = [claim.id for claim in claims]
        if len(set(claim_ids)) != len(claim_ids) or seen_claim_ids & set(claim_ids):
            raise ValueError("GEM scheduling requires module-wide unique claim ids")
        seen_claim_ids.update(claim_ids)
        out[pid] = hazard_predecessors(claims)
    return out


def schedule_eft(
    module: Module,
    durations: dict[int, int],
    target=None,
    locality: bool = True,
    hazards: dict[int, dict[int, list[int]]] | None = None,
) -> GemSchedule:
    """Duration-aware wave scheduling (HEFT-lite) with phase barriers.

    Within each phase: the hazard DAG over EVERY claim of the phase (data hazards
    and fences, `hazard_predecessors`), then LPT priority + earliest-finish-time
    placement + locality tie-breaks + the bandwidth-knee clamp, with the GGG/random
    tail on its own stream inside the same dispatch (an independent tail still
    overlaps the waves: phase span = max(main, tail); a dependent one waits).
    Phases compose serially. `hazards` is `phase_hazards(module)`, precomputed by a
    caller that places the same module many times.
    """
    domains, knee = _streams(target)
    sched = GemSchedule(mode="eft", knee=knee)
    resident: list[set[int]] = [set() for _ in range(domains + 1)]
    pmap = module.phase_map()
    t0 = 0
    if hazards is None:
        hazards = phase_hazards(module)

    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        # The intra-phase hazard DAG over main AND tail claims, before the stream split
        # (the lower claim id is the producer of a conflicting pair).
        preds = hazards[pid]
        domain_free = [t0] * (domains + 1)
        finish_of: dict[int, int] = {}
        _dispatch(
            claims,
            preds,
            durations,
            t0,
            domain_free,
            resident,
            domains,
            knee,
            locality,
            finish_of,
            sched,
        )
        t0 = max([t0] + list(finish_of.values()))
    sched.makespan = t0
    return sched


@dataclass
class _PhaseRecord:
    """One phase's base dispatch under `EftPlacer`: its structure, the residency at entry
    and exit (per stream), what the phase added to each stream, the rids it touches, its
    relative slots (t0 = 0) and span, and the recorded run (push indices; checkpoints once
    a replay asked for them)."""

    dispatch: _PhaseDispatch
    entry: list[set[int]]
    exit: list[set[int]]
    added: list[set[int]]
    touched: set[int]
    slots: list[Slot]
    affinity: dict[int, int]
    span: int
    record: _Record


class EftPlacer:
    """Places one module's duration vectors many times under the phase-barriered dispatch,
    exactly as `schedule_eft` does, re-dispatching only what a change can reach (G2 / S2-A:
    the report's "cache per-phase contributions and reprice only the affected chain").

    The phase-barriered schedule composes phases serially, and one phase's placement is a
    pure function of its claims, its hazard edges, its durations and the residency at its
    entry (the locality tie-break) -- its slots are translation-invariant in the phase
    start. So the placer keeps, per phase, the span, the entry/exit residency and the
    recorded pop order, and prices a `trial` (claim id -> duration, not adopted) as:

      * the phases before the first changed one: their cached spans;
      * a changed phase: replayed from the last checkpoint taken before the first changed
        claim entered the ready heap (the pop order is a function of the edges and the
        durations alone, so nothing before that push differs) -- the checkpoints are taken
        on the first replay of a phase, so a module the sweep never re-places pays nothing;
      * every later phase: skipped, with its cached span, when it touches no rid whose
        stream membership the replay moved (its decisions read only its own rids'
        residency, so its placement is the cached one and the moved rids stay moved), and
        re-dispatched otherwise.

    `adopt` makes a change permanent (re-recording what it reaches), `schedule` assembles
    the artifact -- `schedule() == schedule_eft(module, durations, target, locality)` by
    construction, which the tests hold on every fixture and the sweep relies on.
    """

    def __init__(
        self,
        module: Module,
        durations: dict[int, int],
        target=None,
        locality: bool = True,
        hazards: dict[int, dict[int, list[int]]] | None = None,
    ) -> None:
        self.domains, self.knee = _streams(target)
        self.locality = locality
        self.durations = dict(durations)
        self.pops = 0  # claims re-placed by trials and adoptions (the delta work)
        if hazards is None:
            hazards = phase_hazards(module)
        pmap = module.phase_map()
        self.phase_of: dict[int, int] = {}
        self.records: list[_PhaseRecord] = []
        resident: list[set[int]] = [set() for _ in range(self.domains + 1)]
        for index, pid in enumerate(_topo_phase_ids(module)):
            claims = sorted(pmap[pid].claims, key=lambda c: c.id)
            for claim in claims:
                self.phase_of[claim.id] = index
            dispatch = _PhaseDispatch(claims, hazards[pid], self.domains, self.knee, locality)
            self.records.append(self._record(dispatch, resident))
        self.makespan = sum(record.span for record in self.records)

    # --- the base ------------------------------------------------------------------------

    def _record(self, dispatch: _PhaseDispatch, resident: list[set[int]]) -> _PhaseRecord:
        """Run one phase from `resident` (mutated to the exit state) with the push indices
        recorded; the relative slots are kept for `schedule`."""
        entry = [set(s) for s in resident]
        sched = GemSchedule(mode="eft", knee=self.knee)
        finish_of: dict[int, int] = {}
        record = _Record()
        dispatch.run(
            self.durations, 0, [0] * (self.domains + 1), resident, finish_of, sched, record=record
        )
        exit_ = [set(s) for s in resident]
        return _PhaseRecord(
            dispatch,
            entry,
            exit_,
            [after - before for before, after in zip(entry, exit_)],
            set().union(*dispatch.rids.values()) if dispatch.rids else set(),
            sched.slots,
            sched.affinity,
            max(finish_of.values(), default=0),
            record,
        )

    def _checkpoints(self, record: _PhaseRecord) -> None:
        """Take the phase's checkpoints (once): a second recorded run from its entry under
        the base durations -- a trial never mutates them, so a checkpoint taken during one
        trial is valid for every later one."""
        from math import isqrt

        run = _Record(checkpoints=[], stride=max(1, isqrt(len(record.dispatch.claims))))
        record.dispatch.run(
            self.durations,
            0,
            [0] * (self.domains + 1),
            [set(s) for s in record.entry],
            {},
            None,
            record=run,
        )
        base = record.record
        record.record = _Record(
            base.push_index, base.pop_index, base.pop_keys, run.checkpoints, run.stride
        )

    # --- trials --------------------------------------------------------------------------

    def _replay(
        self,
        record: _PhaseRecord,
        durations: dict[int, int],
        changed: list[int],
        resident: list[set[int]],
    ) -> int:
        """The span of `record`'s phase under `durations` from the cached entry
        (`resident` == entry on call; the exit state on return), continued from the last
        checkpoint at or before the first pop a changed claim can move.

        The pop order is the sorted order of the keys present, so a changed key alters
        nothing until the claim is popped in the base run (`pop_index`) or would be popped
        in the new one: a shortened claim (a lower priority) is popped no earlier than in
        the base; a lengthened one is popped at the first base pop whose key is below its
        new key. Up to the earlier of the two the placements agree, so the checkpoint there
        is the new run's state once the changed claims still in its heap carry their new
        keys (those not yet pushed will be pushed by their predecessors, as recorded)."""
        base = record.record
        if base.checkpoints is None:
            self._checkpoints(record)
            base = record.record
        ordinal = record.dispatch.ordinal
        limit = len(record.dispatch.claims)
        new_keys: dict[int, tuple[int, int, int]] = {}
        for cid in changed:
            new_key = (-max(0, durations.get(cid, 1)), cid, ordinal[cid])
            new_keys[cid] = new_key
            popped = base.pop_index[cid]
            if new_key > base.pop_keys[popped]:  # shortened: popped no earlier than before
                limit = min(limit, popped)
                continue
            first = base.push_index.get(cid, 0)
            for index in range(first, popped):
                if base.pop_keys[index] > new_key:  # the new key would have been the minimum
                    popped = index
                    break
            limit = min(limit, popped)
        resume: _Checkpoint | None = None
        for checkpoint in base.checkpoints:  # ascending pops
            if checkpoint.pops <= limit:
                resume = checkpoint
            else:
                break
        if resume is None:
            finish_of: dict[int, int] = {}
            self.pops += record.dispatch.run(
                durations, 0, [0] * (self.domains + 1), resident, finish_of, None
            )
        else:
            import heapq

            ready = [entry for entry in resume.ready if entry[1] not in new_keys]
            for cid, new_key in new_keys.items():
                if base.push_index.get(cid, 0) <= resume.pops:  # in the heap at the checkpoint
                    ready.append(new_key)
            heapq.heapify(ready)
            for stream, held in zip(resident, resume.resident):
                stream.clear()
                stream |= held
            finish_of = dict(resume.finish_of)
            self.pops -= resume.pops
            self.pops += record.dispatch.run(
                durations,
                0,
                list(resume.domain_free),
                resident,
                finish_of,
                None,
                resume=_Checkpoint(
                    resume.pops,
                    ready,
                    resume.indegree,
                    resume.domain_free,
                    resume.resident,
                    resume.finish_of,
                ),
            )
        return max(finish_of.values(), default=0)

    def _walk(self, durations: dict[int, int], changes: dict[int, int], adopt: bool) -> int:
        """The makespan under `durations` (the base with `changes` applied): the cached
        prefix, the changed phases replayed (from a checkpoint when the entry residency is
        the cached one), the phases after them skipped unless they touch a moved rid."""
        for cid in changes:
            if cid not in self.phase_of:
                raise KeyError(f"claim {cid} is not placed by this placer")
        by_phase: dict[int, list[int]] = {}
        for cid in changes:
            by_phase.setdefault(self.phase_of[cid], []).append(cid)
        first = min(by_phase)
        records = self.records
        total = sum(record.span for record in records[:first])
        resident = [set(s) for s in records[first].entry]
        moved: set[int] = set()  # rids whose stream membership differs from the cached one
        for index in range(first, len(records)):
            record = records[index]
            changed = by_phase.get(index)
            if changed is None and moved.isdisjoint(record.touched):
                total += record.span  # the cached placement stands; the moved rids stay moved
                for stream, added in zip(resident, record.added):
                    stream |= added
                continue
            if adopt:
                new = self._record(record.dispatch, resident)
                records[index] = new
                span = new.span
                self.pops += len(new.dispatch.claims)
            elif changed is not None and not moved:
                span = self._replay(record, durations, changed, resident)
            else:
                finish_of: dict[int, int] = {}
                self.pops += record.dispatch.run(
                    durations, 0, [0] * (self.domains + 1), resident, finish_of, None
                )
                span = max(finish_of.values(), default=0)
            total += span
            moved = set()
            for stream, cached in zip(resident, record.exit):
                moved |= stream ^ cached
        return total

    def trial(self, changes: dict[int, int]) -> int:
        """The makespan `schedule_eft` would report with `changes` (claim id -> duration)
        applied to the current durations; nothing is adopted."""
        changes = {cid: dur for cid, dur in changes.items() if self.durations.get(cid) != dur}
        if not changes:
            return self.makespan
        return self._walk({**self.durations, **changes}, changes, adopt=False)

    def adopt(self, changes: dict[int, int]) -> int:
        """Apply `changes` to the current durations, re-record what they reach, and return
        the new makespan."""
        changes = {cid: dur for cid, dur in changes.items() if self.durations.get(cid) != dur}
        if changes:
            self.durations.update(changes)
            self.makespan = self._walk(self.durations, changes, adopt=True)
        return self.makespan

    def schedule(self) -> GemSchedule:
        """The artifact of the current durations: `schedule_eft(module, durations, ...)`."""
        sched = GemSchedule(mode="eft", knee=self.knee)
        t0 = 0
        for record in self.records:
            for slot in record.slots:
                sched.slots.append(
                    Slot(slot.claim_id, slot.domain, slot.start + t0, slot.finish + t0)
                )
            sched.affinity.update(record.affinity)
            t0 += record.span
        sched.makespan = t0
        return sched


def execute_tokens(
    module: Module, durations: dict[int, int], target=None, locality: bool = True
) -> GemSchedule:
    """Token-DAG execution: phase barriers replaced by `!bcir.token` awaits.

    A claim becomes ready when every claim it awaits has finished (its data
    hazards and the ordering fences, `async_plan`) -- nothing else holds it back,
    so independent claims of later phases overlap earlier phases (pipelined
    phases fall out of the dependency structure). Placement is the same EFT +
    locality + knee + tail-stream dispatch as `schedule_eft`; the barriered
    schedule is its degenerate case when every cross-phase claim conflicts.
    """
    domains, knee = _streams(target)
    sched = GemSchedule(mode="tokens", knee=knee)
    resident: list[set[int]] = [set() for _ in range(domains + 1)]

    plan = async_plan(module)
    pmap = module.phase_map()
    order = {cid: i for i, cid in enumerate(plan.forks)}
    claims = sorted(
        (c for pid in _topo_phase_ids(module) for c in pmap[pid].claims), key=lambda c: order[c.id]
    )

    domain_free = [0] * (domains + 1)
    finish_of: dict[int, int] = {}
    _dispatch(
        claims,
        plan.awaits,
        durations,
        0,
        domain_free,
        resident,
        domains,
        knee,
        locality,
        finish_of,
        sched,
    )
    sched.makespan = max(finish_of.values(), default=0)
    return sched


SCHEDULE_MODES = ("eft", "tokens")


def schedule_plan(
    module: Module, result, target=None, mode: str = "eft", locality: bool = True
) -> GemSchedule:
    """The canonical schedule artifact of a selected plan (G1).

    One placement, read by the scheduled price and by the executors alike: the
    plan's own step costs (`durations_from`) placed by the hazard-honoring LPT/EFT
    dispatch, phase-barriered (`mode="eft"`, the default) or token-pipelined
    (`mode="tokens"`). `gem.overlap.price_scheduled(...).schedule` is this object,
    and `price_scheduled(...).makespan` is its makespan.
    """
    if mode not in SCHEDULE_MODES:
        raise ValueError(f"unknown schedule mode {mode!r}; expected one of {SCHEDULE_MODES}")
    durations = durations_from(result)
    if mode == "eft":
        return schedule_eft(module, durations, target, locality)
    return execute_tokens(module, durations, target, locality)


# --- phase-aware DVFS over the schedule timeline (schedule_power_rail) ------------


@dataclass(frozen=True)
class PowerRailDecision:
    claim_id: int
    start: int
    finish: int
    klass: str  # compute | memory | balanced
    clock_q8: int  # Q8 clock for this slot's interval (256 = nominal)
    reason: str

    @property
    def duration(self) -> int:
        return max(0, self.finish - self.start)


@dataclass(frozen=True)
class PowerRail:
    decisions: tuple

    @property
    def energy_saved_milli(self) -> int:
        """Modeled energy avoided by downclocking memory-bound slots: sum over slots
        of (nominal - clock) x interval, in milli of a nominal slot-cycle. A *model*
        (power ~ clock on a bandwidth-bound slot whose throughput is clock-insensitive)
        -- NOT a measured Joule figure; see docs/kernel/HARDWARE_VALIDATION.md."""
        from .dvfs import NOMINAL

        return sum(
            ((NOMINAL - d.clock_q8) * d.duration * 1000) // NOMINAL
            for d in self.decisions
            if d.clock_q8 < NOMINAL
        )

    @property
    def downclocked(self) -> tuple:
        return tuple(d.claim_id for d in self.decisions if d.clock_q8 < 256)


def schedule_power_rail(sched: GemSchedule, result, theta, h=None) -> PowerRail:
    """A power-orchestration pass over the *placed timeline*: for each scheduled Slot,
    classify its claim's arithmetic intensity and set a per-slot clock for that
    slot's [start, finish) interval. Extended memory-bound slots (e.g. a long
    bandwidth-bound prefetch/stream window) are downclocked -- their throughput is
    bandwidth-bound, so scaling the core clock to the data-arrival bound saves power
    without losing throughput; compute-bound slots overclock (thermal budget
    permitting); balanced hold nominal. Unlike per-phase `plan_dvfs`, this keys off
    the schedule's real slot intervals. Deterministic; sorted by (start, claim_id).
    The energy figure is *modeled* (no RAPL in-sandbox -- docs/kernel/HARDWARE_VALIDATION.md)."""
    from ..kbcir.cost import COMPUTE, MEMORY
    from .dvfs import classify, clock_for

    base_of = {s.claim_id: s.candidate.base.v for s in result.steps}
    out: list[PowerRailDecision] = []
    for slot in sorted(sched.slots, key=lambda s: (s.start, s.claim_id)):
        base = base_of.get(slot.claim_id)
        if base is None:
            continue
        klass = classify(base[COMPUTE], base[MEMORY])
        clock, reason = clock_for(klass, theta)
        out.append(
            PowerRailDecision(
                claim_id=slot.claim_id,
                start=slot.start,
                finish=slot.finish,
                klass=klass,
                clock_q8=clock,
                reason=reason,
            )
        )
    return PowerRail(decisions=tuple(out))
