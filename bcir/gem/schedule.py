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
    """Event-driven LPT list scheduling of `claims` honoring `preds` edges.

    `domain_free` / `resident` carry one entry per affinity domain plus one for the
    tail stream (index `domains`): a sparse GGG/random claim is dispatched on the
    tail inside this same loop, so its hazard edges hold across the streams and its
    slot is reported on `TAIL_STREAM`. A ready heap replaces repeated full scans/
    sorts of the pending list; the heap key is exactly the historical
    ``(-duration, claim_id)`` priority, so placement and tie-breaking are unchanged.
    """
    import heapq

    if len(domain_free) != domains + 1 or len(resident) != domains + 1:
        raise ValueError("GEM dispatch needs one stream per affinity domain plus the tail")

    claim_by_id = {claim.id: claim for claim in claims}
    if len(claim_by_id) != len(claims):
        raise ValueError("GEM dispatch requires unique claim ids")
    ordinal = {claim.id: index for index, claim in enumerate(claims)}
    indegree = {claim.id: 0 for claim in claims}
    successors: dict[int, list[int]] = {claim.id: [] for claim in claims}
    for claim in claims:
        for predecessor in preds.get(claim.id, ()):
            if predecessor in claim_by_id:
                indegree[claim.id] += 1
                successors[predecessor].append(claim.id)
            elif predecessor not in finish_of:
                raise ValueError(
                    f"GEM dispatch predecessor {predecessor} for claim {claim.id} is unavailable"
                )

    def duration_of(claim_id: int) -> int:
        return max(0, durations.get(claim_id, 1))  # an unpriced claim takes a unit

    tail = range(domains, domains + 1)  # the decoupled tail: its own stream, the same edges
    wave = {True: range(knee), False: range(domains)}  # bandwidth: the knee; compute: all
    ready: list[tuple[int, int, int]] = []
    for claim in claims:
        if indegree[claim.id] == 0:
            heapq.heappush(ready, (-duration_of(claim.id), claim.id, ordinal[claim.id]))
    dispatched = 0
    while ready:
        _negative_duration, claim_id, _index = heapq.heappop(ready)
        c = claim_by_id[claim_id]
        dur = duration_of(c.id)
        ready_t = t0
        for p in preds.get(c.id, ()):
            if finish_of[p] > ready_t:
                ready_t = finish_of[p]
        eligible = tail if _is_sparse(c) else wave[c.cost_class == "bandwidth"]
        rids = _rids(c)
        d, start = _pick_domain(rids, ready_t, dur, domain_free, resident, eligible, locality)
        finish = start + dur
        domain_free[d] = finish
        resident[d] |= rids
        finish_of[c.id] = finish
        stream = TAIL_STREAM if d == domains else d
        sched.slots.append(Slot(c.id, stream, start, finish))
        sched.affinity[c.id] = stream
        dispatched += 1
        for successor in successors[c.id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, (-duration_of(successor), successor, ordinal[successor]))
    if dispatched != len(claims):
        raise ValueError("GEM dispatch dependency graph is cyclic")


def _streams(target) -> tuple[int, int]:
    """(affinity domains, bandwidth knee) of a target; one domain and no knee without one."""
    domains = max(1, getattr(target, "affinity_domains", 1)) if target is not None else 1
    knee = bandwidth_knee(target) if target is not None else 1
    return domains, knee


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
