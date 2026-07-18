"""Duration-aware GEM scheduling: EFT waves, the token DAG, locality, the knee.

Four upgrades over the unit-time wave scheduler (`concurrency.schedule_concurrent`),
all deterministic and integer-only:

  1. **Duration-aware waves (HEFT-lite).** Claims carry durations (the K_BCIR
     step costs); within a phase an event-driven list scheduler dispatches the
     ready claim with the longest duration first (LPT priority -- the degenerate
     upward rank of HEFT inside one phase) onto the domain with the earliest
     finish time. Hazard conflicts (lower claim id = producer) serialize.
  2. **Token-DAG execution.** `execute_tokens` consumes the `!bcir.token`
     fork/await plan instead of phase barriers: a claim starts when the claims it
     awaits finish, so independent claims of a *later phase* overlap an earlier
     phase -- software pipelining falls out of the dependency structure.
  3. **Locality-aware affinity.** Among earliest-finish ties, a claim prefers the
     domain already holding the most of its operand RIDs (cache reuse), replacing
     blind round-robin.
  4. **Bandwidth-knee clamp.** Bandwidth-class claims contend for at most
     `bandwidth_knee(H)` concurrent slots (the roofline knee from the target's
     `mem_channels`); compute-class claims use the full domain set. Parallelism
     past the knee only thrashes the memory system, so the scheduler queues it.

The unit-time wave scheduler remains the law for CT2 wave *formation*; this
module is the duration-aware refinement that prices and places real work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Claim, Module
from .async_tokens import async_plan
from .concurrency import _conflict_predecessors, _is_sparse, _topo_phase_ids

TAIL_STREAM = -1  # the decoupled GGG tail executes on its own stream


@dataclass(frozen=True)
class Slot:
    claim_id: int
    domain: int       # affinity domain, or TAIL_STREAM for the decoupled tail
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
    """Claim durations from a K_BCIR realization (the scalarized step costs)."""
    return {s.claim_id: max(1, s.cost) for s in result.steps}


def _rids(c: Claim) -> set[int]:
    return set(c.rd) | set(c.wr)


def _pick_domain(c: Claim, ready_t: int, dur: int, domain_free: list[int],
                 resident: list[set[int]], eligible: range, locality: bool) -> tuple[int, int]:
    """Earliest finish first; ties prefer the domain holding the claim's operands."""
    best_d, best_key = eligible[0], None
    rids = _rids(c)
    for d in eligible:
        start = max(domain_free[d], ready_t)
        score = len(rids & resident[d]) if locality else 0
        key = (start + dur, -score, d)
        if best_key is None or key < best_key:
            best_d, best_key = d, key
    return best_d, max(domain_free[best_d], ready_t)


def _dispatch(claims: list[Claim], preds: dict[int, list[int]], durations: dict[int, int],
              t0: int, domain_free: list[int], resident: list[set[int]],
              domains: int, knee: int, locality: bool,
              finish_of: dict[int, int], sched: GemSchedule) -> None:
    """Event-driven LPT list scheduling of `claims` honoring `preds` edges.

    A ready heap replaces repeated full scans/sorts of the pending list.  The heap
    key is exactly the historical ``(-duration, claim_id)`` priority, so placement
    and tie-breaking are unchanged.
    """
    import heapq

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
                    f"GEM dispatch predecessor {predecessor} for claim {claim.id} is unavailable")
    ready: list[tuple[int, int, int]] = []
    for claim in claims:
        if indegree[claim.id] == 0:
            heapq.heappush(ready, (-max(1, durations.get(claim.id, 1)),
                                   claim.id, ordinal[claim.id]))
    dispatched = 0
    while ready:
        _negative_duration, claim_id, _index = heapq.heappop(ready)
        c = claim_by_id[claim_id]
        dur = max(1, durations.get(c.id, 1))
        ready_t = max([finish_of[p] for p in preds.get(c.id, ())], default=t0)
        # Bandwidth-bound claims contend for the knee; compute-bound for all domains.
        width = knee if c.cost_class == "bandwidth" else domains
        d, start = _pick_domain(c, ready_t, dur, domain_free, resident,
                                range(width), locality)
        finish = start + dur
        domain_free[d] = finish
        resident[d] |= _rids(c)
        finish_of[c.id] = finish
        sched.slots.append(Slot(c.id, d, start, finish))
        sched.affinity[c.id] = d
        dispatched += 1
        for successor in successors[c.id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                next_claim = claim_by_id[successor]
                heapq.heappush(ready, (-max(1, durations.get(successor, 1)),
                                       successor, ordinal[next_claim.id]))
    if dispatched != len(claims):
        raise ValueError("GEM dispatch dependency graph is cyclic")


def schedule_eft(module: Module, durations: dict[int, int], target=None,
                 locality: bool = True) -> GemSchedule:
    """Duration-aware wave scheduling (HEFT-lite) with phase barriers.

    Within each phase: LPT priority + earliest-finish-time placement + locality
    tie-breaks + the bandwidth-knee clamp; the GGG tail runs decoupled on its
    own stream (phase span = max(main, tail)). Phases compose serially.
    """
    domains = max(1, getattr(target, "affinity_domains", 1)) if target is not None else 1
    knee = bandwidth_knee(target) if target is not None else 1
    sched = GemSchedule(mode="eft", knee=knee)
    resident: list[set[int]] = [set() for _ in range(domains)]
    pmap = module.phase_map()
    t0 = 0
    seen_claim_ids: set[int] = set()

    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        claim_ids = [claim.id for claim in claims]
        if len(set(claim_ids)) != len(claim_ids) or seen_claim_ids & set(claim_ids):
            raise ValueError("GEM scheduling requires module-wide unique claim ids")
        seen_claim_ids.update(claim_ids)
        main = [c for c in claims if not _is_sparse(c)]
        tail = [c for c in claims if _is_sparse(c)]

        # Intra-phase hazard DAG: the lower claim id is the producer.
        preds = _conflict_predecessors(main)
        domain_free = [t0] * domains
        finish_of: dict[int, int] = {}
        _dispatch(main, preds, durations, t0, domain_free, resident,
                  domains, knee, locality, finish_of, sched)

        tt = t0
        for c in tail:  # the decoupled tail: a serial chain overlapping the waves
            dur = max(1, durations.get(c.id, 1))
            sched.slots.append(Slot(c.id, TAIL_STREAM, tt, tt + dur))
            sched.affinity[c.id] = TAIL_STREAM
            tt += dur
        t0 = max([t0, tt] + list(finish_of.values()))
    sched.makespan = t0
    return sched


def execute_tokens(module: Module, durations: dict[int, int], target=None,
                   locality: bool = True) -> GemSchedule:
    """Token-DAG execution: phase barriers replaced by `!bcir.token` awaits.

    A claim becomes ready when every claim it awaits has finished -- nothing
    else holds it back, so independent claims of later phases overlap earlier
    phases (pipelined phases fall out of the dependency structure). Placement
    uses the same EFT + locality + knee machinery as `schedule_eft`; the
    schedule is its degenerate case when every cross-phase claim conflicts.
    """
    domains = max(1, getattr(target, "affinity_domains", 1)) if target is not None else 1
    knee = bandwidth_knee(target) if target is not None else 1
    sched = GemSchedule(mode="tokens", knee=knee)
    resident: list[set[int]] = [set() for _ in range(domains)]

    plan = async_plan(module)
    pmap = module.phase_map()
    order = {cid: i for i, cid in enumerate(plan.forks)}
    claims = sorted((c for pid in _topo_phase_ids(module) for c in pmap[pid].claims),
                    key=lambda c: order[c.id])

    domain_free = [0] * domains
    finish_of: dict[int, int] = {}
    _dispatch(claims, plan.awaits, durations, 0, domain_free, resident,
              domains, knee, locality, finish_of, sched)
    sched.makespan = max(finish_of.values(), default=0)
    return sched


# --- phase-aware DVFS over the schedule timeline (schedule_power_rail) ------------

@dataclass(frozen=True)
class PowerRailDecision:
    claim_id: int
    start: int
    finish: int
    klass: str            # compute | memory | balanced
    clock_q8: int         # Q8 clock for this slot's interval (256 = nominal)
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
        return sum(((NOMINAL - d.clock_q8) * d.duration * 1000) // NOMINAL
                   for d in self.decisions if d.clock_q8 < NOMINAL)

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
        out.append(PowerRailDecision(claim_id=slot.claim_id, start=slot.start,
                                     finish=slot.finish, klass=klass, clock_q8=clock,
                                     reason=reason))
    return PowerRail(decisions=tuple(out))
