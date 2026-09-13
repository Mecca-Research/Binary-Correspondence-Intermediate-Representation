"""Bounded exact scheduling and the lower-bound stack (G4 / S2-B): the first TMSAO-2.

The canonical schedule artifact (`schedule.schedule_plan`, G1) is a heuristic placement --
per phase, LPT priority, earliest-finish placement, locality tie-breaks and the bandwidth
knee. The 2026-08-12 report (section 6.1) measured it against an exact scheduler on every
nondecreasing six-job duration multiset with values 1..8: suboptimal on 190 of 1,716 instances
on two domains (worst 17/15), 18 on three (worst 7/6). Until this slice nothing in the tree
could state that distance for a given plan; every result was TMSAO-4 by construction.

`exact_schedule` is the dependency-free bounded exact solver behind the heuristic. It does not
replace the artifact -- the placement every rail reads and the C and MLIR twins reproduce
stays `schedule_eft`'s -- it CERTIFIES it: for one module and one duration vector it returns
the heuristic's makespan `U`, the strongest valid lower bound `L` from the bound stack, the
optimum when the search completes within its budget (then `L == optimum` and the incumbent's
gap is exact), the stop reason and the work spent, all in the solver's own work units (node
expansions), never seconds. "The gap is the product, not the speed" (roadmap G4): a plan that
carries `L`, `U` and both gaps is TMSAO-2 whatever the heuristic did.

The search, per phase (phase barriers compose serially, so the module's optimum is the sum of
its phases' optima and the exact solver runs one phase at a time):

  * the state is a partial active schedule -- the streams' free times, the finish of every
    placed claim, the ready set -- and a node branches on (ready claim, eligible stream) with
    the claim's start the earliest its predecessors and the stream allow (an active schedule:
    every optimal schedule under precedence and release times is among them);
  * eligibility is the artifact's own: a sparse GGG/random claim on the tail stream, a
    bandwidth-class claim on the first `knee` streams, a compute-class claim on every domain
    (`schedule._PhaseDispatch.eligible`, the one place that rule lives);
  * the bound at a node is the strongest of the stack -- the finish so far, the critical
    path from any unplaced claim's earliest start, the remaining work over the streams'
    frontier (bandwidth work over the knee), the tail stream's serial work -- and a node whose
    bound reaches the incumbent is cut;
  * symmetry breaking: streams that are empty at equal free times are interchangeable (only
    the lowest is branched on), and identical unplaced claims (same duration, edges and
    eligibility) are placed in id order;
  * the incumbent starts at the heuristic's placement, so the first answer is a legal plan
    and the search only ever improves on it; a budget stop keeps it and reports the minimum
    bound over the subtrees it never entered, so `L` stays valid.

`exact_selection` is the same discipline for the schedule-aware candidate selection
(section 6.3): every assignment of a bounded module's candidates, priced through the same
serial re-pricing and the same artifact the one-sweep search uses, so the sweep's makespan
can be held to the enumeration's.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

from ..model import Claim, Module
from .concurrency import _topo_phase_ids
from .schedule import (
    GemSchedule,
    Slot,
    _PhaseDispatch,
    _streams,
    phase_hazards,
)

#: Work units (node expansions) a search may spend before it stops on its budget. Counted,
#: never timed: a certificate that says "budget" means the same thing on every host.
DEFAULT_EXACT_BUDGET = 200_000

STOP_REASONS = ("optimal", "budget")


@dataclass(frozen=True)
class Bound:
    """One member of the lower-bound stack: what it is and what it proves."""

    name: str
    value: int


@dataclass
class PhaseSolution:
    """One phase's search: the heuristic span, the best span found, the strongest valid lower
    bound, the stack it came from, why the search stopped and the work it spent."""

    phase_id: int
    heuristic: int
    incumbent: int
    lower_bound: int
    bounds: tuple[Bound, ...]
    stop_reason: str
    expansions: int
    slots: list[Slot]  # the incumbent's placement, relative to the phase start


@dataclass
class ExactSchedule:
    """The certified answer for one module and one duration vector.

    `heuristic` is the artifact's makespan (`schedule_eft`); `incumbent` (U) the best makespan
    the search found, never above it; `lower_bound` (L) the strongest bound proved, never
    above `incumbent`; `optimal` iff every phase closed its gap. `schedule` is the incumbent's
    placement (the artifact itself when nothing improved on it)."""

    heuristic: int
    incumbent: int
    lower_bound: int
    stop_reason: str
    expansions: int
    budget: int
    phases: list[PhaseSolution]
    schedule: GemSchedule

    @property
    def optimal(self) -> bool:
        return self.stop_reason == "optimal"

    @property
    def heuristic_gap(self) -> dict:
        """How far the artifact is from the strongest bound: TMSAO-2's `(U - L) / U` with the
        heuristic as `U` (exact when `optimal`)."""
        from ..kbcir.scope import gap

        return gap(self.heuristic, self.lower_bound)

    @property
    def incumbent_gap(self) -> dict:
        from ..kbcir.scope import gap

        return gap(self.incumbent, self.lower_bound)

    @property
    def bounds(self) -> tuple[Bound, ...]:
        """The module-level stack: each named bound summed over the phases."""
        names = [bound.name for bound in self.phases[0].bounds] if self.phases else []
        return tuple(
            Bound(
                name,
                sum(
                    bound.value
                    for phase in self.phases
                    for bound in phase.bounds
                    if bound.name == name
                ),
            )
            for name in names
        )


# --- the bound stack ----------------------------------------------------------------------


def _critical_path(
    order: list[int], preds: dict[int, tuple[int, ...]], dur: dict[int, int]
) -> dict[int, int]:
    """The longest path ENDING at each claim (its own duration included), in topological order."""
    head: dict[int, int] = {}
    for cid in order:
        best = 0
        for p in preds[cid]:
            if p in head and head[p] > best:
                best = head[p]
        head[cid] = best + dur[cid]
    return head


def _tail_path(
    order: list[int], succs: dict[int, list[int]], dur: dict[int, int]
) -> dict[int, int]:
    """The longest path STARTING at each claim (its own duration included)."""
    tail: dict[int, int] = {}
    for cid in reversed(order):
        best = 0
        for s in succs[cid]:
            if tail[s] > best:
                best = tail[s]
        tail[cid] = dur[cid] + best
    return tail


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# --- the search ----------------------------------------------------------------------------


class _PhaseSearch:
    """Branch-and-bound over one phase's active schedules."""

    def __init__(self, dispatch: _PhaseDispatch, durations: dict[int, int], knee: int) -> None:
        self.dispatch = dispatch
        claims = dispatch.claims
        self.ids = [claim.id for claim in claims]
        self.dur = {cid: max(0, durations.get(cid, 1)) for cid in self.ids}
        self.preds = {
            cid: tuple(p for p in dispatch.preds_of[cid] if p in dispatch.claim_by_id)
            for cid in self.ids
        }
        self.succs = dispatch.successors
        self.domains = dispatch.domains
        self.tail = dispatch.domains  # the tail stream's index
        self.knee = knee
        # a topological order (claim ids are the producer order within a phase)
        self.order = self._topological()
        self.eligible = {cid: tuple(dispatch.eligible[cid]) for cid in self.ids}
        self.on_tail = {cid: self.eligible[cid] == (self.tail,) for cid in self.ids}
        self.tail_path = _tail_path(self.order, self.succs, self.dur)
        # identical-claim classes for symmetry breaking: same duration, edges and eligibility
        signature = {
            cid: (
                self.dur[cid],
                self.preds[cid],
                tuple(sorted(self.succs[cid])),
                self.eligible[cid],
            )
            for cid in self.ids
        }
        self.twin_before: dict[int, int | None] = {}
        seen: dict[tuple, int] = {}
        for cid in self.ids:  # id order: the lower id of two twins is placed first
            key = signature[cid]
            self.twin_before[cid] = seen.get(key)
            seen[key] = cid

    def _topological(self) -> list[int]:
        indegree = {cid: len(self.preds[cid]) for cid in self.ids}
        ready = [cid for cid in self.ids if indegree[cid] == 0]
        heapq.heapify(ready)
        order: list[int] = []
        while ready:
            cid = heapq.heappop(ready)
            order.append(cid)
            for s in self.succs[cid]:
                indegree[s] -= 1
                if indegree[s] == 0:
                    heapq.heappush(ready, s)
        if len(order) != len(self.ids):
            raise ValueError("GEM dispatch dependency graph is cyclic")
        return order

    def root_bounds(self) -> tuple[Bound, ...]:
        """The stack at the root: every bound is valid for any schedule of the phase."""
        head = _critical_path(self.order, self.preds, self.dur)
        critical = max(head.values(), default=0)
        wave = [cid for cid in self.ids if not self.on_tail[cid]]
        bandwidth = [
            cid
            for cid in wave
            if self.eligible[cid] == tuple(range(self.knee)) and self.knee < self.domains
        ]
        work = _ceil_div(sum(self.dur[cid] for cid in wave), max(1, self.domains))
        knee = (
            _ceil_div(sum(self.dur[cid] for cid in bandwidth), max(1, self.knee))
            if bandwidth
            else 0
        )
        tail_work = sum(self.dur[cid] for cid in self.ids if self.on_tail[cid])
        return (
            Bound("critical-path", critical),
            Bound("work/capacity", work),
            Bound("bandwidth/knee", knee),
            Bound("tail-serial", tail_work),
        )

    def heuristic(self) -> tuple[int, list[Slot]]:
        """The artifact's own placement of the phase (from an empty residency, at t0 = 0)."""
        sched = GemSchedule(mode="eft", knee=self.knee)
        finish_of: dict[int, int] = {}
        self.dispatch.run(
            self.dur,
            0,
            [0] * (self.domains + 1),
            [set() for _ in range(self.domains + 1)],
            finish_of,
            sched,
        )
        return max(finish_of.values(), default=0), sched.slots

    def solve(self, budget: int) -> tuple[int, int, str, int, list[Slot], tuple[Bound, ...]]:
        """(incumbent, lower bound, stop reason, expansions, incumbent slots, root bounds)."""
        bounds = self.root_bounds()
        root = max(bound.value for bound in bounds)
        upper, best_slots = self.heuristic()
        if upper <= root or not self.ids:
            return upper, upper, "optimal", 0, best_slots, bounds
        n = len(self.ids)
        dur, preds, succs, eligible = self.dur, self.preds, self.succs, self.eligible
        tail_path, twin_before = self.tail_path, self.twin_before
        streams = self.domains + 1
        remaining_work_all = sum(dur[cid] for cid in self.ids if not self.on_tail[cid])
        remaining_tail = sum(dur[cid] for cid in self.ids if self.on_tail[cid])

        # A node: (free per stream, finish_of, placed count, remaining wave work, remaining tail
        # work, indegree, slots so far). The DFS stack holds child descriptors; each child is
        # materialized only when popped.
        indegree0 = {cid: len(preds[cid]) for cid in self.ids}
        Node = tuple
        root_node: Node = (
            tuple([0] * streams),
            {},
            0,
            remaining_work_all,
            remaining_tail,
            dict(indegree0),
            (),
        )
        stack: list[tuple[int, Node]] = [(root, root_node)]
        open_min = None  # the least bound among nodes cut by the budget, for L
        expansions = 0
        stop = "optimal"
        while stack:
            bound, node = stack.pop()
            if bound >= upper:
                continue
            if expansions >= budget:
                stop = "budget"
                open_min = bound if open_min is None else min(open_min, bound)
                continue
            expansions += 1
            free, finish_of, placed, work_left, tail_left, indegree, slots = node
            if placed == n:
                makespan = max(free)
                if makespan < upper:
                    upper = makespan
                    best_slots = list(slots)
                continue
            # the ready claims, twins in id order
            ready = [cid for cid in self.ids if indegree[cid] == 0 and cid not in finish_of]
            children: list[tuple[int, int, Node]] = []
            for cid in ready:
                twin = twin_before[cid]
                if twin is not None and twin not in finish_of:
                    continue  # its identical lower-id twin is placed first
                release = 0
                for p in preds[cid]:
                    if finish_of[p] > release:
                        release = finish_of[p]
                seen_free: set[int] = set()
                for s in eligible[cid]:
                    if free[s] <= release and release in seen_free:
                        continue  # an interchangeable stream: same start, empty until then
                    if free[s] <= release:
                        seen_free.add(release)
                    start = max(free[s], release)
                    finish = start + dur[cid]
                    new_free = list(free)
                    new_free[s] = finish
                    new_finish = dict(finish_of)
                    new_finish[cid] = finish
                    new_indegree = dict(indegree)
                    for succ in succs[cid]:
                        new_indegree[succ] -= 1
                    on_tail = s == self.tail
                    new_work = work_left - (0 if on_tail else dur[cid])
                    new_tail = tail_left - (dur[cid] if on_tail else 0)
                    # the node bound: the stack, at this node
                    frontier = max(new_free)
                    wave_free = new_free[: self.domains]
                    cap = _ceil_div(sum(wave_free) + new_work, max(1, self.domains))
                    tail_bound = new_free[self.tail] + new_tail
                    path = frontier
                    for other in self.ids:
                        if other in new_finish:
                            continue
                        est = 0
                        for p in preds[other]:
                            if p in new_finish:
                                if new_finish[p] > est:
                                    est = new_finish[p]
                        if est + tail_path[other] > path:
                            path = est + tail_path[other]
                    child_bound = max(frontier, cap, tail_bound, path, root)
                    if child_bound >= upper:
                        continue
                    stream = -1 if on_tail else s
                    child: Node = (
                        tuple(new_free),
                        new_finish,
                        placed + 1,
                        new_work,
                        new_tail,
                        new_indegree,
                        slots + (Slot(cid, stream, start, finish),),
                    )
                    children.append((finish, child_bound, child))
            # earliest finish first (LPT-like dive), so the stack pops the promising child first
            children.sort(key=lambda item: (item[0], item[1]), reverse=True)
            for _finish, child_bound, child in children:
                stack.append((child_bound, child))
        lower = (
            upper
            if stop == "optimal"
            else max(root, min(upper, open_min if open_min is not None else upper))
        )
        return upper, lower, stop, expansions, best_slots, bounds


def exact_schedule(
    module: Module,
    durations: dict[int, int],
    target=None,
    *,
    budget: int = DEFAULT_EXACT_BUDGET,
    hazards: dict[int, dict[int, list[int]]] | None = None,
) -> ExactSchedule:
    """Certify the phase-barriered placement of `durations`: the heuristic's makespan, the
    best makespan found, the strongest valid lower bound and the incumbent's placement. Each
    phase gets the whole `budget` (phases compose serially, so a proof is a proof per phase)."""
    if budget < 0:
        raise ValueError("the exact search budget must be non-negative")
    domains, knee = _streams(target)
    if hazards is None:
        hazards = phase_hazards(module)
    pmap = module.phase_map()
    solutions: list[PhaseSolution] = []
    sched = GemSchedule(mode="eft", knee=knee)
    t0 = 0
    heuristic_total = incumbent_total = lower_total = expansions = 0
    stop = "optimal"
    for pid in _topo_phase_ids(module):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        dispatch = _PhaseDispatch(claims, hazards[pid], domains, knee, True)
        search = _PhaseSearch(dispatch, durations, knee)
        heuristic, _slots = search.heuristic()
        upper, lower, reason, spent, slots, bounds = search.solve(budget)
        if lower > upper or upper > heuristic:  # pragma: no cover - the solver's own invariants
            raise AssertionError("exact search produced an inconsistent bound")
        solutions.append(PhaseSolution(pid, heuristic, upper, lower, bounds, reason, spent, slots))
        for slot in slots:
            sched.slots.append(Slot(slot.claim_id, slot.domain, slot.start + t0, slot.finish + t0))
            sched.affinity[slot.claim_id] = slot.domain
        t0 += upper
        heuristic_total += heuristic
        incumbent_total += upper
        lower_total += lower
        expansions += spent
        if reason != "optimal":
            stop = "budget"
    sched.makespan = t0
    return ExactSchedule(
        heuristic_total, incumbent_total, lower_total, stop, expansions, budget, solutions, sched
    )


# --- the certificate -----------------------------------------------------------------------


@dataclass
class ScheduleCertificate:
    """What a plan's placement may now say about itself, and over what.

    `scope` is the `ExecutionScopeV1` digest the statement ranges over (program, hardware,
    Theta, the admitted policy, the objective `makespan` and the search budget declared);
    `klass` the strongest rung `kbcir.scope.certificate_class_allowed` grants the evidence --
    TMSAO-1 when the search closed (the census of active schedules is complete and the bound
    meets the incumbent), TMSAO-2 when it stopped on its budget with an explicit gap. `L`, `U`
    and both gaps are always present: the heuristic's distance to the bound and the
    incumbent's (zero when optimal)."""

    scope: str
    klass: str
    statement: str
    heuristic: int
    incumbent: int
    lower_bound: int
    heuristic_gap: dict
    incumbent_gap: dict
    stop_reason: str
    expansions: int
    budget: int
    bounds: tuple[Bound, ...]

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "class": self.klass,
            "statement": self.statement,
            "objective": "makespan",
            "heuristic": self.heuristic,
            "incumbent": self.incumbent,
            "lower_bound": self.lower_bound,
            "heuristic_gap": dict(self.heuristic_gap),
            "incumbent_gap": dict(self.incumbent_gap),
            "stop_reason": self.stop_reason,
            "expansions": self.expansions,
            "budget": self.budget,
            "bounds": [[bound.name, bound.value] for bound in self.bounds],
        }


def certify_schedule(
    module: Module,
    result,
    target,
    theta=None,
    policy=None,
    *,
    budget: int = DEFAULT_EXACT_BUDGET,
    hazards: dict[int, dict[int, list[int]]] | None = None,
) -> ScheduleCertificate:
    """Certify the canonical placement of a selected plan (`schedule.schedule_plan`): run the
    bounded exact search over the plan's own step costs and bind `L`, `U`, the gaps, the stop
    reason and the budget to the scope they range over."""
    from ..kbcir.scope import certificate_class_allowed, scope_for
    from .schedule import durations_from

    exact = exact_schedule(module, durations_from(result), target, budget=budget, hazards=hazards)
    scope = scope_for(
        module,
        target,
        theta,
        policy,
        budget={"exact_search_expansions": budget},
        objective={"name": "makespan", "artifact": "schedule_plan(mode=eft)"},
    )
    evidence = {
        "incumbent": True,
        "lower_bound": True,
        "proof": exact.optimal,
        "candidate_census": exact.optimal,  # the census of active schedules, complete iff closed
        "census_complete": exact.optimal,
    }
    klass, statement = certificate_class_allowed(scope, evidence)
    return ScheduleCertificate(
        scope.digest(),
        klass,
        statement,
        exact.heuristic,
        exact.incumbent,
        exact.lower_bound,
        exact.heuristic_gap,
        exact.incumbent_gap,
        exact.stop_reason,
        exact.expansions,
        budget,
        exact.bounds,
    )


# --- exact candidate selection (section 6.3) -----------------------------------------------


@dataclass
class ExactSelection:
    """The exhaustive schedule-aware selection of a bounded module: the best makespan over
    every candidate assignment, the assignment (claim id -> candidate width), the sweep's
    makespan for comparison, the assignments priced and whether the enumeration completed."""

    optimum: int
    widths: dict[int, int]
    sweep: int
    assignments: int
    stop_reason: str

    @property
    def ratio(self) -> float:
        return self.sweep / self.optimum if self.optimum else 1.0


def exact_selection(
    module: Module, h, theta, policy=None, *, limit: int = 20_000
) -> ExactSelection:
    """Enumerate every candidate assignment of `module` (at most `limit`, then stop on the
    budget with the best seen), pricing each through the one-sweep search's own re-pricing
    and artifact, and hold the sweep's makespan to the best."""
    from ..kbcir.realize import fused_candidates
    from ..kbcir.weights import PERF
    from .overlap import _serial_result, optimize_scheduled
    from .schedule import schedule_plan

    if policy is None:
        policy = PERF
    _result, price = optimize_scheduled(module, h, theta, policy)
    cand_map = fused_candidates(module, h)
    ids = [
        claim.id
        for pid in _topo_phase_ids(module)
        for claim in sorted(module.phase_map()[pid].claims, key=lambda c: c.id)
    ]
    best: tuple[int, dict[int, int]] | None = None
    count = 0
    stop = "optimal"
    for choice in itertools.product(*[cand_map[cid] for cid in ids]):
        if count >= limit:
            stop = "budget"
            break
        count += 1
        assignment = dict(zip(ids, choice))
        result = _serial_result(module, assignment, h, theta, policy)
        makespan = schedule_plan(module, result, h, "eft").makespan
        if best is None or makespan < best[0]:
            best = (makespan, {cid: cand.width for cid, cand in assignment.items()})
    if best is None:
        return ExactSelection(price.makespan, {}, price.makespan, 0, stop)
    return ExactSelection(best[0], best[1], price.makespan, count, stop)


__all__ = [
    "DEFAULT_EXACT_BUDGET",
    "STOP_REASONS",
    "Bound",
    "ExactSchedule",
    "ExactSelection",
    "PhaseSolution",
    "ScheduleCertificate",
    "certify_schedule",
    "exact_schedule",
    "exact_selection",
]
