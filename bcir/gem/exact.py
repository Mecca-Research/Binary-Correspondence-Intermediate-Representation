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

import hashlib
import heapq
import itertools
import json
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
    state: "SearchState | None" = field(default=None, compare=False, repr=False)

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

    def solve(
        self, budget: int, frontier: "_Frontier | None" = None
    ) -> tuple[int, int, str, int, list[Slot], tuple[Bound, ...], "_Frontier | None"]:
        """(incumbent, lower bound, stop reason, expansions spent HERE, incumbent slots, root
        bounds, the frontier to resume from -- None when the search closed).

        `budget` is the number of expansions this call may spend; a `frontier` from an
        earlier call continues that search exactly where it stopped (the DFS is
        deterministic, so a resumed run is the uninterrupted run)."""
        bounds = self.root_bounds()
        root = max(bound.value for bound in bounds)
        if frontier is None:
            upper, best_slots = self.heuristic()
            if upper <= root or not self.ids:
                return upper, upper, "optimal", 0, best_slots, bounds, None
            n = len(self.ids)
            remaining_work_all = sum(self.dur[cid] for cid in self.ids if not self.on_tail[cid])
            remaining_tail = sum(self.dur[cid] for cid in self.ids if self.on_tail[cid])
            indegree0 = {cid: len(self.preds[cid]) for cid in self.ids}
            root_node: _Node = (
                tuple([0] * (self.domains + 1)),
                {},
                0,
                remaining_work_all,
                remaining_tail,
                dict(indegree0),
                (),
            )
            stack: list[tuple[int, _Node]] = [(root, root_node)]
            spent_before = 0
        else:
            upper, best_slots, stack, spent_before = (
                frontier.upper,
                list(frontier.best_slots),
                list(frontier.stack),
                frontier.expansions,
            )
        n = len(self.ids)
        dur, preds, succs, eligible = self.dur, self.preds, self.succs, self.eligible
        tail_path, twin_before = self.tail_path, self.twin_before
        expansions = 0
        while stack:
            bound, node = stack[-1]
            if bound >= upper:
                stack.pop()
                continue
            if expansions >= budget:
                break  # the frontier stays on the stack: resumable
            stack.pop()
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
            children: list[tuple[int, int, _Node]] = []
            for cid in ready:
                twin = twin_before[cid]
                if twin is not None and twin not in finish_of:
                    continue  # its identical lower-id twin is placed first
                release = 0
                for p in preds[cid]:
                    if finish_of[p] > release:
                        release = finish_of[p]
                seen_free: set[int] = set()
                for s_ in eligible[cid]:
                    if free[s_] <= release and release in seen_free:
                        continue  # an interchangeable stream: same start, empty until then
                    if free[s_] <= release:
                        seen_free.add(release)
                    start = max(free[s_], release)
                    finish = start + dur[cid]
                    new_free = list(free)
                    new_free[s_] = finish
                    new_finish = dict(finish_of)
                    new_finish[cid] = finish
                    new_indegree = dict(indegree)
                    for succ in succs[cid]:
                        new_indegree[succ] -= 1
                    on_tail = s_ == self.tail
                    new_work = work_left - (0 if on_tail else dur[cid])
                    new_tail = tail_left - (dur[cid] if on_tail else 0)
                    # the node bound: the stack, at this node
                    frontier_t = max(new_free)
                    wave_free = new_free[: self.domains]
                    cap = _ceil_div(sum(wave_free) + new_work, max(1, self.domains))
                    tail_bound = new_free[self.tail] + new_tail
                    path = frontier_t
                    for other in self.ids:
                        if other in new_finish:
                            continue
                        est = 0
                        for p in preds[other]:
                            if p in new_finish and new_finish[p] > est:
                                est = new_finish[p]
                        if est + tail_path[other] > path:
                            path = est + tail_path[other]
                    child_bound = max(frontier_t, cap, tail_bound, path, root)
                    if child_bound >= upper:
                        continue
                    stream = -1 if on_tail else s_
                    child: _Node = (
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
        live = [bound for bound, _node in stack if bound < upper]
        if not live:
            return upper, upper, "optimal", expansions, best_slots, bounds, None
        lower = max(root, min(upper, min(live)))
        return (
            upper,
            lower,
            "budget",
            expansions,
            best_slots,
            bounds,
            _Frontier(
                upper,
                best_slots,
                [(b, nd) for b, nd in stack if b < upper],
                spent_before + expansions,
            ),
        )


_Node = (
    tuple  # (free per stream, finish_of, placed, wave work left, tail work left, indegree, slots)
)


@dataclass
class _Frontier:
    """Where one phase's search stopped: the incumbent, its placement, the open nodes (each
    with its bound) and the expansions spent on the phase so far."""

    upper: int
    best_slots: list[Slot]
    stack: list[tuple[int, _Node]]
    expansions: int


# --- the resumable state -------------------------------------------------------------------


def _inputs_digest(module: Module, durations: dict[int, int], domains: int, knee: int) -> str:
    """What a search is a search OF: the module's identity, the duration vector and the
    stream geometry -- a state resumes only against the inputs it was taken from."""
    from ..kbcir.provenance import module_identity

    body = {
        "module": module_identity(module).digest,
        "durations": sorted((int(cid), int(dur)) for cid, dur in durations.items()),
        "domains": domains,
        "knee": knee,
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _slot_rows(slots) -> list[list[int]]:
    return [[slot.claim_id, slot.domain, slot.start, slot.finish] for slot in slots]


def _slots_of(rows) -> list[Slot]:
    return [Slot(int(a), int(b), int(c), int(d)) for a, b, c, d in rows]


@dataclass
class SearchState:
    """The content-addressed state of an `exact_schedule` search (G12 / S2-C).

    Per phase, in topological order: `closed` (its solution is final), `open` (the frontier
    the search stopped at: the incumbent, its placement, every open node with its bound, the
    expansions spent) or `pending` (not started). `inputs` binds the state to the module, the
    duration vector and the stream geometry it was taken from; `budget_spent` is the module's
    total. `digest` is SHA-256 over the canonical JSON, so a state is an artifact: resuming
    from it -- `exact_schedule(..., resume=state)` -- reproduces the uninterrupted run."""

    inputs: str
    budget_spent: int
    phases: list[dict]

    def to_dict(self) -> dict:
        return {
            "version": "ExactSearchStateV1",
            "inputs": self.inputs,
            "budget_spent": self.budget_spent,
            "phases": self.phases,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "SearchState":
        body = json.loads(text)
        if body.get("version") != "ExactSearchStateV1":
            raise ValueError("not an ExactSearchStateV1 document")
        return cls(str(body["inputs"]), int(body["budget_spent"]), list(body["phases"]))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @property
    def closed(self) -> bool:
        return all(phase["status"] == "closed" for phase in self.phases)


def _node_rows(node: _Node) -> list:
    free, finish_of, placed, work_left, tail_left, indegree, slots = node
    return [
        list(free),
        [[int(cid), int(fin)] for cid, fin in sorted(finish_of.items())],
        placed,
        work_left,
        tail_left,
        [[int(cid), int(deg)] for cid, deg in sorted(indegree.items())],
        _slot_rows(slots),
    ]


def _node_of(rows) -> _Node:
    free, finish_rows, placed, work_left, tail_left, indegree_rows, slot_rows = rows
    return (
        tuple(int(v) for v in free),
        {int(cid): int(fin) for cid, fin in finish_rows},
        int(placed),
        int(work_left),
        int(tail_left),
        {int(cid): int(deg) for cid, deg in indegree_rows},
        tuple(_slots_of(slot_rows)),
    )


def exact_schedule(
    module: Module,
    durations: dict[int, int],
    target=None,
    *,
    budget: int = DEFAULT_EXACT_BUDGET,
    hazards: dict[int, dict[int, list[int]]] | None = None,
    resume: SearchState | None = None,
) -> ExactSchedule:
    """Certify the phase-barriered placement of `durations`: the heuristic's makespan, the
    best makespan found, the strongest valid lower bound and the incumbent's placement.

    `budget` is the module's, in node expansions, consumed phase by phase in topological
    order: a phase that closes hands the remainder to the next, a phase that exhausts it
    leaves the later phases pending (their incumbent is the heuristic's placement, their
    bound the root stack -- still a legal plan with a valid gap). `resume` continues a
    search from its `SearchState` with a fresh `budget`, and the result is what the
    uninterrupted run with the summed budget returns; the state must have been taken from
    these inputs. The returned `state` is the search's state after this call."""
    if budget < 0:
        raise ValueError("the exact search budget must be non-negative")
    domains, knee = _streams(target)
    if hazards is None:
        hazards = phase_hazards(module)
    inputs = _inputs_digest(module, durations, domains, knee)
    if resume is not None and resume.inputs != inputs:
        raise ValueError(
            "the search state was taken from other inputs (module, durations or target)"
        )
    pmap = module.phase_map()
    order = _topo_phase_ids(module)
    if resume is not None and [p["phase_id"] for p in resume.phases] != list(order):
        raise ValueError("the search state does not describe this module's phases")
    solutions: list[PhaseSolution] = []
    states: list[dict] = []
    sched = GemSchedule(mode="eft", knee=knee)
    t0 = 0
    heuristic_total = incumbent_total = lower_total = 0
    spent = 0
    left = budget
    stop = "optimal"
    for index, pid in enumerate(order):
        claims = sorted(pmap[pid].claims, key=lambda c: c.id)
        dispatch = _PhaseDispatch(claims, hazards[pid], domains, knee, True)
        search = _PhaseSearch(dispatch, durations, knee)
        heuristic, heuristic_slots = search.heuristic()
        prior = resume.phases[index] if resume is not None else None
        if prior is not None and prior["status"] == "closed":
            bounds = tuple(Bound(name, int(value)) for name, value in prior["bounds"])
            upper = lower = int(prior["upper"])
            reason, phase_spent, slots, frontier = (
                "optimal",
                int(prior["expansions"]),
                _slots_of(prior["best_slots"]),
                None,
            )
            here = 0
        else:
            frontier_in = None
            if prior is not None and prior["status"] == "open":
                frontier_in = _Frontier(
                    int(prior["upper"]),
                    _slots_of(prior["best_slots"]),
                    [(int(b), _node_of(rows)) for b, rows in prior["stack"]],
                    int(prior["expansions"]),
                )
            if stop == "budget" and frontier_in is None:
                # an earlier phase exhausted the budget: this one is pending -- the heuristic
                # stands and the root stack is its bound (0 expansions)
                bounds = search.root_bounds()
                upper, lower = heuristic, min(heuristic, max(b.value for b in bounds))
                reason, here, slots, frontier = "budget", 0, heuristic_slots, None
                phase_spent = 0
                if upper == lower:
                    reason = "optimal"
                status = "pending" if reason == "budget" else "closed"
            else:
                upper, lower, reason, here, slots, bounds, frontier = search.solve(
                    left, frontier_in
                )
                phase_spent = here + (frontier_in.expansions if frontier_in is not None else 0)
                left -= here
                status = "closed" if reason == "optimal" else "open"
        if lower > upper or upper > heuristic:  # pragma: no cover - the solver's own invariants
            raise AssertionError("exact search produced an inconsistent bound")
        if prior is not None and prior["status"] == "closed":
            status = "closed"
        states.append(
            {
                "phase_id": pid,
                "status": status,
                "heuristic": heuristic,
                "upper": upper,
                "lower": lower,
                "expansions": phase_spent,
                "bounds": [[b.name, b.value] for b in bounds],
                "best_slots": _slot_rows(slots),
                "stack": [[b, _node_rows(nd)] for b, nd in frontier.stack]
                if frontier is not None
                else [],
            }
        )
        solutions.append(
            PhaseSolution(pid, heuristic, upper, lower, bounds, reason, phase_spent, slots)
        )
        for slot in slots:
            sched.slots.append(Slot(slot.claim_id, slot.domain, slot.start + t0, slot.finish + t0))
            sched.affinity[slot.claim_id] = slot.domain
        t0 += upper
        heuristic_total += heuristic
        incumbent_total += upper
        lower_total += lower
        spent += here
        if reason != "optimal":
            stop = "budget"
    sched.makespan = t0
    total_spent = spent + (resume.budget_spent if resume is not None else 0)
    result = ExactSchedule(
        heuristic_total, incumbent_total, lower_total, stop, total_spent, budget, solutions, sched
    )
    result.state = SearchState(inputs, total_spent, states)
    return result


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
    dispatch: object = None  # the DispatchRecord (G12): which rail ran, how it stopped, why

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
            "dispatch": None if self.dispatch is None else self.dispatch.to_dict(),
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
    requested: str = "TMSAO-2",
    resume: SearchState | None = None,
    workload=None,
) -> ScheduleCertificate:
    """Certify the canonical placement of a selected plan (`schedule.schedule_plan`): dispatch
    the rail the law names for (`schedule`, the plan's size, `requested`, `budget`), run it
    over the plan's own step costs and bind `L`, `U`, the gaps, the stop reason, the budget
    and the dispatch record to the scope they range over. The fast rail (a `TMSAO-4` request
    or a zero budget) certifies the heuristic with the root stack as its bound. A declared
    `workload` (`kbcir.workload.Workload`, G13) enters the scope as `W`, so the certificate
    cannot be carried to another workload of the same program."""
    from ..kbcir.scope import certificate_class_allowed, scope_for
    from .dispatch import DispatchRecord, DispatchRequest, dispatch
    from .schedule import durations_from

    durations = durations_from(result)
    decision = dispatch(DispatchRequest("schedule", len(result.steps), requested, budget))
    if decision.rail == "fast":
        exact = exact_schedule(module, durations, target, budget=0, hazards=hazards)
    else:
        exact = exact_schedule(
            module, durations, target, budget=decision.budget, hazards=hazards, resume=resume
        )
    scope = scope_for(
        module,
        target,
        theta,
        policy,
        workload=workload,
        budget={"exact_search_expansions": budget},
        objective={"name": "makespan", "artifact": "schedule_plan(mode=eft)"},
    )
    searched = decision.rail == "proof"
    evidence = {
        "incumbent": True,
        "lower_bound": searched,  # the fast rail places once: no search, no bound of its own
        "proof": searched and exact.optimal,
        "candidate_census": searched and exact.optimal,  # complete iff the search closed
        "census_complete": searched and exact.optimal,
    }
    klass, statement = certificate_class_allowed(scope, evidence)
    strongest = max(exact.bounds, key=lambda bound: bound.value).name if exact.bounds else "none"
    record = DispatchRecord(
        decision,
        "heuristic" if not searched else exact.stop_reason,
        exact.expansions,
        strongest if searched else "none",
        klass,
    )
    return ScheduleCertificate(
        scope.digest(),
        klass,
        statement,
        exact.heuristic,
        exact.incumbent,
        exact.lower_bound,
        exact.heuristic_gap,
        exact.incumbent_gap,
        exact.stop_reason if searched else "heuristic",
        exact.expansions,
        decision.budget,
        exact.bounds,
        record,
    )


@dataclass
class SelectionCertificate:
    """What a selected plan may say about its SELECTION (G6 / S2-D): the min-plus path over
    the declared candidate census is exact, so the plan's score is the optimum of that model
    (`L == U`), and the region graph's structural floor -- what the claims could cost under
    the most favourable coupling -- says how much of the score the context coupling itself
    accounts for. Bound to the scope digest with the dispatch record of the path rail."""

    scope: str
    klass: str
    statement: str
    score: int
    optimum: int
    lower_bound: int
    structural_floor: int
    coupling_cost: int
    regions: dict
    dispatch: object

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "class": self.klass,
            "statement": self.statement,
            "objective": "min_plus",
            "score": self.score,
            "optimum": self.optimum,
            "lower_bound": self.lower_bound,
            "structural_floor": self.structural_floor,
            "coupling_cost": self.coupling_cost,
            "regions": dict(self.regions),
            "dispatch": None if self.dispatch is None else self.dispatch.to_dict(),
        }


def certify_selection(
    module: Module, result, target, theta=None, policy=None, *, workload=None
) -> SelectionCertificate:
    """Certify a plan's selection: re-run the exact min-plus path through the dispatch law's
    path rail, hold the plan's score to it, and report the region graph's structural floor.
    A declared `workload` (G13) enters the scope as `W`."""
    from ..kbcir.regions import module_floor, region_graph
    from ..kbcir.scope import certificate_class_allowed, scope_for
    from ..kbcir.weights import PERF
    from .dispatch import DispatchRequest, solve_path

    used_policy = policy if policy is not None else PERF
    used_theta = (
        theta
        if theta is not None
        else __import__("bcir.kbcir.cost", fromlist=["Theta"]).Theta.cool()
    )
    again, record = solve_path(
        DispatchRequest("path", len(result.steps), "TMSAO-1", 1),
        module,
        target,
        used_theta,
        used_policy,
    )
    optimum = again.score
    if result.score < optimum:  # pragma: no cover - the DP is exact
        raise AssertionError("a plan scored below the exact path optimum")
    graph = region_graph(module)
    floor = module_floor(module, target, used_theta, used_policy)
    scope = scope_for(
        module,
        target,
        theta,
        policy,
        workload=workload,
        objective={"name": "min_plus", "artifact": "realize.optimize"},
        budget={"path_relaxations": record.spent},
    )
    evidence = {
        "incumbent": True,
        "lower_bound": True,
        "proof": True,  # the layered min-plus path is exact over the census
        "candidate_census": True,
        "census_complete": True,
    }
    klass, statement = certificate_class_allowed(scope, evidence)
    return SelectionCertificate(
        scope.digest(),
        klass,
        statement,
        result.score,
        optimum,
        optimum,
        floor,
        result.score - floor,
        graph.kinds(),
        record,
    )


# --- exact candidate selection (section 6.3) -----------------------------------------------


@dataclass
class SelectionSearchState:
    """The content-addressed state of an `exact_selection` enumeration (G12 / S2-C): the
    inputs it ranges over, the next assignment index, the best seen and the count priced."""

    inputs: str
    closed: bool
    next_index: int
    best_makespan: int | None
    best_widths: list[list[int]]
    assignments: int

    def to_dict(self) -> dict:
        return {
            "version": "ExactSelectionStateV1",
            "inputs": self.inputs,
            "closed": self.closed,
            "next_index": self.next_index,
            "best_makespan": self.best_makespan,
            "best_widths": self.best_widths,
            "assignments": self.assignments,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "SelectionSearchState":
        body = json.loads(text)
        if body.get("version") != "ExactSelectionStateV1":
            raise ValueError("not an ExactSelectionStateV1 document")
        best = body["best_makespan"]
        return cls(
            str(body["inputs"]),
            bool(body["closed"]),
            int(body["next_index"]),
            None if best is None else int(best),
            [[int(a), int(b)] for a, b in body["best_widths"]],
            int(body["assignments"]),
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


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
    state: "SelectionSearchState | None" = field(default=None, compare=False, repr=False)

    @property
    def ratio(self) -> float:
        return self.sweep / self.optimum if self.optimum else 1.0


def exact_selection(
    module: Module,
    h,
    theta,
    policy=None,
    *,
    limit: int = 20_000,
    resume: SelectionSearchState | None = None,
) -> ExactSelection:
    """Enumerate every candidate assignment of `module` (at most `limit` per call, then stop
    on the budget with the best seen), pricing each through the one-sweep search's own
    re-pricing and artifact, and hold the sweep's makespan to the best. The sweep's own
    assignment is the incumbent the enumeration starts from, so a budget stop never returns
    worse than the fast rail; `resume` continues an enumeration from its state with a fresh
    `limit` (G12 / S2-C)."""
    from ..kbcir.provenance import hash_policy, hash_target, hash_theta, module_identity
    from ..kbcir.realize import fused_candidates
    from ..kbcir.weights import PERF
    from .overlap import _serial_result, optimize_scheduled
    from .schedule import schedule_plan

    if policy is None:
        policy = PERF
    if limit < 0:
        raise ValueError("the enumeration limit must be non-negative")
    inputs = hashlib.sha256(
        json.dumps(
            {
                "module": module_identity(module).digest,
                "target": hash_target(h),
                "theta": hash_theta(theta),
                "policy": hash_policy(policy),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if resume is not None and resume.inputs != inputs:
        raise ValueError("the selection state was taken from other inputs")
    swept, price = optimize_scheduled(module, h, theta, policy)
    cand_map = fused_candidates(module, h)
    ids = [
        claim.id
        for pid in _topo_phase_ids(module)
        for claim in sorted(module.phase_map()[pid].claims, key=lambda c: c.id)
    ]
    # an incumbent first (G12): the sweep's assignment seeds the enumeration, so a budget stop
    # never returns worse than the fast rail
    best: tuple[int, dict[int, int]] | None = (
        price.makespan,
        {step.claim_id: step.candidate.width for step in swept.steps},
    )
    start = 0
    count = 0
    if resume is not None:
        if resume.best_makespan is not None:
            best = (resume.best_makespan, {int(c): int(w) for c, w in resume.best_widths})
        start = resume.next_index
        count = resume.assignments
        if resume.closed:
            stop = "optimal"
            selection = ExactSelection(
                best[0] if best else price.makespan,
                best[1] if best else {},
                price.makespan,
                count,
                stop,
            )
            selection.state = resume
            return selection
    stop = "optimal"
    index = 0
    priced = 0
    closed = True
    for choice in itertools.product(*[cand_map[cid] for cid in ids]):
        if index < start:
            index += 1
            continue
        if priced >= limit:
            stop = "budget"
            closed = False
            break
        index += 1
        priced += 1
        assignment = dict(zip(ids, choice))
        result = _serial_result(module, assignment, h, theta, policy)
        makespan = schedule_plan(module, result, h, "eft").makespan
        if best is None or makespan < best[0]:
            best = (makespan, {cid: cand.width for cid, cand in assignment.items()})
    count += priced
    state = SelectionSearchState(
        inputs,
        closed,
        index,
        best[0] if best else None,
        sorted(best[1].items()) if best else [],
        count,
    )
    if best is None:
        selection = ExactSelection(price.makespan, {}, price.makespan, 0, stop)
    else:
        selection = ExactSelection(best[0], best[1], price.makespan, count, stop)
    selection.state = state
    return selection


@dataclass
class MeasuredCertificate:
    """What a plan may say about being the MEASURED best (G13 / S2-E), and over what.

    `scope` is the `ExecutionScopeV1` digest with `W` (the declared workload) and `M` (the
    corpus's protocol) inside it; `klass` the ladder's verdict: TMSAO-3 only when the scope
    declares `P`, `H`, `W`, `M`, the census was measured and the prediction interval comes
    from attested silicon on two materially different targets with counters (the two-target
    rule); TMSAO-4 with the reason otherwise. `policy` and `plan` name the measured-best
    candidate and the assignment it re-derived to; `interval` is (min, median, max, MAD) of
    the pooled raw samples; `coverage` the measured share of the admitted census."""

    scope: str
    klass: str
    statement: str
    policy: str
    plan: str
    median_ns: int
    interval: tuple[int, int, int, int]
    coverage: dict
    episodes: int
    samples: int
    tenancy: str
    physical_targets: int
    corpus: str
    dispatch: object = None

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "class": self.klass,
            "statement": self.statement,
            "objective": "wall_ns",
            "policy": self.policy,
            "plan": self.plan,
            "median_ns": self.median_ns,
            "interval": list(self.interval),
            "coverage": dict(self.coverage),
            "episodes": self.episodes,
            "samples": self.samples,
            "tenancy": self.tenancy,
            "physical_targets": self.physical_targets,
            "corpus": self.corpus,
            "dispatch": None if self.dispatch is None else self.dispatch.to_dict(),
        }


def certify_measured(
    module: Module,
    target,
    theta,
    workload,
    corpus,
    policies,
    *,
    measurement: dict | None = None,
    requested: str = "TMSAO-3",
    incumbent=None,
) -> MeasuredCertificate:
    """Certify the measured-best plan among `policies` for (module, target, workload, Theta)
    from the corpus's evidence: dispatch through the law (the measured rail iff `W` is
    declared and the corpus holds evidence for the scope), re-derive the plan the evidence
    names, and bind the class to the scope with `W` and `M` inside it. The corpus informs
    the choice among admitted policies and never decides legality: the plan is `optimize`'s."""
    from ..kbcir.measured import assignment_digest, pooled_median, scope_key, theta_key
    from ..kbcir.realize import optimize
    from ..kbcir.scope import certificate_class_allowed, scope_for
    from .dispatch import DispatchRecord, DispatchRequest, solve_measured

    census = list(policies)
    program, target_key, workload_digest = scope_key(module, target, workload)
    episode = theta_key(theta)
    rows = corpus.lookup(program, target_key, workload_digest, theta=episode)
    size = sum(len(phase.claims) for phase in module.phases)
    request = DispatchRequest(
        "selection", size, requested, 1, workload=workload.digest(), evidence=len(rows)
    )
    (policy, result), record = solve_measured(
        request, corpus, module, target, theta, census, workload=workload, incumbent=incumbent
    )
    plan = assignment_digest(result)
    live = [row for row in rows if row.policy == policy.name and row.plan == plan]
    # Coverage counts LIVE evidence: a candidate whose every row names a plan the planner no
    # longer selects under it is stale, not measured -- the corpus says what a plan cost, and
    # that plan is gone.
    digests = {
        member.name: assignment_digest(optimize(module, target, theta, member)) for member in census
    }
    by_policy: dict[str, list] = {}
    for row in rows:
        if row.policy in digests:
            by_policy.setdefault(row.policy, []).append(row)
    measured = {
        name for name, held in by_policy.items() if any(row.plan == digests[name] for row in held)
    }
    stale = set(by_policy) - measured
    coverage = {"measured": len(measured), "stale": len(stale), "census": len(census)}
    walls = sorted(sample.wall_ns for row in live for sample in row.samples)
    if walls:
        median = pooled_median(live)
        interval = (
            walls[0],
            median,
            walls[-1],
            int(sorted(abs(w - median) for w in walls)[len(walls) // 2]),
        )
    else:
        median, interval = 0, (0, 0, 0, 0)
    tenancy = live[0].tenancy if live else "unattested"
    silicon = bool(live) and all(row.silicon for row in live) and corpus.physical_targets() >= 2
    if measurement is None:
        repeats = sorted(len(row.samples) for row in rows) or [0]
        measurement = {
            "protocol": "warm-up excluded; one lap per repeat under OS counters; PMU when exposed",
            "repeats": [repeats[0], repeats[-1]],
            "outliers": "none discarded",
            "judge": "pooled median wall_ns",
        }
    scope = scope_for(
        module,
        target,
        theta,
        policy,
        workload=workload,
        measurement=measurement,
        objective={"name": "wall_ns", "artifact": "measured corpus"},
        budget={"samples": record.spent},
    )
    evidence = {
        "incumbent": True,
        "search_coverage": (len(measured) / len(census)) if census else 0.0,
        "stale": len(stale),
        "prediction_interval": interval if (walls and silicon) else None,
        "tenancy": tenancy,
    }
    klass, statement = certificate_class_allowed(scope, evidence)
    return MeasuredCertificate(
        scope.digest(),
        klass,
        statement,
        policy.name,
        plan,
        median,
        interval,
        coverage,
        len(corpus.episodes(program, target_key, workload_digest)),
        len(walls),
        tenancy,
        corpus.physical_targets(),
        corpus.head,
        DispatchRecord(
            record.decision, record.stop_reason, record.spent, record.bound_source, klass
        ),
    )


__all__ = [
    "DEFAULT_EXACT_BUDGET",
    "STOP_REASONS",
    "Bound",
    "ExactSchedule",
    "ExactSelection",
    "MeasuredCertificate",
    "PhaseSolution",
    "ScheduleCertificate",
    "SearchState",
    "SelectionCertificate",
    "certify_measured",
    "certify_selection",
    "SelectionSearchState",
    "certify_schedule",
    "exact_schedule",
    "exact_selection",
]
