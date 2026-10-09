"""The joint schedule x memory optimum of a region, exactly, on the CSP rail (G28).

The parent planned a region in two steps -- a schedule first (earliest finish), then a layout
over that schedule's lifetimes -- and each step can be exact on its own while the pair is not:
the schedule that finishes first can keep two large values alive together that a schedule
one tick longer keeps apart, so `w_time x makespan + w_memory x extent` has an optimum neither
step sees (the audit's item 9). This module states the region as one model and solves it on
`kbcir.csp`:

* each task (a claim) a start in `[0, horizon - duration]`, after every task it must wait for
  (its hazard predecessors), at most `domains` running at any instant (`cumulative`);
* each value (a resource) live from the earliest start to the latest end of the tasks that
  touch it -- the union of `[s_i, s_j + d_j)` over every pair of them, one group of spans at
  one offset -- so that values live together are disjoint in memory (`no_overlap_2d` over
  lifetime x address);
* the makespan and the extent as variables bounding the ends and the tops, and
  `w_time x makespan + w_memory x extent` minimized.

`sequential_plan` is the parent's pipeline on the same instance -- the earliest-finish list
schedule in task order, then the exact layout over its lifetimes -- and seeds the search as
its incumbent, so the joint answer is never worse than the parent's and a legal plan exists
at every interruption. Offsets are in units of alignment one; the instance is one region (one
phase) and small: the rail is exact, and its budget counts nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .csp import DEFAULT_CSP_BUDGET, Model, solve
from .static_memory import LayoutItem, exact_layout


class JointError(ValueError):
    """A malformed joint instance."""


@dataclass(frozen=True)
class JointInstance:
    """One region: task durations, what each task touches, the hazard edges, the value sizes,
    the parallel domains and the objective's weights."""

    durations: dict[int, int]  # task -> duration (ticks)
    touches: dict[int, tuple[int, ...]]  # task -> the values it reads or writes
    preds: dict[int, tuple[int, ...]]  # task -> the tasks it must wait for
    sizes: dict[int, int]  # value -> size (units)
    domains: int = 1
    weights: tuple[int, int] = (1, 1)  # (per tick of makespan, per unit of extent)

    def __post_init__(self) -> None:
        if isinstance(self.domains, bool) or not isinstance(self.domains, int) or self.domains < 1:
            raise JointError("domains must be a positive integer")
        for tid, d in self.durations.items():
            if isinstance(d, bool) or not isinstance(d, int) or d < 1:
                raise JointError(f"task {tid} has no positive integer duration")
        for vid, size in self.sizes.items():
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise JointError(f"value {vid} has no positive integer size")
        for tid in (*self.touches, *self.preds):
            if tid not in self.durations:
                raise JointError(f"task {tid} has no duration")
        for tid, vids in self.touches.items():
            for vid in vids:
                if vid not in self.sizes:
                    raise JointError(f"task {tid} touches value {vid}, which has no size")
        for tid, ps in self.preds.items():
            for p in ps:
                if p not in self.durations or p == tid:
                    raise JointError(f"task {tid} waits for {p}, which is not another task")
        wt, wm = self.weights
        for w in (wt, wm):
            if isinstance(w, bool) or not isinstance(w, int) or w < 0:
                raise JointError("the weights are non-negative integers")

    @property
    def tasks(self) -> list[int]:
        return sorted(self.durations)

    @property
    def horizon(self) -> int:
        return sum(self.durations.values())

    def accessors(self, vid: int) -> list[int]:
        return [t for t in self.tasks if vid in self.touches.get(t, ())]

    @property
    def values(self) -> list[int]:
        return [v for v in sorted(self.sizes) if self.accessors(v)]


@dataclass(frozen=True)
class JointPlan:
    """A plan for a region: every start and offset, the makespan, the extent, the weighted
    objective, a lower bound on it, why the solver stopped and its work (nodes)."""

    starts: dict[int, int]
    offsets: dict[int, int]
    makespan: int
    extent: int
    objective: int
    lower_bound: int
    stop_reason: str
    nodes: int

    @property
    def optimal(self) -> bool:
        return self.stop_reason == "optimal"


def lifetimes(inst: JointInstance, starts: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Each value's half-open lifetime under `starts`: from the earliest start to the latest
    end of the tasks touching it."""
    out = {}
    for vid in inst.values:
        tasks = inst.accessors(vid)
        out[vid] = (
            min(starts[t] for t in tasks),
            max(starts[t] + inst.durations[t] for t in tasks),
        )
    return out


def evaluate(inst: JointInstance, starts: dict[int, int], offsets: dict[int, int]):
    """(makespan, extent, objective) of a plan, or None when it breaks a hazard, the domain
    count or the memory disjointness -- the plan's definition, independent of the solver."""
    for t, ps in inst.preds.items():
        if any(starts[t] < starts[p] + inst.durations[p] for p in ps):
            return None
    spans = [(starts[t], starts[t] + inst.durations[t]) for t in inst.tasks]
    if any(sum(1 for a, b in spans if a <= s < b) > inst.domains for s, _ in spans):
        return None
    life = lifetimes(inst, starts)
    vids = list(life)
    for i, a in enumerate(vids):
        for b in vids[i + 1 :]:
            (a0, a1), (b0, b1) = life[a], life[b]
            if a0 < b1 and b0 < a1:
                if (
                    offsets[a] < offsets[b] + inst.sizes[b]
                    and offsets[b] < offsets[a] + inst.sizes[a]
                ):
                    return None
    makespan = max((e for _s, e in spans), default=0)
    extent = max((offsets[v] + inst.sizes[v] for v in vids), default=0)
    wt, wm = inst.weights
    return makespan, extent, wt * makespan + wm * extent


def _layout(inst: JointInstance, starts: dict[int, int]) -> dict[int, int]:
    life = lifetimes(inst, starts)
    items = [LayoutItem(v, inst.sizes[v], 1, a, b) for v, (a, b) in life.items()]
    result = exact_layout(items, 1_000_000)
    if result.stop_reason != "optimal":  # pragma: no cover - regions are small
        raise JointError("the exact layout did not close on the sequential plan's lifetimes")
    return dict(result.offsets)


def sequential_plan(inst: JointInstance) -> JointPlan:
    """The parent's pipeline: the earliest-finish list schedule (tasks in id order, each at
    the earliest instant its predecessors are done and a domain is free), then the exact
    layout over that schedule's lifetimes."""
    starts: dict[int, int] = {}
    busy: list[tuple[int, int]] = []
    for t in inst.tasks:
        d = inst.durations[t]
        s = max((starts[p] + inst.durations[p] for p in inst.preds.get(t, ())), default=0)
        while _covers(busy, s, d, inst.domains):
            s += 1
        starts[t] = s
        busy.append((s, s + d))
    offsets = _layout(inst, starts)
    makespan, extent, objective = evaluate(inst, starts, offsets)
    return JointPlan(starts, offsets, makespan, extent, objective, 0, "heuristic", 0)


def _covers(busy, s: int, d: int, domains: int) -> bool:
    """Whether starting at `s` for `d` would put more than `domains` tasks at some instant."""
    return any(sum(1 for a, b in busy if a <= u < b) >= domains for u in range(s, s + d))


@dataclass(frozen=True)
class JointVars:
    """Where each part of a plan lives in `joint_model`'s variables."""

    start: dict[int, int]
    offset: dict[int, int]
    makespan: int
    extent: int


def joint_model(inst: JointInstance):
    """The CSP of the region (module docstring) and its `JointVars`."""
    m = Model()
    horizon = inst.horizon
    total = sum(inst.sizes[v] for v in inst.values)
    # the schedule first: fixed starts fix every lifetime, and the layout then propagates
    start = {t: m.var(0, horizon - inst.durations[t], f"s{t}", 0) for t in inst.tasks}
    makespan = m.var(0, horizon, "makespan", 1)
    extent = m.var(0, total, "extent", 3)
    for t in inst.tasks:
        m.linear([(1, makespan), (-1, start[t])], ">=", inst.durations[t])
        for p in inst.preds.get(t, ()):
            m.linear([(1, start[t]), (-1, start[p])], ">=", inst.durations[p])
    m.cumulative([(start[t], inst.durations[t]) for t in inst.tasks], inst.domains)
    offset, rects = {}, []
    for v in inst.values:
        size = inst.sizes[v]
        offset[v] = m.var(0, total - size, f"o{v}", 2)
        m.linear([(1, extent), (-1, offset[v])], ">=", size)
        # the lifetime [min start, max end) is the union of [s_i, s_j + d_j) over every pair
        # of the value's tasks: one group of spans at one offset
        tasks = inst.accessors(v)
        for i in tasks:
            for j in tasks:
                rects.append((start[i], 0, start[j], inst.durations[j], offset[v], size, v))
    if len(inst.values) > 1:
        m.no_overlap_2d(rects)
    wt, wm = inst.weights
    m.minimize([(wt, makespan), (wm, extent)])
    return m, JointVars(start, offset, makespan, extent)


def joint_optimum(inst: JointInstance, budget: int = DEFAULT_CSP_BUDGET) -> JointPlan:
    """The exact joint optimum on the CSP rail, seeded with `sequential_plan` -- never worse
    than the parent's pipeline; a budget stop returns the best plan found and a valid bound."""
    m, x = joint_model(inst)
    seed = sequential_plan(inst)
    values = [0] * len(m.lo)
    for t, var in x.start.items():
        values[var] = seed.starts[t]
    for v, var in x.offset.items():
        values[var] = seed.offsets[v]
    values[x.makespan], values[x.extent] = seed.makespan, seed.extent
    result = solve(m, budget, incumbent=values)
    starts = {t: result.values[var] for t, var in x.start.items()}
    offsets = {v: result.values[var] for v, var in x.offset.items()}
    span, top, objective = evaluate(inst, starts, offsets)
    return JointPlan(
        starts, offsets, span, top, objective, result.lower_bound, result.stop_reason, result.nodes
    )


def joint_instance(module, durations: dict[int, int], domains: int = 1, weights=(1, 1)):
    """The joint instance of a one-phase module: its claims as tasks (durations from the
    caller, e.g. `gem.schedule.durations_from`), their hazard predecessors as the edges
    (`gem.concurrency.hazard_predecessors`) and every resource they touch as a value of its
    byte size."""
    from ..gem.concurrency import hazard_predecessors

    if len(module.phases) != 1:
        raise JointError("a joint instance is one region: the module must have one phase")
    claims = sorted(module.phases[0].claims, key=lambda c: c.id)
    preds = hazard_predecessors(claims)
    touches = {c.id: tuple(sorted({*c.rd, *c.wr})) for c in claims}
    sizes = {}
    for vids in touches.values():
        for vid in vids:
            resource = module.resources[vid]
            sizes[vid] = resource.count * resource.elem_bytes
    return JointInstance(
        {c.id: durations[c.id] for c in claims},
        touches,
        {t: tuple(ps) for t, ps in preds.items() if ps},
        sizes,
        domains,
        tuple(weights),
    )


__all__ = [
    "JointError",
    "JointInstance",
    "JointPlan",
    "JointVars",
    "evaluate",
    "joint_instance",
    "joint_model",
    "joint_optimum",
    "lifetimes",
    "sequential_plan",
]
