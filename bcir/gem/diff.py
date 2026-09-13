"""The plan and certificate diff (G12 / S2-C): what a residual gap is made of.

The per-slice analysis protocol asks what changed between two plans under one scope -- which
claims moved streams or times, which candidates were re-selected, which resources were laid
out elsewhere, and whether the bound tightened -- the regret ledger (`kbcir.regret`)
generalized from one number to a structured comparison. `plan_diff` answers it over two
`ExecutionPlan`s (the plan as bytes, G11) or two `GemSchedule`s, with two static-memory plans
and two certificates optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Move:
    claim_id: int
    before: tuple[int, int, int]  # (stream, start, duration)
    after: tuple[int, int, int]


@dataclass(frozen=True)
class Reselection:
    claim_id: int
    before: tuple[str, int]  # (candidate, width)
    after: tuple[str, int]


@dataclass(frozen=True)
class Repricing:
    """The same candidate, a different step cost -- the context coupling moved."""

    claim_id: int
    before: int
    after: int


@dataclass(frozen=True)
class Relayout:
    rid: int
    before: tuple[str, int, int]  # (bank, offset, size)
    after: tuple[str, int, int]


@dataclass
class PlanDiff:
    """The structured difference: every move, re-selection and re-layout, the makespans, the
    bounds when certificates were given, and the regret (how much the second plan lost or won
    against the first, in makespan units; negative is a win)."""

    moved: list[Move] = field(default_factory=list)
    reselected: list[Reselection] = field(default_factory=list)
    repriced: list[Repricing] = field(default_factory=list)
    relaid: list[Relayout] = field(default_factory=list)
    added: list[int] = field(default_factory=list)  # claims only the second plan places
    removed: list[int] = field(default_factory=list)  # claims only the first plan places
    makespan: tuple[int, int] = (0, 0)
    lower_bound: tuple[int, int] | None = None

    @property
    def regret(self) -> int:
        return self.makespan[1] - self.makespan[0]

    @property
    def bound_tightened(self) -> int | None:
        return None if self.lower_bound is None else self.lower_bound[1] - self.lower_bound[0]

    @property
    def empty(self) -> bool:
        return not (
            self.moved
            or self.reselected
            or self.repriced
            or self.relaid
            or self.added
            or self.removed
        )

    def to_dict(self) -> dict:
        return {
            "moved": [
                {"claim": m.claim_id, "before": list(m.before), "after": list(m.after)}
                for m in self.moved
            ],
            "reselected": [
                {"claim": r.claim_id, "before": list(r.before), "after": list(r.after)}
                for r in self.reselected
            ],
            "repriced": [
                {"claim": r.claim_id, "before": r.before, "after": r.after} for r in self.repriced
            ],
            "relaid": [
                {"rid": r.rid, "before": list(r.before), "after": list(r.after)}
                for r in self.relaid
            ],
            "added": list(self.added),
            "removed": list(self.removed),
            "makespan": list(self.makespan),
            "regret": self.regret,
            "lower_bound": None if self.lower_bound is None else list(self.lower_bound),
            "bound_tightened": self.bound_tightened,
        }


def _placements(plan) -> tuple[dict[int, tuple[int, int, int]], int]:
    """claim id -> (stream, start, duration), and the makespan, from a plan or a schedule."""
    steps = getattr(plan, "steps", None)
    if steps is not None:  # an ExecutionPlan
        return {
            step.claim_id: (step.stream, step.start, step.duration) for step in steps
        }, plan.makespan
    slots = getattr(plan, "slots", None)
    if slots is not None:  # a GemSchedule
        return {
            slot.claim_id: (slot.domain, slot.start, slot.finish - slot.start) for slot in slots
        }, plan.makespan
    raise TypeError("plan_diff wants ExecutionPlans or GemSchedules")


def _selections(plan) -> dict[int, tuple[str, int]]:
    steps = getattr(plan, "steps", None)
    if steps is None:
        return {}
    return {step.claim_id: (step.candidate, step.width) for step in steps}


def _costs(plan) -> dict[int, int]:
    steps = getattr(plan, "steps", None)
    if steps is None:
        return {}
    return {step.claim_id: step.cost for step in steps}


def _layouts(memory) -> dict[int, tuple[str, int, int]]:
    if memory is None:
        return {}
    return {row.rid: (row.bank, row.offset, row.size_bytes) for row in memory.allocations}


def plan_diff(before, after, *, memory=None, certificates=None) -> PlanDiff:
    """Compare two plans (ExecutionPlans or GemSchedules). `memory` is an optional pair of
    static-memory plans, `certificates` an optional pair of schedule certificates (their
    lower bounds are compared)."""
    first, makespan_before = _placements(before)
    second, makespan_after = _placements(after)
    diff = PlanDiff(makespan=(makespan_before, makespan_after))
    for cid in sorted(first):
        if cid not in second:
            diff.removed.append(cid)
        elif first[cid] != second[cid]:
            diff.moved.append(Move(cid, first[cid], second[cid]))
    diff.added = sorted(cid for cid in second if cid not in first)
    chosen_before, chosen_after = _selections(before), _selections(after)
    for cid in sorted(chosen_before):
        if cid in chosen_after and chosen_before[cid] != chosen_after[cid]:
            diff.reselected.append(Reselection(cid, chosen_before[cid], chosen_after[cid]))
    cost_before, cost_after = _costs(before), _costs(after)
    for cid in sorted(cost_before):
        if cid in cost_after and cost_before[cid] != cost_after[cid]:
            diff.repriced.append(Repricing(cid, cost_before[cid], cost_after[cid]))
    if memory is not None:
        rows_before, rows_after = _layouts(memory[0]), _layouts(memory[1])
        for rid in sorted(rows_before):
            if rid in rows_after and rows_before[rid] != rows_after[rid]:
                diff.relaid.append(Relayout(rid, rows_before[rid], rows_after[rid]))
    if certificates is not None:
        diff.lower_bound = (certificates[0].lower_bound, certificates[1].lower_bound)
    return diff


__all__ = ["Move", "PlanDiff", "Relayout", "Repricing", "Reselection", "plan_diff"]
