"""K_BCIR constrained selection (RCSP rail): min M(pi) s.t. R(pi, Theta) <= B(H, Theta).

The scalarized tropical optimizer (`realize.optimize`) is the *degenerate case* of
the LangRef central equation: no budgets, serial composition. This module is the
general selection rail: additive resource dimensions (thermal, power, memory, ...)
carry hard caps B, labels accumulate (score, resources) along the same layered DAG
of legal candidates, dominated labels are pruned, and infeasible extensions are cut.

A weighted-sum scalarization can only reach the convex hull of the Pareto front
(Geoffrion); the label DP reaches every non-dominated plan and turns thermal/power
from preferences into constraints -- a hot machine makes wide SIMD *infeasible*,
not merely expensive. All arithmetic stays integer/Q8 and the result is
deterministic (stable candidate order, first-label tie-breaks), so with an
unbounded budget this rail reproduces `optimize` exactly (the pinned scores hold).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Module
from .cost import DIMS, Theta, _INDEX
from .realize import (
    Candidate,
    ChosenStep,
    RealizationResult,
    _context_factor,
    _flatten,
    candidates_for,
)
from .weights import PERF, Policy, weights


class Infeasible(Exception):
    """No legal realization satisfies the budget (LangRef R9: plan legality)."""


@dataclass(frozen=True)
class Budget:
    """B(H, Theta): hard caps on additive resource dimensions of the plan.

    Caps are (dimension index, cap) pairs over `cost.DIMS`, accumulated over the
    *coupled* per-step cost vectors R(pi, Theta) = sum_i T_i (X) f_i(pi).
    """

    caps: tuple[tuple[int, int], ...] = ()

    @staticmethod
    def of(**named: int) -> "Budget":
        caps = []
        for name, cap in named.items():
            if name not in _INDEX:
                raise KeyError(f"unknown cost dim {name!r}")
            caps.append((_INDEX[name], int(cap)))
        return Budget(tuple(sorted(caps)))

    @staticmethod
    def unbounded() -> "Budget":
        return Budget(())

    @property
    def dims(self) -> tuple[int, ...]:
        return tuple(d for d, _ in self.caps)

    def names(self) -> tuple[str, ...]:
        return tuple(DIMS[d] for d, _ in self.caps)


@dataclass
class _Label:
    """A non-dominated partial plan: scalar score + accumulated tracked resources."""

    score: int
    res: tuple[int, ...]
    cand: Candidate | None          # candidate realized at this column (None = source)
    parent: "_Label | None"


def _dominates(a: _Label, b: _Label) -> bool:
    return a.score <= b.score and all(x <= y for x, y in zip(a.res, b.res))


def _insert(labels: list[_Label], new: _Label) -> None:
    """Keep `labels` a non-dominated set (stable order; ties keep the first)."""
    for ex in labels:
        if _dominates(ex, new):
            return
    labels[:] = [ex for ex in labels if not _dominates(new, ex)]
    labels.append(new)


def _expand(module: Module, h, theta: Theta, policy: Policy,
            dims: tuple[int, ...], caps: dict[int, int]):
    """Run the label DP over the layered candidate DAG; return the sink labels
    paired with their (phase_id, claim) trail for reconstruction."""
    flat = _flatten(module)
    prev: list[_Label] = [_Label(0, (0,) * len(dims), None, None)]
    trail: list[tuple[int, object]] = []

    for phase_id, claim in flat:
        trail.append((phase_id, claim))
        w_phase = weights(h, theta, phase_id, policy)
        cost_rid = claim.rd[0] if claim.rd else claim.primary_rid
        resource = module.resource(cost_rid) if cost_rid is not None else None

        column: dict[int, list[_Label]] = {}
        for ci, cand in enumerate(candidates_for(claim, h, resource)):
            slot = column.setdefault(ci, [])
            for lab in prev:
                factor = _context_factor(lab.cand, cand, theta)
                coupled = cand.base.couple(factor)
                res = tuple(lab.res[i] + coupled.v[d] for i, d in enumerate(dims))
                if any(r > caps[d] for r, d in zip(res, dims) if d in caps):
                    continue  # infeasible extension: R(pi) would exceed B
                _insert(slot, _Label(lab.score + coupled.dot(w_phase), res, cand, lab))

        prev = [lab for ci in sorted(column) for lab in column[ci]]
        if not prev:
            raise Infeasible(
                f"claim {claim.id}: no legal realization satisfies the budget "
                f"{{{', '.join(f'{DIMS[d]}<={caps[d]}' for d in dims if d in caps)}}}"
            )
    return prev, trail


def _reconstruct(label: _Label, trail) -> RealizationResult:
    chain: list[_Label] = []
    node = label
    while node.parent is not None:
        chain.append(node)
        node = node.parent
    chain.reverse()
    steps = [
        ChosenStep(claim.id, phase_id, lab.cand, lab.score - (lab.parent.score if lab.parent else 0))
        for (phase_id, claim), lab in zip(trail, chain)
    ]
    return RealizationResult(steps, label.score)


def optimize_constrained(module: Module, h, theta: Theta, policy: Policy = PERF,
                         budget: Budget = Budget.unbounded()) -> RealizationResult:
    """Constrained K_BCIR selection: min score subject to R(pi, Theta) <= budget.

    With `Budget.unbounded()` this is exactly `realize.optimize` (the degenerate
    case of the central equation). Raises `Infeasible` when no legal plan fits.
    """
    if not _flatten(module):
        return RealizationResult([], 0)
    dims = budget.dims
    caps = dict(budget.caps)
    sink, trail = _expand(module, h, theta, policy, dims, caps)
    best = sink[0]
    for lab in sink[1:]:
        if (lab.score, lab.res) < (best.score, best.res):
            best = lab
    return _reconstruct(best, trail)


def pareto_plans(module: Module, h, theta: Theta, policy: Policy = PERF,
                 dims: tuple[str, ...] = ("thermal", "power")) -> list[RealizationResult]:
    """The Pareto front over (score, *dims): every non-dominated plan, by rising score.

    Reaches the non-convex points that no weight vector w can select under the
    scalarized rail (the front is exact over the legal candidate DAG).
    """
    if not _flatten(module):
        return [RealizationResult([], 0)]
    for name in dims:
        if name not in _INDEX:
            raise KeyError(f"unknown cost dim {name!r}")
    didx = tuple(_INDEX[name] for name in dims)
    sink, trail = _expand(module, h, theta, policy, didx, {})
    front: list[_Label] = []
    for lab in sorted(sink, key=lambda l: (l.score, l.res)):
        _insert(front, lab)
    return [_reconstruct(lab, trail) for lab in front]
