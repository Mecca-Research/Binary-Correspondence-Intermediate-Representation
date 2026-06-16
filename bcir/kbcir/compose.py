"""Compositional semantics: functions/calls, control flow, dynamic shapes.

The K_BCIR central equation is already a **constrained series-parallel** optimization
(LangRef §2): `min_pi M(pi, Theta)` over a series-parallel composition. The straight-line
planner (`realize.optimize`) handles one parallel block of claims (a phase); this module
extends planning *along that same grain* to a **region tree** -- the frontier past
straight-line array kernels:

  * `Leaf(claims)`        -- one parallel block of straight-line claims (planned by `optimize`).
  * `Seq(parts...)`       -- sequential composition: cost is the **sum** of the parts.
  * `Cond(pred, t, e)`    -- control flow: a worst-case **max** over the two branches (a hard
                            latency bound) and a probability-weighted **expected** cost.
  * `Call(fn, arg_map)`   -- function reuse: plan the callee's region with the caller's actual
                            resources substituted for its formals (inline-substitution, so the
                            planner sees the real graph; bounded recursion-free depth).

`Function` is a named region. **Dynamic shapes** are a per-claim contract: a claim marked
`dynamic` carries `count` as a *static upper bound*, so its leaf cost is the worst case and
the plan is valid for any actual size <= the bound (`plan_holds_for`). Everything is
deterministic integer and reuses the existing optimizer for the leaves, so the pinned
straight-line scores are unchanged -- a `Leaf([vector_add])` plans to exactly 7808.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Union

from ..model import Claim, Module, Phase, Resource
from .cost import HProfile, Theta
from .realize import RealizationResult, optimize
from .weights import PERF, Policy

# A small fixed cost (in score units) for evaluating a branch predicate -- the control-flow
# overhead the straight-line model does not have. Deterministic; documented, not tuned.
PRED_COST = 8


# --- the region tree -------------------------------------------------------------

@dataclass(frozen=True)
class Leaf:
    """One parallel block of straight-line claims (a phase), planned by `optimize`."""
    claims: tuple[Claim, ...]


@dataclass(frozen=True)
class Seq:
    """Sequential composition: the parts run in order; the cost is their sum."""
    parts: tuple["Region", ...]


@dataclass(frozen=True)
class Cond:
    """Control flow: `if pred then then_ else else_`. Only one branch runs, so the hard
    bound is the max over branches; the expected cost weights by `prob_then_milli`/1000."""
    pred: str
    then_: "Region"
    else_: "Region"
    prob_then_milli: int = 500


@dataclass(frozen=True)
class Call:
    """Invoke a function, substituting the caller's actual RIDs for the callee's formals
    (`arg_map`: formal_rid -> actual_rid). Inline-substitution: the planner sees the real
    expanded graph. Recursion is rejected (bounded compile time)."""
    fn: str
    arg_map: tuple[tuple[int, int], ...] = ()


Region = Union[Leaf, Seq, Cond, Call]


@dataclass(frozen=True)
class Function:
    """A named region -- define once, `Call` many."""
    name: str
    region: Region


@dataclass(frozen=True)
class CompositeResult:
    """The compositional cost of a region: a hard worst-case bound (max over branches) and
    a probability-weighted expected cost, plus the planned-leaf count (a compile-time bound
    witness -- finite iff there are no recursive calls)."""
    worst_cost: int
    expected_cost: int
    leaves: int

    @property
    def branch_spread(self) -> int:
        return self.worst_cost - self.expected_cost


# --- compositional planning ------------------------------------------------------

def _substitute(claim: Claim, amap: dict[int, int]) -> Claim:
    rd = tuple(amap.get(r, r) for r in claim.rd)
    wr = tuple(amap.get(r, r) for r in claim.wr)
    return replace(claim, rd=rd, wr=wr)


def _leaf_module(claims: tuple[Claim, ...], resources: dict[int, Resource]) -> Module:
    """A flat single-phase Module for a leaf's claims (only the resources it touches)."""
    m = Module(name="leaf")
    touched = {rid for c in claims for rid in (tuple(c.rd) + tuple(c.wr))}
    for rid in sorted(touched):
        res = resources.get(rid)
        m.add_resource(res if res is not None else Resource(rid=rid, shape=(1,)))
    m.add_phase(Phase(phase_id=0, deps=(), claims=list(claims)))
    return m


def plan_composite(region: Region, functions: dict[str, Function],
                   resources: dict[int, Resource], h: HProfile, theta: Theta,
                   policy: Policy = PERF, *, _depth: int = 0,
                   _active: frozenset = frozenset()) -> CompositeResult:
    """Plan a region tree compositionally (series sum, branch max/expected, call inline).
    Reuses `optimize` for the leaves, so a single-block region prices exactly like the
    straight-line planner. Raises on recursion (bounded compile time)."""
    if _depth > 64:
        raise RecursionError("compose: region nesting too deep (>64)")

    if isinstance(region, Leaf):
        if not region.claims:
            return CompositeResult(0, 0, 0)
        score = optimize(_leaf_module(region.claims, resources), h, theta, policy).score
        return CompositeResult(score, score, 1)

    if isinstance(region, Seq):
        worst = expected = leaves = 0
        for part in region.parts:
            sub = plan_composite(part, functions, resources, h, theta, policy,
                                 _depth=_depth + 1, _active=_active)
            worst += sub.worst_cost
            expected += sub.expected_cost
            leaves += sub.leaves
        return CompositeResult(worst, expected, leaves)

    if isinstance(region, Cond):
        t = plan_composite(region.then_, functions, resources, h, theta, policy,
                           _depth=_depth + 1, _active=_active)
        e = plan_composite(region.else_, functions, resources, h, theta, policy,
                           _depth=_depth + 1, _active=_active)
        p = max(0, min(1000, region.prob_then_milli))
        expected = PRED_COST + (p * t.expected_cost + (1000 - p) * e.expected_cost) // 1000
        worst = PRED_COST + max(t.worst_cost, e.worst_cost)
        return CompositeResult(worst, expected, t.leaves + e.leaves)

    if isinstance(region, Call):
        if region.fn not in functions:
            raise KeyError(f"compose: call to undefined function {region.fn!r}")
        if region.fn in _active:
            raise RecursionError(f"compose: recursive call to {region.fn!r} (unbounded)")
        body = _inline(functions[region.fn].region, dict(region.arg_map))
        return plan_composite(body, functions, resources, h, theta, policy,
                              _depth=_depth + 1, _active=_active | {region.fn})

    raise TypeError(f"compose: unknown region node {type(region).__name__}")


def _inline(region: Region, amap: dict[int, int]) -> Region:
    """Substitute `amap` (formal -> actual RIDs) through a region's claims (and nested
    calls' arg maps), so a called function plans against the caller's actual resources."""
    if isinstance(region, Leaf):
        return Leaf(tuple(_substitute(c, amap) for c in region.claims))
    if isinstance(region, Seq):
        return Seq(tuple(_inline(p, amap) for p in region.parts))
    if isinstance(region, Cond):
        return Cond(region.pred, _inline(region.then_, amap), _inline(region.else_, amap),
                    region.prob_then_milli)
    if isinstance(region, Call):
        # compose the substitutions: the inner call's actuals are themselves remapped.
        inner = tuple((formal, amap.get(actual, actual)) for formal, actual in region.arg_map)
        return Call(region.fn, inner)
    raise TypeError(f"compose: unknown region node {type(region).__name__}")


# --- dynamic shapes --------------------------------------------------------------

def plan_holds_for(claim: Claim, actual_count: int) -> bool:
    """A `dynamic` claim's plan (priced at the upper bound `claim.count`) is valid for any
    actual size <= the bound -- the worst-case guarantee. A non-dynamic claim holds only at
    its exact declared count."""
    if claim.dynamic:
        return 0 <= actual_count <= claim.count
    return actual_count == claim.count


def worst_case_module(module: Module) -> Module:
    """Return `module` unchanged for planning: a `dynamic` claim already carries its count
    as the static upper bound, so `optimize` already prices the worst case. Provided as the
    explicit, documented entry point for dynamic-shape planning (a hard latency bound that
    holds for every actual size <= the declared bound)."""
    return module
