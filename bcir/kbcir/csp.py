"""A dependency-free exact finite-domain solver: integer CSP / ILP by branch and bound (G28).

The 2026-10-06 audit (items 9 and 14) found no constraint or integer-programming solver in the
code: every exact rail was a special-purpose search (the schedule's node search, the layout's
placement search, the selection's enumeration), so a problem spanning two of them -- the joint
schedule x memory optimum -- had no exact rail at all. This module is that rail, at the scope
declared here and no wider.

**The model.** Integer variables over closed intervals, and three constraint families:

* `linear`: `sum(c * x) <= / >= / == rhs`, integer coefficients;
* `cumulative`: tasks `(start, duration)` of unit demand on `capacity` identical machines --
  at no instant do more than `capacity` of them run (the intervals are half-open);
* `no_overlap_2d`: rectangles `[x0 + a, x1 + b) x [y, y + h)` -- the two x ends each a
  variable plus a constant, the height a positive constant -- pairwise disjoint (an empty
  rectangle overlaps nothing). Lifetime x address, as a memory layout over a schedule is.

and a linear objective to minimize.

**The search.** Depth-first branch and bound on an explicit stack. At every node each
constraint narrows the variables' bounds to a fixpoint (bounds consistency on a linear row;
the compulsory-part timetable on a cumulative; the disjunction on a rectangle pair whose
other axis certainly overlaps); the incumbent's objective is a cut every node must beat; the
branch splits the narrowest open domain of the lowest declared priority at its midpoint, the
lower half first. A fully fixed
node is a solution only if `satisfied` -- the constraints' definitions, sharing no code with
the propagators -- agrees. `budget` counts nodes, never seconds; an exhausted budget returns
the incumbent with `stop_reason="budget"` and a lower bound valid over every unexplored node
(the least objective bound on the stack), never a claimed optimum. An incumbent handed in is
checked against the definitions first, so a legal answer exists at every interruption.

Exact and exponential: the rail for small regions, held to `brute_force` (every assignment
enumerated) on generated models. Not built: linear-relaxation bounds, cutting planes,
learned clauses, and any resumable state.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

STOP_REASONS = ("optimal", "infeasible", "budget")
DEFAULT_CSP_BUDGET = 200_000
#: `brute_force` enumerates at most this many assignments.
BRUTE_FORCE_LIMIT = 2_000_000
_OPS = ("<=", ">=", "==")


class CspError(ValueError):
    """A malformed model, or an incumbent that does not satisfy it."""


@dataclass(frozen=True)
class Linear:
    terms: tuple[tuple[int, int], ...]  # (coefficient, variable)
    op: str
    rhs: int


@dataclass(frozen=True)
class Cumulative:
    tasks: tuple[tuple[int, int], ...]  # (start variable, duration)
    capacity: int


@dataclass(frozen=True)
class NoOverlap2D:
    #: (x0 variable, x0 offset, x1 variable, x1 offset, y variable, height, group): rectangles
    #: of one group are parts of one object (its lifetime as a union of spans) and never
    #: constrain each other
    rects: tuple[tuple[int, int, int, int, int, int, int], ...]


def _int(value, what: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CspError(f"{what} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise CspError(f"{what} must be at least {minimum}, got {value}")
    return value


class Model:
    """A finite-domain model: variables, constraints and a linear objective (minimized)."""

    def __init__(self) -> None:
        self.lo: list[int] = []
        self.hi: list[int] = []
        self.names: list[str] = []
        self.priority: list[int] = []
        self.constraints: list = []
        self.objective: tuple[tuple[int, int], ...] = ()
        self.offset = 0

    def var(self, lo: int, hi: int, name: str = "", priority: int = 0) -> int:
        """A variable over `[lo, hi]`; the search branches on the lowest `priority` first
        (then the narrowest domain), so a model can say which decisions fix the others."""
        _int(lo, "a lower bound")
        _int(hi, "an upper bound")
        self.priority.append(_int(priority, "a branching priority"))
        if lo > hi:
            raise CspError(f"variable {name or len(self.lo)} has an empty domain [{lo}, {hi}]")
        self.lo.append(lo)
        self.hi.append(hi)
        self.names.append(name or f"x{len(self.lo) - 1}")
        return len(self.lo) - 1

    def _var(self, index) -> int:
        _int(index, "a variable")
        if not 0 <= index < len(self.lo):
            raise CspError(f"no variable {index}")
        return index

    def _terms(self, terms) -> tuple[tuple[int, int], ...]:
        merged: dict[int, int] = {}
        for coef, index in terms:
            merged[self._var(index)] = merged.get(index, 0) + _int(coef, "a coefficient")
        return tuple((c, v) for v, c in sorted(merged.items()) if c)

    def linear(self, terms, op: str, rhs: int) -> None:
        if op not in _OPS:
            raise CspError(f"a linear constraint's relation is one of {_OPS}, got {op!r}")
        self.constraints.append(Linear(self._terms(terms), op, _int(rhs, "a right-hand side")))

    def cumulative(self, tasks, capacity: int) -> None:
        rows = tuple((self._var(s), _int(d, "a duration", 0)) for s, d in tasks)
        self.constraints.append(Cumulative(rows, _int(capacity, "a capacity", 1)))

    def no_overlap_2d(self, rects) -> None:
        """Rectangles `(x0, a, x1, b, y, h[, group])`; without a group each is its own. The
        rectangles of one group must share their `y` and `h` (one object, several spans)."""
        rows = []
        shape: dict[int, tuple[int, int]] = {}
        for index, rect in enumerate(rects):
            if len(rect) not in (6, 7):
                raise CspError("a rectangle is (x0, a, x1, b, y, h) or (x0, a, x1, b, y, h, group)")
            x0, a, x1, b, y, h = rect[:6]
            group = _int(rect[6], "a group") if len(rect) == 7 else -1 - index
            row = (
                self._var(x0),
                _int(a, "an offset"),
                self._var(x1),
                _int(b, "an offset"),
                self._var(y),
                _int(h, "a height", 1),
                group,
            )
            if shape.setdefault(group, (row[4], row[5])) != (row[4], row[5]):
                raise CspError(f"the rectangles of group {group} do not share one y and height")
            rows.append(row)
        self.constraints.append(NoOverlap2D(tuple(rows)))

    def minimize(self, terms, offset: int = 0) -> None:
        self.objective = self._terms(terms)
        self.offset = _int(offset, "an objective offset")

    def value(self, values) -> int:
        return self.offset + sum(c * values[v] for c, v in self.objective)


# --- the definitions (no propagation code) -----------------------------------------------------


def satisfied(model: Model, values) -> bool:
    """Whether `values` (one integer per variable) satisfies every domain and constraint,
    from the constraints' definitions alone."""
    if len(values) != len(model.lo):
        return False
    if any(not lo <= x <= hi for x, lo, hi in zip(values, model.lo, model.hi)):
        return False
    for con in model.constraints:
        if isinstance(con, Linear):
            total = sum(c * values[v] for c, v in con.terms)
            if (
                (con.op == "<=" and total > con.rhs)
                or (con.op == ">=" and total < con.rhs)
                or (con.op == "==" and total != con.rhs)
            ):
                return False
        elif isinstance(con, Cumulative):
            spans = [(values[s], values[s] + d) for s, d in con.tasks if d > 0]
            for t, _end in spans:
                if sum(1 for a, b in spans if a <= t < b) > con.capacity:
                    return False
        else:
            boxes = [
                (values[x0] + a, values[x1] + b, values[y], values[y] + h, g)
                for x0, a, x1, b, y, h, g in con.rects
            ]
            for (p0, p1, q0, q1, g), (r0, r1, s0, s1, k) in itertools.combinations(boxes, 2):
                if g == k:
                    continue
                if p0 < p1 and r0 < r1 and p0 < r1 and r0 < p1 and q0 < s1 and s0 < q1:
                    return False
    return True


def brute_force(model: Model):
    """(objective, values) of the optimum by enumerating every assignment, or (None, None)
    when there is none -- the reference `solve` is held to. Refused over BRUTE_FORCE_LIMIT."""
    size = 1
    for lo, hi in zip(model.lo, model.hi):
        size *= hi - lo + 1
    if size > BRUTE_FORCE_LIMIT:
        raise CspError(f"brute force enumerates at most {BRUTE_FORCE_LIMIT} assignments")
    best = None
    for values in itertools.product(*(range(lo, hi + 1) for lo, hi in zip(model.lo, model.hi))):
        if satisfied(model, values):
            obj = model.value(values)
            if best is None or obj < best[0]:
                best = (obj, values)
    return best if best is not None else (None, None)


# --- propagation -------------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -((-a) // b)


def _linear_le(terms, rhs, lo, hi) -> bool | None:
    """Bounds consistency on `sum(c x) <= rhs`: None when infeasible, else whether a bound
    moved (the protocol of every propagator here)."""
    mins = [c * lo[v] if c > 0 else c * hi[v] for c, v in terms]
    total = sum(mins)
    if total > rhs:
        return None
    moved = False
    for (c, v), m in zip(terms, mins):
        slack = rhs - (total - m)
        if c > 0:
            bound = slack // c
            if bound < hi[v]:
                hi[v] = bound
                moved = True
        else:
            bound = _ceil_div(slack, c)
            if bound > lo[v]:
                lo[v] = bound
                moved = True
        if lo[v] > hi[v]:
            return None
    return moved


def _linear(con: Linear, lo, hi) -> bool | None:
    moved = False
    if con.op in ("<=", "=="):
        r = _linear_le(con.terms, con.rhs, lo, hi)
        if r is None:
            return None
        moved |= r
    if con.op in (">=", "=="):
        r = _linear_le(tuple((-c, v) for c, v in con.terms), -con.rhs, lo, hi)
        if r is None:
            return None
        moved |= r
    return moved


def _cumulative(con: Cumulative, lo, hi) -> bool | None:
    """The compulsory-part timetable: a task's [latest start, earliest end) is occupied in every
    solution; no start may cover an instant the other tasks' compulsory parts already fill."""
    parts = []
    for index, (s, d) in enumerate(con.tasks):
        if d > 0 and hi[s] < lo[s] + d:
            parts.append((index, hi[s], lo[s] + d))
    profile: dict[int, int] = {}
    for _owner, a, b in parts:
        for t in range(a, b):
            profile[t] = profile.get(t, 0) + 1
    if any(count > con.capacity for count in profile.values()):
        return None
    moved = False
    for index, (s, d) in enumerate(con.tasks):
        if d == 0:
            continue
        load = dict(profile)
        for owner, a, b in parts:
            if owner == index:  # a task's own compulsory part never blocks it
                for t in range(a, b):
                    load[t] -= 1
        full = {t for t, count in load.items() if count >= con.capacity}
        if not full:
            continue
        t = lo[s]
        while t <= hi[s]:
            hit = [u for u in range(t, t + d) if u in full]
            if not hit:
                break
            t = max(hit) + 1
        if t > hi[s]:
            return None
        if t > lo[s]:
            lo[s] = t
            moved = True
        t = hi[s]
        while t >= lo[s]:
            hit = [u for u in range(t, t + d) if u in full]
            if not hit:
                break
            t = min(hit) - d
        if t < lo[s]:
            return None
        if t < hi[s]:
            hi[s] = t
            moved = True
    return moved


def _before(v_end, end_off, v_start, start_off, lo, hi) -> bool:
    """Whether `v_end + end_off <= v_start + start_off` is still possible."""
    return lo[v_end] + end_off <= hi[v_start] + start_off


def _enforce_before(v_end, end_off, v_start, start_off, lo, hi) -> bool:
    """Narrow to `v_end + end_off <= v_start + start_off`; whether a bound moved."""
    moved = False
    cap = hi[v_start] + start_off - end_off
    if cap < hi[v_end]:
        hi[v_end] = cap
        moved = True
    floor = lo[v_end] + end_off - start_off
    if floor > lo[v_start]:
        lo[v_start] = floor
        moved = True
    return moved


def _no_overlap_2d(con: NoOverlap2D, lo, hi) -> bool | None:
    # energy: the objects certainly live at one instant need their summed heights inside the
    # address window their bounds leave them
    certain = []
    for x0, a, x1, b, y, h, g in con.rects:
        start, end = hi[x0] + a, lo[x1] + b
        if start < end:
            certain.append((start, end, y, h, g))
    for t, _end, _y, _h, _g in certain:
        live = {g: (y, h) for s, e, y, h, g in certain if s <= t < e}
        if len(live) > 1:
            low = min(lo[y] for y, _h in live.values())
            high = max(hi[y] + h for y, h in live.values())
            if sum(h for _y, h in live.values()) > high - low:
                return None
    moved = False
    for (ax0, aa, ax1, ab, ay, ah, ag), (bx0, ba, bx1, bb, by, bh, bg) in itertools.combinations(
        con.rects, 2
    ):
        if ag == bg:
            continue
        x_overlap = (
            hi[ax0] + aa < lo[bx1] + bb
            and hi[bx0] + ba < lo[ax1] + ab
            and hi[ax0] + aa < lo[ax1] + ab
            and hi[bx0] + ba < lo[bx1] + bb
        )
        if x_overlap:  # then the addresses must be disjoint: a below b, or b below a
            below = lo[ay] + ah <= hi[by]
            above = lo[by] + bh <= hi[ay]
            if not below and not above:
                return None
            if below and not above:
                moved |= _enforce_before(ay, ah, by, 0, lo, hi)
            elif above and not below:
                moved |= _enforce_before(by, bh, ay, 0, lo, hi)
        y_overlap = hi[ay] < lo[by] + bh and hi[by] < lo[ay] + ah
        if y_overlap:  # then the lifetimes must be disjoint (or one of them empty)
            a_first = _before(ax1, ab, bx0, ba, lo, hi)
            b_first = _before(bx1, bb, ax0, aa, lo, hi)
            a_empty = lo[ax1] + ab <= hi[ax0] + aa
            b_empty = lo[bx1] + bb <= hi[bx0] + ba
            if not (a_first or b_first or a_empty or b_empty):
                return None
            if a_first and not (b_first or a_empty or b_empty):
                moved |= _enforce_before(ax1, ab, bx0, ba, lo, hi)
            elif b_first and not (a_first or a_empty or b_empty):
                moved |= _enforce_before(bx1, bb, ax0, aa, lo, hi)
        for v in (ax0, ax1, ay, bx0, bx1, by):
            if lo[v] > hi[v]:
                return None
    return moved


def _propagate(model: Model, lo: list[int], hi: list[int], cut: Linear | None) -> bool:
    """Narrow `lo`/`hi` in place to a fixpoint of every constraint (and the objective cut);
    False when the node is infeasible."""
    rows = list(model.constraints)
    if cut is not None:
        rows.append(cut)
    while True:
        moved = False
        for con in rows:
            if isinstance(con, Linear):
                r = _linear(con, lo, hi)
            elif isinstance(con, Cumulative):
                r = _cumulative(con, lo, hi)
            else:
                r = _no_overlap_2d(con, lo, hi)
            if r is None:
                return False
            moved |= r
        if not moved:
            return True


# --- the search --------------------------------------------------------------------------------


@dataclass(frozen=True)
class CspResult:
    """The solver's answer: the best assignment found (None when there is none), its
    objective, a lower bound valid over the whole model, why the search stopped and the
    nodes it expanded."""

    values: tuple[int, ...] | None
    objective: int | None
    lower_bound: int | None
    stop_reason: str
    nodes: int

    @property
    def optimal(self) -> bool:
        return self.stop_reason == "optimal"


def _bound(model: Model, lo, hi) -> int:
    return model.offset + sum(c * lo[v] if c > 0 else c * hi[v] for c, v in model.objective)


def solve(model: Model, budget: int = DEFAULT_CSP_BUDGET, incumbent=None) -> CspResult:
    """Minimize `model`'s objective by branch and bound (module docstring). `incumbent`, a
    known solution, must satisfy the model and seeds the search."""
    _int(budget, "the search budget", 0)
    best = None
    if incumbent is not None:
        values = tuple(incumbent)
        if not satisfied(model, values):
            raise CspError("the incumbent does not satisfy the model")
        best = (model.value(values), values)
    stack = [(list(model.lo), list(model.hi))]
    nodes = 0
    while stack:
        if nodes >= budget:
            bounds = [_bound(model, lo, hi) for lo, hi in stack]
            floor = min(bounds)
            if best is not None:
                floor = min(floor, best[0])
            return CspResult(
                best[1] if best else None, best[0] if best else None, floor, "budget", nodes
            )
        lo, hi = stack.pop()
        nodes += 1
        cut = (
            Linear(model.objective, "<=", best[0] - 1 - model.offset) if best is not None else None
        )
        if not _propagate(model, lo, hi, cut):
            continue
        open_vars = [v for v in range(len(lo)) if lo[v] < hi[v]]
        if not open_vars:
            values = tuple(lo)
            if not satisfied(model, values):  # the propagators admitted a non-solution
                raise CspError(f"internal: propagation accepted {values}, which is not a solution")
            obj = model.value(values)
            if best is None or obj < best[0]:
                best = (obj, values)
            continue
        v = min(open_vars, key=lambda k: (model.priority[k], hi[k] - lo[k], k))
        mid = (lo[v] + hi[v]) // 2
        upper_lo, upper_hi = list(lo), list(hi)
        upper_lo[v] = mid + 1
        hi[v] = mid
        stack.append((upper_lo, upper_hi))
        stack.append((lo, hi))
    if best is None:
        return CspResult(None, None, None, "infeasible", nodes)
    return CspResult(best[1], best[0], best[0], "optimal", nodes)


__all__ = [
    "BRUTE_FORCE_LIMIT",
    "DEFAULT_CSP_BUDGET",
    "STOP_REASONS",
    "CspError",
    "CspResult",
    "Cumulative",
    "Linear",
    "Model",
    "NoOverlap2D",
    "brute_force",
    "satisfied",
    "solve",
]
