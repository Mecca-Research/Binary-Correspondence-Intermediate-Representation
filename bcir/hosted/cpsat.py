"""An optional hosted adapter: a `kbcir.csp.Model` solved by OR-Tools CP-SAT (G28).

Opt-in and import-quarantined like every `bcir.hosted` module: nothing in the oracle imports
it, and `available()` says whether `ortools` is installed. The adapter translates the model
one-for-one -- integer variables, linear rows, a cumulative of unit demands, and every pair of
rectangles of distinct groups as the disjunction "disjoint in x, or disjoint in y, or one of
them empty" (enforced literals, so a span's emptiness is the model's, not a solver default) --
and is held to the in-tree solver: the same optimum, or the same infeasibility, on the
models the in-tree rail is held to brute force on. It never decides a plan BCIR emits; the
in-tree solver is the rail, this is its cross-check.
"""

from __future__ import annotations

import importlib.util
import itertools

from ..kbcir.csp import Cumulative, CspResult, Linear, Model


def available() -> bool:
    return importlib.util.find_spec("ortools") is not None


def solve_cpsat(model: Model, *, seconds: float = 30.0, workers: int = 1) -> CspResult:
    """Solve `model` with CP-SAT. Raises ModuleNotFoundError without `ortools`; a time-out
    before a proof is a `budget` stop with CP-SAT's bound."""
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()
    xs = [m.new_int_var(lo, hi, name) for lo, hi, name in zip(model.lo, model.hi, model.names)]

    def expr(terms):
        return sum(c * xs[v] for c, v in terms)

    for con in model.constraints:
        if isinstance(con, Linear):
            e = expr(con.terms)
            if con.op == "<=":
                m.add(e <= con.rhs)
            elif con.op == ">=":
                m.add(e >= con.rhs)
            else:
                m.add(e == con.rhs)
        elif isinstance(con, Cumulative):
            intervals = []
            for s, d in con.tasks:
                if d > 0:
                    end = m.new_int_var(model.lo[s] + d, model.hi[s] + d, "")
                    intervals.append(m.new_interval_var(xs[s], d, end, ""))
            if intervals:
                m.add_cumulative(intervals, [1] * len(intervals), con.capacity)
        else:
            for (ax0, aa, ax1, ab, ay, ah, ag), (
                bx0,
                ba,
                bx1,
                bb,
                by,
                bh,
                bg,
            ) in itertools.combinations(con.rects, 2):
                if ag == bg:
                    continue
                options = [m.new_bool_var("") for _ in range(6)]
                m.add(xs[ax1] + ab <= xs[bx0] + ba).only_enforce_if(options[0])
                m.add(xs[bx1] + bb <= xs[ax0] + aa).only_enforce_if(options[1])
                m.add(xs[ay] + ah <= xs[by]).only_enforce_if(options[2])
                m.add(xs[by] + bh <= xs[ay]).only_enforce_if(options[3])
                m.add(xs[ax1] + ab <= xs[ax0] + aa).only_enforce_if(options[4])
                m.add(xs[bx1] + bb <= xs[bx0] + ba).only_enforce_if(options[5])
                m.add_bool_or(options)
    m.minimize(model.offset + expr(model.objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_workers = workers
    status = solver.solve(m)
    if status == cp_model.INFEASIBLE:
        return CspResult(None, None, None, "infeasible", int(solver.num_branches))
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return CspResult(None, None, None, "budget", int(solver.num_branches))
    values = tuple(int(solver.value(x)) for x in xs)
    objective = model.value(values)
    if status == cp_model.OPTIMAL:
        return CspResult(values, objective, objective, "optimal", int(solver.num_branches))
    bound = int(solver.best_objective_bound)
    return CspResult(values, objective, bound, "budget", int(solver.num_branches))


__all__ = ["available", "solve_cpsat"]
