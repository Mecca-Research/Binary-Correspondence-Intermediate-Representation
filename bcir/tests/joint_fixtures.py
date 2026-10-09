"""G28 fixtures, shared by `test_csp` and the GEM+ harness so both grade the same way:
generated finite-domain models held to brute force, generated joint schedule x memory regions
held to an independent reference (every start vector enumerated, each laid out by the proved
exact layout), and the parent's pipeline on the same regions."""

from __future__ import annotations

import itertools
import random

from bcir.kbcir.csp import Model, brute_force, solve
from bcir.kbcir.joint import JointInstance, evaluate, joint_optimum, lifetimes, sequential_plan
from bcir.kbcir.static_memory import LayoutItem, exact_layout

#: How many generated models and regions the rows read.
CSP_MODELS = 150
JOINT_REGIONS = 40


def random_model(seed: int) -> Model:
    """A small model: two to four variables, linear rows of every relation, and -- on some
    seeds -- a cumulative or a 2-D no-overlap over the same variables."""
    r = random.Random(seed)
    m = Model()
    n = r.randint(2, 4)
    xs = [m.var(r.randint(-2, 1), r.randint(2, 5)) for _ in range(n)]
    for _ in range(r.randint(1, 3)):
        terms = [(r.randint(-3, 3), x) for x in r.sample(xs, r.randint(1, n))]
        m.linear(terms, r.choice(("<=", ">=", "==")), r.randint(-4, 8))
    kind = r.random()
    if kind < 0.3:
        m.cumulative([(x, r.randint(1, 3)) for x in xs], r.randint(1, 2))
    elif kind < 0.55 and n >= 3:
        a, b, c = xs[:3]
        m.no_overlap_2d([(a, 0, a, r.randint(1, 3), b, r.randint(1, 2)), (c, 0, c, 2, b, 1)])
    m.minimize([(r.randint(-3, 3), x) for x in xs], r.randint(-5, 5))
    return m


def csp_disagreements(count: int = CSP_MODELS) -> int:
    """Models where the solver's verdict -- infeasible, or the optimum's objective -- is not
    brute force's, or its answer does not satisfy the model."""
    from bcir.kbcir.csp import satisfied

    bad = 0
    for seed in range(count):
        m = random_model(seed)
        want, _values = brute_force(m)
        got = solve(m)
        if want is None:
            bad += got.stop_reason != "infeasible"
        else:
            bad += not (
                got.optimal
                and got.objective == want
                and got.lower_bound == want
                and satisfied(m, got.values)
            )
    return bad


def random_region(seed: int) -> JointInstance:
    """Three or four tasks of one to three ticks, a random hazard DAG over them, two to four
    values of one to four units touched by one to three tasks each, one or two domains, and
    weights trading a tick against a unit."""
    r = random.Random(seed)
    n = r.randint(3, 4)
    tasks = list(range(1, n + 1))
    durations = {t: r.randint(1, 3) for t in tasks}
    preds = {}
    for j in tasks:
        ps = tuple(i for i in tasks if i < j and r.random() < 0.3)
        if ps:
            preds[j] = ps
    nvals = r.randint(2, 4)
    sizes = {100 + k: r.randint(1, 4) for k in range(nvals)}
    touches: dict[int, tuple[int, ...]] = {t: () for t in tasks}
    for vid in sizes:
        for t in r.sample(tasks, r.randint(1, 3)):
            touches[t] = tuple(sorted({*touches[t], vid}))
    return JointInstance(
        durations, touches, preds, sizes, r.randint(1, 2), (r.randint(1, 3), r.randint(1, 3))
    )


def joint_reference(inst: JointInstance) -> int:
    """The region's optimum objective by enumerating every start vector in the horizon that
    keeps the hazards and the domain count, each laid out by the exact layout over its
    lifetimes -- sharing no code with the CSP."""
    tasks = inst.tasks
    horizon = inst.horizon
    best = None
    layouts: dict = {}
    ranges = [range(0, horizon - inst.durations[t] + 1) for t in tasks]
    for combo in itertools.product(*ranges):
        starts = dict(zip(tasks, combo))
        if any(
            starts[t] < starts[p] + inst.durations[p] for t, ps in inst.preds.items() for p in ps
        ):
            continue
        spans = [(starts[t], starts[t] + inst.durations[t]) for t in tasks]
        if any(sum(1 for a, b in spans if a <= s < b) > inst.domains for s, _ in spans):
            continue
        life = lifetimes(inst, starts)
        key = tuple(sorted(life.items()))
        if key not in layouts:
            items = [LayoutItem(v, inst.sizes[v], 1, a, b) for v, (a, b) in life.items()]
            result = exact_layout(items, 1_000_000)
            assert result.stop_reason == "optimal"
            layouts[key] = result.extent
        makespan = max(e for _s, e in spans)
        wt, wm = inst.weights
        objective = wt * makespan + wm * layouts[key]
        if best is None or objective < best:
            best = objective
    return best


def joint_rows(count: int = JOINT_REGIONS) -> dict[str, float]:
    """The joint rows: regions whose joint plan is not the reference optimum (or not a legal
    plan, or not certified), and the summed excess of the parent's pipeline -- and of the
    joint planner -- over the optimum."""
    disagreements = parent_excess = joint_excess = 0
    for seed in range(count):
        inst = random_region(seed)
        want = joint_reference(inst)
        plan = joint_optimum(inst)
        legal = evaluate(inst, plan.starts, plan.offsets)
        disagreements += not (
            legal is not None
            and legal[2] == plan.objective == want
            and plan.optimal
            and plan.lower_bound == want
        )
        parent_excess += sequential_plan(inst).objective - want
        joint_excess += plan.objective - want
    return {
        "joint.optimum.disagreements": float(disagreements),
        "joint.excess": float(joint_excess),
        "_joint.parent_excess": float(parent_excess),
    }


def measure() -> dict[str, float]:
    """The G28 harness rows."""
    out = {"csp.optimum.disagreements": float(csp_disagreements())}
    rows = joint_rows()
    out.update({k: v for k, v in rows.items() if not k.startswith("_")})
    return out
