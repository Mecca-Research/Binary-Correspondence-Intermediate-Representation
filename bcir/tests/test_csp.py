"""G28: a dependency-free exact ILP/CSP, the joint schedule x memory optimum, the CP-SAT adapter.

The parent had no constraint or integer-programming solver, and so no exact rail for a
problem spanning the schedule and the memory: it scheduled first and laid out after. These
witnesses hold the solver to brute force (TMSAO-1 on every generated model), show the check
fires on unsound propagators, hold the joint planner to a reference that enumerates every
start vector and lays each out with the proved exact layout, pin a region where scheduling
first loses, and hold the optional CP-SAT adapter to the in-tree solver where `ortools` is
installed (`BCIR_REQUIRE_CPSAT=1` makes its absence a failure, not a skip).
"""

from __future__ import annotations

import os
import sys
import types

from bcir.gem.dispatch import DispatchRequest, solve_joint
from bcir.kbcir import csp
from bcir.kbcir.csp import CspError, Model, brute_force, satisfied, solve
from bcir.kbcir.joint import (
    JointError,
    JointInstance,
    evaluate,
    joint_instance,
    joint_model,
    joint_optimum,
    sequential_plan,
)
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.tests.joint_fixtures import (
    CSP_MODELS,
    JOINT_REGIONS,
    csp_disagreements,
    joint_reference,
    joint_rows,
    random_model,
    random_region,
)


def _mutant(name: str, old: str, new: str):
    source = open(csp.__file__, encoding="utf-8").read()
    assert source.count(old) == 1, name
    module = types.ModuleType(f"bcir.kbcir._csp_{name}")
    module.__package__ = "bcir.kbcir"
    sys.modules[module.__name__] = module
    try:
        exec(compile(source.replace(old, new), module.__name__, "exec"), module.__dict__)
    finally:
        del sys.modules[module.__name__]
    return module


def test_the_solver_is_brute_force_on_every_generated_model():
    """150 generated models -- linear rows of every relation, cumulatives and 2-D no-overlaps
    -- each solved to the brute-force optimum (or proved infeasible where brute force finds
    nothing), certified (bound == objective), its answer a solution by the definitions."""
    assert csp_disagreements(CSP_MODELS) == 0
    feasible = sum(brute_force(random_model(s))[0] is not None for s in range(CSP_MODELS))
    assert 40 <= feasible <= CSP_MODELS - 20, feasible  # both verdicts are exercised


def _rebuild(model: Model, module) -> Model:
    """`model` in `module`'s own classes (a mutant copy's isinstance checks see only its own)."""
    out = module.Model()
    out.lo, out.hi = list(model.lo), list(model.hi)
    out.names, out.priority = list(model.names), list(model.priority)
    out.objective, out.offset = model.objective, model.offset
    for con in model.constraints:
        if isinstance(con, csp.Linear):
            out.constraints.append(module.Linear(con.terms, con.op, con.rhs))
        elif isinstance(con, csp.Cumulative):
            out.constraints.append(module.Cumulative(con.tasks, con.capacity))
        else:
            out.constraints.append(module.NoOverlap2D(con.rects))
    return out


def _wrong(module, models) -> int:
    """Models on which `module`'s solver misses brute force's verdict or optimum, or raises."""
    wrong = 0
    for m in models:
        want = brute_force(m)[0]
        try:
            got = module.solve(_rebuild(m, module))
        except module.CspError:
            wrong += 1
            continue
        wrong += got.objective != want
    return wrong


def test_the_check_fires_on_unsound_propagators():
    """Wrong propagators, each loaded from a copy of the module, never by editing the tree,
    and each against a control copy that agrees everywhere: an over-tight linear bound and an
    over-eager cumulative push lose optima (brute force disagrees); a skipped no-overlap
    admits overlapping values, which the leaf's definition check refuses rather than
    returns."""
    models = [random_model(s) for s in range(CSP_MODELS)]
    regions = [joint_model(random_region(s))[0] for s in range(JOINT_REGIONS)]
    line = "            bound = slack // c\n"
    assert _wrong(_mutant("control", line, line), models) == 0
    for name, old, new in (
        ("tight", line, "            bound = slack // c - 1\n"),
        ("push", "            t = max(hit) + 1\n", "            t = max(hit) + 2\n"),
    ):
        assert _wrong(_mutant(name, old, new), models) > 0, f"the differential cannot see {name}"
    skipped = _mutant(
        "skipped",
        "                r = _no_overlap_2d(con, lo, hi)\n",
        "                r = False\n",
    )
    raised = 0
    for m in regions:
        try:
            skipped.solve(_rebuild(m, skipped))
        except skipped.CspError:
            raised += 1
    assert raised > 0  # an admitted overlap is refused at the leaf, never returned


def test_budgets_incumbents_and_refusals():
    """A budget stop returns a legal incumbent and a bound no greater than the optimum; an
    incumbent that is not a solution is refused; a malformed model is refused."""
    stopped = 0
    for seed in range(JOINT_REGIONS):
        inst = random_region(seed)
        full = joint_optimum(inst)
        cut = joint_optimum(inst, budget=3)
        assert evaluate(inst, cut.starts, cut.offsets) is not None
        assert cut.lower_bound <= full.objective <= cut.objective
        stopped += cut.stop_reason == "budget"
        assert cut.objective <= sequential_plan(inst).objective  # seeded: never worse
    assert stopped >= 5
    m = Model()
    x = m.var(0, 3)
    m.linear([(1, x)], ">=", 2)
    m.minimize([(1, x)])
    assert solve(m).objective == 2 and satisfied(m, (2,)) and not satisfied(m, (1,))
    for bad in (
        lambda: solve(m, incumbent=(1,)),
        lambda: m.var(3, 2),
        lambda: m.var(0, True),
        lambda: m.linear([(1, x)], "<", 1),
        lambda: m.linear([(1, 99)], "<=", 1),
        lambda: m.cumulative([(x, -1)], 1),
        lambda: m.cumulative([(x, 1)], 0),
        lambda: m.no_overlap_2d([(x, 0, x, 1, x, 0)]),
        lambda: m.no_overlap_2d([(x, 0, x, 1, x, 1, 7), (x, 0, x, 1, x, 2, 7)]),
        lambda: solve(m, budget=-1),
    ):
        try:
            bad()
        except CspError:
            continue
        raise AssertionError("a malformed model or incumbent was accepted")


def test_the_joint_optimum_is_the_reference_optimum():
    """Over the generated regions every joint plan is legal, certified optimal, and equal
    to the reference that enumerates every start vector -- and the parent's pipeline
    (schedule first, lay out after) is worse on some of them."""
    rows = joint_rows(JOINT_REGIONS)
    assert rows["joint.optimum.disagreements"] == 0
    assert rows["joint.excess"] == 0
    assert rows["_joint.parent_excess"] > 0
    worse = sum(
        sequential_plan(random_region(s)).objective > joint_reference(random_region(s))
        for s in range(JOINT_REGIONS)
    )
    assert worse >= 5, worse


def test_a_region_where_scheduling_first_loses():
    """Two independent one-tick tasks, each touching its own four-unit value, on two
    domains: scheduling first runs them together (makespan 1) and so keeps both values live
    (extent 8): 1 + 8 = 9. Running them one after the other costs a tick and halves the
    memory: 2 + 4 = 6, the joint optimum."""
    inst = JointInstance({1: 1, 2: 1}, {1: (10,), 2: (11,)}, {}, {10: 4, 11: 4}, 2, (1, 1))
    seq = sequential_plan(inst)
    assert (seq.makespan, seq.extent, seq.objective) == (1, 8, 9)
    plan = joint_optimum(inst)
    assert (plan.makespan, plan.extent, plan.objective, plan.optimal) == (2, 4, 6, True)
    assert joint_reference(inst) == 6
    # weigh time heavily and running together is the optimum again
    heavy = JointInstance({1: 1, 2: 1}, {1: (10,), 2: (11,)}, {}, {10: 4, 11: 4}, 2, (9, 1))
    assert joint_optimum(heavy).objective == joint_reference(heavy) == 9 + 8
    for bad in (
        lambda: JointInstance({1: 0}, {}, {}, {}, 1),
        lambda: JointInstance({1: 1}, {1: (5,)}, {}, {}, 1),
        lambda: JointInstance({1: 1}, {}, {1: (1,)}, {}, 1),
        lambda: JointInstance({1: 1}, {}, {}, {}, 0),
    ):
        try:
            bad()
        except JointError:
            continue
        raise AssertionError("a malformed region was accepted")


def test_a_one_phase_module_is_a_joint_instance():
    """A module's phase becomes a region: its claims the tasks, their hazard predecessors
    the edges, the resources they touch the values at their byte sizes; the joint plan keeps
    every hazard."""
    m = Module(name="joint")
    for rid, n in ((1, 16), (2, 16), (3, 16), (4, 16)):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(n,)))

    def claim(cid, rd, wr):
        return Claim(
            id=cid,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=16,
            rd=rd,
            wr=wr,
            op="vector.add",
        )

    m.add_phase(
        Phase(
            phase_id=0,
            claims=[claim(1, (1, 1), (2,)), claim(2, (3, 3), (4,)), claim(3, (2, 4), (1,))],
        )
    )
    inst = joint_instance(m, {1: 2, 2: 2, 3: 1}, domains=2)
    assert inst.preds[3] == (1, 2) and inst.sizes[1] == 16 * m.resources[1].elem_bytes
    plan = joint_optimum(inst)
    assert plan.optimal and plan.objective == joint_reference(inst)
    assert plan.starts[3] >= max(plan.starts[1] + 2, plan.starts[2] + 2)
    try:
        joint_instance(Module(name="empty"), {})
    except JointError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a module without one phase was made a region")


def test_the_dispatch_law_routes_a_joint_region():
    """The joint kind in the dispatch table: a heuristic request is the parent's pipeline
    (TMSAO-4), a proof request the CSP rail, granted TMSAO-1 when it closes and TMSAO-2 on a
    budget stop with its bound."""
    inst = random_region(12)
    fast, record = solve_joint(DispatchRequest("joint", len(inst.tasks), "TMSAO-4"), inst)
    assert record.granted == "TMSAO-4" and fast.objective == sequential_plan(inst).objective
    best, record = solve_joint(DispatchRequest("joint", len(inst.tasks), "TMSAO-1"), inst)
    assert record.granted == "TMSAO-1" and best.objective == joint_reference(inst)
    assert record.decision.solver == "joint_optimum" and record.spent == best.nodes
    cut, record = solve_joint(DispatchRequest("joint", 9, "TMSAO-1", budget=2), inst)
    assert record.granted == "TMSAO-2" and record.decision.expected == "TMSAO-2"
    assert cut.lower_bound <= best.objective <= cut.objective


def test_the_cpsat_adapter_agrees_with_the_in_tree_solver():
    """Where `ortools` is installed, CP-SAT on the translated model reaches the in-tree
    solver's verdict and optimum on every generated model and region. Without it this is a
    skip -- unless BCIR_REQUIRE_CPSAT=1, under which the absence fails."""
    from bcir.hosted import cpsat

    if not cpsat.available():
        assert os.environ.get("BCIR_REQUIRE_CPSAT") != "1", "CP-SAT was required and is absent"
        return
    for seed in range(CSP_MODELS):
        m = random_model(seed)
        ours, theirs = solve(m), cpsat.solve_cpsat(m)
        assert (ours.stop_reason, ours.objective) == (theirs.stop_reason, theirs.objective), seed
    for seed in range(JOINT_REGIONS):
        inst = random_region(seed)
        m, _x = joint_model(inst)
        theirs = cpsat.solve_cpsat(m)
        assert theirs.optimal and theirs.objective == joint_optimum(inst).objective, seed
