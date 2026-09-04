"""Compositional semantics: functions/calls, control flow, dynamic shapes.

Roadmap (BCIR_MASTER_ROADMAP.md §5.3 / §6): the frontier past straight-line array kernels.
The K_BCIR central equation is already constrained series-parallel, so planning extends
along that grain to a region tree -- `Seq` (series, sum), `Cond` (branch, worst-case max +
expected), `Call`/`Function` (inline-substituted reuse), and `dynamic` claims (count as a
static upper bound, worst-case priced). It reuses `optimize` for the leaves, so the pinned
straight-line scores are exactly preserved.
"""

from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.compose import (
    Call,
    Cond,
    Effect,
    Function,
    FunctionSummary,
    Leaf,
    PRED_COST,
    Seq,
    effect,
    independent,
    plan_composite,
    plan_holds_for,
    summarize,
    worst_case_module,
)
from bcir.kbcir.cost import Theta
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def _res():
    return {
        rid: Resource(rid=rid, domain=Domain.RAM, shape=(1024,)) for rid in (10, 11, 12, 20, 21, 22)
    }


def _add(i, a, b, c, **kw):
    return Claim(
        id=i,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=kw.pop("count", 1024),
        rd=(a, b),
        wr=(c,),
        op="vector.add",
        domain=Domain.RAM,
        **kw,
    )


def test_leaf_prices_exactly_like_the_straight_line_planner():
    """A single parallel block must equal `optimize` -- the compositional layer is a strict
    extension; the pinned 7808 is unchanged."""
    r = plan_composite(Leaf((_add(1, 10, 11, 12),)), {}, _res(), AVX, COOL)
    assert r.worst_cost == r.expected_cost == 7808
    assert r.leaves == 1


def test_seq_is_the_sum():
    seq = Seq((Leaf((_add(1, 10, 11, 12),)), Leaf((_add(2, 20, 21, 22),))))
    r = plan_composite(seq, {}, _res(), AVX, COOL)
    assert r.worst_cost == r.expected_cost == 2 * 7808
    assert r.leaves == 2


def test_cond_is_worst_case_max_and_weighted_expected():
    cond = Cond(
        "pred",
        Leaf((_add(1, 10, 11, 12),)),
        Seq((Leaf((_add(2, 20, 21, 22),)), Leaf((_add(3, 20, 21, 22),)))),
        prob_then_milli=250,
    )
    r = plan_composite(cond, {}, _res(), AVX, COOL)
    assert r.worst_cost == PRED_COST + max(7808, 15616)  # hard latency bound
    assert r.expected_cost == PRED_COST + (250 * 7808 + 750 * 15616) // 1000
    assert r.worst_cost > r.expected_cost and r.branch_spread > 0  # the branch matters


def test_call_inlines_and_substitutes_arguments():
    fn = Function("axpy", Leaf((_add(100, 1, 2, 3),)))  # formals 1,2,3
    call = Call("axpy", ((1, 10), (2, 11), (3, 12)))  # actuals A,B,C
    r = plan_composite(call, {"axpy": fn}, _res(), AVX, COOL)
    assert r.worst_cost == 7808  # priced against the actuals
    # define once, call twice.
    r2 = plan_composite(Seq((call, call)), {"axpy": fn}, _res(), AVX, COOL)
    assert r2.worst_cost == 2 * 7808 and r2.leaves == 2


def test_undefined_call_and_recursion_are_rejected():
    try:
        plan_composite(Call("ghost", ()), {}, _res(), AVX, COOL)
        assert False
    except KeyError:
        pass
    rec = Function("rec", Call("rec", ()))
    try:
        plan_composite(Call("rec", ()), {"rec": rec}, _res(), AVX, COOL)
        assert False, "recursion must be rejected (bounded compile time)"
    except RecursionError:
        pass


def test_nested_call_substitution_composes():
    """A function that calls another: the outer actuals flow through to the inner formals."""
    inner = Function("inner", Leaf((_add(1, 1, 2, 3),)))  # formals 1,2,3
    outer = Function("outer", Call("inner", ((1, 10), (2, 11), (3, 12))))
    r = plan_composite(Call("outer", ()), {"inner": inner, "outer": outer}, _res(), AVX, COOL)
    assert r.worst_cost == 7808


# --- dynamic shapes --------------------------------------------------------------


def test_dynamic_claim_plan_holds_up_to_the_bound():
    dyn = _add(1, 10, 11, 12, dynamic=True, count=1024)
    assert plan_holds_for(dyn, 0) and plan_holds_for(dyn, 512) and plan_holds_for(dyn, 1024)
    assert not plan_holds_for(dyn, 1025)  # beyond the declared bound
    static = _add(2, 10, 11, 12, count=1024)
    assert plan_holds_for(static, 1024) and not plan_holds_for(static, 512)  # exact only


def test_dynamic_module_is_worst_case_priced():
    """A dynamic claim is priced at its upper bound, so the plan is a hard latency bound
    that holds for every actual size <= the bound (a single, reusable plan)."""
    m = Module(name="dyn")
    for rid in (10, 11, 12):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(1024,)))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[_add(1, 10, 11, 12, dynamic=True, count=1024)]))
    # planning the bound is the same as the static-1024 plan (worst case).
    assert optimize(worst_case_module(m), AVX, COOL).score == 7808
    # and it is the bound for any smaller actual run.
    assert plan_holds_for(m.phases[0].claims[0], 700)


def test_a_realistic_composite_program():
    """A function called inside a conditional inside a sequence -- a small program past
    straight-line kernels, planned compositionally."""
    fn = Function("ew", Leaf((_add(1, 1, 2, 3),)))
    prog = Seq(
        (
            Leaf((_add(10, 10, 11, 12),)),  # prologue
            Cond(
                "hot",
                Call("ew", ((1, 10), (2, 11), (3, 12))),  # then: one call
                Seq(
                    (
                        Call("ew", ((1, 20), (2, 21), (3, 22))),  # else: two calls
                        Call("ew", ((1, 20), (2, 21), (3, 22))),
                    )
                ),
                prob_then_milli=600,
            ),
        )
    )
    r = plan_composite(prog, {"ew": fn}, _res(), AVX, COOL)
    assert r.worst_cost == 7808 + PRED_COST + max(7808, 2 * 7808)  # prologue + cond(max)
    assert r.leaves == 1 + 1 + 2  # prologue + both branches
    assert r.expected_cost < r.worst_cost


# --- alias / effect modeling + inter-procedural summary costs (deepening) ---------


def test_effect_is_the_read_write_footprint():
    e = effect(Leaf((_add(1, 10, 11, 12),)), {})
    assert e.reads == frozenset({10, 11}) and e.writes == frozenset({12})
    # Seq/Cond fold; a Call folds through its argument substitution.
    fn = Function("ew", Leaf((_add(1, 1, 2, 3),)))
    ce = effect(Call("ew", ((1, 20), (2, 21), (3, 22))), {"ew": fn})
    assert ce.reads == frozenset({20, 21}) and ce.writes == frozenset({22})


def test_effect_conflict_is_raw_war_waw():
    a = Effect(frozenset({1}), frozenset({2}))
    assert a.conflicts(Effect(frozenset({2}), frozenset()))  # RAW (b reads a's write)
    assert a.conflicts(Effect(frozenset(), frozenset({2})))  # WAW
    assert a.conflicts(Effect(frozenset({2}), frozenset({1})))  # WAR (b writes a's read)
    assert not a.conflicts(Effect(frozenset({9}), frozenset({8})))  # disjoint


def test_independent_calls_commute_across_call_boundaries():
    fn = Function("ew", Leaf((_add(1, 1, 2, 3),)))
    fns = {"ew": fn}
    disjoint = Call("ew", ((1, 20), (2, 21), (3, 22)))  # touches 20,21,22
    base = Call("ew", ((1, 10), (2, 11), (3, 12)))  # touches 10,11,12 (writes 12)
    aliasing = Call("ew", ((1, 12), (2, 13), (3, 14)))  # reads 12 -> RAW with base
    assert independent(base, disjoint, fns)  # commute (overlap/reorder ok)
    assert not independent(base, aliasing, fns)  # do not commute


def test_summary_plans_once_and_reuses_for_compatible_calls():
    fn = Function("ew", Leaf((_add(1, 1, 2, 3),)))
    formals = {r: Resource(rid=r, domain=Domain.RAM, shape=(1024,)) for r in (1, 2, 3)}
    summ = summarize(fn, {"ew": fn}, formals, AVX, COOL)
    assert summ.cost.worst_cost == 7808 and summ.cost.leaves == 1
    prog = Seq(
        tuple(Call("ew", ((1, 10 * k), (2, 10 * k + 1), (3, 10 * k + 2))) for k in range(1, 6))
    )  # 5 cost-compatible calls
    res = {r: Resource(rid=r, domain=Domain.RAM, shape=(1024,)) for r in range(1, 60)}
    inline = plan_composite(prog, {"ew": fn}, res, AVX, COOL)
    with_summary = plan_composite(prog, {"ew": fn}, res, AVX, COOL, summaries={"ew": summ})
    # identical cost...
    assert with_summary.worst_cost == inline.worst_cost == 5 * 7808
    # ...but the summary plans NOTHING (compile-time bounded: 1 summary + N reuses).
    assert inline.leaves == 5 and inline.reused == 0
    assert with_summary.leaves == 0 and with_summary.reused == 5


def test_summary_falls_back_to_inline_for_incompatible_args():
    """The summary cost is only reused when the actuals match the formals' cost-keys
    (domain/count/access). An HBM actual where the formal was DRAM re-plans (sound)."""
    fn = Function("ew", Leaf((_add(1, 1, 2, 3),)))
    formals = {r: Resource(rid=r, domain=Domain.RAM, shape=(1024,)) for r in (1, 2, 3)}
    summ = summarize(fn, {"ew": fn}, formals, AVX, COOL)
    res = {
        40: Resource(rid=40, domain=Domain.HBM, shape=(1024,)),  # different tier
        41: Resource(rid=41, domain=Domain.RAM, shape=(1024,)),
        42: Resource(rid=42, domain=Domain.RAM, shape=(1024,)),
    }
    r = plan_composite(
        Call("ew", ((1, 40), (2, 41), (3, 42))), {"ew": fn}, res, AVX, COOL, summaries={"ew": summ}
    )
    assert r.leaves == 1 and r.reused == 0  # re-planned, not reused


def test_summary_reuse_plus_hbm_reprice_matches_the_mlir_compose_pass():
    """Pins the constants in mlir/test/passes/compose_summary.mlir: a func summarized to 7808,
    reused for a DRAM call and re-priced to 2816 for an HBM call -> worst 10624, reused 1
    (the law-rail -bcir-compose must reproduce this)."""
    fn = Function("ew", Leaf((_add(1, 1, 2, 3),)))
    formals = {r: Resource(rid=r, domain=Domain.RAM, shape=(1024,)) for r in (1, 2, 3)}
    summ = summarize(fn, {"ew": fn}, formals, AVX, COOL)
    assert summ.cost.worst_cost == 7808
    res = {
        10: Resource(rid=10, domain=Domain.RAM, shape=(1024,)),
        11: Resource(rid=11, domain=Domain.RAM, shape=(1024,)),
        12: Resource(rid=12, domain=Domain.RAM, shape=(1024,)),
        20: Resource(rid=20, domain=Domain.HBM, shape=(1024,)),
        21: Resource(rid=21, domain=Domain.HBM, shape=(1024,)),
        22: Resource(rid=22, domain=Domain.HBM, shape=(1024,)),
    }
    prog = Seq(
        (
            Call("ew", ((1, 10), (2, 11), (3, 12))),  # DRAM -> reuse
            Call("ew", ((1, 20), (2, 21), (3, 22))),
        )
    )  # HBM  -> re-price
    r = plan_composite(prog, {"ew": fn}, res, AVX, COOL, summaries={"ew": summ})
    assert r.worst_cost == r.expected_cost == 10624 and r.reused == 1 and r.leaves == 1


# --- RCSP-constrained compose: min M(pi,Theta) s.t. R(pi,Theta) <= B --------------


def test_budget_unbounded_is_exactly_the_unconstrained_plan():
    """No budget (or an unbounded one) is the degenerate case -- the pinned 7808 holds."""
    from bcir.kbcir.rcsp import Budget

    leaf = Leaf((_add(1, 10, 11, 12),))
    assert plan_composite(leaf, {}, _res(), AVX, COOL).worst_cost == 7808
    assert plan_composite(leaf, {}, _res(), AVX, COOL, budget=Budget.unbounded()).worst_cost == 7808


def test_budget_reprices_a_leaf_when_a_thermal_cap_forbids_wide_simd():
    """A loose thermal cap leaves vec16 (7808); a tight one makes vec16 infeasible and the
    parallel block re-prices to the feasible vec8 (9472) -- Theta-feasibility over the tree."""
    from bcir.kbcir.rcsp import Budget

    leaf = Leaf((_add(1, 10, 11, 12),))
    assert (
        plan_composite(leaf, {}, _res(), AVX, COOL, budget=Budget.of(thermal=1200)).worst_cost
        == 7808
    )
    assert (
        plan_composite(leaf, {}, _res(), AVX, COOL, budget=Budget.of(thermal=800)).worst_cost
        == 9472
    )


def test_budget_infeasible_leaf_raises():
    """A cap no legal realization can satisfy makes the composite plan infeasible."""
    from bcir.kbcir.rcsp import Budget, Infeasible

    leaf = Leaf((_add(1, 10, 11, 12),))
    try:
        plan_composite(leaf, {}, _res(), AVX, COOL, budget=Budget.of(thermal=400))
        assert False, "an unsatisfiable budget must raise Infeasible"
    except Infeasible:
        pass


def test_budget_threads_through_seq_and_cond():
    """The cap is enforced per parallel block across the whole region tree (series + branch)."""
    from bcir.kbcir.rcsp import Budget

    b = Budget.of(thermal=800)
    seq = Seq((Leaf((_add(1, 10, 11, 12),)), Leaf((_add(2, 20, 21, 22),))))
    assert plan_composite(seq, {}, _res(), AVX, COOL, budget=b).worst_cost == 2 * 9472
    cond = Cond(
        "p", Leaf((_add(1, 10, 11, 12),)), Leaf((_add(2, 20, 21, 22),)), prob_then_milli=500
    )
    r = plan_composite(cond, {}, _res(), AVX, COOL, budget=b)
    assert r.worst_cost == PRED_COST + max(9472, 9472)
