"""Compositional semantics: functions/calls, control flow, dynamic shapes.

Roadmap (BCIR_MASTER_ROADMAP.md §5.3 / §6): the frontier past straight-line array kernels.
The K_BCIR central equation is already constrained series-parallel, so planning extends
along that grain to a region tree -- `Seq` (series, sum), `Cond` (branch, worst-case max +
expected), `Call`/`Function` (inline-substituted reuse), and `dynamic` claims (count as a
static upper bound, worst-case priced). It reuses `optimize` for the leaves, so the pinned
straight-line scores are exactly preserved.
"""

from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.compose import (Call, Cond, Function, Leaf, PRED_COST, Seq, plan_composite,
                                plan_holds_for, worst_case_module)
from bcir.kbcir.cost import Theta
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def _res():
    return {rid: Resource(rid=rid, domain=Domain.RAM, shape=(1024,))
            for rid in (10, 11, 12, 20, 21, 22)}


def _add(i, a, b, c, **kw):
    return Claim(id=i, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
                 count=kw.pop("count", 1024), rd=(a, b), wr=(c,), op="vector.add",
                 domain=Domain.RAM, **kw)


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
    cond = Cond("pred", Leaf((_add(1, 10, 11, 12),)),
                Seq((Leaf((_add(2, 20, 21, 22),)), Leaf((_add(3, 20, 21, 22),)))),
                prob_then_milli=250)
    r = plan_composite(cond, {}, _res(), AVX, COOL)
    assert r.worst_cost == PRED_COST + max(7808, 15616)            # hard latency bound
    assert r.expected_cost == PRED_COST + (250 * 7808 + 750 * 15616) // 1000
    assert r.worst_cost > r.expected_cost and r.branch_spread > 0  # the branch matters


def test_call_inlines_and_substitutes_arguments():
    fn = Function("axpy", Leaf((_add(100, 1, 2, 3),)))          # formals 1,2,3
    call = Call("axpy", ((1, 10), (2, 11), (3, 12)))            # actuals A,B,C
    r = plan_composite(call, {"axpy": fn}, _res(), AVX, COOL)
    assert r.worst_cost == 7808                                # priced against the actuals
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
    inner = Function("inner", Leaf((_add(1, 1, 2, 3),)))             # formals 1,2,3
    outer = Function("outer", Call("inner", ((1, 10), (2, 11), (3, 12))))
    r = plan_composite(Call("outer", ()), {"inner": inner, "outer": outer}, _res(), AVX, COOL)
    assert r.worst_cost == 7808


# --- dynamic shapes --------------------------------------------------------------

def test_dynamic_claim_plan_holds_up_to_the_bound():
    dyn = _add(1, 10, 11, 12, dynamic=True, count=1024)
    assert plan_holds_for(dyn, 0) and plan_holds_for(dyn, 512) and plan_holds_for(dyn, 1024)
    assert not plan_holds_for(dyn, 1025)                  # beyond the declared bound
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
    prog = Seq((
        Leaf((_add(10, 10, 11, 12),)),                              # prologue
        Cond("hot",
             Call("ew", ((1, 10), (2, 11), (3, 12))),               # then: one call
             Seq((Call("ew", ((1, 20), (2, 21), (3, 22))),          # else: two calls
                  Call("ew", ((1, 20), (2, 21), (3, 22))))),
             prob_then_milli=600),
    ))
    r = plan_composite(prog, {"ew": fn}, _res(), AVX, COOL)
    assert r.worst_cost == 7808 + PRED_COST + max(7808, 2 * 7808)   # prologue + cond(max)
    assert r.leaves == 1 + 1 + 2                                    # prologue + both branches
    assert r.expected_cost < r.worst_cost
