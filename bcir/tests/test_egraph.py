"""The building-blocks engine: e-classes (liked), rewrites (unliked), extraction."""

from bcir.kbcir.egraph import (
    Add,
    C,
    EGraph,
    Expr,
    Mul,
    Sub,
    V,
    module_exprs,
    optimize_expr,
    saturate,
    shared_blocks,
)
from bcir.model import Claim, Module, Opcode, Phase, Resource


# --- liked pairs: hashcons / CSE / congruence -----------------------------------

def test_hashcons_shares_a_class_for_identical_building_blocks():
    eg = EGraph()
    a = eg.add(Add(V("a"), V("b")))
    b = eg.add(Add(V("a"), V("b")))
    assert eg.find(a) == eg.find(b)        # a=a: identical terms are one liked pair


def test_congruence_closure_after_a_merge():
    # f(a) and f(b); proving a=b must make f(a)=f(b) (congruence -- the engine's
    # core invariant, the reason rewrites compose).
    eg = EGraph()
    fa = eg.add(Mul(V("a"), C(3)))
    fb = eg.add(Mul(V("b"), C(3)))
    assert eg.find(fa) != eg.find(fb)
    eg.union(eg.add(V("a")), eg.add(V("b")))
    eg.rebuild()
    assert eg.find(fa) == eg.find(fb)


# --- unliked operators: the algebraic identity rewrites (the user's examples) ----

def test_identity_eliminations():
    assert optimize_expr(Add(V("x"), C(0))).optimized == V("x")     # x + 0 = x
    assert optimize_expr(Add(C(0), V("x"))).optimized == V("x")     # 0 + x = x
    assert optimize_expr(Mul(V("x"), C(1))).optimized == V("x")     # x * 1 = x
    assert optimize_expr(Mul(V("x"), C(0))).optimized == C(0)       # x * 0 = 0
    assert optimize_expr(Sub(V("x"), V("x"))).optimized == C(0)     # x - x = 0


def test_constant_folding_resolves_unliked_constants():
    assert optimize_expr(Add(C(1), C(1))).optimized == C(2)         # 1 + 1 = 2
    assert optimize_expr(Sub(C(1), C(1))).optimized == C(0)         # 1 - 1 = 0
    assert optimize_expr(Add(C(1), C(0))).optimized == C(1)         # 1 + 0 = 1
    assert optimize_expr(Mul(C(2), C(3))).optimized == C(6)         # 2 * 3 = 6


# --- resolution: factoring / fusion + extraction optimality ----------------------

def test_factoring_fuses_shared_building_blocks():
    # a*b + a*c -> a*(b+c): two multiplies + an add (cost 5) recompose into one
    # multiply + one add (cost 3). The combinatorial composition that fuses.
    r = optimize_expr(Add(Mul(V("a"), V("b")), Mul(V("a"), V("c"))))
    assert r.original_cost == 5 and r.optimized_cost == 3 and r.improved == 2
    assert r.optimized == Mul(V("a"), Add(V("b"), V("c")))
    assert r.saturated


def test_rewrites_never_increase_cost():
    # R9 / the rewrite-legality law: extraction returns the min, so the optimized
    # cost is always <= the original, for every expression.
    exprs = [Add(V("a"), C(0)), Mul(V("x"), C(1)), Sub(V("p"), V("p")),
             Add(C(1), C(1)), Add(Mul(V("a"), V("b")), Mul(V("a"), V("c"))),
             Mul(Add(V("a"), C(0)), C(1))]
    for e in exprs:
        r = optimize_expr(e)
        assert r.optimized_cost <= r.original_cost


def test_nothing_to_rewrite_is_a_fixpoint():
    r = optimize_expr(Mul(V("a"), V("b")))     # already minimal
    assert r.optimized == Mul(V("a"), V("b"))
    assert r.optimized_cost == r.original_cost == 2 and r.saturated


def test_optimization_is_deterministic():
    e = Add(Mul(V("a"), V("b")), Mul(V("a"), V("c")))
    assert optimize_expr(e) == optimize_expr(e)


def test_saturation_respects_the_budget():
    # A 1-iteration budget stops early without claiming a fixpoint (the blow-up
    # guard); the result is still cost-non-increasing.
    eg = EGraph()
    eg.add(Add(Mul(V("a"), V("b")), Mul(V("a"), V("c"))))
    st = saturate(eg, budget=1)
    assert st.iterations == 1


# --- the BCIR bridge: claims compose as expressions; CSE is the liked memory -----

def _two_adds_over(rd_a, rd_b):
    m = Module(name="cse")
    for rid in set(rd_a) | set(rd_b) | {90, 91}:
        m.add_resource(Resource(rid=rid, shape=(8,)))
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ADD, rd=rd_a, wr=(90,), op="vector.add"),
        Claim(id=2, opcode=Opcode.ADD, rd=rd_b, wr=(91,), op="vector.add")]))
    return m


def test_module_claims_map_to_expressions():
    m = _two_adds_over((10, 11), (12, 13))
    exprs = module_exprs(m)
    assert exprs[1].op == "add" and exprs[2].op == "add"


def test_identical_claims_share_one_building_block():
    # Two claims computing a+b over the SAME operands: CSE collapses them to one
    # shared add e-class (2 atoms + 1 op = 3 blocks). The liked-pair memory.
    same = _two_adds_over((10, 11), (10, 11))
    assert shared_blocks(same) == 3
    # Distinct operands: no sharing (4 atoms + 2 ops = 6 blocks).
    distinct = _two_adds_over((10, 11), (12, 13))
    assert shared_blocks(distinct) == 6
