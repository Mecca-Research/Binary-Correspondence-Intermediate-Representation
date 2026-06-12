"""Constrained (RCSP) selection tests: budgets, dominance, the degenerate-case law."""

from bcir.examples import PROGRAMS, vector_add
from bcir.kbcir import (
    Budget,
    Infeasible,
    TARGETS,
    optimize,
    optimize_constrained,
    pareto_plans,
)
from bcir.kbcir.cost import Theta
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify_plan


def test_unbounded_budget_is_the_degenerate_case():
    # LangRef Sec. 2: with B = infinity the constrained rail IS the scalarized
    # tropical optimizer -- same scores, same chosen realizations, everywhere.
    for h in TARGETS.values():
        for build in PROGRAMS.values():
            m = build()
            base = optimize(m, h, Theta.cool())
            rcsp = optimize_constrained(m, h, Theta.cool())
            assert rcsp.score == base.score
            assert {cid: c.width for cid, c in rcsp.by_claim().items()} == \
                   {cid: c.width for cid, c in base.by_claim().items()}


def test_thermal_budget_makes_vec16_infeasible_selects_vec8():
    # vec16 on AVX-512 costs thermal 1088; a 700 cap forbids it. The constrained
    # optimum is vec8 (thermal 640) at score 9472 -- a plan the PERF weight
    # vector alone can never select (thermal weight is 0 under PERF).
    m = vector_add(1024)
    r = optimize_constrained(m, TARGETS["x86_avx512"], Theta.cool(),
                             budget=Budget.of(thermal=700))
    assert r.by_claim()[1000].width == 8
    assert r.score == 9472


def test_power_budget_constrains_like_thermal():
    m = vector_add(1024)
    r = optimize_constrained(m, TARGETS["x86_avx512"], Theta.cool(),
                             budget=Budget.of(power=700))
    assert r.by_claim()[1000].width == 8


def test_unsatisfiable_budget_raises_infeasible():
    # Every realization of vector_add on AVX-512 (scalar/vec8/vec16) costs
    # thermal >= 640: a 500 cap admits no legal plan.
    m = vector_add(1024)
    try:
        optimize_constrained(m, TARGETS["x86_avx512"], Theta.cool(),
                             budget=Budget.of(thermal=500))
        assert False, "expected Infeasible"
    except Infeasible:
        pass


def test_budget_accumulates_across_claims():
    # Two independent claims: vec16+vec16 costs thermal 2176, one-wide mixes
    # 1728 -- a 1500 cap admits only vec8+vec8 (1280).
    m = Module(name="two")
    for rid in (10, 11, 12, 20, 21, 22):
        m.add_resource(Resource(rid=rid, shape=(1024,)))
    a = Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=1024, rd=(10, 11), wr=(12,), op="vector.add")
    b = Claim(id=2, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=1024, rd=(20, 21), wr=(22,), op="vector.add")
    m.add_phase(Phase(phase_id=0, claims=[a, b]))
    r = optimize_constrained(m, TARGETS["x86_avx512"], Theta.cool(),
                             budget=Budget.of(thermal=1500))
    widths = {cid: c.width for cid, c in r.by_claim().items()}
    assert widths == {1: 8, 2: 8}


def test_pareto_front_reaches_the_nonconvex_point():
    # Over (score, thermal) the front is exactly {vec16 (7808, hot), vec8
    # (9472, cool)}; scalar is dominated by vec16 (worse score, equal thermal).
    m = vector_add(1024)
    front = pareto_plans(m, TARGETS["x86_avx512"], Theta.cool(), dims=("thermal",))
    assert [r.score for r in front] == [7808, 9472]
    assert [r.by_claim()[1000].width for r in front] == [16, 8]


def test_constrained_plans_satisfy_R8_R9():
    # The verifier does not care who proposed the plan: RCSP output passes
    # verify_plan (coverage, lane legality, score == sum of step costs).
    m = vector_add(1024)
    r = optimize_constrained(m, TARGETS["x86_avx512"], Theta.cool(),
                             budget=Budget.of(thermal=700, power=700))
    assert verify_plan(m, r) == []


def test_unknown_budget_dim_is_rejected():
    try:
        Budget.of(zeal=1)
        assert False, "expected KeyError"
    except KeyError:
        pass
