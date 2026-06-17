"""Generated metamorphic differential for the compositional planner (hardens the rail).

The optimizer-core differential (`test_differential`) fuzzes `optimize` against the law rail
on randomized *modules*. This is its compose analog: it generates randomized **region trees**
(Seq / Cond / Leaf / Call over a small pool of leaf functions) and asserts the metamorphic
laws `plan_composite` must satisfy -- determinism, the worst >= expected bound, the
unbounded-budget degeneracy, RCSP-budget monotonicity (a feasible cap never lowers the cost),
and inter-procedural summary consistency (reuse changes the *plan count*, never the cost). It
is the property net under both the oracle and -- by construction equality -- the law-rail
`-bcir-compose` it conforms to.
"""

import random

from bcir.kbcir import TARGETS
from bcir.kbcir.compose import (Call, Cond, Function, Leaf, Seq, plan_composite, summarize)
from bcir.kbcir.cost import Theta
from bcir.kbcir.rcsp import Budget, Infeasible
from bcir.model import Claim, Domain, Lane, Opcode, Resource, StrideClass

AVX = TARGETS["x86_avx512"]
THETAS = (Theta.cool(), Theta.hot(), Theta.mem_bound())
RES = {r: Resource(rid=r, domain=Domain.RAM, shape=(1024,)) for r in range(64)}

_OPS = (("vector.add", Opcode.ADD), ("vector.mul", Opcode.MUL), ("reduce.add", Opcode.ADD))
_STRIDES = (StrideClass.UNIT, StrideClass.STRIDED, StrideClass.CACHELINE)


class _Gen:
    """A bounded random region-tree generator (claim ids stay unique via a counter)."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.cid = 0

    def claim(self) -> Claim:
        self.cid += 1
        op, opc = self.rng.choice(_OPS)
        a, b, c = self.rng.sample(range(64), 3)
        return Claim(id=self.cid, opcode=opc, lane=Lane.U,
                     stride_class=self.rng.choice(_STRIDES),
                     count=self.rng.choice((256, 1024, 4096)), rd=(a, b), wr=(c,),
                     op=op, domain=Domain.RAM)

    def leaf(self) -> Leaf:
        return Leaf(tuple(self.claim() for _ in range(self.rng.randint(1, 3))))

    def region(self, funcs, depth: int = 0):
        if depth >= 3 or self.rng.random() < 0.45:
            if funcs and self.rng.random() < 0.3:
                return Call(self.rng.choice(list(funcs)), ())   # empty arg_map -> formals
            return self.leaf()
        if self.rng.random() < 0.5:
            return Seq(tuple(self.region(funcs, depth + 1)
                             for _ in range(self.rng.randint(1, 3))))
        return Cond("p", self.region(funcs, depth + 1), self.region(funcs, depth + 1),
                    prob_then_milli=self.rng.randint(0, 1000))


def test_compose_metamorphic_laws_hold_over_generated_region_trees():
    for seed in range(250):
        g = _Gen(seed)
        theta = THETAS[seed % len(THETAS)]
        funcs = {f"f{k}": Function(f"f{k}", g.leaf()) for k in range(g.rng.randint(0, 2))}
        region = g.region(funcs)

        base = plan_composite(region, funcs, RES, AVX, theta)
        # 1. determinism: the same inputs reproduce the same plan exactly.
        assert plan_composite(region, funcs, RES, AVX, theta) == base
        # 2. the bounds: worst-case >= probability-weighted expected >= 0; counts non-negative.
        assert base.worst_cost >= base.expected_cost >= 0
        assert base.leaves >= 0 and base.reused >= 0
        # 3. an unbounded budget is exactly the unconstrained plan (the degenerate case).
        ub = plan_composite(region, funcs, RES, AVX, theta, budget=Budget.unbounded())
        assert (ub.worst_cost, ub.expected_cost) == (base.worst_cost, base.expected_cost)
        # 4. RCSP monotonicity: a feasible thermal cap never lowers the cost (the constrained
        #    optimum is >= the unconstrained one), and an unsatisfiable cap raises -- never a
        #    silently-cheaper plan.
        try:
            cap = plan_composite(region, funcs, RES, AVX, theta, budget=Budget.of(thermal=1500))
            assert cap.worst_cost >= base.worst_cost
            assert cap.expected_cost >= base.expected_cost
        except Infeasible:
            pass
        # 5. summary consistency: reuse for cost-compatible calls (empty arg_map -> the formals
        #    themselves) changes the *plan count*, never the cost.
        summaries = {n: summarize(fn, funcs, RES, AVX, theta) for n, fn in funcs.items()}
        s = plan_composite(region, funcs, RES, AVX, theta, summaries=summaries)
        assert (s.worst_cost, s.expected_cost) == (base.worst_cost, base.expected_cost)
        assert s.reused >= base.reused


def test_compose_budget_is_monotone_in_the_cap():
    """Tightening a cap can only raise the cost or make the plan infeasible -- never cheaper."""
    g = _Gen(12345)
    region = Seq(tuple(g.leaf() for _ in range(4)))
    last = 0
    for cap in (2000, 1500, 1200, 1000, 900):
        try:
            r = plan_composite(region, {}, RES, AVX, Theta.cool(), budget=Budget.of(thermal=cap))
            assert r.worst_cost >= last
            last = r.worst_cost
        except Infeasible:
            break  # once infeasible, tighter caps stay infeasible -- monotone
