"""The verification cost dimension (the 12th K_BCIR cost axis): a real producer.

Roadmap (BCIR_MASTER_ROADMAP.md §5.1 / §6): give the `verification` cost axis -- modeled
as 0 until now -- a real producer on both rails (the C++ mirror is BCIRCostModel.h
verifyCostFor; cross-checked by mlir/test/passes/cost_model_verify.mlir). The contract
cost is the cost of discharging the claim's verify obligation: `none`/`bounds` are free
(the bounds check fuses into the access the memory axis already prices), while `exact`
(recompute + compare) and `hash` (digest every output) carry an O(n) cost. It is
width-independent, so it never perturbs the per-claim selection; it is a tradeable plan
resource (RCSP can cap it, a policy can weight it). Pins: the producer values, that the
common `bounds`/`none` contract stays 0 (so every existing score is unchanged), and that
the plan score / budget feasibility now reflect the contract.
"""

from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import VERIFICATION, TargetProfile, Theta
from bcir.kbcir.realize import _verify_cost, candidates_for
from bcir.kbcir.rcsp import Budget, Infeasible, optimize_constrained
from bcir.kbcir.weights import PERF
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

AVX = TargetProfile.x86_avx512()
COOL = Theta.cool()


def _claim(verify: str):
    return Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
                 count=1024, rd=(10, 11), wr=(12,), op="vector.add", domain=Domain.RAM,
                 verify=verify)


def _module(verify: str):
    m = Module(name=f"v_{verify}")
    for rid in (10, 11, 12):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(1024,)))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[_claim(verify)]))
    return m


def test_verify_cost_producer_values():
    assert _verify_cost(_claim("none")) == 0
    assert _verify_cost(_claim("bounds")) == 0          # the cheap, default contract: free
    assert _verify_cost(_claim("exact")) == 1024        # recompute + compare every element
    assert _verify_cost(_claim("hash")) == 1024         # digest every output element
    # scales with the problem size (O(n)).
    c = Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=4096, rd=(10, 11), wr=(12,), op="vector.add", domain=Domain.RAM, verify="exact")
    assert _verify_cost(c) == 4096


def test_every_candidate_carries_the_verification_cost():
    """Width-independent: the contract cost is on every realization equally, so the 12th
    axis is set on all candidates and the per-claim argmin (vec16) is unchanged."""
    cands = candidates_for(_claim("exact"), AVX)
    assert cands and all(cd.base.v[VERIFICATION] == 1024 for cd in cands)
    # bounds: the axis stays 0 on every candidate (the existing scores are untouched).
    assert all(cd.base.v[VERIFICATION] == 0 for cd in candidates_for(_claim("bounds"), AVX))


def test_bounds_score_is_unchanged_and_exact_adds_the_cost():
    bounds = optimize(_module("bounds"), AVX, COOL, PERF)
    exact = optimize(_module("exact"), AVX, COOL, PERF)
    assert bounds.score == 7808                          # the historic pin: unchanged
    assert exact.score == 7808 + 1024                    # + O(n) verification (weight 1)
    # selection (lane width) is identical -- verification is not a realization knob.
    assert bounds.by_claim()[1].width == exact.by_claim()[1].width == 16


def test_verification_is_a_tracked_rcsp_resource():
    """A budget that caps verification below the contract's cost is infeasible -- the
    optimizer now accounts for verification as a real, constrainable plan resource."""
    m = _module("exact")
    # feasible: a generous cap admits the plan.
    ok = optimize_constrained(m, AVX, COOL, PERF, Budget.of(verification=2048))
    assert ok.score == 8832
    # infeasible: no realization can discharge `exact` for less than n=1024.
    try:
        optimize_constrained(m, AVX, COOL, PERF, Budget.of(verification=1023))
        assert False, "a sub-contract verification cap must be infeasible"
    except Infeasible:
        pass


def test_default_corpus_claim_is_free():
    """vector_add (and the whole corpus) uses the default `bounds` contract, so the
    verification producer leaves its score exactly where it was."""
    assert _module("bounds").phases[0].claims[0].verify == "bounds"
    assert optimize(vector_add(1024), AVX, COOL, PERF).score == 7808
