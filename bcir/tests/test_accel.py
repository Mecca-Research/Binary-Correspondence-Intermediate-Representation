"""The propose-verify search accelerator: learned ordering, same optimum, faster."""

from bcir.examples import PROGRAMS, vector_add
from bcir.kbcir import TARGETS, optimize_constrained
from bcir.kbcir.accel import (
    AccelCertificate,
    FrozenRanker,
    accelerator_certificate,
    greedy_order,
    identity_order,
    optimize_ordered,
    ranker_samples,
    train_ranker,
    worst_order,
)
from bcir.kbcir.cost import Theta
from bcir.kbcir.rcsp import Budget
from bcir.kbcir.weights import PERF
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify_plan, verify_provenance

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()
THETAS = [Theta.cool(), Theta.hot(), Theta.mem_bound()]


def _multi(n_claims=3):
    """Independent unit-stride adds: a multi-column DAG that exercises ordering."""
    m = Module(name="multi")
    claims = []
    for k in range(n_claims):
        a, b, c = 10 * (k + 1), 10 * (k + 1) + 1, 10 * (k + 1) + 2
        for rid in (a, b, c):
            m.add_resource(Resource(rid=rid, shape=(1024,)))
        claims.append(Claim(id=k + 1, opcode=Opcode.ADD, lane=Lane.U,
                            stride_class=StrideClass.UNIT, count=1024,
                            rd=(a, b), wr=(c,), op="vector.add"))
    m.add_phase(Phase(phase_id=0, claims=claims))
    return m


def _laws(diags):
    return {d.law for d in diags}


# --- exactness: the optimum is invariant to candidate order (the safety law) -----

def test_ordered_search_matches_the_exact_optimum_everywhere():
    for name, build in PROGRAMS.items():
        m = build()
        exact = optimize_constrained(m, AVX, COOL).score
        for order in (identity_order, greedy_order, worst_order):
            fast, _ = optimize_ordered(m, AVX, COOL, order=order)
            assert fast.score == exact, f"{name}/{order.__name__}"


def test_exactness_holds_across_targets():
    m = vector_add(1024)
    for h in TARGETS.values():
        exact = optimize_constrained(m, h, COOL).score
        fast, _ = optimize_ordered(m, h, COOL, order=greedy_order)
        assert fast.score == exact


def test_exactness_holds_under_a_budget():
    m = vector_add(1024)
    b = Budget.of(thermal=700)
    exact = optimize_constrained(m, AVX, COOL, PERF, b).score
    fast, _ = optimize_ordered(m, AVX, COOL, PERF, b, greedy_order)
    assert fast.score == exact == 9472          # the RCSP budgeted optimum (vec8)


def test_a_bad_order_still_returns_the_exact_optimum():
    # The whole safety point: even worst-first ordering cannot corrupt the result.
    m = _multi(3)
    exact = optimize_constrained(m, AVX, COOL).score
    bad, _ = optimize_ordered(m, AVX, COOL, order=worst_order)
    assert bad.score == exact


def test_returned_plan_is_legal_and_optimal():
    m = _multi(3)
    fast, _ = optimize_ordered(m, AVX, COOL, order=greedy_order)
    assert verify_plan(m, fast) == []
    assert fast.score == optimize_constrained(m, AVX, COOL).score


# --- acceleration: good ordering finds the optimum first and prunes more ---------

def test_good_order_finds_the_optimum_first():
    m = _multi(3)
    exact = optimize_constrained(m, AVX, COOL).score
    _, gs = optimize_ordered(m, AVX, COOL, order=greedy_order)
    _, ws = optimize_ordered(m, AVX, COOL, order=worst_order)
    assert gs.first_complete_cost == exact          # greedy nails it immediately
    assert ws.first_complete_cost > exact           # worst surfaces a bad plan first


def test_good_order_expands_no_more_than_a_bad_one():
    m = _multi(3)
    _, gs = optimize_ordered(m, AVX, COOL, order=greedy_order)
    _, ws = optimize_ordered(m, AVX, COOL, order=worst_order)
    assert gs.expansions <= ws.expansions


# --- the learned ranker ----------------------------------------------------------

def test_ranker_trains_and_preserves_the_optimum():
    r = train_ranker(ranker_samples(vector_add(1024), AVX, THETAS))
    m = _multi(3)
    exact = optimize_constrained(m, AVX, COOL).score
    fast, stats = optimize_ordered(m, AVX, COOL, order=r.order)
    assert fast.score == exact
    assert stats.first_complete_cost == exact       # the learned order nails it too


def test_ranker_training_is_deterministic():
    a = train_ranker(ranker_samples(vector_add(1024), AVX, THETAS), epochs=100)
    b = train_ranker(ranker_samples(vector_add(1024), AVX, THETAS), epochs=100)
    assert a.w == b.w


def test_frozen_ranker_is_deterministic_integer_and_exact():
    fr = train_ranker(ranker_samples(vector_add(1024), AVX, THETAS)).freeze()
    assert isinstance(fr, FrozenRanker)
    m = _multi(3)
    f1, _ = optimize_ordered(m, AVX, COOL, order=fr.order)
    f2, _ = optimize_ordered(m, AVX, COOL, order=fr.order)
    assert f1.score == f2.score == optimize_constrained(m, AVX, COOL).score


# --- the equivalence certificate -------------------------------------------------

def _cases():
    return [(vector_add(1024), AVX, t, PERF, Budget.unbounded()) for t in THETAS] + \
           [(_multi(3), AVX, COOL, PERF, Budget.unbounded())]


def test_certificate_admits_every_order_because_all_are_exact():
    # The safety guarantee in one assertion: ANY order (even worst) reproduces the
    # exact optimum, so its equivalence certificate is admitted (0 mismatches).
    fr = train_ranker(ranker_samples(vector_add(1024), AVX, THETAS)).freeze()
    for order, nm in ((greedy_order, "greedy"), (worst_order, "worst"), (fr.order, "learned")):
        cert = accelerator_certificate(_cases(), order, nm)
        assert cert.admitted and cert.mismatches == 0 and cert.checked == 4


def test_certificate_satisfies_R13():
    cert = accelerator_certificate(_cases(), greedy_order, "greedy")
    assert verify_provenance(None, accel_certificates=[cert]) == []


def test_forged_accelerator_certificate_is_R13():
    # A certificate claiming deployment but with a mismatch (the accelerator
    # changed the optimum) is illegal -- the accelerator must not alter the result.
    forged = AccelCertificate(order_name="learned", checked=10, mismatches=1)
    assert not forged.admitted
    assert "R13" in _laws(verify_provenance(None, accel_certificates=[forged]))
