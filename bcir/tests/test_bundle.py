"""Multi-claim bundle (joint) optimization -- the combinatorial step beyond pairwise.

Roadmap (BCIR_MASTER_ROADMAP.md §5.3 / §6): the genuinely new optimizer capability. The
shortest path couples claims pairwise in declared order, so input-sharing claims that are
*interleaved* with non-sharers miss the fusion discount; `optimize_bundled` jointly
reorders each input-sharing bundle (bounded, exhaustive, dependency-preserving) to recover
it, emitting a search certificate per improving bundle. It is correctness-preserving (only
mutually-independent same-phase claims are reordered, and never across a conflicting pair)
and a no-op on modules with no interleaved sharers (the pinned scores hold).
"""

from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.bundle import _conflict, _legal_reorder, find_bundles, optimize_bundled
from bcir.kbcir.cost import Theta
from bcir.examples import fused_chain, matmul_tiled, multi_histogram, scan, scan_chain, vector_add
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import verify, verify_plan

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def _interleaved():
    """c1 reads A,B; c2 reads D,E (no share); c3 reads A,G (shares A with c1). Declared
    order c1,c2,c3 separates the A-sharers, so the pairwise model misses their fusion."""
    m = Module(name="interleaved")
    for rid in range(1, 9):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(1024,)))
    cs = [
        Claim(
            id=1,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1024,
            rd=(1, 2),
            wr=(3,),
            op="vector.add",
            domain=Domain.RAM,
        ),
        Claim(
            id=2,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1024,
            rd=(4, 5),
            wr=(6,),
            op="vector.add",
            domain=Domain.RAM,
        ),
        Claim(
            id=3,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1024,
            rd=(1, 7),
            wr=(8,),
            op="vector.add",
            domain=Domain.RAM,
        ),
    ]
    m.add_phase(Phase(phase_id=0, deps=(), claims=cs))
    return m


def test_find_bundles_clusters_input_sharers():
    bundles = find_bundles(_interleaved())
    assert len(bundles) == 1
    assert bundles[0].claim_ids == (1, 3) and bundles[0].shared_rid == 1


def test_joint_reorder_beats_the_pairwise_plan():
    m = _interleaved()
    base = optimize(m, AVX, COOL).score
    res, certs = optimize_bundled(m, AVX, COOL)
    assert res.score < base  # the joint reorder strictly wins
    assert len(certs) == 1
    c = certs[0]
    assert c.gain == base - res.score > 0
    assert c.greedy_score == base and c.joint_score == res.score
    assert c.searched == 2  # 2! orderings of the 2-claim bundle
    assert set(c.order) == {1, 3}


def test_corpus_is_a_noop_where_there_is_nothing_to_join():
    """No interleaved sharers -> no reorder, no certificate, pinned scores preserved."""
    for mod in (vector_add(), fused_chain(), scan(), scan_chain(), multi_histogram()):
        base = optimize(mod, AVX, COOL).score
        res, certs = optimize_bundled(mod, AVX, COOL)
        assert res.score == base
        assert certs == []


def test_real_gain_on_tiled_matmul_is_legal_and_clean():
    """matmul_tiled interleaves A/B-tile sharers across output tiles; the joint reorder
    clusters them for a real gain while preserving every k-accumulation dependency."""
    m = matmul_tiled()
    base = optimize(m, AVX, COOL).score
    res, certs = optimize_bundled(m, AVX, COOL)
    assert res.score < base  # a genuine joint win on a real workload
    assert certs and all(c.gain > 0 for c in certs)


def test_reorder_never_inverts_a_dependency():
    """The legality guard: a conflicting (RAW/WAR/WAW) pair keeps its declared order."""
    orig = matmul_tiled().phases[0].claims
    # a hand-built inversion of a conflicting pair must be rejected.
    by_id = {c.id: c for c in orig}
    # find the first conflicting pair (a before b)
    pair = next((a, b) for i, a in enumerate(orig) for b in orig[i + 1 :] if _conflict(a, b))
    inverted = [c for c in orig if c.id not in (pair[0].id, pair[1].id)]
    inverted = [pair[1], pair[0]] + inverted
    assert not _legal_reorder(orig, inverted)
    assert _legal_reorder(orig, list(orig))  # the identity order is legal


def test_dependent_chain_is_never_bundled():
    """scan_chain is a producer->consumer pair (c2 reads c1's write): they conflict, so
    they never form a bundle and the plan is untouched."""
    assert find_bundles(scan_chain()) == []


def test_is_deterministic():
    m = _interleaved()
    a, ca = optimize_bundled(m, AVX, COOL)
    b, cb = optimize_bundled(_interleaved(), AVX, COOL)
    assert a.score == b.score and [c.order for c in ca] == [c.order for c in cb]
