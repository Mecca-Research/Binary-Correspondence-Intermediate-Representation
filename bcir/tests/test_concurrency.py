"""CT2: concurrent wave scheduling + GGG decoupling + affinity tests."""

from bcir.gem import schedule_concurrent
from bcir.kbcir.cost import TargetProfile
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass


def _mod(claims, rids=(10, 11, 12, 13, 14)) -> Module:
    m = Module(name="cc")
    for r in rids:
        m.add_resource(Resource(rid=r, name=f"r{r}"))
    m.add_phase(Phase(phase_id=0, deps=(), claims=claims))
    return m


def _c(cid, rd, wr, lane=Lane.U, stride=StrideClass.UNIT, op=Opcode.ADD):
    return Claim(id=cid, opcode=op, lane=lane, stride_class=stride, count=64, rd=rd, wr=wr)


def test_independent_claims_share_a_wave():
    # Three claims with disjoint writes / no shared reads co-execute.
    m = _mod([_c(1, (10,), (12,)), _c(2, (10,), (13,)), _c(3, (11,), (14,))])
    s = schedule_concurrent(m, TargetProfile.x86_avx512())  # 8 affinity domains
    assert s.waves == [[1, 2, 3]]
    assert s.max_parallelism() == 3
    assert s.contention == 0
    assert s.affinity == {1: 0, 2: 1, 3: 2}  # round-robin over domains


def test_conflicting_claims_serialize():
    # B reads what A writes (RAW) -> B must follow A in a later wave.
    m = _mod([_c(1, (10,), (12,)), _c(2, (12,), (13,))])
    s = schedule_concurrent(m, TargetProfile.x86_avx512())
    assert s.waves == [[1], [2]]


def test_ggg_tail_is_decoupled():
    # A random/gather claim is pulled out of the sequential waves.
    m = _mod([_c(1, (10,), (12,)), _c(9, (11,), (13,), lane=Lane.GGG, stride=StrideClass.RANDOM)])
    s = schedule_concurrent(m, TargetProfile.x86_avx512())
    assert s.ggg_tail == [9]
    assert s.waves == [[1]]


def test_oversubscription_is_contention():
    # A wave wider than the affinity domains thrashes the cache -> contention.
    claims = [_c(i, (10,), (10 + i,)) for i in range(1, 4)]  # 3 independent
    m = _mod(claims, rids=(10, 11, 12, 13))
    s = schedule_concurrent(m, TargetProfile.riscv_rvv())  # affinity_domains = 4
    assert s.contention == 0
    # NEON has 8 domains; force a small-domain target via a tweaked profile.
    from dataclasses import replace

    narrow = replace(TargetProfile.x86_avx512(), affinity_domains=2)
    s2 = schedule_concurrent(m, narrow)
    assert s2.contention == 1  # wave of 3 on 2 domains -> 1 oversubscribed


# --- G1 / S1-A: the ONE hazard predicate, built before the stream split --------------


def _fenced(cid, rd, wr, **kw):
    return Claim(
        id=cid,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=64,
        rd=rd,
        wr=wr,
        **kw,
    )


def test_hazard_edges_cross_the_stream_split():
    """A gather (the decoupled tail) that reads what a wave claim writes is a RAW across the
    streams, and the edge exists in both directions -- the parent build split the streams
    first and built the DAG over the main claims only (assessment row 6)."""
    from bcir.gem.concurrency import hazard_predecessors

    tail_reads = [
        _c(1, (10,), (12,)),
        _c(2, (12,), (13,), lane=Lane.GGG, stride=StrideClass.RANDOM),
    ]
    assert hazard_predecessors(tail_reads) == {1: [], 2: [1]}
    tail_writes = [
        _c(1, (12,), (10,), lane=Lane.GGG, stride=StrideClass.RANDOM),
        _c(2, (10,), (13,)),
    ]
    assert hazard_predecessors(tail_writes) == {1: [], 2: [1]}
    war = [_c(1, (10,), (13,)), _c(2, (12,), (10,), lane=Lane.GGG, stride=StrideClass.RANDOM)]
    assert hazard_predecessors(war) == {1: [], 2: [1]}


def test_a_fence_waits_for_everything_before_it_and_holds_everything_after():
    """A barriered or volatile claim conflicts with every other claim: it waits for every claim
    since the previous fence (and for that fence), and every later claim waits for it. No data
    hazard anywhere in this module -- the fence is the only edge."""
    from bcir.gem.concurrency import hazard_conflict, hazard_predecessors, is_fence

    a = _fenced(1, (10,), (11,))
    f = _fenced(2, (20,), (21,), hazard="barriered")
    b = _fenced(3, (30,), (31,))
    g = _fenced(4, (40,), (41,), hazard="barriered", volatile=True)
    c = _fenced(5, (50,), (51,))
    assert [is_fence(x) for x in (a, f, b, g, c)] == [False, True, False, True, False]
    assert hazard_predecessors([a, f, b, g, c]) == {1: [], 2: [1], 3: [2], 4: [2, 3], 5: [4]}
    assert hazard_conflict(f, c) and hazard_conflict(c, f) and hazard_conflict(g, a)
    assert not hazard_conflict(a, c)  # the control: independent unique claims do not conflict


def test_hazard_predecessors_without_a_fence_are_the_data_dag():
    """On a fence-free module the DAG is byte-identical to the RAW/WAR/WAW predecessor lists --
    the existing pins (`kbcir.async_awaits`, the audit checksums) do not move."""
    from bcir.gem.concurrency import _conflict_predecessors, hazard_predecessors

    tangle = [_c(i, (10 + i % 3, 20 + i % 7), (30 + i % 5,)) for i in range(1, 60)]
    assert hazard_predecessors(tangle) == _conflict_predecessors(tangle)


def test_the_bundle_reorderer_and_the_schedulers_share_the_hazard_predicate():
    """`kbcir.bundle._conflict` (the reorder fence of ASM3b) IS `concurrency.hazard_conflict`:
    one predicate decides what may be reordered, bundled, overlapped or awaited (L14)."""
    from bcir.gem.concurrency import hazard_conflict
    from bcir.kbcir.bundle import _conflict as bundle_conflict

    cases = [
        (_fenced(1, (10,), (11,)), _fenced(2, (11,), (12,))),  # RAW
        (_fenced(1, (10,), (11,)), _fenced(2, (20,), (21,))),  # independent
        (_fenced(1, (10,), (11,), hazard="barriered"), _fenced(2, (20,), (21,))),  # fence
        (_fenced(1, (10,), (11,)), _fenced(2, (20,), (21,), volatile=True)),  # volatile fence
    ]
    assert [bundle_conflict(a, b) for a, b in cases] == [hazard_conflict(a, b) for a, b in cases]
    assert [bundle_conflict(a, b) for a, b in cases] == [True, False, True, True]
