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
