"""Phase 8: !bcir.token async-dependency plan tests."""

from bcir.examples import vector_add
from bcir.gem import async_plan
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass


def _c(cid, rd, wr):
    return Claim(id=cid, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
                 count=8, rd=rd, wr=wr)


def _mod(claims, phases=None, rids=(10, 11, 12, 13)):
    m = Module(name="async")
    for r in rids:
        m.add_resource(Resource(rid=r, name=f"r{r}"))
    if phases is None:
        m.add_phase(Phase(phase_id=0, deps=(), claims=claims))
    else:
        for p in phases:
            m.add_phase(p)
    return m


def test_single_claim_is_independent():
    plan = async_plan(vector_add(1024))
    assert plan.forks == [1000]
    assert plan.is_independent(1000)
    assert plan.awaits[1000] == []


def test_independent_claims_await_nothing():
    m = _mod([_c(1, (10,), (12,)), _c(2, (11,), (13,))])
    plan = async_plan(m)
    assert plan.forks == [1, 2]
    assert plan.awaits[1] == [] and plan.awaits[2] == []


def test_raw_chain_awaits_producer():
    # 1: write r12 ; 2: read r12 write r13 ; 3: read r13 -> a minimal chain.
    m = _mod([_c(1, (10,), (12,)), _c(2, (12,), (13,)), _c(3, (13,), (11,))])
    plan = async_plan(m)
    assert plan.awaits[1] == []
    assert plan.awaits[2] == [1]   # 2 depends on 1's write
    assert plan.awaits[3] == [2]   # 3 depends on 2's write (not transitively on 1)


def test_cross_phase_dependency():
    load = _c(1, (10,), (10,))
    calc = _c(2, (10,), (12,))   # reads what phase-0 wrote
    m = _mod(None, phases=[Phase(0, (), [load]), Phase(1, (0,), [calc])])
    plan = async_plan(m)
    assert plan.awaits[2] == [1]
