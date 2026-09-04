"""GEM StreamPack hydration + deterministic phase-executor tests (ir/ fold)."""

from bcir.examples import vector_add
from bcir.gem import execute, hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass


def _two_phase_module() -> Module:
    m = Module(name="two_phase")
    m.add_resource(Resource(rid=1, name="A"))
    m.add_resource(Resource(rid=2, name="C"))
    # phase 1 (calc) depends on phase 0 (load); claims given out of id order on purpose.
    load = Claim(
        id=20,
        opcode=Opcode.LOAD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=8,
        rd=(1,),
        wr=(1,),
    )
    c_hi = Claim(
        id=11,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=8,
        rd=(1,),
        wr=(2,),
    )
    c_lo = Claim(
        id=10,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=8,
        rd=(1,),
        wr=(2,),
    )
    m.add_phase(Phase(phase_id=0, deps=(), claims=[load]))
    m.add_phase(Phase(phase_id=1, deps=(0,), claims=[c_hi, c_lo]))
    return m


def test_execute_is_deterministic_and_phase_ordered():
    m = _two_phase_module()
    r = execute(m)
    # Topological phase order (load before calc), and within the calc phase claims
    # dispatch by ascending id (10 before 11) under deterministic mode.
    assert r.phase_order == [0, 1]
    assert r.order == [20, 10, 11]
    assert r.executed == 3
    assert [p.scheduled for p in r.phases] == [1, 2]
    # Reproducible run to run.
    assert execute(m).order == r.order


def test_execute_invokes_kernels():
    m = _two_phase_module()
    hits: list[int] = []
    kernels = {cid: (lambda i=cid: hits.append(i)) for cid in (20, 10, 11)}
    r = execute(m, kernels=kernels)
    assert hits == [20, 10, 11]
    assert r.executed == 3


def test_hydrate_then_execute_vector_add():
    m = vector_add(1024)
    res = optimize(m, TargetProfile.x86_avx512(), Theta.cool())
    pack = hydrate(m, res)
    assert pack.provenance_ok()
    r = execute(m)
    assert r.order == [1000] and r.executed == 1
