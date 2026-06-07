"""Verifier (R1-R12 subset) tests."""

from bcir.examples import vector_add
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.verify import is_legal, verify


def test_vector_add_is_legal():
    assert is_legal(vector_add(1024))


def test_undeclared_rid_is_R2():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1))
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=4, rd=(1, 99), wr=(1,))]))
    laws = {d.law for d in verify(m)}
    assert "R2" in laws


def test_illegal_lane_for_stride_is_R6():
    m = Module(name="bad")
    m.add_resource(Resource(rid=1))
    # UNIT access on a GGG lane is illegal geometry.
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ADD, lane=Lane.GGG, stride_class=StrideClass.UNIT,
              count=4, rd=(1,), wr=(1,))]))
    laws = {d.law for d in verify(m)}
    assert "R6" in laws


def test_phase_cycle_is_R4():
    m = Module(name="cyclic")
    m.add_phase(Phase(phase_id=0, deps=(1,)))
    m.add_phase(Phase(phase_id=1, deps=(0,)))
    laws = {d.law for d in verify(m)}
    assert "R4" in laws
