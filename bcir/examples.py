"""Canonical BCIR example programs (the goal graphs G).

Each builder returns a `Module`. `PROGRAMS` maps a CLI/test name to a builder, in
the spirit of the LangRef's `examples/*.mlir` corpus.
"""

from __future__ import annotations

from .model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass


def vector_add(n: int = 1024) -> Module:
    """C = A + B over RAM (the LangRef canonical correspondence-stack example)."""
    m = Module(name="vec_add")
    m.add_resource(Resource(rid=10, domain=Domain.RAM, shape=(n,), name="A"))
    m.add_resource(Resource(rid=11, domain=Domain.RAM, shape=(n,), name="B"))
    m.add_resource(Resource(rid=12, domain=Domain.RAM, shape=(n,), name="C"))
    add = Claim(id=1000, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
                count=n, rd=(10, 11), wr=(12,), op="vector.add", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[add]))
    return m


def vector_add_hbm(n: int = 1024) -> Module:
    """vector_add with the read source A resident in HBM (cheaper memory tier)."""
    m = Module(name="vec_add_hbm")
    m.add_resource(Resource(rid=10, domain=Domain.HBM, shape=(n,), name="A"))
    m.add_resource(Resource(rid=11, domain=Domain.HBM, shape=(n,), name="B"))
    m.add_resource(Resource(rid=12, domain=Domain.HBM, shape=(n,), name="C"))
    add = Claim(id=1000, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
                count=n, rd=(10, 11), wr=(12,), op="vector.add", domain=Domain.HBM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[add]))
    return m


def saxpy_strided(n: int = 1024, k: int = 4) -> Module:
    """A strided AXPY-like claim: exercises strided-vs-gather candidate choice."""
    m = Module(name="saxpy_strided")
    m.add_resource(Resource(rid=20, domain=Domain.RAM, shape=(n * k,), name="X"))
    m.add_resource(Resource(rid=21, domain=Domain.RAM, shape=(n,), name="Y"))
    claim = Claim(id=2000, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.STRIDED,
                  count=n, stride_k=k, rd=(20,), wr=(21,), op="vector.axpy", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[claim]))
    return m


def histogram_gather(n: int = 1024, ham: bool = False) -> Module:
    """A random-access (gather) claim. With ``ham=True`` the table uses O(log n) access."""
    m = Module(name="histogram_ham" if ham else "histogram_gather")
    m.add_resource(Resource(rid=30, domain=Domain.RAM, shape=(n,),
                            access="ham" if ham else "flat", name="TABLE"))
    m.add_resource(Resource(rid=31, domain=Domain.RAM, shape=(n,), name="OUT"))
    claim = Claim(id=3000, opcode=Opcode.GGG_LOAD, lane=Lane.GGG, stride_class=StrideClass.RANDOM,
                  count=n, rd=(30,), wr=(31,), op="histogram.scatter", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[claim]))
    return m


def tiled_matmul(n: int = 256) -> Module:
    """A tile matmul-accumulate claim (T lane)."""
    m = Module(name="tiled_matmul")
    m.add_resource(Resource(rid=40, domain=Domain.RAM, shape=(n, n), name="A"))
    m.add_resource(Resource(rid=41, domain=Domain.RAM, shape=(n, n), name="B"))
    m.add_resource(Resource(rid=42, domain=Domain.HBM, shape=(n, n), name="C"))
    claim = Claim(id=4000, opcode=Opcode.T_MACC, lane=Lane.T, stride_class=StrideClass.TILE,
                  count=n * n, rd=(40, 41), wr=(42,), op="linalg.matmul", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[claim]))
    return m


PROGRAMS = {
    "vector_add": vector_add,
    "vector_add_hbm": vector_add_hbm,
    "saxpy_strided": saxpy_strided,
    "histogram_gather": histogram_gather,
    "tiled_matmul": tiled_matmul,
}
