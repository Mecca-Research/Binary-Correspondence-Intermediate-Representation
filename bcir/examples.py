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


def gather_reduce(n: int = 1024) -> Module:
    """A reduction over a permutation gather: ACC = sum_i TABLE[perm[i]].

    Because + is commutative and perm is a permutation, this equals sum_i TABLE[i]
    -- so the gather (random TABLE access) has a semantically identical *blocked*
    (sequential) realization. BCIR's cost model picks blocked (no gather_penalty);
    a naive compiler, unable to prove perm is a permutation, must gather. The
    `reduce.gather` op signals the reducible-permutation property to the planner."""
    m = Module(name="gather_reduce")
    m.add_resource(Resource(rid=50, domain=Domain.RAM, shape=(n,), name="TABLE"))
    m.add_resource(Resource(rid=51, domain=Domain.RAM, shape=(1,), name="ACC"))
    # The access is a permutation, but because + is commutative the claim is
    # *realizable* unit-stride (the blocked sum) -- that is its canonical lane; the
    # gather is the de-optimized alternative the cost model rejects. (UNIT keeps the
    # claim and the selected blocked plan legal under the lane law R6/R9.)
    claim = Claim(id=5000, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
                  count=n, rd=(50,), wr=(51,), op="reduce.gather", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[claim]))
    return m


def fused_chain(n: int = 1024) -> Module:
    """Two independent elementwise claims that share a read operand (A): the
    multi-claim case where (max,+) overlap pricing and fusion earn their keep --
    C = A + B and D = A + E run as concurrent waves (makespan < serial), and
    back-to-back in a bin they reuse A's loads (the fusion discount)."""
    m = Module(name="fused_chain")
    for rid, nm in ((60, "A"), (61, "B"), (62, "C"), (63, "E"), (64, "D")):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(n,), name=nm))
    c1 = Claim(id=6001, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
               count=n, rd=(60, 61), wr=(62,), op="vector.add", domain=Domain.RAM)
    c2 = Claim(id=6002, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
               count=n, rd=(60, 63), wr=(64,), op="vector.add", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[c1, c2]))
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
    "gather_reduce": gather_reduce,
    "fused_chain": fused_chain,
    "tiled_matmul": tiled_matmul,
}
