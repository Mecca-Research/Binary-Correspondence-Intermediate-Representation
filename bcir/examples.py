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



def scan_chain(n: int = 1024) -> Module:
    """A two-stage dependency chain (a scan/pipeline): T = A + B, then O = T + C.
    The second claim reads the first's output (RAW), so the two *serialize* -- the
    scheduler cannot overlap them (overlap_gain 0), the counterpart to fused_chain
    where independent claims do overlap. They DO fuse, though: c2 consumes T which c1
    produced, so producer->consumer **deforestation** discounts c2's memory (the
    intermediate T need not round-trip) -- a lower plan score even though the schedule
    still serializes (makespan == serial: the gain is fusion, not overlap)."""
    m = Module(name="scan_chain")
    for rid, nm in ((80, "A"), (81, "B"), (82, "C"), (83, "T"), (84, "O")):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(n,), name=nm))
    c1 = Claim(id=8001, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
               count=n, rd=(80, 81), wr=(83,), op="vector.add", domain=Domain.RAM)
    c2 = Claim(id=8002, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
               count=n, rd=(83, 82), wr=(84,), op="vector.add", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[c1, c2]))
    return m


def tiled_matmul(n: int = 256) -> Module:
    """A tile matmul-accumulate claim (T lane) -- the single-claim *skeleton* kept
    for back-compat. `matmul_tiled` is the real, multi-tile blocked kernel."""
    m = Module(name="tiled_matmul")
    m.add_resource(Resource(rid=40, domain=Domain.RAM, shape=(n, n), name="A"))
    m.add_resource(Resource(rid=41, domain=Domain.RAM, shape=(n, n), name="B"))
    m.add_resource(Resource(rid=42, domain=Domain.HBM, shape=(n, n), name="C"))
    claim = Claim(id=4000, opcode=Opcode.T_MACC, lane=Lane.T, stride_class=StrideClass.TILE,
                  count=n * n, rd=(40, 41), wr=(42,), op="linalg.matmul", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[claim]))
    return m


# --- the widened corpus: real tiled matmul / scan / multi-claim histogram --------
#
# The optimization claims must generalize past toy kernels. These three builders
# replace the single-claim skeletons with the real workload geometry: a blocked
# matmul whose K-accumulation is a register-resident dependency chain, a multi-stage
# scan/prefix pipeline, and a map/reduce histogram of independent gathers feeding a
# reduction. Each carries Python<->MLIR parity across the six TARGETS (see
# bcir.kbcir.differential + mlir/test/passes/gem_corpus.mlir).

def matmul_tiled(n: int = 256, tile: int = 128) -> Module:
    """A real blocked matmul C[n,n] = A[n,n] @ B[n,n], tiled into `tile`x`tile`
    blocks (the T lane). Each output tile C[i,j] accumulates over the K dimension as
    a register-resident chain of T_MACC claims -- the accumulator reads *and* writes
    the same C tile, so producer->consumer **deforestation** keeps it off the memory
    bus (k>0 reads the partial from k-1); distinct output tiles are independent
    (wave overlap). This is the structured workload the single-T_MACC `tiled_matmul`
    skeleton stood in for: the optimizer now sees real tile geometry, not one
    mega-claim."""
    g = max(1, n // max(1, tile))
    if g * g > 900_000:
        raise ValueError(f"matmul_tiled grid {g}x{g} too large (n//tile must keep g^2 < 9e5)")
    m = Module(name="matmul_tiled")
    tcount = tile * tile

    # Non-overlapping per-array RID spaces (each holds up to g^2 < 1e6 tiles), so the
    # builder stays R1-legal for any g, not just the default 2x2.
    def aid(i, k): return 1_000_000 + i * g + k
    def bid(k, j): return 2_000_000 + k * g + j
    def cid(i, j): return 3_000_000 + i * g + j

    for i in range(g):
        for k in range(g):
            m.add_resource(Resource(rid=aid(i, k), domain=Domain.RAM,
                                    shape=(tile, tile), name=f"A_{i}_{k}"))
    for k in range(g):
        for j in range(g):
            m.add_resource(Resource(rid=bid(k, j), domain=Domain.RAM,
                                    shape=(tile, tile), name=f"B_{k}_{j}"))
    for i in range(g):
        for j in range(g):
            m.add_resource(Resource(rid=cid(i, j), domain=Domain.HBM,
                                    shape=(tile, tile), name=f"C_{i}_{j}"))

    claims = []
    claim_id = 4000
    for i in range(g):
        for j in range(g):
            for k in range(g):
                # k>0 accumulates into the existing C tile (read it back -> fuse it).
                rd = (aid(i, k), bid(k, j)) + ((cid(i, j),) if k > 0 else ())
                claims.append(Claim(id=claim_id, opcode=Opcode.T_MACC, lane=Lane.T,
                                    stride_class=StrideClass.TILE, count=tcount, rd=rd,
                                    wr=(cid(i, j),), op="linalg.matmul", domain=Domain.RAM))
                claim_id += 1
    m.add_phase(Phase(phase_id=0, deps=(), claims=claims))
    return m


def scan(n: int = 4096, stages: int = 4) -> Module:
    """A multi-stage scan / prefix pipeline: O0 = A + B, then Ok = O(k-1) + Ck for
    k = 1..stages-1. Each stage reads the previous stage's output (a RAW hazard), so
    the stages **serialize** -- but they fuse: producer->consumer deforestation keeps
    each intermediate off the memory bus. The widened counterpart to the 2-claim
    `scan_chain`: a real dependency chain the scheduler must respect."""
    stages = max(1, stages)
    m = Module(name="scan")
    m.add_resource(Resource(rid=8000, domain=Domain.RAM, shape=(n,), name="A"))
    m.add_resource(Resource(rid=8001, domain=Domain.RAM, shape=(n,), name="B"))
    claims = []
    prev_out = None
    for s in range(stages):
        out_rid = 8100 + s
        m.add_resource(Resource(rid=out_rid, domain=Domain.RAM, shape=(n,), name=f"O{s}"))
        if s == 0:
            rd = (8000, 8001)
        else:
            c_rid = 8200 + s
            m.add_resource(Resource(rid=c_rid, domain=Domain.RAM, shape=(n,), name=f"C{s}"))
            rd = (prev_out, c_rid)
        claims.append(Claim(id=8300 + s, opcode=Opcode.ADD, lane=Lane.U,
                            stride_class=StrideClass.UNIT, count=n, rd=rd, wr=(out_rid,),
                            op="vector.add", domain=Domain.RAM))
        prev_out = out_rid
    m.add_phase(Phase(phase_id=0, deps=(), claims=claims))
    return m


def multi_histogram(n: int = 1024, bins: int = 3) -> Module:
    """A multi-claim histogram: `bins` partial-histogram gathers over a shared input
    (random access -> the GGG lane) run as independent waves, then a reduction merges
    the partials in a second phase. Exercises the gather lane across *multiple* claims
    and a cross-phase reduce -- the map/reduce histogram the single-claim
    `histogram_gather` stood in for."""
    bins = max(1, bins)
    m = Module(name="multi_histogram")
    m.add_resource(Resource(rid=9000, domain=Domain.RAM, shape=(n,), name="INP"))
    part_rids = []
    gathers = []
    for b in range(bins):
        prid = 9100 + b
        m.add_resource(Resource(rid=prid, domain=Domain.RAM, shape=(n,), name=f"PART{b}"))
        part_rids.append(prid)
        gathers.append(Claim(id=9300 + b, opcode=Opcode.GGG_LOAD, lane=Lane.GGG,
                             stride_class=StrideClass.RANDOM, count=n, rd=(9000,),
                             wr=(prid,), op="histogram.scatter", domain=Domain.RAM))
    m.add_resource(Resource(rid=9200, domain=Domain.RAM, shape=(n,), name="OUT"))
    merge = Claim(id=9400, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
                  count=n, rd=tuple(part_rids), wr=(9200,), op="reduce.add", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=gathers))
    m.add_phase(Phase(phase_id=1, deps=(0,), claims=[merge]))
    return m


PROGRAMS = {
    "vector_add": vector_add,
    "vector_add_hbm": vector_add_hbm,
    "saxpy_strided": saxpy_strided,
    "histogram_gather": histogram_gather,
    "gather_reduce": gather_reduce,
    "fused_chain": fused_chain,
    "scan_chain": scan_chain,
    "tiled_matmul": tiled_matmul,
    "matmul_tiled": matmul_tiled,
    "scan": scan,
    "multi_histogram": multi_histogram,
}

# The widened corpus (the real, multi-claim workloads), carried with Python<->MLIR
# parity across the six TARGETS by bcir.kbcir.differential.
CORPUS = ("matmul_tiled", "scan", "multi_histogram")
