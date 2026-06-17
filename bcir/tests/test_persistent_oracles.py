"""Persistent/active oracle deltas (the second pass over the adaptive layer).

Tests the refinements that hold gains: the C-side zero-copy telemetry ring
(memory_model + telemetry), a-priori ranker-confidence-gated sensing (accel +
sensing), liveness-based pool allocation (allocator), timeline DVFS
(schedule_power_rail), and CIM/PIM spatial partitioning (c_kernel). The JIT
specialist is included as a *correctness* check only -- it is measured-neutral on
bandwidth-bound kernels (see the PR / strategy doc), so no speedup is pinned."""

import mmap
import os
import subprocess
import tempfile
from shutil import which

from bcir.examples import gather_reduce, vector_add
from bcir.kbcir import TARGETS, optimize, pool_plan, sense_by_ranker
from bcir.kbcir.allocator import live_intervals
from bcir.kbcir.accel import FrozenRanker
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.realize import candidates_for
from bcir.gem.schedule import bandwidth_knee, durations_from, schedule_eft, schedule_power_rail
from bcir.gem.dvfs import DOWNCLOCK, NOMINAL
from bcir.lower.c_kernel import is_pim_target, optimize_spatial
from bcir.lower.memory_model import c_memory_order, emit_ring_header_c
from bcir.model import Claim, Domain, Module, Opcode, Phase, Resource
from bcir.telemetry import DataDNA, TelemetryRing, parse_shared_ring

AVX = TARGETS["x86_avx512"]


def _cc():
    return which("clang") or which("cc") or which("gcc")


# --- #7: the C-side zero-copy telemetry ring -------------------------------------

def test_ring_header_publishes_with_release_ordering():
    c = emit_ring_header_c()
    assert "memory_order_release" in c            # publish the record before bumping head
    assert "atomic_store_explicit" in c and "bcir_ring_write" in c
    assert c_memory_order("release") == "memory_order_release"


def test_parse_shared_ring_reads_the_python_written_layout():
    # build the shared layout in Python and read it back (no compiler needed).
    import struct
    cap = 4
    buf = bytearray(32 + cap * 56)
    struct.pack_into("<4Q", buf, 0, 3, cap, 56, 0)            # head=3, capacity=4
    for i in range(3):
        struct.pack_into("<7q", buf, 32 + i * 56, 200 + i, 10 * (i + 1), 0, i, 0, 0, 0)
    recs = parse_shared_ring(buf)
    assert [r.claim_id for r in recs] == [200, 201, 202]
    assert [r.cycles for r in recs] == [10, 20, 30]


def test_c_kernel_writes_ring_python_reads_zero_copy():
    cc = _cc()
    if cc is None:
        return
    main = r'''
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
int main(int c, char**v){
  uint64_t cap=8; size_t sz = BCIR_RING_HEADER + cap*BCIR_RING_RECORD;
  int fd=open(v[1],O_RDWR|O_CREAT,0644); if(ftruncate(fd,(off_t)sz)) return 2;
  void*base=mmap(0,sz,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  bcir_ring_init(base,cap);
  for(int i=0;i<5;++i) bcir_ring_write(base, 100+i, 1000*(i+1), 0, i, 0,0,50);
  msync(base,sz,MS_SYNC); munmap(base,sz); close(fd); return 0;
}
'''
    with tempfile.TemporaryDirectory() as d:
        src, exe, shm = (os.path.join(d, x) for x in ("r.c", "r", "ring.bin"))
        open(src, "w").write("#define _GNU_SOURCE\n" + emit_ring_header_c() + "\n" + main)
        b = subprocess.run([cc, "-std=gnu23", "-O2", src, "-o", exe], capture_output=True, text=True)
        assert b.returncode == 0, b.stderr
        assert subprocess.run([exe, shm]).returncode == 0
        with open(shm, "rb") as f:
            recs = parse_shared_ring(mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ))
        assert [r.claim_id for r in recs] == [100, 101, 102, 103, 104]
        assert [r.cycles for r in recs] == [1000, 2000, 3000, 4000, 5000]   # C->Python, zero-copy


# --- #2: a-priori ranker-confidence-gated sensing --------------------------------

def _ranker():
    # a frozen ranker that weights the static-score feature -- so a clear best score
    # gives a wide margin, near-equal scores give a narrow one.
    return FrozenRanker(wq=[256, 0, 0, 0, 0])


def test_ranker_confidence_is_margin_between_top_two():
    r = _ranker()
    m = vector_add(1024)
    cands = candidates_for(m.phases[0].claims[0], AVX, m.resource(1))
    one = r.confidence(cands[:1], [10])
    assert one >= (1 << 29)                                   # nothing to resolve -> max
    assert r.confidence(cands, [100] * len(cands)) >= 0       # equal scores -> small margin


def test_sense_by_ranker_instruments_only_uncertain_columns():
    r = _ranker()
    m = vector_add(1024)
    cands = candidates_for(m.phases[0].claims[0], AVX, m.resource(1))
    # confident column: one score dominates -> wide margin -> telemetry off.
    confident = ("hot", cands, [1000] + [1] * (len(cands) - 1))
    # uncertain column: all scores equal -> ~zero margin -> instrument.
    uncertain = ("cold", cands, [50] * len(cands))
    decisions = {d.path: d.resolution for d in sense_by_ranker(r, [confident, uncertain],
                                                              margin_threshold=64)}
    assert decisions["cold"] == "high" and decisions["hot"] == "off"


# --- #4: liveness-based pool allocation ------------------------------------------

def _chain_module():
    # three tensors with DISJOINT lifetimes across 3 phases: t1 (p0), t2 (p1), t3 (p2).
    m = Module(name="chain")
    for rid in (1, 2, 3):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(1024,)))
    for pid, rid in enumerate((1, 2, 3)):
        deps = (pid - 1,) if pid else ()
        m.add_phase(Phase(phase_id=pid, deps=deps, claims=[
            Claim(id=10 + pid, opcode=Opcode.ADD, count=1024, rd=(rid,), wr=(rid,),
                  op="vector.add")]))
    return m


def test_pool_plan_reuses_arenas_for_disjoint_lifetimes():
    pp = pool_plan(_chain_module())
    # three single-phase tensors never overlap -> they share ONE arena.
    assert len(pp.pools) == 1 and pp.saved_bytes > 0
    assert pp.peak_bytes < pp.naive_bytes                     # a real footprint win


def test_pool_plan_is_a_noop_when_everything_is_simultaneously_live():
    # vector_add: A,B,C all live in the single phase -> no reuse possible.
    pp = pool_plan(vector_add(1024))
    assert pp.is_noop and pp.peak_bytes == pp.naive_bytes     # gains-only: never worse
    assert pp.peak_bytes <= pp.naive_bytes


def test_live_intervals_span_the_phase_range():
    iv = live_intervals(_chain_module())
    assert iv[1] == (0, 0) and iv[3] == (2, 2)


# --- #8: timeline-aware DVFS (schedule_power_rail) -------------------------------

def test_schedule_power_rail_downclocks_a_memory_bound_timeline():
    m = vector_add(1 << 16)
    r = optimize(m, AVX, Theta.cool())
    sched = schedule_eft(m, durations_from(r), AVX)
    rail = schedule_power_rail(sched, r, Theta.cool(), AVX)
    assert rail.decisions                                     # one per scheduled slot
    # vector_add is bandwidth-bound -> the rail downclocks it (power saved, throughput
    # unaffected); the modeled energy figure is non-negative.
    assert any(d.clock_q8 <= DOWNCLOCK for d in rail.decisions)
    assert rail.energy_saved_milli > 0
    assert all(64 <= d.clock_q8 <= 512 for d in rail.decisions)   # legal DVFS range


def test_schedule_power_rail_is_deterministic():
    m = vector_add(4096)
    r = optimize(m, AVX, Theta.cool())
    sched = schedule_eft(m, durations_from(r), AVX)
    a = schedule_power_rail(sched, r, Theta.cool(), AVX)
    b = schedule_power_rail(sched, r, Theta.cool(), AVX)
    assert a == b


# --- #5: CIM/PIM spatial partitioning (modeled; non-PIM target is a no-op) -------

def test_optimize_spatial_binds_a_reduction_to_pim_on_a_pim_target():
    pim_target = TargetProfile.x86_avx512()
    import dataclasses
    pim_target = dataclasses.replace(pim_target, isa_features=frozenset({"pim"}))
    assert is_pim_target(pim_target) and not is_pim_target(AVX)
    m = gather_reduce(1 << 20)                                # a large reduction
    r = optimize(m, pim_target, Theta.cool())
    plan = optimize_spatial(m, r, pim_target)
    assert plan.pim_capable and plan.offloaded                 # the reduction offloads
    assert plan.total_bytes_saved > 0                          # operand stream stays in memory


def test_optimize_spatial_is_a_noop_on_a_non_pim_target():
    m = gather_reduce(1 << 20)
    r = optimize(m, AVX, Theta.cool())
    plan = optimize_spatial(m, r, AVX)
    assert not plan.pim_capable and plan.is_noop and plan.total_bytes_saved == 0


# --- #6: JIT specialist is correctness-preserving but NOT a measured speedup ------

def test_specialist_is_correctness_preserving_only():
    # The specialist bakes n + unrolls; on bandwidth-bound elementwise it is
    # measured-neutral (generic/spec ~ 1.0 -- see the PR), so we assert CORRECTNESS,
    # not speed: it computes the same result as the generic kernel.
    cc = _cc()
    if cc is None:
        return
    from bcir.lower.c_kernel import emit_kernel_c
    from bcir.lower.specialist import emit_specialist_c
    n = 256
    m = vector_add(n)
    r = optimize(m, AVX, Theta.cool())
    spec = emit_specialist_c("vector.add", "+", n, "f32", "k_spec")
    gen = emit_kernel_c(m, r, "k_gen", hw_width=AVX.vector_width)
    main = f'''
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
int main(void){{ size_t n={n};
  float*A=malloc(n*4),*B=malloc(n*4),*Cs=malloc(n*4),*Cg=malloc(n*4);
  for(size_t i=0;i<n;++i){{A[i]=(float)i;B[i]=2.0f*(float)i;Cs[i]=Cg[i]=-1;}}
  k_spec(A,B,Cs); k_gen(A,B,Cg,n);
  for(size_t i=0;i<n;++i) if(Cs[i]!=Cg[i]){{printf("FAIL %zu\\n",i);return 1;}}
  puts("OK"); return 0; }}
'''
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "s.c"), os.path.join(d, "s")
        open(src, "w").write(spec + "\n" + gen + "\n" + main)
        b = subprocess.run([cc, "-std=c23", "-O2", src, "-o", exe], capture_output=True, text=True)
        assert b.returncode == 0, b.stderr
        run = subprocess.run([exe], capture_output=True, text=True)
        assert run.returncode == 0 and "OK" in run.stdout       # specialist == generic


def test_mlir_alloc_pool_constants_are_in_sync():
    """Pins the constants -bcir-alloc-pool annotates in mlir/test/passes/alloc_pool.mlir."""
    from bcir.kbcir.allocator import pool_plan
    from bcir.model import Module, Claim, Domain, Lane, Opcode, Resource, StrideClass, Phase
    m = Module(name="a")
    for r in (10, 11, 12, 13, 14):
        m.add_resource(Resource(rid=r, domain=Domain.RAM, shape=(1024,)))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[Claim(id=1, opcode=Opcode.ADD, lane=Lane.U,
        stride_class=StrideClass.UNIT, count=1024, rd=(10, 11), wr=(12,), op="vector.add", domain=Domain.RAM)]))
    m.add_phase(Phase(phase_id=1, deps=(0,), claims=[Claim(id=2, opcode=Opcode.ADD, lane=Lane.U,
        stride_class=StrideClass.UNIT, count=1024, rd=(12, 13), wr=(14,), op="vector.add", domain=Domain.RAM)]))
    pp = pool_plan(m)
    assert pp.naive_bytes == 20480 and pp.peak_bytes == 12288 and pp.saved_bytes == 8192
    assert pp.pool_of(10) == 0 and pp.pool_of(13) == 0   # A and D share an arena
    assert pp.pool_of(11) == 1 and pp.pool_of(14) == 1   # B and E share an arena
    assert pp.pool_of(12) == 2                            # C bridges both phases -> its own
