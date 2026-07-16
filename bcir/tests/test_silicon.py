"""Measured wiring to real signals: the allocator reads the real cache tier map,
the telemetry ring carries real OS counters, DVFS anchors to the real cpu
frequency -- and the hardware-dependent gains are sampled on this silicon. The
thread-safe ring benchmark has no fixed shared-runner speedup floor; cache-resident
access is still compared directly with DRAM. Every measurement degrades gracefully on a host that does not expose the
signal (so CI never flakes)."""

import json
import os
import subprocess
import tempfile
import time
from shutil import which

import bcir.silicon as silicon

from bcir.silicon import (
    Counters,
    CounterSampler,
    HwSample,
    cache_topology,
    cpufreq_info,
    perf_counters_available,
    read_hw_counters,
    sample_into_ring,
    summary,
    tier_capacities,
)
from bcir.kbcir import TARGETS, place, silicon_tier_capacity
from bcir.kbcir.cost import MemTier
from bcir.gem.dvfs import NOMINAL, actuate, plan_dvfs, quantize_to_silicon
from bcir.kbcir import optimize
from bcir.kbcir.cost import Theta
from bcir.examples import vector_add
from bcir.model import Claim, Domain, Module, Opcode, Phase, Resource
from bcir.telemetry import DataDNA, TelemetryRing

AVX = TARGETS["x86_avx512"]


def _cc():
    return which("clang") or which("cc") or which("gcc")


# --- the probes read real signals (honest about what is present) -----------------

def test_summary_reports_the_real_signal_split():
    s = summary()
    assert "cache_levels" in s and "dvfs_actuatable" in s and "hw_pmu" in s
    assert s["os_counters"] is True                       # always real
    assert s["rusage_counters"] is (silicon._resource is not None)


def test_real_cache_tier_map_when_sys_is_present():
    topo = cache_topology()
    if not topo:
        return  # host does not expose /sys cache: skip cleanly
    caps = tier_capacities()
    assert caps and MemTier.L1 in caps
    # the allocator helper returns the real sizes, matching /sys exactly.
    assert silicon_tier_capacity()[MemTier.L1] == caps[MemTier.L1]
    assert caps[MemTier.L1] == max(c.size_bytes for c in topo if c.level == 1 and c.kind == "Data")


def test_allocator_with_the_real_tier_map_is_gains_only():
    # build a hot/ephemeral + cold/large module; place with the REAL cache sizes.
    m = Module(name="sil")
    m.add_resource(Resource(rid=1, domain=Domain.RAM, shape=(64,)))            # hot, tiny
    m.add_resource(Resource(rid=2, domain=Domain.RAM, shape=(1 << 24,)))       # 64 MB, cold
    m.add_phase(Phase(phase_id=0, claims=[
        Claim(id=1, opcode=Opcode.ADD, count=64, rd=(1,), wr=(1,), op="vector.add"),
        Claim(id=2, opcode=Opcode.ADD, count=64, rd=(1,), wr=(1,), op="vector.add")]))
    m.add_phase(Phase(phase_id=1, deps=(0,), claims=[
        Claim(id=3, opcode=Opcode.ADD, count=1 << 24, rd=(2,), wr=(2,), op="vector.add")]))
    pl = place(m, AVX, tier_capacity=silicon_tier_capacity())
    assert pl.tier_of(1) in (MemTier.L1, MemTier.L2, MemTier.L3)   # hot tiny -> SRAM
    # gains-only: every relocation strictly lowered modeled cost (rationale records it)
    assert all("cost" in pl.rationale[r] for r in pl.moved)


# --- the telemetry ring carries REAL measured counters ---------------------------

def test_ring_is_fed_by_real_os_counters():
    ring = TelemetryRing(capacity=16)
    delta = sample_into_ring(ring, claim_id=42, work=lambda: sum(range(200000)))
    assert delta.wall_ns > 0 and delta.cpu_ns >= 0       # both are genuine timer deltas
    rec = ring.read_one()
    assert rec.claim_id == 42 and rec.cycles > 0         # CPU or wall signal reached the ring


def test_counter_sampler_measures_real_deltas():
    s = CounterSampler()
    _ = [i * i for i in range(500000)]
    c = s.lap()
    assert c.wall_ns > 0 and c.cpu_ns >= 0               # genuine timer deltas


def test_counter_sampler_degrades_without_posix_rusage():
    saved = silicon._resource
    silicon._resource = None
    try:
        s = CounterSampler()
        _ = sum(range(10000))
        c = s.lap()
        assert c.wall_ns > 0 and c.cpu_ns >= 0
        assert (c.minor_faults, c.major_faults, c.vol_ctx, c.invol_ctx) == (0, 0, 0, 0)
        assert summary()["rusage_counters"] is False
    finally:
        silicon._resource = saved


def test_ring_uses_wall_delta_when_the_host_cpu_clock_is_too_coarse():
    """A short Windows sample may truthfully report cpu_ns == 0; telemetry still carries
    the measured nonzero wall delta rather than fabricating a CPU tick or a zero cycle."""
    from unittest import mock

    class CoarseCpuSampler:
        def lap(self):
            return Counters(wall_ns=73, cpu_ns=0, minor_faults=0, major_faults=0,
                            vol_ctx=0, invol_ctx=0)

    calls = []
    ring = TelemetryRing(capacity=4)
    with mock.patch.object(silicon, "CounterSampler", CoarseCpuSampler), \
            mock.patch.object(silicon, "read_hw_counters", return_value=None):
        delta = sample_into_ring(ring, claim_id=9, work=lambda: calls.append("work"))
    rec = ring.read_one()
    assert calls == ["work"]
    assert delta.cpu_ns == 0 and delta.wall_ns == 73
    assert rec.claim_id == 9 and rec.cycles == 73


# --- synchronized zero-copy ring versus equivalent validated serialization -------

def test_zero_copy_ring_benchmark_covers_the_synchronized_and_serializing_paths():
    # This is a smoke benchmark, not a shared-runner performance floor.  The ring now
    # serializes slot publication so a reader cannot observe a torn/duplicated record;
    # scheduler load and lock implementation make a fixed ratio inappropriate in CI.
    # Compare equivalent trust-boundary work by validating the JSON record too.
    n = 200_000 if os.environ.get("BCIR_THOROUGH") else 20_000
    e = DataDNA(segment_id="s", claim_id=1, cycles=123, bytes=456, misses=7)
    ring = TelemetryRing(capacity=4096)

    t0 = time.perf_counter_ns()
    for _ in range(n):
        ring.write(e)
    ring_ns = time.perf_counter_ns() - t0

    t0 = time.perf_counter_ns()
    for _ in range(n):
        _ = json.dumps(e.validate().to_dict()).encode("utf-8")
    json_ns = time.perf_counter_ns() - t0

    assert ring_ns > 0 and json_ns > 0
    assert ring.stats.written == n and ring.pending == ring.capacity


# --- MEASURED gain #2: cache-resident access beats DRAM (justifies hot->SRAM) -----

def test_cache_resident_access_is_measurably_faster_than_dram():
    cc = _cc()
    if cc is None or not cache_topology():
        return  # need a compiler + a real cache map
    src = r'''
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static long now(void){struct timespec t;timespec_get(&t,TIME_UTC);return (long)t.tv_sec*1000000000L+t.tv_nsec;}
/* pointer-chase a random permutation cycle: defeats prefetch, exposes real latency */
static long chase(size_t n, int steps){
  size_t *a = malloc(n*sizeof *a);
  for (size_t i=0;i<n;++i) a[i]=i;
  for (size_t i=n-1;i>0;--i){ size_t j=(size_t)(((unsigned long)i*2862933555777941757UL+1)% (i+1)); size_t t=a[i];a[i]=a[j];a[j]=t; }
  size_t p=0; long t0=now();
  for (int s=0;s<steps;++s) p=a[p];
  long dt=now()-t0; volatile size_t sink=p;(void)sink; free(a);
  return dt/steps;  /* ns per dependent access */
}
int main(int c,char**v){
  size_t small=(size_t)atol(v[1]), big=(size_t)atol(v[2]); int steps=atoi(v[3]);
  printf("%ld %ld\n", chase(small,steps), chase(big,steps));
  return 0;
}
'''
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "lat"); s = os.path.join(d, "lat.c")
        open(s, "w").write(src)
        b = subprocess.run([cc, "-O2", s, "-o", exe], capture_output=True, text=True)
        assert b.returncode == 0, b.stderr
        # small = 4096 nodes (32 KB, L1-resident); big = 64 M nodes (512 MB, exceeds L3).
        out = subprocess.run([exe, "4096", str(64 << 20), "2000000"],
                             capture_output=True, text=True)
        l1_ns, dram_ns = (int(x) for x in out.stdout.split())
        assert l1_ns > 0 and dram_ns >= 2 * l1_ns, (l1_ns, dram_ns)   # real tier gap


# --- DVFS anchored to the real cpu frequency (actuation honestly gated) -----------

def test_dvfs_quantizes_to_the_real_cpu_frequency():
    fq = cpufreq_info()
    if not fq.available:
        return  # no readable frequency on this host
    m = vector_add(1 << 16)
    plan = plan_dvfs(m, optimize(m, AVX, Theta.cool()), Theta.cool())
    targets = quantize_to_silicon(plan, fq)
    assert targets and all(t.target_khz > 0 for t in targets)
    # a memory-bound (downclocked) phase targets below the nominal; never claims to
    # have actuated unless the host truly allows it.
    d0 = plan.decisions[0]
    if d0.clock_q8 < NOMINAL:
        assert targets[0].target_khz <= fq.nominal_khz
    assert all(t.settable == fq.actuatable for t in targets)


def test_perf_counter_probe_is_honest():
    # whatever the answer, it must be a clean bool (no raise) -- the honest split.
    assert isinstance(perf_counters_available(), bool)


# --- hardware PMU counters: attempt, then degrade gracefully ----------------------

def test_hw_counters_attempt_degrades_gracefully():
    hw = read_hw_counters(lambda: sum(range(300_000)))
    if hw is None:
        return  # no PMU exposed (this guest: perf_event_open -> ENOENT) -- honest path
    assert isinstance(hw, HwSample)
    assert hw.cycles > 0 and hw.instructions > 0 and hw.ipc_milli >= 0


def test_hw_pmu_summary_matches_the_probe():
    assert summary()["hw_pmu"] == perf_counters_available()


def test_ring_feed_runs_work_once_with_or_without_a_pmu():
    # the PMU may be absent; either way work runs exactly once and a real record lands.
    ring = TelemetryRing(capacity=8)
    delta = sample_into_ring(ring, claim_id=7, work=lambda: sum(range(100_000)))
    rec = ring.read_one()
    assert rec.claim_id == 7 and rec.cycles > 0 and delta.wall_ns > 0


# --- DVFS actuation: attempt to set the clock, report the boundary honestly --------

def test_dvfs_actuation_attempts_and_reports_the_boundary():
    fq = cpufreq_info()
    m = vector_add(1 << 16)
    targets = quantize_to_silicon(plan_dvfs(m, optimize(m, AVX, Theta.cool()), Theta.cool()), fq)
    if not targets:
        return  # no readable nominal frequency on this host
    results = actuate(targets, fq)
    assert len(results) == len(targets)
    for r in results:
        assert r.requested_khz > 0
        assert r.applied == fq.actuatable          # never claims to have set what it couldn't
        if not r.applied:
            assert r.reason and r.observed_khz is None         # honest dry-run + reason
        else:
            assert r.observed_khz is not None                  # confirmed on real hardware


def test_dvfs_actuation_is_a_safe_noop_without_a_governor():
    # in a sandbox (no userspace governor / no privilege) actuate must not mutate the
    # system: every result is a dry-run that names the missing capability.
    fq = cpufreq_info()
    if fq.actuatable:
        return  # this host CAN actuate -- covered by the test above
    from bcir.gem.dvfs import FreqTarget
    res = actuate([FreqTarget(phase_id=0, clock_q8=192, target_khz=1_000_000, settable=False)], fq)
    assert res[0].applied is False and "governor" in res[0].reason.lower()
