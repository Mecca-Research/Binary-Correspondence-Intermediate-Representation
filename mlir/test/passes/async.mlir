// RUN: bcir-opt -bcir-async %s | FileCheck %s
//
// -bcir-async ports gem.async_tokens.async_plan + schedule.execute_tokens: the !bcir.token
// fork/await DAG drives a single cross-phase dispatch (no phase barriers). @c1 (phase 0) writes
// @C. @c2 (phase 1) is independent of @c1 (disjoint) -> it awaits nothing and starts at 0,
// OVERLAPPING phase 0 -- software pipelining. @c3 (phase 1) reads @C (which @c1 writes) -> it
// awaits @c1 and starts at 7808. The pipelined makespan is 15616 (vs 2*7808 phase-barriered).

bcir.module @async {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>,
    affinity_domains = 8 : i32, mem_channels = 4 : i32
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @D { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @E { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @F { rid = 15 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @G { rid = 16 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @H { rid = 17 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.phase @p1 { id = 1 : i32, deps = [@p0] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, cost_class = #bcir.cost_class<compute>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p1, op = "vector.add", reads = [@D, @E], writes = [@F],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, cost_class = #bcir.cost_class<compute>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c3 attributes {
    claim_id = 3 : i32, phase = @p1, op = "vector.add", reads = [@C, @G], writes = [@H],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, cost_class = #bcir.cost_class<compute>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// CHECK-LABEL: bcir.module @async
// CHECK-SAME: kbcir.async_makespan = 15616 : i64
// c1: phase 0, awaits nothing.
// CHECK: bcir.claim @c1
// CHECK-SAME: kbcir.async_awaits = []
// CHECK-SAME: kbcir.async_finish = 7808 : i64
// c2: phase 1 but INDEPENDENT -> overlaps phase 0 (starts at 0).
// CHECK: bcir.claim @c2
// CHECK-SAME: kbcir.async_awaits = []
// CHECK-SAME: kbcir.async_finish = 7808 : i64
// CHECK-SAME: kbcir.async_start = 0 : i64
// c3: awaits c1 (RAW on @C) -> starts after it.
// CHECK: bcir.claim @c3
// CHECK-SAME: kbcir.async_awaits = [1
// CHECK-SAME: kbcir.async_start = 7808 : i64
