// RUN: bcir-opt -bcir-alloc-pool %s | FileCheck %s
//
// -bcir-alloc-pool ports kbcir.allocator.pool_plan: resources whose live ranges (the
// [first_phase, last_phase] span) are disjoint share a memory arena. @A/@B are live only in
// phase 0 (read by @c1); @D/@E only in phase 1; @C bridges both (written in 0, read in 1). So
// @A and @D can share an arena (pool 0), @B and @E another (pool 1), @C its own (pool 2) ->
// peak 3*4096 = 12288 vs naive 5*4096 = 20480, saving 8192. Greedy left-edge, deterministic.

bcir.module @alloc {
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @D { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @E { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.phase @p1 { id = 1 : i32, deps = [@p0] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p1, op = "vector.add", reads = [@C, @D], writes = [@E],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// CHECK-LABEL: bcir.module @alloc
// CHECK-SAME: kbcir.pool_naive_bytes = 20480 : i64
// CHECK-SAME: kbcir.pool_peak_bytes = 12288 : i64
// CHECK-SAME: kbcir.pool_saved = 8192 : i64
// CHECK: bcir.resource @A
// CHECK-SAME: kbcir.pool_id = 0 : i64
// CHECK: bcir.resource @B
// CHECK-SAME: kbcir.pool_id = 1 : i64
// CHECK: bcir.resource @C
// CHECK-SAME: kbcir.pool_id = 2 : i64
// CHECK: bcir.resource @D
// CHECK-SAME: kbcir.pool_id = 0 : i64
// CHECK: bcir.resource @E
// CHECK-SAME: kbcir.pool_id = 1 : i64
