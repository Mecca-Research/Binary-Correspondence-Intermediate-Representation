// RUN: bcir-opt -bcir-bundle %s | FileCheck %s
//
// -bcir-bundle joint-reorder: detection + the re-priced joint reorder. c1 and c3 both read
// @r1 (independent: disjoint writes), but they are *interleaved* with c2 (which shares
// nothing), so the declared-order plan c1,c2,c3 never places the two sharers adjacent and
// misses the shared-input fusion discount. The pass reorders the cost-model columns so the
// bundle is contiguous (c1,c3) and re-runs the min-plus shortest path: the wide c3, whose
// wide predecessor c1 now shares the read @r1, earns the x0.75 memory discount (7808 ->
// 5888), so the plan drops 23424 -> 21504 -- a joint gain of 1920 the pairwise plan cannot
// see. The transformation is a re-price only: it annotates the gain, it never mutates the IR.

bcir.module @reorder {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @r1 { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r2 { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r3 { rid = 3 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r4 { rid = 4 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r5 { rid = 5 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r6 { rid = 6 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r7 { rid = 7 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r8 { rid = 8 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@r1, @r2], writes = [@r3],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@r4, @r5], writes = [@r6],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c3 attributes {
    claim_id = 3 : i32, phase = @p0, op = "vector.add", reads = [@r1, @r7], writes = [@r8],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// CHECK-LABEL: bcir.module @reorder
// exactly one bundle, and the joint reorder recovers a strictly positive gain.
// CHECK: kbcir.bundle_count = 1
// CHECK-SAME: kbcir.bundle_total_gain = {{[1-9][0-9]*}}
// c1 carries its bundle index + the re-priced joint gain (the sharers, contiguous, win).
// CHECK: bcir.claim @c1 attributes
// CHECK-SAME: kbcir.bundle = 0 : i64
// CHECK-SAME: kbcir.bundle_gain = {{[1-9][0-9]*}}
// CHECK-SAME: kbcir.bundle_shared = @r1
// c3 is the other member of the same bundle.
// CHECK: bcir.claim @c3 attributes
// CHECK-SAME: kbcir.bundle = 0 : i64
// CHECK-SAME: kbcir.bundle_shared = @r1
