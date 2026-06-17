// RUN: bcir-opt -bcir-cost-model %s | FileCheck %s
//
// Intra-phase fusion in the C++ cost model (step 2: realize.fused_candidates ported).
// Processing claims in (phase, declared) order with value-numbering + a produced-rid
// set, -bcir-cost-model applies the two redundancy credits, matching the oracle's
// fused_candidates bit-for-bit:
//   * @c1  (A+B -> T)          : first occurrence, full price                 -> 7808
//   * @c2  (T+C -> O)          : consumes c1's T  => producer->consumer        -> 5888
//                               deforestation (x0.75 memory: 3840 -> 2880)
//   * @dup (A+B -> E)          : duplicates c1's (op, operand-versions) => CSE -> 5100
//                               (compute zeroed, memory copy-priced 3840 -> 2550)

bcir.module @cost_model_fusion {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @T { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @O { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @E { rid = 15 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@T],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@T, @C], writes = [@O],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @dup attributes {
    claim_id = 3 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@E],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// CHECK-LABEL: bcir.module @cost_model_fusion
// c1: full price, no fusion credit (vec16 @ 7808, memory 3840).
// CHECK: bcir.claim @c1
// CHECK-SAME: kbcir.cm_min_cost = #bcir.costvec<compute = 64, memory = 3840
// CHECK-SAME: kbcir.cm_min_score = 7808
// c2: producer->consumer deforestation (x0.75 memory: 3840 -> 2880).
// CHECK: bcir.claim @c2
// CHECK-SAME: kbcir.cm_fusion = "deforest"
// CHECK-SAME: kbcir.cm_min_cost = #bcir.costvec<compute = 64, memory = 2880
// CHECK-SAME: kbcir.cm_min_score = 5888
// dup: CSE -- compute zeroed, memory copy-priced (3840 -> 2550).
// CHECK: bcir.claim @dup
// CHECK-SAME: kbcir.cm_fusion = "cse"
// CHECK-SAME: kbcir.cm_min_cost = #bcir.costvec<compute = 0, memory = 2550
// CHECK-SAME: kbcir.cm_min_score = 5100
