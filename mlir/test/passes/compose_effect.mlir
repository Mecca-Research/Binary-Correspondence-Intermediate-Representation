// RUN: bcir-opt -bcir-compose %s | FileCheck %s
//
// -bcir-compose alias/effect modeling + dynamic shapes (the Tier-1 compose remainder):
//   * kbcir.effect_reads / effect_writes  -- a func's read/write footprint, folded through its
//     calls' argument substitution (compose.effect).
//   * kbcir.commutes_with_prev            -- on each call with a preceding sibling: true iff its
//     footprint is disjoint from the predecessor's (compose.independent -- the two reorder/overlap).
//   * kbcir.compose_dynamic               -- true iff a leaf is a dynamic-shape claim, so the
//     cost is a worst-case bound that holds for any actual size <= the count (plan_holds_for).

bcir.module @compose {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @f1 { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @f2 { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @f3 { rid = 3 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @X { rid = 20 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @Y { rid = 21 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @Z { rid = 22 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // a leaf function over three formals: footprint reads {f1,f2}, writes {f3}; not dynamic.
  bcir.kbcir.func @ew attributes { formals = [@f1, @f2, @f3] } {
    bcir.claim @ec attributes {
      claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@f1, @f2], writes = [@f3],
      count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
      stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
      verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
    } { %i = bcir.index_range 0 to 1024 step 1 }
  }
  // two calls over disjoint resources -> the second commutes with the first.
  bcir.kbcir.func @indep {
    bcir.kbcir.call @ew { formals = [@f1, @f2, @f3], actuals = [@A, @B, @C] }
    bcir.kbcir.call @ew { formals = [@f1, @f2, @f3], actuals = [@X, @Y, @Z] }
  }
  // the second call reads @C, which the first writes (RAW) -> it does NOT commute.
  bcir.kbcir.func @dep {
    bcir.kbcir.call @ew { formals = [@f1, @f2, @f3], actuals = [@A, @B, @C] }
    bcir.kbcir.call @ew { formals = [@f1, @f2, @f3], actuals = [@C, @Y, @Z] }
  }
  // a dynamic-shape leaf: count is a static upper bound -> the plan is a worst-case bound.
  bcir.kbcir.func @dyn {
    bcir.claim @dc attributes {
      claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
      count = 1024 : i64, dynamic = true, lane = #bcir.lane<u>,
      stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32, domain = #bcir.domain<ram>,
      hazard = #bcir.hazard<unique>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
    } { %i = bcir.index_range 0 to 1024 step 1 }
  }
}

// CHECK-LABEL: bcir.kbcir.func @ew
// CHECK-SAME: kbcir.compose_dynamic = false
// CHECK-SAME: kbcir.effect_reads = [@f1, @f2]
// CHECK-SAME: kbcir.effect_writes = [@f3]
// CHECK-LABEL: bcir.kbcir.func @indep
// CHECK: kbcir.commutes_with_prev = true
// CHECK-LABEL: bcir.kbcir.func @dep
// CHECK: kbcir.commutes_with_prev = false
// CHECK-LABEL: bcir.kbcir.func @dyn
// CHECK-SAME: kbcir.compose_dynamic = true
