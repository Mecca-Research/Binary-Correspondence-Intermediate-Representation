// RUN: bcir-opt -bcir-cost-model %s | FileCheck %s
//
// The verification cost axis (the 12th K_BCIR cost dim) now has a real producer on the
// MLIR rail (mirrors realize._verify_cost). Two identical vec16 elementwise claims differ
// only in their verify contract:
//   * verify=bounds -> discharged for free (the bounds check fuses into the access the
//     memory axis already prices): verification = 0, score 7808 (unchanged).
//   * verify=exact  -> recompute + compare every element, O(n): verification = 1024,
//     so the score rises to 8832. (hash is priced the same.)
// The cost is width-independent, so the per-claim selection (vec16) is unchanged -- it is
// a tradeable resource on the plan (an RCSP cap or a verification-weighted policy), not a
// realization knob. Disjoint operands keep the two claims out of CSE/deforestation.

bcir.module @cost_model_verify {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx2", "avx512f", "fma"],
    lane_widths = array<i64: 1, 8, 16>, cacheline = 64 : i32, gather_penalty = 32 : i32
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    %a = bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %b = bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %c = bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %d = bcir.resource @D { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %e = bcir.resource @E { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %f = bcir.resource @F { rid = 15 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @add_bounds attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @add_exact attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@D, @E], writes = [@F],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<exact>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// CHECK-LABEL: bcir.module @cost_model_verify
// bounds: verification discharged free -> the classic vec16 cost vector (verification 0)
// and score 7808 (unchanged).
// CHECK-DAG: kbcir.cm_min_cost = #bcir.costvec<compute = 64, memory = 3840, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
// CHECK-DAG: kbcir.cm_min_score = 7808
// exact: same geometry + O(n) verification = 1024 -> score 8832 (same vec16 selection).
// CHECK-DAG: kbcir.cm_min_cost = #bcir.costvec<compute = 64, memory = 3840, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 1024>
// CHECK-DAG: kbcir.cm_min_score = 8832
