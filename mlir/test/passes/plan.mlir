// RUN: bcir-opt -bcir-plan %s | FileCheck %s
//
// The layered min-plus shortest path (the full realize.optimize in C++, step 3).
// Two independent elementwise claims that SHARE a read operand (A): c1 = A+B -> C,
// c2 = A+E -> D. Each picks vec16; in the coupled shortest path the wide c2, whose
// wide predecessor c1 shares the read A, earns the path-based shared-input fusion
// discount (x0.75 memory: 3840 -> 2880), so its edge prices 5888 instead of 7808.
// Total 13696 -- exactly the oracle's optimize(fused_chain) (a coupling the per-claim
// argmin of -bcir-select-realization cannot see).

bcir.module @plan {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    %a = bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %b = bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %c = bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %e = bcir.resource @E { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %d = bcir.resource @D { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@A, @E], writes = [@D],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// CHECK-LABEL: bcir.module @plan
// the coupled plan total: 7808 + 5888.
// CHECK: kbcir.plan_score = 13696
// c1: full price, vec16.
// CHECK: bcir.claim @c1
// CHECK-SAME: kbcir.plan_cost = 7808
// CHECK-SAME: kbcir.plan_width = 16
// c2: shared-input fusion discount (x0.75 memory) along the chosen path.
// CHECK: bcir.claim @c2
// CHECK-SAME: kbcir.plan_cost = 5888
// CHECK-SAME: kbcir.plan_width = 16
