// RUN: bcir-opt -bcir-compose %s | FileCheck %s
//
// -bcir-compose under a budget: the central K_BCIR equation min M(pi,Theta) s.t.
// R(pi,Theta) <= B over the region tree. With a kbcir.budget present, each Leaf is priced by
// the constrained label DP (rcsp.optimize_constrained).
//
// @budget: a thermal cap of 800 makes wide vec16 (thermal 1088) INFEASIBLE, so @ew's leaf
// re-prices to the feasible vec8 -> 9472 (still feasible). @tight: a cap of 400 that no legal
// realization can satisfy -> @hot is infeasible.

bcir.module @budget {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.kbcir.budget @b { dims = ["thermal"], caps = array<i64: 800> }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.kbcir.func @ew {
    bcir.claim @ec attributes {
      claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
      count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
      stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
      verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
    } { %i = bcir.index_range 0 to 1024 step 1 }
  }
}

bcir.module @tight {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.kbcir.budget @b { dims = ["thermal"], caps = array<i64: 400> }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.kbcir.func @hot {
    bcir.claim @hc attributes {
      claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
      count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
      stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
      verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
    } { %i = bcir.index_range 0 to 1024 step 1 }
  }
}

// the thermal cap forbids vec16; the leaf re-prices to the feasible vec8 (9472).
// CHECK: bcir.kbcir.func @ew
// CHECK-SAME: kbcir.compose_feasible = true
// CHECK-SAME: kbcir.compose_worst = 9472 : i64
// the tighter cap admits no realization -> infeasible.
// CHECK: bcir.kbcir.func @hot
// CHECK-SAME: kbcir.compose_feasible = false
