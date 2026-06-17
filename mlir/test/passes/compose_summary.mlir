// RUN: bcir-opt -bcir-compose %s | FileCheck %s
//
// -bcir-compose inter-procedural summary costs: @ew is planned ONCE over its formals
// (@f1/@f2/@f3, DRAM) -> a summary of 7808. @prog calls it twice:
//   * actuals @A/@B/@C (DRAM)   -- cost-compatible (same domain/count/access) -> REUSE 7808.
//   * actuals @H1/@H2/@H3 (HBM) -- the cost-key differs -> the body is re-priced with the HBM
//     resources substituted -> 2816 (HBM's cheaper memory tier).
// So @prog worst = 7808 + 2816 = 10624, and exactly one leaf plan was saved by reuse
// (compose_reused = 1) -- precisely plan_composite(summaries=...) on the oracle.

bcir.module @compose {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    %f1 = bcir.resource @f1 { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %f2 = bcir.resource @f2 { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %f3 = bcir.resource @f3 { rid = 3 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %a = bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %b = bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %c = bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %h1 = bcir.resource @H1 { rid = 20 : i32, domain_kind = #bcir.domain<hbm>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %h2 = bcir.resource @H2 { rid = 21 : i32, domain_kind = #bcir.domain<hbm>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %h3 = bcir.resource @H3 { rid = 22 : i32, domain_kind = #bcir.domain<hbm>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // a function over the three formals: a single elementwise add -> the summary prices to 7808.
  bcir.kbcir.func @ew attributes { formals = [@f1, @f2, @f3] } {
    bcir.claim @ec attributes {
      claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@f1, @f2], writes = [@f3],
      count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
      stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
      verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
    } { %i = bcir.index_range 0 to 1024 step 1 }
  }
  // one cost-compatible call (reuses the summary) + one HBM call (re-priced).
  bcir.kbcir.func @prog {
    bcir.kbcir.call @ew { formals = [@f1, @f2, @f3], actuals = [@A, @B, @C] }
    bcir.kbcir.call @ew { formals = [@f1, @f2, @f3], actuals = [@H1, @H2, @H3] }
  }
}

// CHECK: bcir.kbcir.func @ew
// CHECK-SAME: kbcir.compose_expected = 7808 : i64
// CHECK-SAME: kbcir.compose_reused = 0 : i64
// CHECK-SAME: kbcir.compose_worst = 7808 : i64
// CHECK: bcir.kbcir.func @prog
// CHECK-SAME: kbcir.compose_expected = 10624 : i64
// CHECK-SAME: kbcir.compose_reused = 1 : i64
// CHECK-SAME: kbcir.compose_worst = 10624 : i64
