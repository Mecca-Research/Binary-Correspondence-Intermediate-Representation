// RUN: bcir-opt -bcir-compose %s | FileCheck %s
//
// -bcir-compose: compositional cost over the kbcir.func/call/cond region tree (the law-rail
// compose.plan_composite). @leaf is a single vector_add (a Leaf -> 7808). @prog is a prologue
// Leaf followed by a Cond("hot", then = call @leaf, else = call @leaf twice, p = 600):
//   worst    = prologue 7808 + PRED 8 + max(7808, 2*7808)               = 23432
//   expected = prologue 7808 + PRED 8 + (600*7808 + 400*15616)/1000      = 18747
// exactly plan_composite's worst / expected (the leaf reuses the K_BCIR cost model -> 7808).

bcir.module @compose {
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
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // a single elementwise add -> the Leaf prices to 7808.
  bcir.kbcir.func @leaf {
    bcir.claim @lc attributes {
      claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
      count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
      stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
      verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
    } { %i = bcir.index_range 0 to 1024 step 1 }
  }
  // a prologue Leaf followed by a Cond: the hot branch calls @leaf once, the cold branch twice.
  bcir.kbcir.func @prog {
    bcir.claim @pc attributes {
      claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
      count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
      stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
      verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
    } { %i = bcir.index_range 0 to 1024 step 1 }
    bcir.kbcir.cond "hot" attributes { prob_then_milli = 600 : i64 } then {
      bcir.kbcir.call @leaf { }
    } else {
      bcir.kbcir.call @leaf { }
      bcir.kbcir.call @leaf { }
    }
  }
}

// CHECK: bcir.kbcir.func @leaf
// CHECK-SAME: kbcir.compose_expected = 7808 : i64
// CHECK-SAME: kbcir.compose_worst = 7808 : i64
// CHECK: bcir.kbcir.func @prog
// CHECK-SAME: kbcir.compose_expected = 18747 : i64
// CHECK-SAME: kbcir.compose_worst = 23432 : i64
