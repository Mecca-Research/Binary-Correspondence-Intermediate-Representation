// RUN: bcir-opt -bcir-overlap %s | FileCheck %s
//
// The (max,+) scheduled price M(pi,Theta) (step 4: gem/overlap.py ported). Two
// independent same-phase claims (no read/write conflict) that each pick vec16 fan out
// across the target's 8 affinity domains -- one per bin -- so they co-execute: the
// wave makespan is max(7808, 7808) = 7808, not the serial sum 13696. overlap_gain =
// 5888 (exactly the oracle's price_scheduled(fused_chain)). A bin alone has no in-bin
// predecessor, so neither claim takes the shared-input discount here (that is a
// back-to-back effect).

bcir.module @overlap {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>,
    affinity_domains = 8 : i32
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

// CHECK-LABEL: bcir.module @overlap
// CHECK-DAG: kbcir.overlap_makespan = 7808
// CHECK-DAG: kbcir.overlap_serial = 13696
// CHECK-DAG: kbcir.overlap_gain = 5888
