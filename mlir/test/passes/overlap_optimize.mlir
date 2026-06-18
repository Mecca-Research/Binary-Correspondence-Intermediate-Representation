// RUN: bcir-opt -bcir-overlap-optimize %s | FileCheck %s
//
// The makespan-driven re-selection sweep (gem/overlap.py::optimize_scheduled). Starting from
// the serial optimum, each claim is swept once and the alternative that strictly lowers the
// scheduled makespan is adopted. Here the serial optimum (both claims vec16) is already
// makespan-optimal -- the two independent same-phase claims fan out across the 8 affinity
// domains and co-execute (makespan max(7808,7808)=7808 < serial 13696), so no narrower lane
// improves it: the sweep is a no-op and must not churn the plan (opt_makespan == the priced
// makespan, opt_width stays 16). Re-priced serially so R9 holds (makespan <= serial).

bcir.module @overlap_opt {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>,
    affinity_domains = 8 : i32
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @E { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @D { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
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

// CHECK-LABEL: bcir.module @overlap_opt
// CHECK-DAG: kbcir.overlap_opt_makespan = 7808
// CHECK-DAG: kbcir.overlap_opt_serial = 13696
// CHECK-DAG: kbcir.overlap_opt_gain = 5888
// CHECK-DAG: kbcir.overlap_opt_width = 16
