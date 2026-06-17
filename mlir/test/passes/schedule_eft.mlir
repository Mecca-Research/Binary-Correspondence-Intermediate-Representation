// RUN: bcir-opt -bcir-schedule-eft %s | FileCheck %s
//
// -bcir-schedule-eft ports gem.schedule.schedule_eft: duration-aware EFT wave scheduling. Two
// independent compute claims share a read (@B) so the plan prices them 7808 and 5888 (the
// shared-input discount). With 8 affinity domains they run in parallel: the LPT scheduler
// dispatches the longer @c1 (7808) to domain 0 and @c2 (5888) to domain 1, both starting at 0,
// so the phase makespan is max(7808, 5888) = 7808. The bandwidth knee is min(8, 4) = 4.

bcir.module @sched {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>,
    affinity_domains = 8 : i32, mem_channels = 4 : i32
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @D { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @E { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, cost_class = #bcir.cost_class<compute>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@D, @B], writes = [@E],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, cost_class = #bcir.cost_class<compute>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// CHECK-LABEL: bcir.module @sched
// CHECK-SAME: kbcir.sched_knee = 4 : i64
// CHECK-SAME: kbcir.sched_makespan = 7808 : i64
// CHECK: bcir.claim @c1
// CHECK-SAME: kbcir.sched_domain = 0 : i64
// CHECK-SAME: kbcir.sched_finish = 7808 : i64
// CHECK: bcir.claim @c2
// CHECK-SAME: kbcir.sched_domain = 1 : i64
// CHECK-SAME: kbcir.sched_finish = 5888 : i64
