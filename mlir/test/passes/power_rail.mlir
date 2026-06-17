// RUN: bcir-opt -bcir-power-rail %s | FileCheck %s
//
// -bcir-power-rail ports gem.schedule.schedule_power_rail: a per-slot DVFS overlay on the EFT
// placed timeline (the join of -bcir-schedule-eft and -bcir-dvfs). The two compute claims place on
// domains 0/1 at [0,7808) and [0,5888) (the schedule_eft timeline). Each slot is classified by its
// base compute:memory mix -- both are bandwidth-bound (compute 64 vs memory 3840 -> intensity 16
// <= 250 -> memory), so each slot downclocks to Q8 192 (x0.75) over its own interval. The modeled
// energy saved is sum over downclocked slots of ((256-192) * duration * 1000) / 256
//   = (64*7808*1000)/256 + (64*5888*1000)/256 = 1952000 + 1472000 = 3424000.

bcir.module @rail {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>,
    affinity_domains = 8 : i32, mem_channels = 4 : i32
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    %a = bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %b = bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %c = bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %d = bcir.resource @D { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %e = bcir.resource @E { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
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

// CHECK-LABEL: bcir.module @rail
// CHECK-SAME: kbcir.rail_energy_saved = 3424000 : i64
// CHECK-SAME: kbcir.rail_knee = 4 : i64
// CHECK-SAME: kbcir.rail_makespan = 7808 : i64
// c1: domain 0, the [0,7808) slot -- memory-bound -> downclock to 192.
// CHECK: bcir.claim @c1
// CHECK-SAME: kbcir.rail_class = "memory"
// CHECK-SAME: kbcir.rail_clock = 192 : i64
// CHECK-SAME: kbcir.rail_domain = 0 : i64
// CHECK-SAME: kbcir.rail_finish = 7808 : i64
// c2: domain 1, the shorter [0,5888) slot -- also memory-bound -> downclock.
// CHECK: bcir.claim @c2
// CHECK-SAME: kbcir.rail_class = "memory"
// CHECK-SAME: kbcir.rail_clock = 192 : i64
// CHECK-SAME: kbcir.rail_domain = 1 : i64
// CHECK-SAME: kbcir.rail_finish = 5888 : i64
