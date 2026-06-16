// RUN: bcir-opt -bcir-rcsp-plan %s | FileCheck %s
//
// Plan-level constrained selection (step 5: rcsp.optimize_constrained ported). Two
// independent vec16 claims accumulate thermal 1088 + 1088 = 2176; a plan-wide cap of
// 2000 makes {vec16, vec16} infeasible, but {vec16, vec8} (thermal 1088 + 640 = 1728)
// fits -- the accumulated-budget label DP narrows just ONE claim, a decision the
// per-claim cap (-bcir-rcsp) cannot make. Constrained optimum 17280, exactly the
// oracle's optimize_constrained.

bcir.module @rcsp_plan {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.kbcir.budget @cap { dims = ["thermal"], caps = array<i64: 2000> }
  bcir.registry @RES {
    %r1 = bcir.resource @r1 { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %r2 = bcir.resource @r2 { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %r3 = bcir.resource @r3 { rid = 3 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %r4 = bcir.resource @r4 { rid = 4 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %r5 = bcir.resource @r5 { rid = 5 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %r6 = bcir.resource @r6 { rid = 6 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@r1, @r2], writes = [@r3],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@r4, @r5], writes = [@r6],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// CHECK-LABEL: bcir.module @rcsp_plan
// CHECK: kbcir.rcsp_plan_score = 17280
// c1 keeps vec16, c2 narrows to vec8 (the accumulated-budget decision).
// CHECK: bcir.claim @c1
// CHECK-SAME: kbcir.rcsp_width = 16
// CHECK: bcir.claim @c2
// CHECK-SAME: kbcir.rcsp_width = 8
