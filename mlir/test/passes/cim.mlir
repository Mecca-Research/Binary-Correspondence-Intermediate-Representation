// RUN: bcir-opt -bcir-cim %s | FileCheck %s
//
// -bcir-cim recomputes gem.cim.cim_decision: for a reduction, core_cost (stream every operand
// byte + reduce, = count*(elem_bytes + mem_unit)) vs pim_cost (in-memory compute x1.5 + a fixed
// 4096 dispatch setup + a 1-element result). Offload iff pim strictly wins -- only large
// reductions. @big (count 4096) offloads (core 20480 > pim 10244); @small (count 1024) does not
// (core 5120 < pim 5636); @va is not a reduction and is never annotated.

bcir.module @cim {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.registry @RES {
    %t = bcir.resource @T { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4096>, layout = #bcir.layout<soa> } : !bcir.resource
    %acc = bcir.resource @ACC { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @big attributes {
    claim_id = 1 : i32, phase = @p0, op = "reduce.add", reads = [@T], writes = [@ACC],
    count = 4096 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 4096 step 1 }
  bcir.claim @small attributes {
    claim_id = 2 : i32, phase = @p0, op = "reduce.add", reads = [@T], writes = [@ACC],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @va attributes {
    claim_id = 3 : i32, phase = @p0, op = "vector.add", reads = [@T], writes = [@ACC],
    count = 4096 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 4096 step 1 }
}

// CHECK: bcir.claim @big
// CHECK-SAME: kbcir.cim_core_cost = 20480 : i64
// CHECK-SAME: kbcir.cim_offload = true
// CHECK-SAME: kbcir.cim_pim_cost = 10244 : i64
// CHECK: bcir.claim @small
// CHECK-SAME: kbcir.cim_offload = false
// the non-reduction is never offloaded (no cim annotation at all).
// CHECK: bcir.claim @va
// CHECK-NOT: kbcir.cim_offload
