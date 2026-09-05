// RUN: bcir-opt -bcir-schedule %s | FileCheck %s
//
// S0-6 (row 9 of the 2026-07/08 assessment): ONE canonical phase order on both rails. The
// phase ids here are declared OUT of dependency order -- @p0 (id 0) depends on @p1 (id 1) --
// so the oracle's topological_phase_ids yields [1, 0] and every oracle consumer (the planner,
// R9, the GEM scheduler, overlap.py::_makespan) runs p1's claims first. -bcir-schedule used
// to sort by the numeric phase id and gave @c1 (phase @p0) exec_order 0, BEFORE the phase it
// depends on: an exec order that violated the phase DAG. Every law-rail consumer now ranks
// phases by BCIRPassSupport.h canonicalPhaseOrder (dependency-first, roots in textual order),
// the twin of the oracle's order, so @c2 runs first.
//
// CHECK-LABEL: bcir.module @out_of_order_ids
// CHECK: bcir.claim @c1 attributes {{.*}}kbcir.exec_order = 1
// CHECK: bcir.claim @c2 attributes {{.*}}kbcir.exec_order = 0
// CHECK: bcir.claim @c3 attributes {{.*}}kbcir.exec_order = 2

bcir.module @out_of_order_ids {
  bcir.registry @RES {
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p1 { id = 1 : i32, deps = [] }
  bcir.phase @p0 { id = 0 : i32, deps = [@p1] }
  bcir.phase @p2 { id = 2 : i32, deps = [@p0] }
  bcir.claim @c1 attributes { claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A], writes = [@A], count = 4 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32, domain = #bcir.domain<ram>,
    hazard = #bcir.hazard<unique>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict> } { %i = bcir.index_range 0 to 4 step 1 }
  bcir.claim @c2 attributes { claim_id = 2 : i32, phase = @p1, op = "vector.add", reads = [@A], writes = [@A], count = 4 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32, domain = #bcir.domain<ram>,
    hazard = #bcir.hazard<unique>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict> } { %i = bcir.index_range 0 to 4 step 1 }
  bcir.claim @c3 attributes { claim_id = 3 : i32, phase = @p2, op = "vector.add", reads = [@A], writes = [@A], count = 4 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32, domain = #bcir.domain<ram>,
    hazard = #bcir.hazard<unique>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict> } { %i = bcir.index_range 0 to 4 step 1 }
}
