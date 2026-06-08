// RUN: bcir-opt -bcir-promote-lanes %s | FileCheck %s
//
// The opt-law GGG -> UX promotion (Phase 6): a GGG/Random claim carrying the
// `bcir.bucketable` proof attribute is promoted to UX/Cacheline.

bcir.module @m {
  // CHECK: bcir.claim @gather
  bcir.claim @gather attributes {
    claim_id = 1 : i32, phase = @p0, op = "gather",
    reads = [@A], writes = [@B], count = 1024 : i64,
    lane = #bcir.lane<ggg>, stride_class = #bcir.stride_class<random>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    bcir.bucketable = true
  } {
    %i = bcir.index_range 0 to 1024 step 1
  }
  // CHECK-DAG: lane = #bcir.lane<ux>
  // CHECK-DAG: stride_class = #bcir.stride_class<cacheline>

  // A GGG/Random claim WITHOUT the proof is left untouched.
  // CHECK: bcir.claim @unproven
  // CHECK-DAG: lane = #bcir.lane<ggg>
  bcir.claim @unproven attributes {
    claim_id = 2 : i32, phase = @p0, op = "gather",
    reads = [@A], writes = [@B], count = 1024 : i64,
    lane = #bcir.lane<ggg>, stride_class = #bcir.stride_class<random>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } {
    %i = bcir.index_range 0 to 1024 step 1
  }
}
