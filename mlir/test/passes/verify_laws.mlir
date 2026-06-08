// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// The semantic verifier (Phase 6): R1/R2/R4/R6 as a real MLIR pass. Each section
// is an independent module whose single law violation is asserted via
// -verify-diagnostics (mirrors bcir/verify in the oracle).

// R1: duplicate RID.
bcir.module @r1 {
  bcir.registry @RES {
    %a = bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa> } : !bcir.resource
    // expected-error @+1 {{R1: duplicate RID 10}}
    %b = bcir.resource @B { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa> } : !bcir.resource
  }
}

// -----

// R2: claim references an undeclared resource.
bcir.module @r2 {
  // expected-error @+1 {{R2: claim c reads undeclared resource @Missing}}
  bcir.claim @c attributes {
    claim_id = 1 : i32, phase = @p0, op = "x", reads = [@Missing], writes = [], count = 4 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 4 step 1 }
}

// -----

// R4: cyclic phase dependencies.
bcir.module @r4 {
  // expected-error @+1 {{R4: phase dependency cycle through @p0}}
  bcir.phase @p0 { id = 0 : i32, deps = [@p1] }
  bcir.phase @p1 { id = 1 : i32, deps = [@p0] }
}

// -----

// R6: lane illegal for the declared stride class (unit requires U, not GGG).
bcir.module @r6 {
  // expected-error @+1 {{R6: claim bad lane illegal for its stride class}}
  bcir.claim @bad attributes {
    claim_id = 1 : i32, phase = @p0, op = "x", reads = [], writes = [], count = 4 : i64,
    lane = #bcir.lane<ggg>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 4 step 1 }
}
