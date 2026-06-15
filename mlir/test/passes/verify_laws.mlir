// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// The semantic verifier: module/claim laws R1-R7 as a real MLIR pass. Each
// section is an independent module whose single law violation is asserted via
// -verify-diagnostics (mirrors bcir/verify in the oracle; the plan/stream/
// lowering laws R8-R12 live in verify_laws_deep.mlir).

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

// -----

// R3: the claim's domain contract is not backed by any touched resource.
bcir.module @r3 {
  bcir.registry @RES {
    %a = bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R3: claim c declares a domain not backed by any touched resource}}
  bcir.claim @c attributes {
    claim_id = 1 : i32, phase = @p0, op = "x", reads = [@A], writes = [], count = 4 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<hbm>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 4 step 1 }
}

// -----

// R5: atomic semantics (lane A / atomic op) without an ordered hazard contract.
bcir.module @r5 {
  bcir.registry @RES {
    %t = bcir.resource @T { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 64>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R5: claim c atomic semantics require an atomic/barriered hazard}}
  bcir.claim @c attributes {
    claim_id = 1 : i32, phase = @p0, op = "atomic.add", reads = [@T], writes = [@T], count = 64 : i64,
    lane = #bcir.lane<a>, stride_class = #bcir.stride_class<random>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 64 step 1 }
}

// -----

// R5: a same-phase conflict through the decoupled GGG tail (CT2) loses its wave
// serialization; the unordered end is flagged (the reader carries `atomic`).
bcir.module @r5_tail {
  bcir.registry @RES {
    %t = bcir.resource @T { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %o = bcir.resource @O { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R5: claim w conflicts across the decoupled GGG tail in phase @p0 without an atomic/barriered hazard}}
  bcir.claim @w attributes {
    claim_id = 1 : i32, phase = @p0, op = "histogram.scatter", reads = [], writes = [@T], count = 1024 : i64,
    lane = #bcir.lane<ggg>, stride_class = #bcir.stride_class<random>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @r attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@T], writes = [@O], count = 1024 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<atomic>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// -----

// R7: a strict-bounds affine claim statically overruns its resource.
bcir.module @r7 {
  bcir.registry @RES {
    %a = bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R7: claim c read of @A overruns the resource (extent 8 > 4)}}
  bcir.claim @c attributes {
    claim_id = 1 : i32, phase = @p0, op = "x", reads = [@A], writes = [], count = 8 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 8 step 1 }
}

// -----

// R7: a data-dependent (random) access with strict bounds and no runtime verify
// contract cannot discharge its bounds obligation.
bcir.module @r7_random {
  bcir.registry @RES {
    %a = bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %o = bcir.resource @O { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R7: claim c data-dependent access with strict bounds requires a runtime verify contract}}
  bcir.claim @c attributes {
    claim_id = 1 : i32, phase = @p0, op = "histogram.gather", reads = [@A], writes = [@O], count = 1024 : i64,
    lane = #bcir.lane<ggg>, stride_class = #bcir.stride_class<random>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<none>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// -----

// R7 (reduction): a `reduce.*` claim accumulates count reads into a single
// location, so its write extent is one element, not count. Writing a 1-element
// accumulator with count 1024 is therefore clean (mirrors bcir/verify R7).
bcir.module @r7_reduce_ok {
  bcir.registry @RES {
    %t = bcir.resource @TABLE { rid = 50 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %acc = bcir.resource @ACC { rid = 51 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @r attributes {
    claim_id = 1 : i32, phase = @p0, op = "reduce.gather", reads = [@TABLE], writes = [@ACC], count = 1024 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// -----

// R7 (the rule is reduction-specific): a NON-reduction claim writing count
// elements into the same 1-element resource still overruns it.
bcir.module @r7_nonreduce_overrun {
  bcir.registry @RES {
    %t = bcir.resource @TABLE { rid = 50 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %acc = bcir.resource @ACC { rid = 51 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R7: claim w write of @ACC overruns the resource (extent 1024 > 1)}}
  bcir.claim @w attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.scatter", reads = [@TABLE], writes = [@ACC], count = 1024 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}
