// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// §5.14 Phase 2: the VOLATILE qualifier on the claim rail is a structural ordering law -- R5 requires
// an ordered (atomic/barriered) hazard on a volatile access (a volatile access -- an MMIO register or a
// volatile DMA/shared buffer -- must never ride the default `unique` hazard, which carries no
// ordering contract; a RAM-domain volatile buffer isolates R5 from the R3 MMIO-write rules). Mirrors the oracle's
// verify.verify() R5 volatile clause; vacuous unless a claim opts into `is_volatile`
// (the non-disturbance invariant -- the whole existing corpus carries the false default).

// (1 -- NEGATIVE) a volatile claim with the default `unique` hazard: no ordering contract -> R5.
bcir.module @r5_volatile_unique {
  bcir.registry @RES {
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R5: claim v volatile access requires an atomic/barriered hazard}}
  bcir.claim @v attributes {
    claim_id = 1 : i32, phase = @p0, op = "c.load", reads = [@A], writes = [@B],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<assumed_safe>, is_volatile = true
  } { %i = bcir.index_range 0 to 8 step 1 }
}

// -----

// (2 -- POSITIVE) the same volatile claim under a `barriered` hazard is legal (the ordering contract the
// cfront volatile lowering always emits: hazard=barriered + volatile=true).
bcir.module @r5_volatile_barriered {
  bcir.registry @RES {
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @v attributes {
    claim_id = 1 : i32, phase = @p0, op = "c.load", reads = [@A], writes = [@B],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<barriered>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<assumed_safe>, is_volatile = true
  } { %i = bcir.index_range 0 to 8 step 1 }
}
