// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// §5.14 Phase 2: the EXTENT-PROVENANCE signal + the dual-railed R7 masked clause. A masked
// (runtime-bounds-checked) access must DECLARE the `bounds` verify contract -- previously an
// oracle-only clause; the MLIR rail now enforces it too, and surfaces WHY the access is checked
// (the `bounds_provenance` attr the cfront lowering stamps) in the diagnostic when carried.

// (1 -- NEGATIVE) a masked claim with verify<none>: the promotion would emit no guard; the
// provenance names the extent source in the diagnostic.
bcir.module @r7_masked_no_guard {
  bcir.registry @RES {
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 64>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 64>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R7: claim m masked (runtime-bounds-checked) access must carry a 'bounds' verify contract (extent provenance: declared_extent)}}
  bcir.claim @m attributes {
    claim_id = 1 : i32, phase = @p0, op = "c.load", reads = [@A], writes = [@C],
    count = 64 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<none>, bounds = #bcir.bounds<masked>,
    bounds_provenance = "declared_extent"
  } { %i = bcir.index_range 0 to 64 step 1 }
}

// -----

// (2 -- POSITIVE) the promoted access as the cfront lowering emits it: masked + verify<bounds>
// + the provenance carried -- legal, and the attr round-trips.
bcir.module @r7_masked_guarded {
  bcir.registry @RES {
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 64>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 64>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @m attributes {
    claim_id = 1 : i32, phase = @p0, op = "c.load", reads = [@A], writes = [@C],
    count = 64 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<masked>,
    bounds_provenance = "recovered_count"
  } { %i = bcir.index_range 0 to 64 step 1 }
}
