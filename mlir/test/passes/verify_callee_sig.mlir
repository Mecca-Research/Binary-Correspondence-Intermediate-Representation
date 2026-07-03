// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// §5.14 Phase 2 (indirect-call effect): a dispatch claim's DECLARED callee signature, when
// carried, must be well-formed "ret(params)" -- the R18 clause, dual-rail with the oracle's
// verify() check. Vacuous when the attr is absent (the opaque-edge default: the whole existing
// corpus). The conservative global effect set is the oracle-side half of the same record.

// (1 -- NEGATIVE) a malformed signature on an indirect dispatch claim.
bcir.module @r18_malformed_sig {
  bcir.registry @RES {
    bcir.resource @P { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @T { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R18: claim d malformed indirect-callee signature 'not-a-signature' (expected 'ret(params)')}}
  bcir.claim @d attributes {
    claim_id = 1 : i32, phase = @p0, op = "c.call.indirect", reads = [@P], writes = [@T],
    count = 1 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<scalar>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<assumed_safe>,
    callee_sig = "not-a-signature"
  } { %i = bcir.index_range 0 to 1 step 1 }
}

// -----

// (2 -- POSITIVE) the well-formed record the cfront lowering stamps.
bcir.module @r18_wellformed_sig {
  bcir.registry @RES {
    bcir.resource @P { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @T { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @d attributes {
    claim_id = 1 : i32, phase = @p0, op = "c.call.indirect", reads = [@P], writes = [@T],
    count = 1 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<scalar>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<assumed_safe>,
    callee_sig = "unsigned int(unsigned int)"
  } { %i = bcir.index_range 0 to 1 step 1 }
}
