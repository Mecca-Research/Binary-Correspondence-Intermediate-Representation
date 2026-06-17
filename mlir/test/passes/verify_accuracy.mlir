// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// R17 (accuracy-contract legality), first-class in -bcir-verify and dual-rail with
// verify.verify_accuracy. A claim that declares a precision tolerance (tol > 0) must
// realize within it: a reduce.* over `count` terms drifts up to `count` ULP with the
// naive accumulator (exact = false) but only 1 ULP when compensated (exact = true), so a
// tight tolerance on a long reduction is the law that forces the compensated realization.

// NEGATIVE: a 1000-term naive reduction with a 1-ULP tolerance -> bound 1000 > 1.
bcir.module @r17_naive_too_loose {
  bcir.registry @RES {
    bcir.resource @T { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1000>, layout = #bcir.layout<soa> }
    bcir.resource @ACC { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1000>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R17: accuracy bound 1000 ULP exceeds tolerance 1 ULP @reduce_naive}}
  bcir.claim @reduce_naive attributes {
    claim_id = 1 : i32, phase = @p0, op = "reduce.add", reads = [@T], writes = [@ACC],
    count = 1000 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    precision = #bcir.precision<f32, exact = false, tol = 1>
  } { %i = bcir.index_range 0 to 1000 step 1 }
}

// -----

// POSITIVE: the SAME reduction compensated (exact = true) -> bound 1 <= 1, legal.
bcir.module @r17_compensated_ok {
  bcir.registry @RES {
    bcir.resource @T { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1000>, layout = #bcir.layout<soa> }
    bcir.resource @ACC { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1000>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @reduce_comp attributes {
    claim_id = 1 : i32, phase = @p0, op = "reduce.add", reads = [@T], writes = [@ACC],
    count = 1000 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    precision = #bcir.precision<f32, exact = true, tol = 1>
  } { %i = bcir.index_range 0 to 1000 step 1 }
}

// -----

// POSITIVE: a generous tolerance admits the naive reduction (bound 1000 <= 2000).
bcir.module @r17_loose_tolerance_ok {
  bcir.registry @RES {
    bcir.resource @T { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1000>, layout = #bcir.layout<soa> }
    bcir.resource @ACC { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1000>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @reduce_loose attributes {
    claim_id = 1 : i32, phase = @p0, op = "reduce.add", reads = [@T], writes = [@ACC],
    count = 1000 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    precision = #bcir.precision<f32, exact = false, tol = 2000>
  } { %i = bcir.index_range 0 to 1000 step 1 }
}
