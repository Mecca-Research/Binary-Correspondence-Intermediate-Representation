// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// R19 (synchronous-timing legality) + R20 (clock-domain-crossing) over the OPTIONAL
// claim.timing metadata (§5.11), and R21 (pointer-lifetime: use-after-free / double-free)
// over the OPTIONAL claim.lifetime (§5.12). First-class in -bcir-verify, dual-rail with the
// oracle's verify.verify_timing / verify.verify_lifetime. A claim that carries no timing /
// lifetime is unconstrained (the non-disturbance invariant, exactly like R14-R17), so the whole
// scalar/C subset verifies identically -- every existing verify*.mlir module is unaffected.

// =============================== R19 (timing legality) ===============================

// NEGATIVE: a synchronous claim must declare a positive clock_frequency_mhz.
bcir.module @r19_synchronous_no_clock {
  bcir.registry @RES {
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R19: claim s a synchronous claim needs a positive clock_frequency_mhz}}
  bcir.claim @s attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A], writes = [@C],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    timing = #bcir.timing<clock_domain = "clkA", latency_cycles = 4, min_throughput_q16 = 0, critical_path = false, power_domain = "", sync_type = "synchronous", clock_frequency_mhz = 0, setup_hold_margin = 0>
  } { %i = bcir.index_range 0 to 8 step 1 }
}

// -----

// NEGATIVE: the setup/hold margin cannot exceed the stage's own latency budget.
bcir.module @r19_setup_exceeds_latency {
  bcir.registry @RES {
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R19: claim sh setup_hold_margin 5 exceeds the stage latency_cycles 2}}
  bcir.claim @sh attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A], writes = [@C],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    timing = #bcir.timing<clock_domain = "", latency_cycles = 2, min_throughput_q16 = 0, critical_path = false, power_domain = "", sync_type = "", clock_frequency_mhz = 0, setup_hold_margin = 5>
  } { %i = bcir.index_range 0 to 8 step 1 }
}

// -----

// POSITIVE: a well-formed synchronous claim (positive clock, margin within latency) is legal.
bcir.module @r19_synchronous_ok {
  bcir.registry @RES {
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @ok attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A], writes = [@C],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    timing = #bcir.timing<clock_domain = "clkA", latency_cycles = 4, min_throughput_q16 = 0, critical_path = true, power_domain = "vdd", sync_type = "synchronous", clock_frequency_mhz = 200, setup_hold_margin = 1>
  } { %i = bcir.index_range 0 to 8 step 1 }
}

// -----

// ========================= R20 (clock-domain-crossing) =========================

// NEGATIVE: producer @w writes @X in clock domain "clkA"; consumer @r reads @X into "clkB"
// with no synchronizer (sync_type != mixed, hazard != barriered) -- an unguarded CDC.
bcir.module @r20_unguarded_crossing {
  bcir.registry @RES {
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @X { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @Y { rid = 3 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @w attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A], writes = [@X],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    timing = #bcir.timing<clock_domain = "clkA", latency_cycles = 0, min_throughput_q16 = 0, critical_path = false, power_domain = "", sync_type = "", clock_frequency_mhz = 0, setup_hold_margin = 0>
  } { %i = bcir.index_range 0 to 8 step 1 }
  // expected-error @+1 {{R20: claim r reads @X from clock domain 'clkA' into 'clkB'}}
  bcir.claim @r attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@X], writes = [@Y],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    timing = #bcir.timing<clock_domain = "clkB", latency_cycles = 0, min_throughput_q16 = 0, critical_path = false, power_domain = "", sync_type = "", clock_frequency_mhz = 0, setup_hold_margin = 0>
  } { %i = bcir.index_range 0 to 8 step 1 }
}

// -----

// POSITIVE: the same crossing, but the consumer declares sync_type = "mixed" (the synchronizer)
// -- a guarded crossing is legal.
bcir.module @r20_guarded_ok {
  bcir.registry @RES {
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @X { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @Y { rid = 3 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @w attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A], writes = [@X],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    timing = #bcir.timing<clock_domain = "clkA", latency_cycles = 0, min_throughput_q16 = 0, critical_path = false, power_domain = "", sync_type = "", clock_frequency_mhz = 0, setup_hold_margin = 0>
  } { %i = bcir.index_range 0 to 8 step 1 }
  bcir.claim @r attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@X], writes = [@Y],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    timing = #bcir.timing<clock_domain = "clkB", latency_cycles = 0, min_throughput_q16 = 0, critical_path = false, power_domain = "", sync_type = "mixed", clock_frequency_mhz = 0, setup_hold_margin = 0>
  } { %i = bcir.index_range 0 to 8 step 1 }
}

// -----

// ===================== R21 (pointer-lifetime legality) =====================

// NEGATIVE: @f frees @P (event = "free"); a later claim @u reads @P -> use-after-free.
bcir.module @r21_use_after_free {
  bcir.registry @RES {
    bcir.resource @P { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @OUT { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @f attributes {
    claim_id = 1 : i32, phase = @p0, op = "c.free", reads = [@P], writes = [],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    lifetime = #bcir.lifetime<event = "free", epoch = 0>
  } { %i = bcir.index_range 0 to 8 step 1 }
  // expected-error @+1 {{R21: claim u use-after-free of @P}}
  bcir.claim @u attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@P], writes = [@OUT],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 8 step 1 }
}

// -----

// NEGATIVE: @f1 frees @P, then @f2 frees @P again -> double-free.
bcir.module @r21_double_free {
  bcir.registry @RES {
    bcir.resource @P { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @f1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "c.free", reads = [@P], writes = [],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    lifetime = #bcir.lifetime<event = "free", epoch = 0>
  } { %i = bcir.index_range 0 to 8 step 1 }
  // expected-error @+1 {{R21: claim f2 double-free of @P}}
  bcir.claim @f2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "c.free", reads = [@P], writes = [],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    lifetime = #bcir.lifetime<event = "free", epoch = 0>
  } { %i = bcir.index_range 0 to 8 step 1 }
}

// -----

// POSITIVE: free @P, then re-allocate it (a WRITE / event = "alloc" re-validates), then read --
// free(p); p = malloc(...); *p is legal (reassignment), so no use-after-free.
bcir.module @r21_realloc_ok {
  bcir.registry @RES {
    bcir.resource @P { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @SRC { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
    bcir.resource @OUT { rid = 3 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 8>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @free attributes {
    claim_id = 1 : i32, phase = @p0, op = "c.free", reads = [@P], writes = [],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    lifetime = #bcir.lifetime<event = "free", epoch = 0>
  } { %i = bcir.index_range 0 to 8 step 1 }
  bcir.claim @alloc attributes {
    claim_id = 2 : i32, phase = @p0, op = "c.alloc", reads = [@SRC], writes = [@P],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    lifetime = #bcir.lifetime<event = "alloc", epoch = 1>
  } { %i = bcir.index_range 0 to 8 step 1 }
  bcir.claim @use attributes {
    claim_id = 3 : i32, phase = @p0, op = "vector.add", reads = [@P], writes = [@OUT],
    count = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 8 step 1 }
}
