// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// R9 over the plan -bcir-plan emits (S0-3 / #761 review): the checkpointed pipelines verify
// the module AFTER planning, so the verifier reads kbcir.plan_width / plan_cost / plan_score
// and re-derives them from the scope -- the oracle's verify_plan R9 on the law rail. The
// module is plan.mlir's: two elementwise claims sharing the read A, whose coupled plan is
// c1 vec16 at 7808 and c2 vec16 at 5888 (the shared-input discount), 13696 in total.
// The first case is that plan, annotated exactly as -bcir-plan writes it: accepted. Each of
// the others corrupts one thing the trailing verifier used to accept.

bcir.module @plan_ok attributes { kbcir.plan_score = 13696 : i64 } {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @E { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @D { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    kbcir.plan_width = 16 : i64, kbcir.plan_cost = 7808 : i64
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@A, @E], writes = [@D],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    kbcir.plan_width = 16 : i64, kbcir.plan_cost = 5888 : i64
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// -----

// c2 priced at the uncoupled 7808 (its shared-input discount dropped) with the score summed
// to match: consistent as a sum, wrong as a re-derivation -- only the scope check sees it.
bcir.module @plan_cost_forged attributes { kbcir.plan_score = 15616 : i64 } {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @E { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @D { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    kbcir.plan_width = 16 : i64, kbcir.plan_cost = 7808 : i64
  } { %i = bcir.index_range 0 to 1024 step 1 }
  // expected-error @+1 {{R9: claim @c2 realized cost 7808 does not re-derive from the scope (expected 5888 for width 16)}}
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@A, @E], writes = [@D],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    kbcir.plan_width = 16 : i64, kbcir.plan_cost = 7808 : i64
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// -----

// The score is not the sum of the realized costs.
// expected-error @+1 {{R9: plan score 13697 != sum of realized costs 13696}}
bcir.module @plan_score_forged attributes { kbcir.plan_score = 13697 : i64 } {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @E { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @D { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    kbcir.plan_width = 16 : i64, kbcir.plan_cost = 7808 : i64
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@A, @E], writes = [@D],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    kbcir.plan_width = 16 : i64, kbcir.plan_cost = 5888 : i64
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// -----

// A width the target does not issue for the claim: not among the planner's offer.
bcir.module @plan_width_not_offered attributes { kbcir.plan_score = 13696 : i64 } {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @E { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @D { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R9: claim @c1 realized at width 3, not among the 3 candidate width(s) this target admits: 1, 8, 16}}
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    kbcir.plan_width = 3 : i64, kbcir.plan_cost = 7808 : i64
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@A, @E], writes = [@D],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    kbcir.plan_width = 16 : i64, kbcir.plan_cost = 5888 : i64
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// -----

// A planned module realizes every claim: c2 carries no annotation.
bcir.module @plan_incomplete attributes { kbcir.plan_score = 7808 : i64 } {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @E { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @D { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    kbcir.plan_width = 16 : i64, kbcir.plan_cost = 7808 : i64
  } { %i = bcir.index_range 0 to 1024 step 1 }
  // expected-error @+1 {{R9: plan does not realize claim @c2 (a planned module annotates every claim with kbcir.plan_width and kbcir.plan_cost together)}}
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@A, @E], writes = [@D],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// -----

// Annotations with no policy in scope: nothing to re-derive the scalarization under.
// expected-error @+1 {{R9: plan annotations without a 12-d non-negative kbcir.policy to re-derive the scalarization under}}
bcir.module @plan_without_policy attributes { kbcir.plan_score = 7808 : i64 } {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    kbcir.plan_width = 16 : i64, kbcir.plan_cost = 7808 : i64
  } { %i = bcir.index_range 0 to 1024 step 1 }
}
