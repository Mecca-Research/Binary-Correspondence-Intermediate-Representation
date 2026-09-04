// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// S0-5 (module scope): the laws quantify over ONE `bcir.module` -- the symbol table that
// owns its registries, phases, paths, plans and policies -- never over whatever root the
// pass was handed. Before this fixture the verifier built its maps with a root-global walk:
// a claim in one module "resolved" a resource declared only in another (R2 held vacuously
// across modules), and two modules that each owned RID 10 tripped R1 against each other.
// The 2026-07/08 assessment's "MLIR module scope violated" finding; dual-rail with the
// oracle, whose `verify(module)` has only ever seen one module at a time. The one scope
// predicate lives in mlir/lib/passes/BCIRPassSupport.h (forEachScope / walkScope).

// NEGATIVE: a claim may not resolve a resource declared in a DIFFERENT module. @a declares
// @X; @b's claim reads @X and must be refused, whatever @a declares.
bcir.module @a {
  bcir.registry @RES {
    bcir.resource @X { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa> }
  }
}
bcir.module @b {
  // expected-error @+1 {{R2: claim c reads undeclared resource @X}}
  bcir.claim @c attributes {
    claim_id = 1 : i32, phase = @p0, op = "x", reads = [@X], writes = [], count = 4 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 4 step 1 }
}

// -----

// POSITIVE: two modules each own RID 10 under the name @X; neither sees the other, so R1
// (duplicate RID) is silent and each claim resolves its own @X. No diagnostic expected.
bcir.module @a {
  bcir.registry @RES {
    bcir.resource @X { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c attributes {
    claim_id = 1 : i32, phase = @p0, op = "x", reads = [@X], writes = [], count = 4 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 4 step 1 }
}
bcir.module @b {
  bcir.registry @RES {
    bcir.resource @X { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c attributes {
    claim_id = 1 : i32, phase = @p0, op = "x", reads = [@X], writes = [], count = 4 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 4 step 1 }
}

// -----

// OUTER SCOPE: operations outside any module form one scope of their own, so a module-free
// file verifies exactly as before -- and its laws are not vacuous: this top-level claim
// resolves nothing and must still be refused.
// expected-error @+1 {{R2: claim top reads undeclared resource @Y}}
bcir.claim @top attributes {
  claim_id = 1 : i32, phase = @p0, op = "x", reads = [@Y], writes = [], count = 4 : i64,
  lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
  domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
  verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
} { %i = bcir.index_range 0 to 4 step 1 }
