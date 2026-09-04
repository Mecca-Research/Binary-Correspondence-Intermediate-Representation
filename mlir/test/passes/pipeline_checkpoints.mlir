// RUN: bcir-opt --bcir-optimize -verify-diagnostics %s
// RUN: bcir-opt --bcir-hydrate -verify-diagnostics %s
//
// S0-3 (verifier checkpoints): the named pipelines `bcir-optimize` and `bcir-hydrate` are
// checkpointed on entry -- and bcir-optimize again on the plan it emits -- the way
// bcir-audit, bcir-lower-llvm and bcir-aot always were. Before this fixture they ran no
// verifier at all: an illegal module planned, hydrated and lowered without a diagnostic
// (the 2026-07/08 assessment's "advertised checkpoints missing" finding). Both pipelines
// must refuse this module at the door with the same R2; check_passes.sh runs the file
// under each.
bcir.module @illegal {
  // expected-error @+1 {{R2: claim c reads undeclared resource @Missing}}
  bcir.claim @c attributes {
    claim_id = 1 : i32, phase = @p0, op = "x", reads = [@Missing], writes = [], count = 4 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 4 step 1 }
}
