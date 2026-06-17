//===- bcir.irdl.mlir - pure-IRDL BCIR projection (LLVM 22) ---------------===//
//
// The portability rail: a structural BCIR dialect definition shipped as IR data
// and loaded by stock `mlir-opt` -- no BCIR-authored C++ compiled first. The
// closest MLIR-native version of the WASM-like property (artifacts are pure data;
// the standard engine loads them).
//
//   mlir-opt --irdl-file=mlir/irdl/bcir.irdl.mlir <program-in-bcir-generic.mlir>
//
// Validated on the latest LLVM/MLIR (22): this file loads and the generic-syntax
// corpus in test/irdl/ round-trips against it.
//
// Notes on the form:
//  * IRDL uses *named* `irdl.operands/results/regions` (`name: %value`); the
//    variadicity marker precedes the value (`name: variadic %v`). (LLVM <=19 used
//    a positional form; MLIR 22 made the names mandatory.)
//  * MLIR 22's IRDL forbids dots in operation names (`isValidName`: lowercase /
//    digits / underscores only). The real ODS + `bcir-opt` dialect keeps its dotted
//    op names (`bcir.target.capability`, `bcir.kbcir.policy`, ...); this *projection*
//    flattens the dot to an underscore (`bcir.target_capability`,
//    `bcir.kbcir_policy`, ...) so the structural rail still loads under stock
//    `mlir-opt`. The generic-syntax corpus in test/irdl/ uses the same flattened
//    names. The dotted taxonomy remains the source of truth on the compiled rail.
//  * An IRDL dialect shares one symbol table across types and ops, so a type and
//    an op cannot both be named `resource`. We use a single opaque `!bcir.handle`
//    type for every handle-producing op (resource / kbcir_path / kbcir_select /
//    gem_stream_pack); ops are otherwise unconstrained `irdl.any`.
//  * Structural only: no `irdl.c_pred` (it requires compiled C++ and blocks
//    runtime registration). Deep semantics stay in the ODS + bcir-opt rail and in
//    the bcir/ oracle.
//
//===----------------------------------------------------------------------===//

irdl.dialect @bcir {
  // One opaque handle type for resources / paths / streams (see header note).
  irdl.type @handle

  // ---- BCIR-0..2: core containers + claims ----
  irdl.operation @module {
    %body = irdl.region
    irdl.regions(body: %body)
  }
  irdl.operation @registry {
    %body = irdl.region
    irdl.regions(body: %body)
  }
  irdl.operation @resource {
    %h = irdl.any
    irdl.results(h: %h)
  }
  irdl.operation @phase
  irdl.operation @claim {
    %body = irdl.region
    irdl.regions(body: %body)
  }
  irdl.operation @index_range {
    %idx = irdl.is index
    irdl.results(idx: %idx)
  }
  irdl.operation @load {
    %idx = irdl.is index
    %value = irdl.any
    irdl.operands(idx: %idx)
    irdl.results(value: %value)
  }
  irdl.operation @store {
    %idx = irdl.is index
    %value = irdl.any
    irdl.operands(value: %value, idx: %idx)
  }
  irdl.operation @compute {
    %value = irdl.any
    irdl.operands(arg: variadic %value)
    irdl.results(res: %value)
  }
  irdl.operation @barrier

  // ---- CT1: target-open container + memory hierarchy ----
  irdl.operation @target_capability
  irdl.operation @mem_tier
  irdl.operation @mem_ham
  irdl.operation @mem_cxl_swap

  // ---- BCIR-3: K_BCIR planning ----
  irdl.operation @kbcir_policy
  irdl.operation @kbcir_plan {
    %body = irdl.region
    irdl.regions(body: %body)
  }
  irdl.operation @kbcir_path {
    %h = irdl.any
    irdl.results(h: %h)
  }
  irdl.operation @kbcir_select {
    %h = irdl.any
    irdl.results(h: %h)
  }
  // Live runtime state Theta (the context op -bcir-plan/-overlap read for the multiplicative
  // coupling) + the compositional func/call/cond family (compose.Function/Call/Cond) -- the
  // K_BCIR ops the projection previously omitted. Regions are presence-only (the loose rail).
  irdl.operation @kbcir_theta
  irdl.operation @kbcir_func {
    %body = irdl.region
    irdl.regions(body: %body)
  }
  irdl.operation @kbcir_call
  irdl.operation @kbcir_cond {
    %then_region = irdl.region
    %else_region = irdl.region
    irdl.regions(then_region: %then_region, else_region: %else_region)
  }
  // Constrained (RCSP) rail: B(H,Theta) caps + the (max,+) overlap price.
  irdl.operation @kbcir_budget
  irdl.operation @kbcir_scheduled_price
  // The temperature dial: the soft log-sum-exp twin of the tropical select.
  irdl.operation @kbcir_soft_select
  // Learning placement (LangRef Sec. 13): L1 frozen tables + L2 portfolio/gate
  // + the L3 regret ledger (the boundary dashboard).
  irdl.operation @kbcir_calibration
  irdl.operation @kbcir_portfolio
  irdl.operation @kbcir_replay_certificate
  irdl.operation @kbcir_regret_ledger
  // The learned MoE gate: a GNN router over the claim graph (the ensemble).
  irdl.operation @kbcir_moe_gate
  // The propose-verify search accelerator: a learned candidate ordering.
  irdl.operation @kbcir_search_accel
  // The provenance manifest: the commit hash of a plan (R13 reproducibility).
  irdl.operation @kbcir_provenance_manifest
  // The L1 cost throttle: a learned component's amortization certificate.
  irdl.operation @kbcir_amortization
  // Memory module: the frozen, generation-tagged resolution fixpoint a = Lim(Res(U)).
  irdl.operation @kbcir_memory_module

  // ---- BCIR-4: GEM StreamPack ----
  irdl.operation @gem_stream_pack {
    %h = irdl.any
    %body = irdl.region
    irdl.results(h: %h)
    irdl.regions(body: %body)
  }
  irdl.operation @gem_prefetch
  irdl.operation @gem_block
  irdl.operation @gem_lane_segment
  // Duration-aware schedule certificate (EFT waves / token DAG, knee, pipeline).
  irdl.operation @gem_schedule
  irdl.operation @trace_note
  irdl.operation @trace_data_dna

  // ---- M5: Event Transduction Layer (structural; loose by design) ----
  irdl.operation @event_stream
  irdl.operation @event_kind
  irdl.operation @event_emit
  irdl.operation @event_consume

  irdl.operation @fsm_machine {
    %body = irdl.region
    irdl.regions(body: %body)
  }
  irdl.operation @fsm_state
  irdl.operation @fsm_transition
  irdl.operation @fsm_stack
  irdl.operation @fsm_capture
  irdl.operation @fsm_reduce

  irdl.operation @parse_grammar {
    %body = irdl.region
    irdl.regions(body: %body)
  }
  irdl.operation @parse_token
  irdl.operation @parse_rule
  irdl.operation @parse_lower_to_fsm

  irdl.operation @binary_format {
    %body = irdl.region
    irdl.regions(body: %body)
  }
  irdl.operation @binary_field
  irdl.operation @binary_record
  irdl.operation @binary_decode

  // ---- M1: verifier obligations as IR (R1-R12) ----
  irdl.operation @verify_registry_symbols
  irdl.operation @verify_resource_domain
  irdl.operation @verify_phase_dag
  irdl.operation @verify_claim_contract
  irdl.operation @verify_lane_stride
  irdl.operation @verify_bounds
  irdl.operation @verify_mem_tier
  irdl.operation @verify_cost_vector
  irdl.operation @verify_plan_selection
  irdl.operation @verify_stream_provenance
  irdl.operation @verify_generation_tags
  irdl.operation @verify_policy_provenance

  // ---- M2: optimization law as IR ----
  irdl.operation @opt_pipeline
  irdl.operation @opt_rewrite_rule
  irdl.operation @opt_layout_rule
  irdl.operation @opt_mem_rule
  irdl.operation @opt_choice
  // The building-blocks engine: equality-saturation extraction.
  irdl.operation @egraph_extract

  // ---- M3: target lowering contracts as IR ----
  irdl.operation @isa_family
  irdl.operation @isa_feature
  irdl.operation @isa_register_class
  irdl.operation @isa_opcode
  irdl.operation @packet_format
  irdl.operation @target_lower_contract

  // ---- Phase 8: async dependency tokens ----
  irdl.type @token
  irdl.operation @async_fork {
    %t = irdl.any
    irdl.results(t: %t)
  }
  irdl.operation @async_await {
    %t = irdl.any
    irdl.operands(t: variadic %t)
  }
}
