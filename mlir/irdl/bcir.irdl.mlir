//===- bcir.irdl.mlir - pure-IRDL BCIR projection (LLVM 18) ---------------===//
//
// The portability rail: a structural BCIR dialect definition shipped as IR data
// and loaded by stock `mlir-opt` -- no BCIR-authored C++ compiled first. The
// closest MLIR-native version of the WASM-like property (artifacts are pure data;
// the standard engine loads them).
//
//   mlir-opt --irdl-file=mlir/irdl/bcir.irdl.mlir <program-in-bcir-generic.mlir>
//
// Validated on LLVM 18 (mlir-opt 18.1.3): this file loads and the generic-syntax
// corpus in test/irdl/ round-trips against it.
//
// Notes on the form:
//  * LLVM 18 IRDL uses *positional* `irdl.operands/results/regions`.
//  * An IRDL dialect shares one symbol table across types and ops, so a type and
//    an op cannot both be named `resource`. We use a single opaque `!bcir.handle`
//    type for every handle-producing op (resource / kbcir.path / kbcir.select /
//    gem.stream_pack); ops are otherwise unconstrained `irdl.any`.
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
    irdl.regions(%body)
  }
  irdl.operation @registry {
    %body = irdl.region
    irdl.regions(%body)
  }
  irdl.operation @resource {
    %h = irdl.any
    irdl.results(%h)
  }
  irdl.operation @phase
  irdl.operation @claim {
    %body = irdl.region
    irdl.regions(%body)
  }
  irdl.operation @index_range {
    %idx = irdl.is index
    irdl.results(%idx)
  }
  irdl.operation @load {
    %idx = irdl.is index
    %value = irdl.any
    irdl.operands(%idx)
    irdl.results(%value)
  }
  irdl.operation @store {
    %idx = irdl.is index
    %value = irdl.any
    irdl.operands(%value, %idx)
  }
  irdl.operation @compute {
    %value = irdl.any
    irdl.operands(variadic %value)
    irdl.results(%value)
  }
  irdl.operation @barrier

  // ---- CT1: target-open container + memory hierarchy ----
  irdl.operation @"target.capability"
  irdl.operation @"mem.tier"
  irdl.operation @"mem.ham"
  irdl.operation @"mem.cxl_swap"

  // ---- BCIR-3: K_BCIR planning ----
  irdl.operation @"kbcir.policy"
  irdl.operation @"kbcir.plan" {
    %body = irdl.region
    irdl.regions(%body)
  }
  irdl.operation @"kbcir.path" {
    %h = irdl.any
    irdl.results(%h)
  }
  irdl.operation @"kbcir.select" {
    %h = irdl.any
    irdl.results(%h)
  }

  // ---- BCIR-4: GEM StreamPack ----
  irdl.operation @"gem.stream_pack" {
    %h = irdl.any
    %body = irdl.region
    irdl.results(%h)
    irdl.regions(%body)
  }
  irdl.operation @"gem.prefetch"
  irdl.operation @"gem.block"
  irdl.operation @"gem.lane_segment"
  irdl.operation @"trace.note"
  irdl.operation @"trace.data_dna"

  // ---- M5: Event Transduction Layer (structural; loose by design) ----
  irdl.operation @"event.stream"
  irdl.operation @"event.kind"
  irdl.operation @"event.emit"
  irdl.operation @"event.consume"

  irdl.operation @"fsm.machine" {
    %body = irdl.region
    irdl.regions(%body)
  }
  irdl.operation @"fsm.state"
  irdl.operation @"fsm.transition"
  irdl.operation @"fsm.stack"
  irdl.operation @"fsm.capture"
  irdl.operation @"fsm.reduce"

  irdl.operation @"parse.grammar" {
    %body = irdl.region
    irdl.regions(%body)
  }
  irdl.operation @"parse.token"
  irdl.operation @"parse.rule"
  irdl.operation @"parse.lower_to_fsm"

  irdl.operation @"binary.format" {
    %body = irdl.region
    irdl.regions(%body)
  }
  irdl.operation @"binary.field"
  irdl.operation @"binary.record"
  irdl.operation @"binary.decode"

  // ---- M1: verifier obligations as IR (R1-R12) ----
  irdl.operation @"verify.registry_symbols"
  irdl.operation @"verify.resource_domain"
  irdl.operation @"verify.phase_dag"
  irdl.operation @"verify.claim_contract"
  irdl.operation @"verify.lane_stride"
  irdl.operation @"verify.bounds"
  irdl.operation @"verify.mem_tier"
  irdl.operation @"verify.target_capability"
  irdl.operation @"verify.cost_vector"
  irdl.operation @"verify.plan_selection"
  irdl.operation @"verify.stream_provenance"
  irdl.operation @"verify.generation_tags"

  // ---- M2: optimization law as IR ----
  irdl.operation @"opt.pipeline"
  irdl.operation @"opt.rewrite_rule"
  irdl.operation @"opt.layout_rule"
  irdl.operation @"opt.mem_rule"
  irdl.operation @"opt.choice"

  // ---- M3: target lowering contracts as IR ----
  irdl.operation @"isa.family"
  irdl.operation @"isa.feature"
  irdl.operation @"isa.register_class"
  irdl.operation @"isa.opcode"
  irdl.operation @"packet.format"
  irdl.operation @"target.lower_contract"
}
