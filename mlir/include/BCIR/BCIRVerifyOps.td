//===- BCIRVerifyOps.td - verifier obligations as IR (M1) ---*- tablegen -*-===//
//
// Verification obligations encoded as IR certificates/checkpoints (LangRef
// R1-R12), not handwritten C++. They can be parsed, printed, tested, lowered,
// audited, or validated by later engines. The oracle's bcir/verify is the
// executable reference (docs/PARITY.md).
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_VERIFY_OPS
#define BCIR_VERIFY_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_VerifyRegistrySymbolsOp : BCIR_Op<"verify.registry_symbols", [Symbol]> {
  let summary = "R1/R2: registry symbol + RID uniqueness/resolution certificate";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$registry,
                       FlatSymbolRefArrayAttr:$resources, DenseI64ArrayAttr:$rids);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_VerifyResourceDomainOp : BCIR_Op<"verify.resource_domain", [Symbol]> {
  let summary = "R3: resource domain legality certificate";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$resource,
                       BCIR_DomainAttr:$expected_domain);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_VerifyPhaseDAGOp : BCIR_Op<"verify.phase_dag", [Symbol]> {
  let summary = "R4: phase DAG legality certificate";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefArrayAttr:$phases,
                       ArrayAttr:$edges, DenseI64ArrayAttr:$topological_order);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_VerifyClaimContractOp : BCIR_Op<"verify.claim_contract", [Symbol]> {
  let summary = "R5: claim read/write/hazard/verify contract certificate";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$claim,
                       FlatSymbolRefAttr:$phase, FlatSymbolRefArrayAttr:$reads,
                       FlatSymbolRefArrayAttr:$writes, BCIR_HazardModeAttr:$hazard,
                       BCIR_VerifyAttr:$verify, BCIR_BoundsAttr:$bounds);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_VerifyLaneStrideOp : BCIR_Op<"verify.lane_stride", [Symbol]> {
  let summary = "R6: lane/stride execution-geometry legality certificate";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$claim,
                       BCIR_LaneAttr:$lane, BCIR_StrideClassAttr:$stride_class,
                       DefaultValuedAttr<I32Attr, "1">:$stride_k);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_VerifyBoundsOp : BCIR_Op<"verify.bounds", [Symbol]> {
  let summary = "R7: bounds obligation certificate";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$claim,
                       FlatSymbolRefArrayAttr:$resources, I64Attr:$offset, I64Attr:$count,
                       BCIR_BoundsAttr:$bounds);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_VerifyMemTierOp : BCIR_Op<"verify.mem_tier", [Symbol]> {
  let summary = "CT1: memory tier legality certificate";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$resource,
                       OptionalAttr<FlatSymbolRefAttr>:$tier_symbol, BCIR_MemTierAttr:$tier,
                       BCIR_AccessAttr:$access, I32Attr:$priority);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_VerifyCostVectorOp : BCIR_Op<"verify.cost_vector", [Symbol]> {
  let summary = "R8: K_BCIR cost-vector completeness certificate";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$path,
                       BCIR_CostVectorAttr:$cost, DenseI64ArrayAttr:$weights);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_VerifyPlanSelectionOp : BCIR_Op<"verify.plan_selection", [Symbol]> {
  let summary = "R9: K_BCIR plan-selection legality certificate";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$plan,
                       FlatSymbolRefAttr:$claim, FlatSymbolRefArrayAttr:$candidates,
                       FlatSymbolRefAttr:$selected, BCIR_SemiringAttr:$semiring, I64Attr:$score);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_VerifyStreamProvenanceOp : BCIR_Op<"verify.stream_provenance", [Symbol]> {
  let summary = "R10: GEM StreamPack provenance certificate";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$stream_pack,
                       FlatSymbolRefAttr:$source_plan, FlatSymbolRefArrayAttr:$claims,
                       FlatSymbolRefArrayAttr:$segments, FlatSymbolRefArrayAttr:$trace_notes);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_VerifyGenerationTagsOp : BCIR_Op<"verify.generation_tags", [Symbol]> {
  let summary = "R11: StreamPack generation-tag freshness certificate";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$stream_pack,
                       I64Attr:$topo_gen, I64Attr:$map_gen, I64Attr:$data_gen,
                       I64Attr:$required_map_gen, I64Attr:$required_data_gen);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_VERIFY_OPS
