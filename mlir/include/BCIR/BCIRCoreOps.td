//===- BCIRCoreOps.td - core BCIR ops (BCIR-0/1/2) ----------*- tablegen -*-===//
//
// Modules own registries; registries own resources; claims declare contracts;
// phases own ordering. Mirrors bcir/model + bcir/verify. A resource is a pure
// symbol (addressed by @name; MLIR 22 forbids a Symbol op from having an SSA
// result, so the old !bcir.resource handle is gone). The claim body holds the
// realization region directly (no bcir.region wrapper); module/registry default
// string attrs are properly quoted.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_CORE_OPS
#define BCIR_CORE_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRTypes.td"
include "BCIR/BCIRAttrs.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Contract bundle carried by memory/compute claims (LangRef Sec. 5).
defvar BCIR_ClaimAttrs = (ins
  BCIR_LaneAttr:$lane,
  BCIR_StrideClassAttr:$stride_class,
  DefaultValuedAttr<I32Attr, "1">:$stride_k,
  BCIR_DomainAttr:$domain,
  BCIR_HazardModeAttr:$hazard,
  BCIR_VerifyAttr:$verify,
  BCIR_BoundsAttr:$bounds,
  OptionalAttr<BCIR_PrecisionAttr>:$precision,
  OptionalAttr<BCIR_CostClassAttr>:$cost_class,
  // The model.Opcode integer (model/opcodes.py); carried so -bcir-verify can recompute the
  // provenance hash_module (R13) from the IR. Defaulted (NOP) so existing IR parses.
  DefaultValuedAttr<I32Attr, "0">:$opcode,
  // Dynamic shape (compose): `count` is a STATIC UPPER BOUND, so the plan priced at it is a
  // worst-case latency bound that holds for any actual size <= the bound (compose.dynamic /
  // plan_holds_for). Defaulted false (an exact static count) so existing IR parses.
  DefaultValuedAttr<BoolAttr, "false">:$dynamic,
  // OPTIONAL timing (§5.11, laws R19/R20) + lifetime (§5.12, law R21) metadata, mirroring
  // model.graph Timing/Lifetime. Absent for the entire scalar/C subset, so R19/R20/R21 are
  // vacuous over it (the non-disturbance invariant -- existing IR parses and verifies unchanged).
  OptionalAttr<BCIR_TimingAttr>:$timing,
  OptionalAttr<BCIR_LifetimeAttr>:$lifetime);

def BCIR_ModuleOp : BCIR_Op<"module",
    [IsolatedFromAbove, NoTerminator, SymbolTable, Symbol]> {
  let summary = "A BCIR module: registries -> claims -> phases -> plan -> stream";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DefaultValuedAttr<StrAttr, "\"0.1\"">:$version,
    DefaultValuedAttr<StrAttr, "\"registry-first\"">:$target_model,
    DefaultValuedAttr<StrAttr, "\"gem\"">:$execution_model,
    DefaultValuedAttr<StrAttr, "\"k_bcir\"">:$cost_model,
    // The goal-graph cacheline/align (model.Module); carried for the R13 hash_module
    // recompute. Defaulted to the model defaults so existing IR is unchanged.
    DefaultValuedAttr<I32Attr, "64">:$cacheline,
    DefaultValuedAttr<I32Attr, "64">:$align);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_RegistryOp : BCIR_Op<"registry", [NoTerminator, Symbol, SymbolTable]> {
  let summary = "A registry namespace of resources";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_ResourceOp : BCIR_Op<"resource", [Symbol]> {
  let summary = "Declare a registry-governed resource (RID-addressed, not a pointer)";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I32Attr:$rid,
    BCIR_DomainAttr:$domain_kind,
    DenseI64ArrayAttr:$shape,
    BCIR_LayoutAttr:$layout,
    DefaultValuedAttr<I32Attr, "64">:$align,
    OptionalAttr<BCIR_AccessAttr>:$access,
    DefaultValuedAttr<I32Attr, "0">:$priority,
    DefaultValuedAttr<I64Attr, "0">:$map_gen,
    DefaultValuedAttr<I64Attr, "0">:$data_gen,
    // Allocator placement tier (append-only; mirrors kbcir.allocator.Placement).
    // R16 (lower-to-llvm): an on-chip placement must fit -- a resource placed in L1
    // must be <= 64 KiB, in L2 <= 4 MiB (static caps; size = product(shape) * 4 B).
    OptionalAttr<BCIR_MemTierAttr>:$placement);
  // MLIR 22's Symbol verifier forbids a symbol op from producing an SSA result; a
  // resource is addressed by its registry symbol (@A) -- the old !bcir.resource handle
  // was vestigial (never consumed), so this op is now a pure symbol.
  let assemblyFormat = "$sym_name attr-dict";
  // Parse-time structural well-formedness (align is a positive power of two; the shape
  // extents are positive). Cheaper + earlier than -bcir-verify (which carries the
  // cross-op R-laws); runs on every parse/builder, not only when the pass is invoked.
  let hasVerifier = 1;
}

def BCIR_PhaseOp : BCIR_Op<"phase", [Symbol]> {
  let summary = "A scheduling phase with explicit dependency edges (LangRef Sec. 6)";
  let arguments = (ins SymbolNameAttr:$sym_name, I32Attr:$id, FlatSymbolRefArrayAttr:$deps);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_ClaimOp : BCIR_Op<"claim", [Symbol, NoTerminator]> {
  let summary = "A declarative transition: op + resources + contract + phase";
  let arguments = !con(
    (ins SymbolNameAttr:$sym_name,
         I32Attr:$claim_id,
         FlatSymbolRefAttr:$phase,
         StrAttr:$op,
         FlatSymbolRefArrayAttr:$reads,
         FlatSymbolRefArrayAttr:$writes,
         DefaultValuedAttr<I64Attr, "0">:$offset,
         DefaultValuedAttr<I64Attr, "1">:$count),
    BCIR_ClaimAttrs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  // Parse-time well-formedness: the iteration extent is non-negative and the stride
  // multiplier is positive.
  let hasVerifier = 1;
}

def BCIR_IndexRangeOp : BCIR_Op<"index_range", [Pure]> {
  let summary = "An affine iteration domain [lo, hi) step s";
  let arguments = (ins I64Attr:$lo, I64Attr:$hi, DefaultValuedAttr<I64Attr, "1">:$step);
  let results = (outs Index:$iv);
  let assemblyFormat = "$lo `to` $hi `step` $step attr-dict";
}

def BCIR_LoadOp : BCIR_Op<"load"> {
  let summary = "Load through (RID, index) with a lane/bounds contract";
  let arguments = !con((ins FlatSymbolRefAttr:$src, Index:$index), BCIR_ClaimAttrs);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$src `[` $index `]` attr-dict `:` type($value)";
}

def BCIR_StoreOp : BCIR_Op<"store"> {
  let summary = "Store through (RID, index) with a lane/bounds contract";
  let arguments = !con((ins AnyType:$value, FlatSymbolRefAttr:$dst, Index:$index), BCIR_ClaimAttrs);
  let assemblyFormat = "$value `,` $dst `[` $index `]` attr-dict `:` type($value)";
}

def BCIR_ComputeOp : BCIR_Op<"compute", [Pure]> {
  let summary = "Pure elementwise computation (kind = fadd/fsub/fmul/...)";
  let arguments = (ins StrAttr:$kind, Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$kind `(` $operands `)` attr-dict `:` functional-type($operands, $result)";
}

def BCIR_BarrierOp : BCIR_Op<"barrier"> {
  let summary = "Memory or execution fence at a given scope (H lane)";
  // `ordering` (Phase 8 memory model) maps to the LLVM fence ordering; default seq_cst.
  let arguments = (ins I32Attr:$scope, OptionalAttr<BCIR_MemOrderingAttr>:$ordering);
  let assemblyFormat = "$scope attr-dict";
}

#endif // BCIR_CORE_OPS
