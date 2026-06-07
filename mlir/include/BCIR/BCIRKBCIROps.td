//===- BCIRKBCIROps.td - K_BCIR planning ops (BCIR-3) -------*- tablegen -*-===//
#ifndef BCIR_KBCIR_OPS
#define BCIR_KBCIR_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRTypes.td"
include "BCIR/BCIRAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_KBCIRPolicyOp : BCIR_Op<"kbcir.policy", [Symbol]> {
  let summary = "K_BCIR scalarization policy (12 Q16 weights)";
  let arguments = (ins SymbolNameAttr:$sym_name, BCIR_PolicyModeAttr:$mode,
                       DenseI64ArrayAttr:$weights);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_KBCIRPlanOp : BCIR_Op<"kbcir.plan", [Symbol, NoTerminator]> {
  let summary = "K_BCIR candidate-path + selected-plan region";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_KBCIRPathOp : BCIR_Op<"kbcir.path", [Symbol]> {
  let summary = "A feasible realization path for a BCIR claim";
  let arguments = (ins
    SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$claim, StrAttr:$realization,
    BCIR_LaneAttr:$lane, BCIR_LayoutAttr:$layout, BCIR_CostVectorAttr:$cost);
  let results = (outs BCIR_PathType:$path);
  let assemblyFormat = "$sym_name attr-dict `:` type($path)";
}

def BCIR_KBCIRSelectOp : BCIR_Op<"kbcir.select"> {
  let summary = "Select a realization path under a K_BCIR policy (min-plus)";
  let arguments = (ins
    FlatSymbolRefAttr:$claim, FlatSymbolRefArrayAttr:$from,
    BCIR_PolicyModeAttr:$policy, BCIR_SemiringAttr:$semiring,
    FlatSymbolRefAttr:$selected, I64Attr:$score);
  let results = (outs BCIR_PathType:$result);
  let assemblyFormat = "$claim `from` $from attr-dict `:` type($result)";
}

#endif // BCIR_KBCIR_OPS
