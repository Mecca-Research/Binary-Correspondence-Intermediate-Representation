//===- BCIROptOps.td - optimization-law ops as IR (M2) ------*- tablegen -*-===//
//
// Optimization rules and planning stages as IR objects, keeping K_BCIR/GEM
// optimization visible and target-open (LangRef Sec. 11). A rewrite is legal
// only if it does not increase the selected K_BCIR cost (or strictly improves
// legality) and the post-rewrite module still passes R1-R12.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_OPT_OPS
#define BCIR_OPT_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_OptPipelineOp : BCIR_Op<"opt.pipeline", [Symbol]> {
  let summary = "Declarative BCIR optimization pipeline (ordered stages as data)";
  let arguments = (ins SymbolNameAttr:$sym_name, ArrayAttr:$stages);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_OptRewriteRuleOp : BCIR_Op<"opt.rewrite_rule", [Symbol]> {
  let summary = "Declarative lane/stride rewrite law (e.g. GGG->UX, UX->U)";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kind,
                       BCIR_LaneAttr:$from_lane, BCIR_LaneAttr:$to_lane,
                       BCIR_StrideClassAttr:$from_stride, BCIR_StrideClassAttr:$to_stride,
                       StrAttr:$requires, StrAttr:$preserves, StrAttr:$cost_axis);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_OptLayoutRuleOp : BCIR_Op<"opt.layout_rule", [Symbol]> {
  let summary = "Declarative layout transformation law (e.g. AoS->SoA)";
  let arguments = (ins SymbolNameAttr:$sym_name, BCIR_LayoutAttr:$from_layout,
                       BCIR_LayoutAttr:$to_layout, StrAttr:$requires, StrAttr:$preserves,
                       StrAttr:$cost_axis);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_OptMemRuleOp : BCIR_Op<"opt.mem_rule", [Symbol]> {
  let summary = "Declarative memory-tier movement law (e.g. hot CXL->HBM)";
  let arguments = (ins SymbolNameAttr:$sym_name, BCIR_MemTierAttr:$from_tier,
                       BCIR_MemTierAttr:$to_tier, I32Attr:$min_priority,
                       StrAttr:$requires, StrAttr:$cost_axis);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_OptChoiceOp : BCIR_Op<"opt.choice", [Symbol]> {
  let summary = "A chosen optimization action for trace/replay";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$rule,
                       FlatSymbolRefAttr:$claim, FlatSymbolRefAttr:$before_path,
                       FlatSymbolRefAttr:$after_path, I64Attr:$delta_cost);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_OPT_OPS
