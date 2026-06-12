//===- BCIRTargetOps.td - open target capability ops --------*- tablegen -*-===//
#ifndef BCIR_TARGET_OPS
#define BCIR_TARGET_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

// The open container: any ISA is data, not optimizer code. Mirrors
// bcir/kbcir/cost.py TargetProfile.
def BCIR_TargetCapabilityOp : BCIR_Op<"target.capability", [Symbol]> {
  let summary = "Open target capability descriptor (x86 / ARM / RISC-V / GPU)";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$triple,
    ArrayAttr:$isa_features,
    DenseI64ArrayAttr:$lane_widths,
    DefaultValuedAttr<I32Attr, "0">:$warp,
    DefaultValuedAttr<I32Attr, "64">:$cacheline,
    DefaultValuedAttr<I32Attr, "0">:$gather_penalty,
    DefaultValuedAttr<I32Attr, "1">:$affinity_domains,
    // Roofline knee: concurrent bandwidth-bound streams the memory sustains.
    DefaultValuedAttr<I32Attr, "4">:$mem_channels);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_TARGET_OPS
