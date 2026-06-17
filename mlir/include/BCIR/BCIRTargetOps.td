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
    // The TargetProfile name + scalable flag (cost.py); carried so -bcir-verify can recompute
    // the provenance hash_target (R13). `target_name` is the profile name (distinct from the
    // triple and the sym_name). Defaulted so existing IR parses.
    DefaultValuedAttr<StrAttr, "\"\"">:$target_name,
    DefaultValuedAttr<BoolAttr, "false">:$scalable,
    // Roofline knee: concurrent bandwidth-bound streams the memory sustains.
    DefaultValuedAttr<I32Attr, "4">:$mem_channels,
    // K_BCIR cost-model seeds (mirror bcir/kbcir/cost.py TargetProfile defaults).
    // Defaulted to the CPU seeds so existing IR parses unchanged; a GPU/PIM target
    // overrides them. -bcir-cost-model reads these to recompute candidate costs from
    // first principles instead of trusting the emitter-baked path costs.
    DefaultValuedAttr<I32Attr, "1">:$mem_unit,          // bandwidth term / element / stream
    DefaultValuedAttr<I32Attr, "4">:$base_overhead,     // latency term / access (unit stride)
    DefaultValuedAttr<I32Attr, "64">:$thermal_density,  // heat ~ vector width
    DefaultValuedAttr<I32Attr, "64">:$power_density,     // current draw ~ vector width
    DefaultValuedAttr<I32Attr, "1">:$per_op_heat,
    DefaultValuedAttr<I32Attr, "4">:$elem_bytes,
    // Cost-table generation (0 = seeded constants; >=1 = a frozen
    // kbcir.calibration table was applied -- L1 freshness).
    DefaultValuedAttr<I64Attr, "0">:$cal_gen);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_TARGET_OPS
