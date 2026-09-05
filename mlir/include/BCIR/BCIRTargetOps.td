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
    DefaultValuedAttr<I64Attr, "0">:$cal_gen,
    // The memory hierarchy (S0-1 / staged plan S0-D), two parallel arrays in the hierarchy's
    // declared order: the tier NAMES (free strings -- the seven MemTier names for a shipped
    // profile, a hardware channel's own tiers such as "sram" otherwise) and four i64 per
    // named tier -- (latency_cyc, bw_factor, lat_factor, capacity) -- the mirror of cost.py
    // MemoryHierarchy.tiers (`tier_names` / `tier_values`). Absent = MemoryHierarchy.default(),
    // the hierarchy every shipped profile carries, so IR written before the attributes
    // hashes as it did. -bcir-verify folds them into the R13 hash_target recompute: scaling
    // one tier's factors moved a plan's score thirty-fold under an unchanged digest, and no
    // attribute existed to recompute the widened hash from.
    OptionalAttr<StrArrayAttr>:$mem_tier_names,
    OptionalAttr<DenseI64ArrayAttr>:$mem_tier_values);
  let assemblyFormat = "$sym_name attr-dict";
  // Parse-time well-formedness: bounded cost-model fields have the signs their
  // arithmetic requires, the cacheline is a positive power of two, lane widths
  // are nonempty/positive, and ISA features are strings.
  let hasVerifier = 1;
}

#endif // BCIR_TARGET_OPS
