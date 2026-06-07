//===- BCIRMemOps.td - memory hierarchy ops (CT1) -----------*- tablegen -*-===//
#ifndef BCIR_MEM_OPS
#define BCIR_MEM_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

// Tier descriptor: Q16 bandwidth/latency factors vs a DRAM baseline. Mirrors
// bcir/kbcir/cost.py Tier / MemoryHierarchy.
def BCIR_MemTierOp : BCIR_Op<"mem.tier", [Symbol]> {
  let summary = "Memory hierarchy tier descriptor";
  let arguments = (ins
    SymbolNameAttr:$sym_name, BCIR_MemTierAttr:$tier,
    I64Attr:$latency_cyc, I64Attr:$bw_factor, I64Attr:$lat_factor);
  let assemblyFormat = "$sym_name attr-dict";
}

// Hierarchical Access Memory: O(log n) addressing instead of flat O(1).
def BCIR_MemHamOp : BCIR_Op<"mem.ham", [Symbol]> {
  let summary = "Hierarchical Access Memory descriptor";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$resource,
                       DefaultValuedAttr<I32Attr, "2">:$fanout);
  let assemblyFormat = "$sym_name attr-dict";
}

// CXL semantic swap: a memory-semantic tier with priority-aware promotion.
def BCIR_MemCxlSwapOp : BCIR_Op<"mem.cxl_swap", [Symbol]> {
  let summary = "CXL semantic swap descriptor";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$resource,
                       DefaultValuedAttr<I32Attr, "0">:$priority);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_MEM_OPS
