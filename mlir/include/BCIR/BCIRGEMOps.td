//===- BCIRGEMOps.td - GEM StreamPack ops (BCIR-4) ----------*- tablegen -*-===//
#ifndef BCIR_GEM_OPS
#define BCIR_GEM_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRTypes.td"
include "BCIR/BCIRAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_GEMStreamPackOp : BCIR_Op<"gem.stream_pack", [Symbol, NoTerminator]> {
  let summary = "Hydrated GEM StreamPack (hot executable artifact)";
  let arguments = (ins
    SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$source_plan,
    I64Attr:$topo_gen, I64Attr:$map_gen, I64Attr:$data_gen);
  let results = (outs BCIR_StreamType:$stream);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body `:` type($stream)";
}

def BCIR_GEMPrefetchOp : BCIR_Op<"gem.prefetch", [Symbol]> {
  let summary = "GEM prefetch profile (operand data, not descriptors)";
  let arguments = (ins SymbolNameAttr:$sym_name, I32Attr:$distance,
                       FlatSymbolRefArrayAttr:$targets, StrAttr:$hint, StrAttr:$pattern);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_GEMBlockOp : BCIR_Op<"gem.block", [Symbol]> {
  let summary = "GEM block descriptor";
  let arguments = (ins SymbolNameAttr:$sym_name, I64Attr:$base, I64Attr:$count,
                       I64Attr:$strideA, I64Attr:$strideB, I64Attr:$strideD);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_GEMLaneSegmentOp : BCIR_Op<"gem.lane_segment", [Symbol]> {
  let summary = "Lane-specialized GEM executable segment";
  let arguments = (ins
    SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$claim, FlatSymbolRefAttr:$phase,
    BCIR_LaneAttr:$lane, DefaultValuedAttr<I32Attr, "1">:$stride_k, I32Attr:$width,
    StrAttr:$opcode, FlatSymbolRefArrayAttr:$reads, FlatSymbolRefArrayAttr:$writes,
    OptionalAttr<FlatSymbolRefAttr>:$prefetch,
    FlatSymbolRefArrayAttr:$fence_before, FlatSymbolRefArrayAttr:$fence_after);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_GEM_OPS
