//===- BCIRGEMOps.td - GEM StreamPack ops (BCIR-4) ----------*- tablegen -*-===//
#ifndef BCIR_GEM_OPS
#define BCIR_GEM_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRTypes.td"
include "BCIR/BCIRAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_GEMStreamPackOp : BCIR_Op<"gem.stream_pack", [Symbol, NoTerminator]> {
  let summary = "Hydrated GEM StreamPack (hot executable artifact)";
  // `pipeline_depth` (StreamPack ABI v2, append-only): phases in flight; 2 =
  // double-buffered software pipelining (mirrors gem.streampack.hydrate_pipelined).
  let arguments = (ins
    SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$source_plan,
    I64Attr:$topo_gen, I64Attr:$map_gen, I64Attr:$data_gen,
    DefaultValuedAttr<I32Attr, "1">:$pipeline_depth);
  let results = (outs BCIR_StreamType:$stream);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body `:` type($stream)";
}

def BCIR_GEMPrefetchOp : BCIR_Op<"gem.prefetch", [Symbol]> {
  let summary = "GEM prefetch profile (operand data, not descriptors)";
  // `buffers` (ABI v2, append-only): 2 = a double-buffer contract feeding the
  // next pipelined phase while the current one computes.
  let arguments = (ins SymbolNameAttr:$sym_name, I32Attr:$distance,
                       FlatSymbolRefArrayAttr:$targets, StrAttr:$hint, StrAttr:$pattern,
                       DefaultValuedAttr<I32Attr, "1">:$buffers);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_GEMScheduleOp : BCIR_Op<"gem.schedule", [Symbol]> {
  let summary = "Duration-aware schedule certificate (EFT waves / token DAG)";
  // mode: "waves" (unit-time CT2), "eft" (HEFT-lite: LPT + earliest finish +
  // locality + the bandwidth knee), "tokens" (!bcir.token DAG, pipelined
  // phases). Mirrors bcir/gem/schedule.py; consistency is verified under R9.
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$plan,
                       StrAttr:$mode, I64Attr:$makespan,
                       DefaultValuedAttr<I32Attr, "1">:$knee,
                       DefaultValuedAttr<I32Attr, "1">:$pipeline_depth);
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
    FlatSymbolRefArrayAttr:$fence_before, FlatSymbolRefArrayAttr:$fence_after,
    // CT2: thread->cache affinity domain (-1 = unpinned) and cache-unroll factor.
    DefaultValuedAttr<I32Attr, "-1">:$affinity,
    DefaultValuedAttr<I32Attr, "1">:$unroll,
    // Duration-aware scheduling: the K_BCIR step cost as execution duration.
    DefaultValuedAttr<I64Attr, "0">:$duration);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_GEM_OPS
