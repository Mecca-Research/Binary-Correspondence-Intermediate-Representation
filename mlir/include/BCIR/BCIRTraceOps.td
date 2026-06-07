//===- BCIRTraceOps.td - trace / provenance ops -------------*- tablegen -*-===//
#ifndef BCIR_TRACE_OPS
#define BCIR_TRACE_OPS

include "BCIR/BCIRDialect.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_TraceNoteOp : BCIR_Op<"trace.note", [Symbol]> {
  let summary = "Trace/provenance note for replay and audit (R10)";
  let arguments = (ins SymbolNameAttr:$sym_name, I64Attr:$src_hash, I64Attr:$trace_hash);
  let assemblyFormat = "$sym_name attr-dict";
}

// CT4: the "data DNA" telemetry record emitted per segment (cycles/bytes/misses +
// thermal/voltage/utilization + provenance). Feeds the K_BCIR cost calibrator.
def BCIR_TraceDataDnaOp : BCIR_Op<"trace.data_dna", [Symbol]> {
  let summary = "Per-segment data-DNA telemetry record (CT4)";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$segment,
                       I64Attr:$cycles, I64Attr:$bytes, I32Attr:$misses,
                       I32Attr:$thermal, I32Attr:$voltage, I32Attr:$utilization,
                       DefaultValuedStrAttr<StrAttr, "">:$provenance);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_TRACE_OPS
