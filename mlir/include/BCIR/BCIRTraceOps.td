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

#endif // BCIR_TRACE_OPS
