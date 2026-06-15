//===- BCIRTransducerOps.td - finite-state transducer ops (M5) -*- tablegen -*-===//
//
// Finite-state / pushdown transducers: machine, state, transition, stack,
// capture, reduce. Mirrors bcir/etl/fsm.py. The reduce action turns captured
// input into BCIR objects (claims/resources/tokens/events).
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_TRANSDUCER_OPS
#define BCIR_TRANSDUCER_OPS

include "BCIR/BCIRDialect.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_FSMMachineOp : BCIR_Op<"fsm.machine", [Symbol, SymbolTable, NoTerminator]> {
  let summary = "Finite-state or pushdown transducer machine";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kind,
                       FlatSymbolRefAttr:$input_stream,
                       OptionalAttr<FlatSymbolRefAttr>:$output_stream,
                       FlatSymbolRefAttr:$start_state);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_FSMStateOp : BCIR_Op<"fsm.state", [Symbol]> {
  let summary = "Finite-state machine state";
  let arguments = (ins SymbolNameAttr:$sym_name,
                       DefaultValuedAttr<BoolAttr, "false">:$accepting,
                       DefaultValuedAttr<BoolAttr, "false">:$error);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_FSMTransitionOp : BCIR_Op<"fsm.transition", [Symbol]> {
  let summary = "Finite-state transition (on byte/range/token/event/predicate)";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$from,
                       FlatSymbolRefAttr:$to, StrAttr:$on, StrAttr:$guard, StrAttr:$action);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_FSMStackOp : BCIR_Op<"fsm.stack", [Symbol]> {
  let summary = "Pushdown stack descriptor (for context-free parsing)";
  let arguments = (ins SymbolNameAttr:$sym_name, I64Attr:$max_depth, StrAttr:$element_shape);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_FSMCaptureOp : BCIR_Op<"fsm.capture", [Symbol]> {
  let summary = "Capture a span/field from the input stream";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$stream,
                       StrAttr:$capture_kind, I64Attr:$begin, I64Attr:$end);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_FSMReduceOp : BCIR_Op<"fsm.reduce", [Symbol]> {
  let summary = "Reduce captured input into a BCIR object (claim/resource/token/event)";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$rule,
                       FlatSymbolRefArrayAttr:$captures, StrAttr:$emits);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_TRANSDUCER_OPS
