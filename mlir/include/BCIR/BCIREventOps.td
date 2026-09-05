//===- BCIREventOps.td - BCIR event-machine ops (M5) --------*- tablegen -*-===//
//
// The event substrate of GEM-E: streams, event kinds, emit/consume. Mirrors
// bcir/etl/events.py. Text, binary, packet, and telemetry all become event
// streams into the same transducer machinery.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_EVENT_OPS
#define BCIR_EVENT_OPS

include "BCIR/BCIRDialect.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_EventStreamOp : BCIR_Op<"event.stream", [Symbol]> {
  let summary = "Event stream descriptor (text/binary/telemetry/packet/driver/token)";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kind, StrAttr:$encoding,
                       I64Attr:$element_bits, I64Attr:$max_window);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def BCIR_EventKindOp : BCIR_Op<"event.kind", [Symbol]> {
  let summary = "A named event class with a payload shape";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$name, StrAttr:$payload_shape);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def BCIR_EventEmitOp : BCIR_Op<"event.emit"> {
  let summary = "Emit an event from a transducer action";
  let arguments = (ins FlatSymbolRefAttr:$event, ArrayAttr:$payload);
  let assemblyFormat = "$event attr-dict";
}

def BCIR_EventConsumeOp : BCIR_Op<"event.consume"> {
  let summary = "Consume an event from a stream";
  let arguments = (ins FlatSymbolRefAttr:$stream, FlatSymbolRefAttr:$event);
  let assemblyFormat = "$stream `->` $event attr-dict";
}

#endif // BCIR_EVENT_OPS
