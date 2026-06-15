//===- BCIRBinaryFormatOps.td - binary / layout parsing ops (M5) -*- tablegen -*-===//
//
// Binary file / packet formats: format, field, record, decode. Mirrors
// bcir/etl/binary.py. Driver-packet decoding (NVMe SQE, DDR5 timing, MMIO maps)
// is the binary twin of text parsing: bytes -> records -> events.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_BINARY_FORMAT_OPS
#define BCIR_BINARY_FORMAT_OPS

include "BCIR/BCIRDialect.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_BinaryFormatOp : BCIR_Op<"binary.format", [Symbol, SymbolTable, NoTerminator]> {
  let summary = "Binary file or packet format container";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$endianness, I64Attr:$alignment_bits);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

def BCIR_BinaryFieldOp : BCIR_Op<"binary.field", [Symbol]> {
  let summary = "Binary field descriptor (offset/width bits, kind, semantic)";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$name, I64Attr:$offset_bits,
                       I64Attr:$width_bits, StrAttr:$kind, StrAttr:$semantic);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_BinaryRecordOp : BCIR_Op<"binary.record", [Symbol]> {
  let summary = "Binary record descriptor";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$name,
                       FlatSymbolRefArrayAttr:$fields, I64Attr:$size_bits);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_BinaryDecodeOp : BCIR_Op<"binary.decode", [Symbol]> {
  let summary = "Decode a binary stream into records/events";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$format,
                       FlatSymbolRefAttr:$stream, StrAttr:$emits);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_BINARY_FORMAT_OPS
