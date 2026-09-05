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
  let hasVerifier = 1;
}

def BCIR_BinaryFieldOp : BCIR_Op<"binary.field", [Symbol]> {
  let summary = "Binary field descriptor (offset/width bits, kind, semantic)";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$name, I64Attr:$offset_bits,
                       I64Attr:$width_bits, StrAttr:$kind, StrAttr:$semantic);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def BCIR_BinaryRecordOp : BCIR_Op<"binary.record", [Symbol]> {
  let summary = "Binary record descriptor";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$name,
                       FlatSymbolRefArrayAttr:$fields, I64Attr:$size_bits);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def BCIR_BinaryDecodeOp : BCIR_Op<"binary.decode", [Symbol]> {
  let summary = "Decode a binary stream into records/events";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$format,
                       FlatSymbolRefAttr:$stream, StrAttr:$emits);
  let assemblyFormat = "$sym_name attr-dict";
}

// BCAB v1 metadata in the IR. Payload bytes remain in their standard external
// formats; this surface carries the verified directory/selection contract into
// MLIR without treating ELF/WASM/JVM/SPIR-V as BCIR-native encodings.
def BCIR_ArtifactBundleOp : BCIR_Op<"artifact.bundle",
    [IsolatedFromAbove, NoTerminator, SymbolTable, Symbol]> {
  let summary = "Validated BCIR Artifact Bundle directory";
  let arguments = (ins SymbolNameAttr:$sym_name, I32Attr:$version,
                       StrAttr:$root_variant, StrAttr:$default_variant,
                       StrAttr:$provenance_digest, I64Attr:$generation,
                       I64Attr:$wire_bytes, StrAttr:$artifact_sha256,
                       I64Attr:$body_crc32, I64Attr:$header_crc32);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def BCIR_ArtifactVariantOp : BCIR_Op<"artifact.variant", [Symbol]> {
  let summary = "One standard-format image in a BCIR Artifact Bundle";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kind, StrAttr:$format,
                       StrAttr:$triple, StrAttr:$architecture, StrAttr:$os_abi,
                       StrAttr:$channel, StrAttr:$entry_symbol,
                       StrAttr:$endianness, I32Attr:$pointer_bits,
                       I64Attr:$machine, I32Attr:$priority, I32Attr:$flags,
                       StrAttr:$provenance_digest,
                       ArrayAttr:$required_features, ArrayAttr:$prohibited_features,
                       I64Attr:$payload_offset, I64Attr:$payload_size,
                       I64Attr:$payload_crc32, StrAttr:$payload_sha256,
                       StrAttr:$target_manifest_sha256, I64Attr:$cal_gen);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def BCIR_ArtifactSelectionOp : BCIR_Op<"artifact.selection", [Symbol]> {
  let summary = "Content-addressed artifact selection decision";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$bundle,
                       StrAttr:$variant, StrAttr:$classification,
                       StrAttr:$envelope_sha256, I64Attr:$generation);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

#endif // BCIR_BINARY_FORMAT_OPS
