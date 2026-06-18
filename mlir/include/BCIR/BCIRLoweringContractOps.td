//===- BCIRLoweringContractOps.td - target lowering contracts (M3) -*- tablegen -*-===//
//
// How BCIR/GEM maps to ISA/opcode/packet realities (LangRef Sec. 12). This is
// driver/compiler law, not C++ runtime policy: ISA families, features, register
// classes, opcodes, packet formats, and a contract tying a BCIR/GEM semantic op
// to an ISA opcode/packet path.
//
//===----------------------------------------------------------------------===//
#ifndef BCIR_LOWERING_CONTRACT_OPS
#define BCIR_LOWERING_CONTRACT_OPS

include "BCIR/BCIRDialect.td"
include "BCIR/BCIRAttrs.td"
include "mlir/IR/SymbolInterfaces.td"

def BCIR_ISAFamilyOp : BCIR_Op<"isa.family", [Symbol]> {
  let summary = "ISA family descriptor (e.g. aarch64, riscv64, nvptx)";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$name, StrAttr:$endianness,
                       I32Attr:$word_bits, I32Attr:$pointer_bits);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_ISAFeatureOp : BCIR_Op<"isa.feature", [Symbol]> {
  let summary = "ISA feature descriptor";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$isa,
                       StrAttr:$feature, StrAttr:$kind);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_ISARegisterClassOp : BCIR_Op<"isa.register_class", [Symbol]> {
  let summary = "ISA register class descriptor";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$isa, StrAttr:$name,
                       I32Attr:$count, I32Attr:$width_bits, StrAttr:$role);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_ISAOpcodeOp : BCIR_Op<"isa.opcode", [Symbol]> {
  let summary = "ISA opcode-level lowering target";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$isa, StrAttr:$mnemonic,
                       StrAttr:$semantic_op, BCIR_LaneAttr:$lane, BCIR_StrideClassAttr:$stride_class,
                       StrAttr:$operand_shape, I32Attr:$latency_cyc, I32Attr:$throughput_q16);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_PacketFormatOp : BCIR_Op<"packet.format", [Symbol]> {
  let summary = "Backend packet or command-buffer format";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kind, FlatSymbolRefAttr:$target,
                       I32Attr:$header_bits, I32Attr:$payload_bits, ArrayAttr:$fields);
  let assemblyFormat = "$sym_name attr-dict";
}

def BCIR_TargetLowerContractOp : BCIR_Op<"target.lower_contract", [Symbol]> {
  let summary = "Lowering contract from a BCIR/GEM semantic op to a backend form (R12)";
  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$target, StrAttr:$bcir_op,
                       BCIR_LaneAttr:$lane, BCIR_StrideClassAttr:$stride_class,
                       FlatSymbolRefAttr:$opcode, OptionalAttr<FlatSymbolRefAttr>:$packet,
                       StrAttr:$legal_if, StrAttr:$preserves,
                       // MOPC objective-support refinement (R12; kbcir/mapping.py): the
                       // cost-dimension indices (0..11) where the source objective J and the
                       // lowered objective J' are nonzero, and the indices this contract
                       // explicitly *discharges*. f(Supp(J)) must lie in Supp(J') (identity
                       // dim-map) unless discharged -- a lowering may rescale/fuse a dimension
                       // but not silently drop a nonzero one. Optional + defaulted absent, so
                       // existing contracts parse and are unchecked.
                       OptionalAttr<DenseI64ArrayAttr>:$source_support,
                       OptionalAttr<DenseI64ArrayAttr>:$target_support,
                       OptionalAttr<DenseI64ArrayAttr>:$discharges);
  let assemblyFormat = "$sym_name attr-dict";
}

#endif // BCIR_LOWERING_CONTRACT_OPS
