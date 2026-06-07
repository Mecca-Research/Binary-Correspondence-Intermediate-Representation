// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// M5 Event Transduction layer in generic syntax, round-tripped through the IRDL
// projection on stock mlir-opt (the "mouth" of BCIR as pure data). Structural
// rail: region-bearing ops use empty regions; leaf ops are siblings.

"bcir.event.stream"() {sym_name = "src", kind = "text", encoding = "utf8",
  element_bits = 8 : i64, max_window = 4096 : i64} : () -> ()
"bcir.parse.grammar"() ({}) {sym_name = "g", syntax = "ebnf", start_symbol = "claim"} : () -> ()
"bcir.parse.token"() {sym_name = "IDENT", pattern = "[A-Za-z_][A-Za-z0-9_]*",
  skip = false, precedence = 0 : i32} : () -> ()
"bcir.parse.rule"() {sym_name = "claim_rule", lhs = "claim",
  rhs = ["claim", "IDENT", "{", "body", "}"], action = "emit_bcir_claim"} : () -> ()
"bcir.fsm.machine"() ({}) {sym_name = "m", kind = "transducer",
  input_stream = "src", start_state = "s0"} : () -> ()
"bcir.binary.format"() ({}) {sym_name = "nvme", endianness = "little", alignment_bits = 32 : i64} : () -> ()
"bcir.binary.field"() {sym_name = "opcode", name = "opcode", offset_bits = 0 : i64,
  width_bits = 8 : i64, kind = "u", semantic = "command_opcode"} : () -> ()
"bcir.binary.decode"() {sym_name = "decode", format = "nvme", stream = "src",
  emits = "event.nvme_command"} : () -> ()

// CHECK: "bcir.event.stream"
// CHECK: "bcir.parse.grammar"
// CHECK: "bcir.parse.token"
// CHECK: "bcir.fsm.machine"
// CHECK: "bcir.binary.format"
// CHECK: "bcir.binary.decode"
