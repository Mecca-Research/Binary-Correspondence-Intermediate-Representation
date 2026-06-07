// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// M3 target lowering contracts (BCIR-5 bridge) in generic syntax, round-tripped
// through the IRDL projection on stock mlir-opt. ARM-first ISA/opcode/contract.

"bcir.target.capability"() {sym_name = "cpu", triple = "aarch64-linux-gnu",
  isa_features = ["neon"], lane_widths = [1, 4], warp = 0 : i32, cacheline = 64 : i32,
  gather_penalty = 8 : i32, affinity_domains = 8 : i32} : () -> ()
"bcir.isa.family"() {sym_name = "aarch64", name = "aarch64", endianness = "little",
  word_bits = 64 : i32, pointer_bits = 64 : i32} : () -> ()
"bcir.isa.feature"() {sym_name = "neon", isa = "aarch64", feature = "neon", kind = "simd"} : () -> ()
"bcir.isa.register_class"() {sym_name = "vreg128", isa = "aarch64", name = "v",
  count = 32 : i32, width_bits = 128 : i32, role = "vector"} : () -> ()
"bcir.isa.opcode"() {sym_name = "aarch64_fadd_v4f32", isa = "aarch64",
  mnemonic = "fadd vD.4s, vA.4s, vB.4s", semantic_op = "f32.add", lane = "u",
  stride_class = "unit", operand_shape = "v4f32,v4f32,v4f32", latency_cyc = 4 : i32,
  throughput_q16 = 65536 : i32} : () -> ()
"bcir.target.lower_contract"() {sym_name = "lower_vec_add_neon", target = "cpu",
  bcir_op = "vector.add", lane = "u", stride_class = "unit", opcode = "aarch64_fadd_v4f32",
  legal_if = "target.feature.neon && dtype.f32 && width == 4",
  preserves = "bounds,hazard,precision"} : () -> ()

// CHECK: "bcir.target.capability"
// CHECK: "bcir.isa.family"
// CHECK: "bcir.isa.opcode"
// CHECK: "bcir.target.lower_contract"
