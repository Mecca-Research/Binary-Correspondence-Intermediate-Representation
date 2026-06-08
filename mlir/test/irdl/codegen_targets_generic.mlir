// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// Phase 9: the per-target lower contracts (the law form of bcir/codegen). Each
// `bcir.target.lower_contract` is the declarative twin of a CodegenTarget --
// ARM/RISC-V/GPU/eBPF, the targets bcir/codegen emits real artifacts for.

"bcir.target.capability"() {sym_name = "riscv", triple = "riscv64",
  isa_features = ["rvv"], lane_widths = [1, 8, 16], warp = 0 : i32, cacheline = 64 : i32} : () -> ()
"bcir.target.lower_contract"() {sym_name = "lc_riscv", target = "riscv", bcir_op = "vector.add",
  lane = "u", stride_class = "unit", opcode = "vadd.vv", legal_if = "rvv",
  preserves = "bounds,hazard,precision"} : () -> ()

"bcir.target.capability"() {sym_name = "gpu", triple = "nvptx64-nvidia-cuda",
  isa_features = ["ptx"], lane_widths = [1, 32], warp = 32 : i32, cacheline = 128 : i32} : () -> ()
"bcir.target.lower_contract"() {sym_name = "lc_gpu", target = "gpu", bcir_op = "vector.add",
  lane = "u", stride_class = "unit", opcode = "add.f32", legal_if = "ptx",
  preserves = "bounds,hazard,precision"} : () -> ()

"bcir.target.capability"() {sym_name = "bpf", triple = "bpfel",
  isa_features = [], lane_widths = [1], warp = 0 : i32, cacheline = 64 : i32} : () -> ()
"bcir.target.lower_contract"() {sym_name = "lc_bpf", target = "bpf", bcir_op = "vector.add",
  lane = "u", stride_class = "unit", opcode = "alu64.add", legal_if = "integer_only",
  preserves = "bounds,hazard"} : () -> ()

// CHECK: "bcir.target.capability"
// CHECK: "bcir.target.lower_contract"
// CHECK: triple = "riscv64"
// CHECK: triple = "nvptx64-nvidia-cuda"
// CHECK: triple = "bpfel"
