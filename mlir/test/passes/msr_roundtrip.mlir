// RUN: bcir-opt %s | bcir-opt | FileCheck %s
//
// bcir.msr_read / bcir.msr_write parse/print round-trip (D1.4): the boot/CPU-state MSR asm edge prints
// back identically through two bcir-opt passes (custom assemblyFormat is stable). The index is a runtime
// i32 operand (the MSR number, placed in ECX) and the value is i64 (rdmsr/wrmsr's EDX:EAX pair as a clean
// 64-bit register). x86-only; lowered to `rdmsr`/`wrmsr` by -convert-bcir-to-llvm (see msr.mlir).

// CHECK-LABEL: func.func @msr_rd
func.func @msr_rd(%idx: i32) -> i64 {
  // CHECK: %{{.*}} = bcir.msr_read %arg0 : i32 -> i64
  %v = bcir.msr_read %idx : i32 -> i64
  return %v : i64
}

// CHECK-LABEL: func.func @msr_wr
func.func @msr_wr(%idx: i32, %val: i64) {
  // CHECK: bcir.msr_write %arg0, %arg1 : i32, i64
  bcir.msr_write %idx, %val : i32, i64
  return
}
