// RUN: bcir-opt -convert-bcir-to-llvm %s | FileCheck %s
//
// bcir.creg_read / bcir.creg_write -> llvm.inline_asm (boot/CPU-state asm edge, D1.3): an x86-64
// control-register access. Unlike bcir.portio (which mirrors the cfront GCC C-asm templates
// byte-identically), the creg ops have NO cfront dual-rail twin, so the template is authored directly in
// LLVM-IR inline-asm syntax ($0 operand, single-% register) -- the exact form clang emits for
// `__asm__("mov %%cr3, %0" : "=r"(v))`. has_side_effects is always set (a control register is not a
// pure value; a write reloads page tables / toggles paging).

// (1) read CR3 -> "mov %cr3, $0", "=r", returns i64.
// CHECK-LABEL: func.func @rd_cr3
func.func @rd_cr3() -> i64 {
  // CHECK: %[[V:.*]] = llvm.inline_asm has_side_effects "mov %cr3, $0", "=r,~{memory}" : () -> i64
  %v = bcir.creg_read <cr3> -> i64
  // CHECK: return %[[V]] : i64
  return %v : i64
}

// (2) read CR2 (the faulting-address register) -> "mov %cr2, $0".
// CHECK-LABEL: func.func @rd_cr2
func.func @rd_cr2() -> i64 {
  // CHECK: llvm.inline_asm has_side_effects "mov %cr2, $0", "=r,~{memory}" : () -> i64
  %v = bcir.creg_read <cr2> -> i64
  return %v : i64
}

// (3) write CR3 -> "mov $0, %cr3", "r,~{memory}", no result (a CR3 write reloads the page tables).
// CHECK-LABEL: func.func @wr_cr3
func.func @wr_cr3(%v: i64) {
  // CHECK: llvm.inline_asm has_side_effects "mov $0, %cr3", "r,~{memory}" %arg0 : (i64) -> ()
  // CHECK-NOT: = llvm.inline_asm
  bcir.creg_write <cr3>, %v : i64
  return
}

// (4) write CR0 (paging/protection control bits) -> "mov $0, %cr0".
// CHECK-LABEL: func.func @wr_cr0
func.func @wr_cr0(%v: i64) {
  // CHECK: llvm.inline_asm has_side_effects "mov $0, %cr0", "r,~{memory}" %arg0 : (i64) -> ()
  bcir.creg_write <cr0>, %v : i64
  return
}
