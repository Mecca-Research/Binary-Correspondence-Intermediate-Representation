// RUN: bcir-opt -convert-bcir-to-llvm %s | FileCheck %s
//
// The typed x86 driver foundation has two lowering forms:
//   * entry/interrupt symbols become LLVM module-level assembly (no compiler prologue); and
//   * descriptor/task/segment state changes become side-effecting llvm.inline_asm.
// The companion asm_lowering_smoke gate translates this file family through LLVM object emission and
// disassembly; this test pins the MLIR-level contract and constraints.

// CHECK: llvm.module_asm = [
bcir.entry @bcir_boot stack @bcir_stack_top target @bcir_kernel_main
bcir.interrupt_trampoline @bcir_irq32 vector 32 handler @bcir_irq_dispatch {swapgs_on_user = true}
bcir.interrupt_trampoline @bcir_page_fault vector 14 handler @bcir_page_fault_dispatch

llvm.func @descriptor_edges(%addr: i64, %selector: i16) {
  // CHECK-LABEL: llvm.func @descriptor_edges
  // CHECK: llvm.inline_asm has_side_effects "lgdt ($0)", "r,~{memory}" %arg0 : (i64) -> ()
  bcir.descriptor_load <gdt>, %addr : i64
  // CHECK: llvm.inline_asm has_side_effects "lidt ($0)", "r,~{memory}" %arg0 : (i64) -> ()
  bcir.descriptor_load <idt>, %addr : i64
  // CHECK: llvm.inline_asm has_side_effects "ltr ${0:w}", "r,~{memory}" %arg1 : (i16) -> ()
  bcir.task_register_load %selector : i16
  llvm.return
}

llvm.func @segment_edge(%data: i16, %code: i16) {
  // CHECK-LABEL: llvm.func @segment_edge
  // CHECK: %[[CODE64:.*]] = llvm.zext %arg1 : i16 to i64
  // CHECK: llvm.inline_asm has_side_effects "movw ${0:w}, %ds; movw ${0:w}, %es; movw ${0:w}, %ss; pushq ${1:q}; leaq .Lbcir_seg_${:uid}(%rip), %rax; pushq %rax; lretq; .Lbcir_seg_${:uid}:", "r,r,~{rax},~{memory}" %arg0, %[[CODE64]] : (i16, i64) -> ()
  bcir.segment_reload %data, %code : i16, i16
  llvm.return
}
