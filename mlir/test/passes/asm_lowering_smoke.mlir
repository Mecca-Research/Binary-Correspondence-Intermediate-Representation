// Assemble-smoke-test fixtures for the asm-edge -convert-bcir-to-llvm lowerings (ASM1/ASM2/D1.2/D1.3).
//
// NOT a FileCheck test: this is the input to tools/wsl/check_asm_lowering.sh, which pipes the lowered IR
// through the REAL backend (mlir-translate-20 --mlir-to-llvmir | llc-20 -filetype=obj) and greps the
// -filetype=asm output for the expected instruction -- catching lowerings that text-check but do NOT
// assemble (the masking the FileCheck-only tests allowed: `%w1`/`=a,Nd` text-pass but llc rejects them).
//
// WRINKLE: `-convert-bcir-to-llvm` has no bundled func->llvm conversion, so a `func.func` body would
// leave the module mixed-dialect and mlir-translate --mlir-to-llvmir would reject it. The bcir ops sit
// inside `llvm.func` bodies here: after lowering, the WHOLE module is LLVM dialect, so mlir-translate
// gets an all-LLVM module. Each function is named so the harness can `llc -filetype=asm` the module once
// and grep per-instruction.

// --- top-level entry and interrupt trampolines (module asm: no generated function prologue) ---
bcir.entry @bcir_boot stack @bcir_stack_top target @bcir_kernel_main
bcir.interrupt_trampoline @bcir_irq32 vector 32 handler @bcir_irq_dispatch {swapgs_on_user = true}
bcir.interrupt_trampoline @bcir_page_fault vector 14 handler @bcir_page_fault_dispatch
// Every hardware-error vector accepted by the normal trampoline is pinned independently. #DF (8) and
// #VC (29) are classified by the architectural table but refused because they require the future
// paranoid/IST op.
// A missing accepted entry would add a synthetic error word and corrupt the C frame / iretq return.
bcir.interrupt_trampoline @bcir_error10 vector 10 handler @bcir_error_dispatch
bcir.interrupt_trampoline @bcir_error11 vector 11 handler @bcir_error_dispatch
bcir.interrupt_trampoline @bcir_error12 vector 12 handler @bcir_error_dispatch
bcir.interrupt_trampoline @bcir_error13 vector 13 handler @bcir_error_dispatch
bcir.interrupt_trampoline @bcir_error17 vector 17 handler @bcir_error_dispatch
bcir.interrupt_trampoline @bcir_error21 vector 21 handler @bcir_error_dispatch
bcir.interrupt_trampoline @bcir_error30 vector 30 handler @bcir_error_dispatch

// --- bcir.portio in, b/w/l (x86 `in`; assembles to `inb/inw/inl %dx, %al/%ax/%eax`) ---
llvm.func @portio_in_b(%port: i16) -> i8 {
  %r = bcir.portio <in> width 8 (%port) : (i16) -> i8
  llvm.return %r : i8
}
llvm.func @portio_in_w(%port: i16) -> i16 {
  %r = bcir.portio <in> width 16 (%port) : (i16) -> i16
  llvm.return %r : i16
}
llvm.func @portio_in_l(%port: i16) -> i32 {
  %r = bcir.portio <in> width 32 (%port) : (i16) -> i32
  llvm.return %r : i32
}

// --- bcir.portio out, b/w/l (x86 `out`; assembles to `outb/outw/outl %al/%ax/%eax, %dx`) ---
llvm.func @portio_out_b(%val: i8, %port: i16) {
  bcir.portio <out> width 8 (%val, %port) : (i8, i16) -> ()
  llvm.return
}
llvm.func @portio_out_w(%val: i16, %port: i16) {
  bcir.portio <out> width 16 (%val, %port) : (i16, i16) -> ()
  llvm.return
}
llvm.func @portio_out_l(%val: i32, %port: i16) {
  bcir.portio <out> width 32 (%val, %port) : (i32, i16) -> ()
  llvm.return
}

// --- bcir.asm: a representative GCC-template asm that TRANSLATES (`%1`->`$1`, `%0`->`$0`) and
// assembles to a real `movl` (proves the GCC->LLVM-IR translation reaches llc, not just FileCheck) ---
llvm.func @asm_mov(%dst: i32, %a: i32) -> i32 {
  %r = bcir.asm "movl %1, %0" outs ["=r"] ins ["r"] clobbers [] (%dst, %a) : (i32, i32) -> i32
  llvm.return %r : i32
}

// --- bcir.creg_read / bcir.creg_write (x86-64 control register; assembles to `mov %cr3, ...` /
// `mov ..., %cr3`) ---
llvm.func @creg_read() -> i64 {
  %v = bcir.creg_read <cr3> -> i64
  llvm.return %v : i64
}
llvm.func @creg_write(%v: i64) {
  bcir.creg_write <cr3>, %v : i64
  llvm.return
}

// --- bcir.volatile_load / bcir.volatile_store (first-class MMIO; assembles to a real load/store) ---
llvm.func @volatile_load(%addr: i64) -> i32 {
  %v = bcir.volatile_load %addr : i64 -> i32
  llvm.return %v : i32
}
llvm.func @volatile_store(%val: i32, %addr: i64) {
  bcir.volatile_store %val, %addr : i32, i64
  llvm.return
}

// --- bcir.msr_read / bcir.msr_write (x86-64 model-specific register; assembles to `rdmsr` / `wrmsr`,
// with the runtime index marshalled to ECX and the i64 value reassembled/split across EDX:EAX) ---
llvm.func @msr_read(%idx: i32) -> i64 {
  %v = bcir.msr_read %idx : i32 -> i64
  llvm.return %v : i64
}
llvm.func @msr_write(%idx: i32, %val: i64) {
  bcir.msr_write %idx, %val : i32, i64
  llvm.return
}

// --- descriptor-table, task-register, and segment reloads (x86 boot/interrupt foundation) ---
llvm.func @descriptor_loads(%addr: i64, %selector: i16) {
  bcir.descriptor_load <gdt>, %addr : i64
  bcir.descriptor_load <idt>, %addr : i64
  bcir.task_register_load %selector : i16
  llvm.return
}
llvm.func @segment_reload(%data: i16, %code: i16) {
  bcir.segment_reload %data, %code : i16, i16
  llvm.return
}
