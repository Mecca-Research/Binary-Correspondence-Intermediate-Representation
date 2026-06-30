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
