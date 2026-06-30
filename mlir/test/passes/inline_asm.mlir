// RUN: bcir-opt -convert-bcir-to-llvm %s | FileCheck %s
//
// bcir.asm -> llvm.inline_asm (ASM1 lowering, SEG8.1): the SINGLE LLVM constraint string is the out
// constraints (each already "=..."), then the in constraints, then each clobber rendered as a
// "~{<clobber>}" entry -- ALL comma-joined (LLVM has no separate clobber field; clobbers ARE "~{...}"
// constraint entries). asm is conservatively side-effecting (has_side_effects), never align-stack,
// default (AT&T) dialect. The LLVM `call asm` operand list is the INPUT operands ONLY: a write-only
// "=" output is the RESULT, not an asm-call argument. This establishes the inline-asm lowering the
// SEG8.2 port-I/O op (bcir.portio) will reuse.

// (1) the 0-output "memory"-clobber fence form: void llvm.inline_asm, constraints = just "~{memory}".
// CHECK-LABEL: func.func @asm_fence
func.func @asm_fence() {
  // CHECK: llvm.inline_asm has_side_effects "mfence", "~{memory}"
  bcir.asm volatile "mfence" outs [] ins [] clobbers ["memory"]
  return
}

// (2) the 1-output 1-input form: constraints = "=r,r" (out then in), side-effecting, returning the
// output type. Only the input %arg1 is an operand (%arg0, the output destination, is NOT an asm-call
// argument -- the "=r" output is the result).
// CHECK-LABEL: func.func @asm_mov
func.func @asm_mov(%dst: i32, %a: i32) -> i32 {
  // CHECK: %[[R:.*]] = llvm.inline_asm has_side_effects "movl $1, $0", "=r,r" %arg1 : (i32) -> i32
  %r = bcir.asm "movl $1, $0" outs ["=r"] ins ["r"] clobbers [] (%dst, %a) : (i32, i32) -> i32
  // CHECK: return %[[R]] : i32
  return %r : i32
}

// (3) the out + in + clobber ordering in one constraint string: "=r,r,~{memory}" (out, then in, then
// the "~{...}" clobber entry), proving the three sections concatenate in the right order.
// CHECK-LABEL: func.func @asm_mov_clobber
func.func @asm_mov_clobber(%dst: i32, %a: i32) -> i32 {
  // CHECK: llvm.inline_asm has_side_effects "movl $1, $0", "=r,r,~{memory}" %arg1 : (i32) -> i32
  %r = bcir.asm "movl $1, $0" outs ["=r"] ins ["r"] clobbers ["memory"] (%dst, %a) : (i32, i32) -> i32
  return %r : i32
}
