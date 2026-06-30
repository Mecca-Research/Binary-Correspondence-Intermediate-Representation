// RUN: bcir-opt -convert-bcir-to-llvm %s | FileCheck %s
//
// bcir.asm -> llvm.inline_asm (ASM1 lowering, SEG8.1): the SINGLE LLVM constraint string is the out
// constraints (each already "=..."), then the in constraints, then each clobber rendered as a
// "~{<clobber>}" entry -- ALL comma-joined (LLVM has no separate clobber field; clobbers ARE "~{...}"
// constraint entries). asm is conservatively side-effecting (has_side_effects), never align-stack,
// default (AT&T) dialect. The LLVM `call asm` operand list is the INPUT operands ONLY: a write-only
// "=" output is the RESULT, not an asm-call argument. This establishes the inline-asm lowering the
// SEG8.2 port-I/O op (bcir.portio) will reuse.
//
// The asm_template is the user's (cfront/C source) GCC operand syntax (`%0`, `%1`, `%w1`); the lowering
// TRANSLATES it to the LLVM-IR `$`-syntax `llvm.inline_asm` / `llc` require (a verbatim GCC `%1`
// otherwise fails `llc: invalid register name`). The bounded rule set: `%%`->`%`, `%<letter><N>`->
// `${N:<letter>}`, `%<N>`->`$N` (incl. multi-digit), a literal `$`->`$$`; exotic `%[name]`/`%P`/`%=`
// are REJECTED (see inline_asm_verify_neg.mlir).

// (1) the 0-output "memory"-clobber fence form: void llvm.inline_asm, constraints = just "~{memory}".
// CHECK-LABEL: func.func @asm_fence
func.func @asm_fence() {
  // CHECK: llvm.inline_asm has_side_effects "mfence", "~{memory}"
  bcir.asm volatile "mfence" outs [] ins [] clobbers ["memory"]
  return
}

// (2) the 1-output 1-input form: GCC template `movl %1, %0` -> translated `movl $1, $0`, constraints =
// "=r,r" (out then in), side-effecting, returning the output type. Only the input %arg1 is an operand
// (%arg0, the output destination, is NOT an asm-call argument -- the "=r" output is the result).
// CHECK-LABEL: func.func @asm_mov
func.func @asm_mov(%dst: i32, %a: i32) -> i32 {
  // CHECK: %[[R:.*]] = llvm.inline_asm has_side_effects "movl $1, $0", "=r,r" %arg1 : (i32) -> i32
  %r = bcir.asm "movl %1, %0" outs ["=r"] ins ["r"] clobbers [] (%dst, %a) : (i32, i32) -> i32
  // CHECK: return %[[R]] : i32
  return %r : i32
}

// (3) the out + in + clobber ordering in one constraint string: "=r,r,~{memory}" (out, then in, then
// the "~{...}" clobber entry), proving the three sections concatenate in the right order.
// CHECK-LABEL: func.func @asm_mov_clobber
func.func @asm_mov_clobber(%dst: i32, %a: i32) -> i32 {
  // CHECK: llvm.inline_asm has_side_effects "movl $1, $0", "=r,r,~{memory}" %arg1 : (i32) -> i32
  %r = bcir.asm "movl %1, %0" outs ["=r"] ins ["r"] clobbers ["memory"] (%dst, %a) : (i32, i32) -> i32
  return %r : i32
}

// (4) GCC->LLVM-IR operand-syntax translation, the full rule set exercised in one template:
//   * `%w1` (size-modified operand) -> `${1:w}`
//   * `%0`  (bare operand)          -> `$0`
//   * `%%`  (literal percent)       -> `%`
//   * `$`   (literal dollar)        -> `$$`   (here the x86 AT&T immediate `$8` -> `$$8`)
// Confirms the lowering carries the LLVM-IR-correct asm_string, not the verbatim GCC template.
// CHECK-LABEL: func.func @asm_gcc_translate
func.func @asm_gcc_translate(%dst: i16, %a: i16) -> i16 {
  // CHECK: %[[R:.*]] = llvm.inline_asm has_side_effects "addw $$8, ${1:w}; movw ${1:w}, $0 %eax", "=r,r" %arg1 : (i16) -> i16
  %r = bcir.asm "addw $8, %w1; movw %w1, %0 %%eax" outs ["=r"] ins ["r"] clobbers [] (%dst, %a) : (i16, i16) -> i16
  return %r : i16
}

// (5) multi-digit bare operand: `%10` -> `$10` (NOT `$1` then `0`). A 0-output 0-input fence-shaped op
// can still carry a template; here a contrived multi-operand template proves the multi-digit scan.
// CHECK-LABEL: func.func @asm_multidigit
func.func @asm_multidigit() {
  // CHECK: llvm.inline_asm has_side_effects "nop $10 $2", "~{memory}"
  bcir.asm "nop %10 %2" outs [] ins [] clobbers ["memory"]
  return
}
