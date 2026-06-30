// RUN: bcir-opt -convert-bcir-to-llvm -verify-diagnostics -split-input-file %s
//
// bcir.asm LOWERING negatives (ASM1, -convert-bcir-to-llvm): the op VERIFIER admits forms the LLVM
// lowering scopes out, because they lower to LLVM IR that opt/llc REJECT (the masking the
// assemble-smoke-test closes -- these are caught at lowering time with a clear diagnostic instead of
// shipping invalid LLVM). Each section carries a SINGLE rejected form (the gem_autodiff_verify_neg.mlir
// style). These are LOWERING diagnostics, so the pass flag is `-convert-bcir-to-llvm`, not the bare
// op verifier. The dialect-conversion driver emits a second "failed to legalize" diagnostic after the
// pattern's emitError, so each section matches BOTH.

// (1 -- READ-WRITE '+' OUTPUT) the verifier admits a '+' (read-write) out-constraint, but a '+' output
// is NOT a write-only result: lowering it as-is yields `llvm.inline_asm ... "+r,r" %x : (i32) -> i32`,
// which opt/llc reject ("inline asm without outputs must return void"). The full '+' handling (the tied
// input + matching constraint) is a follow-on (SEG8.x); rejected here.
func.func @plus_output(%dst: i32, %a: i32) -> i32 {
  // expected-error @+2 {{bcir.asm: read-write '+' output lowering is a follow-on (SEG8.x)}}
  // expected-error @+1 {{failed to legalize operation 'bcir.asm'}}
  %r = bcir.asm "movl %1, %0" outs ["+r"] ins ["r"] clobbers [] (%dst, %a) : (i32, i32) -> i32
  return %r : i32
}

// -----

// (2 -- MULTI-OUTPUT) a >1-output asm makes LLVM return a struct needing extractvalue unpacking -- a
// follow-on (SEG8.x), rejected rather than shipping a wrong struct-unpack.
func.func @multi_output(%d0: i32, %d1: i32, %a: i32) -> (i32, i32) {
  // expected-error @+2 {{bcir.asm: multi-output inline asm lowering is a follow-on (SEG8.x)}}
  // expected-error @+1 {{failed to legalize operation 'bcir.asm'}}
  %r:2 = bcir.asm "movl %2, %0" outs ["=r", "=r"] ins ["r"] clobbers [] (%d0, %d1, %a) : (i32, i32, i32) -> (i32, i32)
  return %r#0, %r#1 : i32, i32
}

// -----

// (3 -- EXOTIC GCC NAMED OPERAND) `%[name]` (a named operand) cannot be translated by the bounded
// numeric-operand rule set; rejected with a clear diagnostic rather than mistranslated.
func.func @exotic_named(%dst: i32, %a: i32) -> i32 {
  // expected-error @+2 {{bcir.asm: unsupported GCC asm operand syntax '%[' (translate or use $-form)}}
  // expected-error @+1 {{failed to legalize operation 'bcir.asm'}}
  %r = bcir.asm "movl %[src], %0" outs ["=r"] ins ["r"] clobbers [] (%dst, %a) : (i32, i32) -> i32
  return %r : i32
}

// -----

// (4 -- EXOTIC GCC OPERAND MODIFIER) `%P0` / `%c0` (non-size operand modifiers: a letter NOT forming a
// size-modifier-then-index, i.e. a letter followed by a non-digit) is rejected. Here `%Pa` -- a letter
// modifier followed by a non-digit -- trips the "unsupported operand modifier" diagnostic.
func.func @exotic_modifier(%dst: i32, %a: i32) -> i32 {
  // expected-error @+2 {{bcir.asm: unsupported GCC asm operand modifier '%P' (translate or use $-form)}}
  // expected-error @+1 {{failed to legalize operation 'bcir.asm'}}
  %r = bcir.asm "call %Pa" outs ["=r"] ins ["r"] clobbers [] (%dst, %a) : (i32, i32) -> i32
  return %r : i32
}

// -----

// (5 -- EXOTIC GCC UNIQUE-LABEL) `%=` (the GCC unique-id form) is neither `%%`, a digit, nor a letter:
// rejected.
func.func @exotic_uniqid(%dst: i32, %a: i32) -> i32 {
  // expected-error @+2 {{bcir.asm: unsupported GCC asm operand syntax '%=' (translate or use $-form)}}
  // expected-error @+1 {{failed to legalize operation 'bcir.asm'}}
  %r = bcir.asm "jmp .L%=" outs ["=r"] ins ["r"] clobbers [] (%dst, %a) : (i32, i32) -> i32
  return %r : i32
}
