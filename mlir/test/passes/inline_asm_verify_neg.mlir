// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// The bcir.asm op verifier (hasVerifier; the ASM1 structural well-formedness check, mirrors
// bcir/frontends/cfront/lower.py::_asm_stmt + AsmInfo -- the op-level check the sibling law ops run
// inline, NOT a new globally-numbered R-law). The operands are outputs-then-inputs, so their count is
// out + in; the results are one per output constraint. Each section carries a SINGLE violation
// (mirrors the gem_autodiff_verify_neg.mlir style).

// (1 -- ARG/CONSTRAINT COUNT) the operand count must equal out_constraints + in_constraints. Here two
// input constraints are declared but only one operand is given -- rejected.
func.func @bad_argcount(%a: i32) {
  // expected-error @+1 {{'bcir.asm' op args count 1 must equal out_constraints + in_constraints = 0 + 2 = 2 (operands are outputs-then-inputs)}}
  bcir.asm "nop" outs [] ins ["r", "r"] clobbers [] (%a) : (i32) -> ()
  return
}

// -----

// (2 -- RESULT/OUT COUNT) the result count must equal out_constraints count (one result per output).
// Here zero output constraints are declared but the op produces one result -- rejected. The operand
// count (2) is kept consistent with out+in (0+2) so it is purely the result/out law that trips.
func.func @bad_rescount(%dst: i32, %a: i32) -> i32 {
  // expected-error @+1 {{'bcir.asm' op results count 1 must equal out_constraints count 0 (one result per output constraint)}}
  %r = bcir.asm "movl $1, $0" outs [] ins ["r", "r"] clobbers [] (%dst, %a) : (i32, i32) -> i32
  return %r : i32
}

// -----

// (3 -- OUTPUT-CONSTRAINT PREFIX) every out_constraints entry must itself be an output constraint, i.e.
// begin with LLVM's write-only '=' or read-write '+'. Here a bare "r" (an INPUT spelling) is given as an
// output: the counts are all consistent (nArgs 2 == out 1 + in 1, nResults 1 == out 1) so it is purely
// the prefix law that trips. Without it this lowers to a result-returning llvm.inline_asm whose
// constraint string has no '=' output -- structurally-invalid asm the LLVM translator rejects far away.
func.func @bad_out_prefix(%dst: i32, %a: i32) -> i32 {
  // expected-error @+1 {{'bcir.asm' op out_constraints[0] must be an output constraint beginning with '=' or '+' (got 'r')}}
  %r = bcir.asm "movl $1, $0" outs ["r"] ins ["r"] clobbers [] (%dst, %a) : (i32, i32) -> i32
  return %r : i32
}
