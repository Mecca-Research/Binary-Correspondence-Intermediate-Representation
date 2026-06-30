// RUN: bcir-opt %s | bcir-opt | FileCheck %s
//
// The bcir.asm op (ASM1, SEG8.1): the law-rail twin of the cfront c.asm: / c.asm.volatile: claim
// (bcir/frontends/cfront/lower.py::_asm_stmt + AsmInfo). GNU extended inline asm is a TRUSTED OPAQUE
// EFFECT EDGE -- the template + per-operand constraints + clobbers are carried VERBATIM (ISA-neutral),
// the operand SSA values ride the variadic operands (outputs-then-inputs), and the results are one per
// output constraint. The closest sibling op is bcir.barrier: a "memory"-clobber / volatile form is an
// ordering fence. This is the op + verifier + round-trip slice; the LLVM lowering is inline_asm.mlir.
// The op auto-registers via the dialect .td; this round-trips it (parse -> print -> parse -> print)
// byte-identically.

// (1) a 0-OUTPUT volatile "memory"-clobber form -- the fence-like asm (e.g. a verbatim `mfence`). No
// operands, no results: it carries pure ordering effect, exactly like the barriered c.asm.volatile:.
// CHECK-LABEL: func.func @asm_fence
func.func @asm_fence() {
  // CHECK: bcir.asm volatile "mfence" outs [] ins [] clobbers ["memory"]
  bcir.asm volatile "mfence" outs [] ins [] clobbers ["memory"]
  return
}

// (2) a 1-OUTPUT 1-INPUT form (`"=r"` output, `"r"` input). The operands are outputs-then-inputs: the
// first operand (%dst) is the write-only output lvalue's SSA value, the second (%a) the input value;
// the single result is the value the asm writes back. Round-trips byte-identically.
// CHECK-LABEL: func.func @asm_mov
func.func @asm_mov(%dst: i32, %a: i32) -> i32 {
  // CHECK: %{{.*}} = bcir.asm "movl $1, $0" outs ["=r"] ins ["r"] clobbers [](%arg0, %arg1) : (i32, i32) -> i32
  %r = bcir.asm "movl $1, $0" outs ["=r"] ins ["r"] clobbers [] (%dst, %a) : (i32, i32) -> i32
  return %r : i32
}
