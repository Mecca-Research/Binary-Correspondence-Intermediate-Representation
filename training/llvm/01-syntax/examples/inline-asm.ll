; Inline asm is target-specific. These examples are x86-64-flavored and are
; intended for llvm-as / verifier training, not portable llc smoke lowering.
source_filename = "inline-asm.ll"
target triple = "x86_64-unknown-linux-gnu"

; Output constraint (=r) with a tied input constraint (0).
define i32 @bswap_reg(i32 %x) {
entry:
  %y = call i32 asm "bswap $0", "=r,0"(i32 %x)
  ret i32 %y
}

; Early-clobber output (=&r), a tied input (0), another register input (r),
; and a condition-code clobber (~{cc}).
define i32 @add_tied_early_clobber(i32 %x, i32 %y) {
entry:
  %sum = call i32 asm "addl $2, $0", "=&r,0,r,~{cc}"(i32 %x, i32 %y)
  ret i32 %sum
}

; Memory input. The elementtype marker documents the memory operand type for
; opaque pointers.
define i32 @load_with_memory_constraint(ptr %p) {
entry:
  %v = call i32 asm "movl $1, $0", "=r,*m"(ptr elementtype(i32) %p)
  ret i32 %v
}

; A side-effecting empty asm with a memory clobber is a compiler barrier.
define void @compiler_barrier() {
entry:
  call void asm sideeffect "", "~{memory}"()
  ret void
}

; callbr models asm-goto-style control flow. The !i label constraint consumes
; the indirect destination label from the bracketed callbr label list.
define i32 @asm_goto_flag(i32 %x) {
entry:
  callbr void asm sideeffect "testl $0, $0; jne ${1:l}", "r,!i,~{cc}"(i32 %x)
          to label %fallthrough [label %taken]

fallthrough:
  ret i32 0

taken:
  ret i32 1
}
