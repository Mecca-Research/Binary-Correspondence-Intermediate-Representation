target triple = "x86_64-unknown-linux-gnu"

; Examples for 06-fast-math-flags.md.

; Strict operation: preserves ordinary floating-point semantics as far as this
; IR operation is concerned.
define float @strict_add(float %a, float %b) {
entry:
  %sum = fadd float %a, %b
  ret float %sum
}

; Narrow promises: this operation excludes NaN and infinity cases, but does not
; permit arbitrary reassociation or reciprocal transforms.
define float @finite_add(float %a, float %b) {
entry:
  %sum = fadd nnan ninf float %a, %b
  ret float %sum
}

; Contract alone permits multiply-add fusion where target and pipeline options
; allow it, without granting all other fast-math permissions.
define float @contracted_mul_add(float %a, float %b, float %c) {
entry:
  %prod = fmul contract float %a, %b
  %sum = fadd contract float %prod, %c
  ret float %sum
}

; Reassociation changes the permitted reduction tree. This is useful for
; numerical kernels that opted into relaxed order, but wrong for exact replay.
define float @reassoc_tree_sum(float %a, float %b, float %c, float %d) {
entry:
  %s01 = fadd reassoc float %a, %b
  %s23 = fadd reassoc float %c, %d
  %sum = fadd reassoc float %s01, %s23
  ret float %sum
}

; The fast shorthand grants the broad aggressive fast-math set. BCIR should emit
; this only when strict NaN, infinity, zero-sign, rounding, and ordering details
; are intentionally irrelevant.
define float @fully_fast(float %a, float %b, float %c) {
entry:
  %prod = fmul fast float %a, %b
  %sum = fadd fast float %prod, %c
  ret float %sum
}

; Vector flags apply to lane operations. A single observed lane that can contain
; NaN, infinity, or meaningful signed zero is enough to make these flags unsafe.
define <4 x float> @vector_finite_add(<4 x float> %x, <4 x float> %y) {
entry:
  %sum = fadd nnan ninf nsz <4 x float> %x, %y
  ret <4 x float> %sum
}
