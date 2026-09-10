; Run:
;   opt -S -passes=instcombine opt-diff-instcombine-before.ll -o opt-diff-instcombine.after-instcombine.ll
;
; Algebraic identities below are deliberately written in non-canonical forms.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "opt-diff-instcombine-before.ll"

define i32 @combine_identities(i32 %x) {
entry:
  %plus_zero = add i32 %x, 0
  %times_one = mul i32 %plus_zero, 1
  %known_true = icmp eq i32 %times_one, %x
  %as_int = zext i1 %known_true to i32
  %answer = add i32 %times_one, %as_int
  ret i32 %answer
}
