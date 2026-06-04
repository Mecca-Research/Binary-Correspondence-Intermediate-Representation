; Run:
;   opt -S -passes=instcombine instcombine-canonical-before.ll -o instcombine-canonical-after.ll
;
; Non-canonical arithmetic and comparisons that instcombine can fold without
; needing control-flow or interprocedural information.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "instcombine-canonical-before.ll"

define i1 @canonical_predicate(i32 %x) {
entry:
  %double = shl i32 %x, 1
  %plus = add i32 %double, 0
  %same = icmp eq i32 %plus, %double
  ret i1 %same
}

define i32 @canonical_arithmetic(i32 %x) {
entry:
  %masked = and i32 %x, -1
  %shifted = shl i32 %masked, 2
  %scaled = mul i32 %x, 4
  %combined = add i32 %shifted, %scaled
  ret i32 %combined
}
