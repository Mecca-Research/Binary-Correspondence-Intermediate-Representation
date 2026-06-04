; Run:
;   opt -S -passes=gvn gvn-before.ll -o gvn-after.ll
;
; The second add computes the same value as the first and can be reused.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "gvn-before.ll"

define i32 @remove_redundant_add(i32 %a, i32 %b) {
entry:
  %sum1 = add i32 %a, %b
  %sum2 = add i32 %a, %b
  %doubled = add i32 %sum1, %sum2
  ret i32 %doubled
}
