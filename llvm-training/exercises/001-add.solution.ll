; Exercise 001: Add two i32 values.
; Verification: llvm-as -disable-output llvm-training/exercises/001-add.solution.ll

source_filename = "001-add.solution.ll"

define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
