; Exercise 004: Load from and store to a global variable.
; Verification: llvm-as -disable-output llvm-training/exercises/004-global-load-store.solution.ll

source_filename = "004-global-load-store.solution.ll"

@counter = global i32 0

define i32 @bump_global(i32 %delta) {
entry:
  %old = load i32, ptr @counter
  %new = add i32 %old, %delta
  store i32 %new, ptr @counter
  ret i32 %new
}
