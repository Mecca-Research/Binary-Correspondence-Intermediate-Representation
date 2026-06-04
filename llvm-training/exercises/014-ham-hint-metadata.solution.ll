; Exercise 014: Attach HAM hint metadata to a lowered memory operation.
; Verification: llvm-as -disable-output llvm-training/exercises/014-ham-hint-metadata.solution.ll

source_filename = "014-ham-hint-metadata.solution.ll"

define i32 @bcir.exercise.load_with_ham_hint(ptr %base, i64 %offset) {
entry:
  %addr = getelementptr i8, ptr %base, i64 %offset
  %value = load i32, ptr %addr, align 4, !bcir.ham.hint !1
  ret i32 %value
}

!bcir.ham.hints = !{!0}
!0 = !{!"BCIR HAM hint metadata", !"domain", !"HBM", !"lane", !"H", !"hazard_mode", i32 1, !"locality", i32 3}
!1 = !{!"bcir.ham.hint", !"domain", !"HBM", !"lane", i32 5, !"hazard_mode", i32 1, !"locality", i32 3}
