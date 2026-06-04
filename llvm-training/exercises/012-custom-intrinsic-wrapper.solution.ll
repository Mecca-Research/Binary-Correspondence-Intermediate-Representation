; Exercise 012: Wrap an LLVM intrinsic behind a BCIR operation.
; Verification: llvm-as -disable-output llvm-training/exercises/012-custom-intrinsic-wrapper.solution.ll

source_filename = "012-custom-intrinsic-wrapper.solution.ll"

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)

define i32 @bcir.exercise.uadd_checked.i32(i32 %lhs, i32 %rhs, ptr %overflow_out) {
entry:
  %pair = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %lhs, i32 %rhs)
  %sum = extractvalue { i32, i1 } %pair, 0
  %overflow = extractvalue { i32, i1 } %pair, 1
  store i1 %overflow, ptr %overflow_out, align 1
  ret i32 %sum
}
