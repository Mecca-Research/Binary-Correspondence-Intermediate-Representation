; Exercise 007: Vector reduction intrinsic.
; Verification: llvm-as -disable-output llvm-training/exercises/007-vector-reduction.solution.ll

source_filename = "007-vector-reduction.solution.ll"

declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)

define i32 @sum4(<4 x i32> %v) {
entry:
  %sum = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %v)
  ret i32 %sum
}
