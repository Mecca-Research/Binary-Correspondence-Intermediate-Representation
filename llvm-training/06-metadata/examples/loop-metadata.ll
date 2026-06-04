; Standalone example: loop metadata and branch-weight profile metadata.
; Assemble: llvm-as llvm-training/06-metadata/examples/loop-metadata.ll -o /dev/null

source_filename = "loop-metadata.c"
target triple = "x86_64-unknown-linux-gnu"

define i32 @sum_to_n(i32 %n) {
entry:
  %has_work = icmp sgt i32 %n, 0
  br i1 %has_work, label %loop, label %exit, !prof !0

loop:
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %next_sum, %loop ]
  %next_sum = add i32 %sum, %i
  %next_i = add i32 %i, 1
  %keep_going = icmp slt i32 %next_i, %n
  br i1 %keep_going, label %loop, label %exit, !prof !1, !llvm.loop !2

exit:
  %result = phi i32 [ 0, %entry ], [ %next_sum, %loop ]
  ret i32 %result
}

!0 = !{!"branch_weights", i32 90, i32 10}
!1 = !{!"branch_weights", i32 1000, i32 1}
!2 = distinct !{!2, !3, !4}
!3 = !{!"llvm.loop.unroll.count", i32 4}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
