; Run:
;   opt -S -passes=loop-unroll loop-unroll-before.ll -o loop-unroll-after.ll
;
; The loop metadata requests full unrolling of this fixed-trip-count loop.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "loop-unroll-before.ll"

define i32 @unroll_sum_to_four() {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %next_sum, %loop ]
  %next_sum = add i32 %sum, %i
  %next_i = add i32 %i, 1
  %keep_going = icmp slt i32 %next_i, 4
  br i1 %keep_going, label %loop, label %exit, !llvm.loop !0

exit:
  ret i32 %next_sum
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.full"}
