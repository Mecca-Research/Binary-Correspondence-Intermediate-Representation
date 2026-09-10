; Run:
;   opt -S -passes=loop-rotate opt-diff-loop-rotate-before.ll -o opt-diff-loop-rotate.after-loop-rotate.ll
;
; A while-shaped counted loop. Loop rotation moves the test to the latch.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "opt-diff-loop-rotate-before.ll"

define i32 @rotated_sum(i32 %n) {
entry:
  br label %loop.header

loop.header:
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop.body ]
  %sum = phi i32 [ 0, %entry ], [ %next_sum, %loop.body ]
  %keep_going = icmp slt i32 %i, %n
  br i1 %keep_going, label %loop.body, label %exit

loop.body:
  %next_sum = add i32 %sum, %i
  %next_i = add i32 %i, 1
  br label %loop.header

exit:
  ret i32 %sum
}
