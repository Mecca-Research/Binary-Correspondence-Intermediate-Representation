; Run:
;   opt -S -passes=loop-unroll loop-before.ll -o -
;   opt -passes='print<loops>' loop-before.ll -disable-output
;
; A small counted loop for loop analysis, scalar evolution, and unroll demos.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "loop-before.ll"

define i32 @sum_to_four() {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %sum = phi i32 [ 0, %entry ], [ %next_sum, %loop ]
  %next_sum = add i32 %sum, %i
  %next_i = add i32 %i, 1
  %keep_going = icmp slt i32 %next_i, 4
  br i1 %keep_going, label %loop, label %exit

exit:
  ret i32 %next_sum
}
