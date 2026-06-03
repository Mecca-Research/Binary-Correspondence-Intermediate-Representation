; Exercise 003: Loop counter with a loop-carried phi node.
; Verification: llvm-as -disable-output llvm-training/exercises/003-loop-counter.solution.ll

source_filename = "003-loop-counter.solution.ll"

define i32 @count_to_n(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %body ]
  %keep_going = icmp slt i32 %i, %n
  br i1 %keep_going, label %body, label %exit

body:
  %next = add i32 %i, 1
  br label %loop

exit:
  ret i32 %i
}
