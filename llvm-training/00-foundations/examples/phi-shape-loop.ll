; A loop phi lists exactly one incoming value for each predecessor.
source_filename = "phi-shape-loop.c"

define i32 @count_to_n(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %next
}
