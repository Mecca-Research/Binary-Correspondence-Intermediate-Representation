; ssa-phi.ll — phi nodes for if/else merge and loop counter
;
; Assemble with: llvm-as ssa-phi.ll -o /dev/null
; Verify  with: opt -passes=verify ssa-phi.ll -o /dev/null

source_filename = "ssa-phi.ll"

; if (cond) x = 1 else x = 2; return x + 3
define i32 @if_else(i1 %cond) {
entry:
  br i1 %cond, label %if_true, label %if_false

if_true:
  br label %merge

if_false:
  br label %merge

merge:
  %x = phi i32 [ 1, %if_true ], [ 2, %if_false ]
  %y = add i32 %x, 3
  ret i32 %y
}

; sum 0..n-1
define i32 @sum_to(i32 %n) {
entry:
  br label %loop

loop:
  %i        = phi i32 [ 0, %entry ], [ %i_next,   %loop ]
  %sum      = phi i32 [ 0, %entry ], [ %sum_next, %loop ]
  %i_next   = add i32 %i, 1
  %sum_next = add i32 %sum, %i
  %done     = icmp eq i32 %i_next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %sum_next
}
