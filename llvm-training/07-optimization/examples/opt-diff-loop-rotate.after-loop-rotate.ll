; ModuleID = 'llvm-training/07-optimization/examples/opt-diff-loop-rotate-before.ll'
source_filename = "opt-diff-loop-rotate-before.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @rotated_sum(i32 %n) {
entry:
  %keep_going1 = icmp slt i32 0, %n
  br i1 %keep_going1, label %loop.body.lr.ph, label %exit

loop.body.lr.ph:                                  ; preds = %entry
  br label %loop.body

loop.body:                                        ; preds = %loop.body.lr.ph, %loop.body
  %sum3 = phi i32 [ 0, %loop.body.lr.ph ], [ %next_sum, %loop.body ]
  %i2 = phi i32 [ 0, %loop.body.lr.ph ], [ %next_i, %loop.body ]
  %next_sum = add i32 %sum3, %i2
  %next_i = add i32 %i2, 1
  %keep_going = icmp slt i32 %next_i, %n
  br i1 %keep_going, label %loop.body, label %loop.header.exit_crit_edge

loop.header.exit_crit_edge:                       ; preds = %loop.body
  %split = phi i32 [ %next_sum, %loop.body ]
  br label %exit

exit:                                             ; preds = %loop.header.exit_crit_edge, %entry
  %sum.lcssa = phi i32 [ %split, %loop.header.exit_crit_edge ], [ 0, %entry ]
  ret i32 %sum.lcssa
}
