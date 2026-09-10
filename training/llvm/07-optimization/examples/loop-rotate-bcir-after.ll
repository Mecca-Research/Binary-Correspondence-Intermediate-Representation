; ModuleID = 'loop-rotate-bcir-before.ll'
source_filename = "loop-rotate-bcir-before.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @bcir_counted_sum(ptr %values, i32 %n) {
entry:
  %keep_going1 = icmp slt i32 0, %n, !bcir.diag !0
  br i1 %keep_going1, label %bcir.loop.body.lr.ph, label %bcir.loop.exit, !bcir.diag !1

bcir.loop.body.lr.ph:                             ; preds = %entry
  br label %bcir.loop.body

bcir.loop.body:                                   ; preds = %bcir.loop.body.lr.ph, %bcir.loop.body
  %sum3 = phi i32 [ 0, %bcir.loop.body.lr.ph ], [ %next_sum, %bcir.loop.body ]
  %i2 = phi i32 [ 0, %bcir.loop.body.lr.ph ], [ %next_i, %bcir.loop.body ]
  %addr = getelementptr inbounds i32, ptr %values, i32 %i2
  %value = load i32, ptr %addr, align 4, !bcir.diag !2
  %next_sum = add i32 %sum3, %value, !bcir.diag !3
  %next_i = add i32 %i2, 1, !bcir.diag !4
  %keep_going = icmp slt i32 %next_i, %n, !bcir.diag !0
  br i1 %keep_going, label %bcir.loop.body, label %bcir.loop.header.bcir.loop.exit_crit_edge, !bcir.diag !1

bcir.loop.header.bcir.loop.exit_crit_edge:        ; preds = %bcir.loop.body
  %split = phi i32 [ %next_sum, %bcir.loop.body ]
  br label %bcir.loop.exit

bcir.loop.exit:                                   ; preds = %bcir.loop.header.bcir.loop.exit_crit_edge, %entry
  %sum.lcssa = phi i32 [ %split, %bcir.loop.header.bcir.loop.exit_crit_edge ], [ 0, %entry ]
  ret i32 %sum.lcssa
}

!0 = !{!"bcir.node", i32 32, !"source header predicate"}
!1 = !{!"bcir.edge", i32 33, !"source header branch"}
!2 = !{!"bcir.node", i32 34, !"body load"}
!3 = !{!"bcir.node", i32 35, !"body accumulation"}
!4 = !{!"bcir.node", i32 36, !"body increment"}
