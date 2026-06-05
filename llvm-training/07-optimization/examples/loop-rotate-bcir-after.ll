; One possible result of:
;   opt -S -passes=loop-rotate loop-rotate-bcir-before.ll -o loop-rotate-bcir-after.ll
;
; The source-shaped header test has become an entry/preheader test, and the
; repeated loop condition now lives on the latch. BCIR lowering or diagnostics
; that expect the original header shape must account for this canonical form.

source_filename = "loop-rotate-bcir-before.ll"
target triple = "x86_64-unknown-linux-gnu"

define i32 @bcir_counted_sum(ptr %values, i32 %n) {
entry:
  %keep_going.entry = icmp slt i32 0, %n, !bcir.diag !0
  br i1 %keep_going.entry, label %bcir.loop.body.lr.ph, label %bcir.loop.exit, !bcir.diag !1

bcir.loop.body.lr.ph:
  br label %bcir.loop.body

bcir.loop.body:
  %sum.rot = phi i32 [ 0, %bcir.loop.body.lr.ph ], [ %next_sum, %bcir.loop.body ], !bcir.diag !2
  %i.rot = phi i32 [ 0, %bcir.loop.body.lr.ph ], [ %next_i, %bcir.loop.body ], !bcir.diag !3
  %addr = getelementptr inbounds i32, ptr %values, i32 %i.rot
  %value = load i32, ptr %addr, align 4, !bcir.diag !4
  %next_sum = add i32 %sum.rot, %value, !bcir.diag !5
  %next_i = add i32 %i.rot, 1, !bcir.diag !6
  %keep_going.latch = icmp slt i32 %next_i, %n, !bcir.diag !0
  br i1 %keep_going.latch, label %bcir.loop.body, label %bcir.loop.exit.loopexit, !bcir.diag !1

bcir.loop.exit.loopexit:
  %sum.lcssa = phi i32 [ %next_sum, %bcir.loop.body ]
  br label %bcir.loop.exit

bcir.loop.exit:
  %result = phi i32 [ 0, %entry ], [ %sum.lcssa, %bcir.loop.exit.loopexit ]
  ret i32 %result
}

!0 = !{!"bcir.node", i32 32, !"rotated predicate from source header"}
!1 = !{!"bcir.edge", i32 33, !"rotated branch from source header"}
!2 = !{!"bcir.node", i32 31, !"rotated accumulator phi"}
!3 = !{!"bcir.node", i32 30, !"rotated index phi"}
!4 = !{!"bcir.node", i32 34, !"body load"}
!5 = !{!"bcir.node", i32 35, !"body accumulation"}
!6 = !{!"bcir.node", i32 36, !"body increment"}
