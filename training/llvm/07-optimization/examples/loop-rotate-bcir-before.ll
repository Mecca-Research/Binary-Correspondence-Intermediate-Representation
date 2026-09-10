; Run:
;   opt -S -passes=loop-rotate loop-rotate-bcir-before.ll -o loop-rotate-bcir-after.ll
;
; BCIR-shaped while loop before canonical rotation. The header carries the
; source-shaped loop test and metadata used by diagnostics/mapping.

source_filename = "loop-rotate-bcir-before.ll"
target triple = "x86_64-unknown-linux-gnu"

define i32 @bcir_counted_sum(ptr %values, i32 %n) {
entry:
  br label %bcir.loop.header

bcir.loop.header:
  %i = phi i32 [ 0, %entry ], [ %next_i, %bcir.loop.body ], !bcir.diag !0
  %sum = phi i32 [ 0, %entry ], [ %next_sum, %bcir.loop.body ], !bcir.diag !1
  %keep_going = icmp slt i32 %i, %n, !bcir.diag !2
  br i1 %keep_going, label %bcir.loop.body, label %bcir.loop.exit, !bcir.diag !3

bcir.loop.body:
  %addr = getelementptr inbounds i32, ptr %values, i32 %i
  %value = load i32, ptr %addr, align 4, !bcir.diag !4
  %next_sum = add i32 %sum, %value, !bcir.diag !5
  %next_i = add i32 %i, 1, !bcir.diag !6
  br label %bcir.loop.header

bcir.loop.exit:
  ret i32 %sum
}

!0 = !{!"bcir.node", i32 30, !"loop index phi"}
!1 = !{!"bcir.node", i32 31, !"loop accumulator phi"}
!2 = !{!"bcir.node", i32 32, !"source header predicate"}
!3 = !{!"bcir.edge", i32 33, !"source header branch"}
!4 = !{!"bcir.node", i32 34, !"body load"}
!5 = !{!"bcir.node", i32 35, !"body accumulation"}
!6 = !{!"bcir.node", i32 36, !"body increment"}
