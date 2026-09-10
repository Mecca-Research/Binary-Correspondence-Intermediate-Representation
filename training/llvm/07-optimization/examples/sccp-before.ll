; Run:
;   opt -S -passes='sccp,simplifycfg' sccp-before.ll -o sccp-after.ll
;
; SCCP proves the diagnostic branch is always true. The cold diagnostic side
; shows metadata that a BCIR pipeline may want to preserve elsewhere before the
; optimizer erases the unreachable block.

source_filename = "sccp-before.ll"
target triple = "x86_64-unknown-linux-gnu"

declare void @bcir_record(i32)

define i32 @sccp_constant_branch(i32 %x) {
entry:
  %known = add i32 40, 2, !bcir.diag !0
  %is_answer = icmp eq i32 %known, 42, !bcir.diag !1
  br i1 %is_answer, label %fast, label %diagnostic, !bcir.diag !2

fast:
  %sum = add i32 %x, %known, !bcir.diag !3
  ret i32 %sum

diagnostic:
  call void @bcir_record(i32 %known), !bcir.diag !4
  ret i32 -1
}

!0 = !{!"bcir.node", i32 20, !"constant candidate"}
!1 = !{!"bcir.node", i32 21, !"branch predicate"}
!2 = !{!"bcir.edge", i32 22, !"diagnostic reachability"}
!3 = !{!"bcir.node", i32 23, !"fast result"}
!4 = !{!"bcir.node", i32 24, !"cold diagnostic"}
