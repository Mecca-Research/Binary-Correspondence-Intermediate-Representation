; One possible result of:
;   opt -S -passes='sccp,simplifycfg' sccp-before.ll -o sccp-after.ll
;
; The constant branch and unreachable diagnostic call are gone. This is useful
; optimization, but it also demonstrates why BCIR diagnostics cannot rely only
; on metadata attached to instructions that may become unreachable.

source_filename = "sccp-before.ll"
target triple = "x86_64-unknown-linux-gnu"

declare void @bcir_record(i32)

define i32 @sccp_constant_branch(i32 %x) {
entry:
  %sum = add i32 %x, 42, !bcir.diag !0
  ret i32 %sum
}

!0 = !{!"bcir.node", i32 23, !"fast result"}
