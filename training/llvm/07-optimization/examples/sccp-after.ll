; ModuleID = 'sccp-before.ll'
source_filename = "sccp-before.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @bcir_record(i32)

define i32 @sccp_constant_branch(i32 %x) {
entry:
  %sum = add i32 %x, 42, !bcir.diag !0
  ret i32 %sum
}

!0 = !{!"bcir.node", i32 23, !"fast result"}
