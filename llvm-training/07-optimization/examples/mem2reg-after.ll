; ModuleID = 'llvm-training/07-optimization/examples/mem2reg-before.ll'
source_filename = "mem2reg-before.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @promote_local(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  %answer = add i32 %sum, 1
  ret i32 %answer
}
