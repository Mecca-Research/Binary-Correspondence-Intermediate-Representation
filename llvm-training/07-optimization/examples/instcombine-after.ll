; ModuleID = 'llvm-training/07-optimization/examples/instcombine-before.ll'
source_filename = "instcombine-before.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @combine_identities(i32 %x) {
entry:
  %answer = add i32 %x, 1
  ret i32 %answer
}
