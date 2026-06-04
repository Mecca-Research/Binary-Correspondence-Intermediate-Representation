; ModuleID = 'llvm-training/07-optimization/examples/gvn-before.ll'
source_filename = "gvn-before.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @remove_redundant_add(i32 %a, i32 %b) {
entry:
  %sum1 = add i32 %a, %b
  %doubled = add i32 %sum1, %sum1
  ret i32 %doubled
}
