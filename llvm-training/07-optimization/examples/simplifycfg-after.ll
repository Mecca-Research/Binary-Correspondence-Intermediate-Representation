; ModuleID = 'llvm-training/07-optimization/examples/simplifycfg-before.ll'
source_filename = "simplifycfg-before.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @fold_known_branch(i32 %x) {
entry:
  %kept = add i32 %x, 1
  ret i32 %kept
}
