; ModuleID = 'llvm-training/07-optimization/examples/dead-code-before.ll'
source_filename = "dead-code-before.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @dead_code(i32 %x) {
entry:
  %plus_zero = add i32 %x, 0
  %always = icmp eq i32 1, 1
  br i1 %always, label %then, label %else

then:                                             ; preds = %entry
  ret i32 %plus_zero

else:                                             ; preds = %entry
  ret i32 0
}
