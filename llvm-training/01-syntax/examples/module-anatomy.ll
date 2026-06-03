; module-anatomy.ll — every common top-level construct in one place
;
; Assemble: llvm-as module-anatomy.ll -o /dev/null

source_filename = "module-anatomy.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; ---- Globals ----
@counter   = global i32 0, align 4
@.msg      = private unnamed_addr constant [12 x i8] c"Hello, ABI\0A\00"

; ---- External declarations ----
declare i32 @printf(ptr, ...)

; ---- Function: simple add ----
define i32 @add(i32 %a, i32 %b) {
entry:
  %r = add i32 %a, %b
  ret i32 %r
}

; ---- Function: control flow with two basic blocks ----
define i32 @max(i32 %a, i32 %b) {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %take_a, label %take_b

take_a:
  ret i32 %a

take_b:
  ret i32 %b
}

; ---- Function: uses a global, calls printf ----
define i32 @main() {
entry:
  store i32 42, ptr @counter, align 4
  %v   = load i32, ptr @counter, align 4
  %ret = call i32 (ptr, ...) @printf(ptr @.msg)
  ret i32 0
}

; ---- Module flags ----
!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
