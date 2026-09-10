; Run:
;   opt -S -passes=mem2reg mem2reg-before.ll -o -
;
; The stack slot %x is promotable because it is a simple local alloca whose
; address does not escape.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "mem2reg-before.ll"

define i32 @promote_local(i32 %a, i32 %b) {
entry:
  %x = alloca i32, align 4
  %sum = add i32 %a, %b
  store i32 %sum, ptr %x, align 4
  %loaded = load i32, ptr %x, align 4
  %answer = add i32 %loaded, 1
  ret i32 %answer
}
