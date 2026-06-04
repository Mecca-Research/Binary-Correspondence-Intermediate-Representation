; Run:
;   opt -S -passes=gvn gvn-load-before.ll -o gvn-load-after.ll
;
; With no intervening writes and a readonly helper call, the second load from
; %p is redundant. GVN can reuse the first loaded value.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "gvn-load-before.ll"

declare void @readonly_observer(ptr nocapture readonly) readnone

define i32 @redundant_load(ptr nocapture readonly %p) {
entry:
  %first = load i32, ptr %p, align 4
  call void @readonly_observer(ptr %p)
  %second = load i32, ptr %p, align 4
  %sum = add i32 %first, %second
  ret i32 %sum
}
