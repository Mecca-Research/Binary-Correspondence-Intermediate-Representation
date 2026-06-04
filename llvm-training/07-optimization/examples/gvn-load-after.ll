; Teaching snapshot for:
;   opt -S -passes=gvn gvn-load-before.ll -o gvn-load-after.ll

target triple = "x86_64-unknown-linux-gnu"
source_filename = "gvn-load-before.ll"

declare void @readonly_observer(ptr nocapture readonly) readnone

define i32 @redundant_load(ptr nocapture readonly %p) {
entry:
  %first = load i32, ptr %p, align 4
  call void @readonly_observer(ptr %p)
  %sum = add i32 %first, %first
  ret i32 %sum
}
