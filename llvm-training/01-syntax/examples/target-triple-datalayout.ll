; Module-level target information is syntax, not an instruction.
source_filename = "target-triple-datalayout.c"
target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @pointer_sized_integer(i64 %x) {
entry:
  %y = add i64 %x, 8
  ret i64 %y
}
