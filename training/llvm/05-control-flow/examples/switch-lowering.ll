; switch models multi-way control flow before target-specific lowering.
source_filename = "switch-lowering.c"

define i32 @classify(i32 %x) {
entry:
  switch i32 %x, label %default [
    i32 0, label %zero
    i32 1, label %one
    i32 7, label %lucky
  ]

zero:
  ret i32 10

one:
  ret i32 11

lucky:
  ret i32 77

default:
  ret i32 -1
}
