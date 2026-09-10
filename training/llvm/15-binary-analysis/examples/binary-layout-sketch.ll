; Globals and functions provide a compact input for object layout experiments.
source_filename = "binary-layout-sketch.c"
target triple = "x86_64-unknown-linux-gnu"

@hot_table = constant [2 x i32] [i32 3, i32 5], align 16
@cold_error_code = global i32 -1, align 4

define i32 @hot_function(i32 %x) {
entry:
  %idx = and i32 %x, 1
  %p = getelementptr inbounds [2 x i32], ptr @hot_table, i32 0, i32 %idx
  %v = load i32, ptr %p, align 4
  ret i32 %v
}

define i32 @cold_function() {
entry:
  %v = load i32, ptr @cold_error_code, align 4
  ret i32 %v
}
