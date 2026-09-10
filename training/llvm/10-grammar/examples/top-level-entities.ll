; Grammar focus: source_filename, target triple, global, declare, define.
source_filename = "top-level-entities.ll"
target triple = "x86_64-unknown-linux-gnu"

@g = global i32 0, align 4

declare void @sink(i32)

define void @top_level_entities() {
entry:
  %v = load i32, ptr @g, align 4
  call void @sink(i32 %v)
  ret void
}
