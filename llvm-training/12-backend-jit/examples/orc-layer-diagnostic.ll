; ORC layer diagnostics often start with unresolved or duplicate symbol sketches.
source_filename = "orc-layer-diagnostic.c"

declare i32 @missing_runtime_symbol(i32)

define i32 @needs_runtime_symbol(i32 %x) {
entry:
  %y = call i32 @missing_runtime_symbol(i32 %x)
  ret i32 %y
}
