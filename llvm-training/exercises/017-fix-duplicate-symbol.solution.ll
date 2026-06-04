source_filename = "017-fix-duplicate-symbol.solution.ll"

define i32 @adjust_up(i32 %x) {
entry:
  %y = add i32 %x, 1
  ret i32 %y
}

define i32 @adjust_down(i32 %x) {
entry:
  %y = sub i32 %x, 1
  ret i32 %y
}
