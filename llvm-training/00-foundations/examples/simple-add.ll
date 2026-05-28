; simple-add.ll — the minimal LLVM IR function
;
; Assemble with: llvm-as simple-add.ll -o /dev/null
; Verify  with: opt -passes=verify simple-add.ll -o /dev/null

source_filename = "simple-add.ll"

define i32 @add(i32 %a, i32 %b) {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}
