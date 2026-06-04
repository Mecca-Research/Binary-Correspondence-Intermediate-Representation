; Grammar focus: named result instructions, terminators, and call syntax.
source_filename = "instruction-forms.ll"

declare i32 @callee(i32)

define i32 @instruction_forms(i32 %x) {
entry:
  %a = add nsw i32 %x, 1
  %b = call i32 @callee(i32 %a)
  ret i32 %b
}
