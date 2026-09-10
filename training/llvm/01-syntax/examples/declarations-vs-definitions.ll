; A declaration has no body; a definition owns basic blocks and instructions.
source_filename = "declarations-vs-definitions.c"

declare i32 @external_add(i32, i32)

define i32 @uses_declaration(i32 %x) {
entry:
  %sum = call i32 @external_add(i32 %x, i32 7)
  ret i32 %sum
}
