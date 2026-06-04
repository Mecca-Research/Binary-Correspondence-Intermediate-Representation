; SSA renaming sketch: each assignment gets a fresh value name.
source_filename = "ssa-renaming-before.c"

define i32 @ssa_renaming_before(i32 %x) {
entry:
  %x1 = add i32 %x, 1
  %x2 = mul i32 %x1, 2
  %x3 = sub i32 %x2, %x1
  ret i32 %x3
}
