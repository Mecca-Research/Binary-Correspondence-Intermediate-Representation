; External symbols become relocation records for object emission or JIT linking.
source_filename = "relocation-symbols.c"
target triple = "x86_64-unknown-linux-gnu"

@extern_data = external global i32

declare i32 @extern_func(i32)

define i32 @relocation_user(i32 %x) {
entry:
  %g = load i32, ptr @extern_data, align 4
  %sum = add i32 %g, %x
  %call = call i32 @extern_func(i32 %sum)
  ret i32 %call
}
