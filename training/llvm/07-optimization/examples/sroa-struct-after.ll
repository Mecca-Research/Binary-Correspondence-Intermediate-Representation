; After SROA/mem2reg: scalar fields replace the aggregate alloca.
source_filename = "sroa-struct-after.c"

define i32 @sroa_struct_after(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
