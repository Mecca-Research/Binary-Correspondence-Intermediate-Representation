; Before SROA: aggregate alloca with field accesses.
source_filename = "sroa-struct-before.c"

%Pair = type { i32, i32 }

define i32 @sroa_struct_before(i32 %a, i32 %b) {
entry:
  %p = alloca %Pair, align 4
  %pa = getelementptr inbounds %Pair, ptr %p, i32 0, i32 0
  %pb = getelementptr inbounds %Pair, ptr %p, i32 0, i32 1
  store i32 %a, ptr %pa, align 4
  store i32 %b, ptr %pb, align 4
  %la = load i32, ptr %pa, align 4
  %lb = load i32, ptr %pb, align 4
  %sum = add i32 %la, %lb
  ret i32 %sum
}
