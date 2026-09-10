; Values defined in entry dominate both arms and the merge block.
source_filename = "dominance-diamond.c"

define i32 @dominance_diamond(i1 %cond, i32 %x) {
entry:
  %base = add i32 %x, 10
  br i1 %cond, label %then, label %else

then:
  %a = mul i32 %base, 2
  br label %merge

else:
  %b = sub i32 %base, 3
  br label %merge

merge:
  %joined = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %joined
}
