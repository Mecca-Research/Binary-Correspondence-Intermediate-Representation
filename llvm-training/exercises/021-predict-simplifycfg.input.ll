source_filename = "021-predict-simplifycfg.input.ll"

define i32 @select_like_diamond(i1 %cond, i32 %x, i32 %y) {
entry:
  br i1 %cond, label %then, label %else

then:
  %tx = add i32 %x, 1
  br label %merge

else:
  %ey = add i32 %y, 1
  br label %merge

merge:
  %r = phi i32 [ %tx, %then ], [ %ey, %else ]
  ret i32 %r
}
