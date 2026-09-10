source_filename = "020-predict-mem2reg.input.ll"

define i32 @choose_and_add(i1 %cond, i32 %a, i32 %b) {
entry:
  %slot = alloca i32, align 4
  store i32 %a, ptr %slot, align 4
  br i1 %cond, label %then, label %else

then:
  %t = add i32 %a, 1
  store i32 %t, ptr %slot, align 4
  br label %merge

else:
  %e = add i32 %b, 2
  store i32 %e, ptr %slot, align 4
  br label %merge

merge:
  %v = load i32, ptr %slot, align 4
  %r = add i32 %v, 10
  ret i32 %r
}
