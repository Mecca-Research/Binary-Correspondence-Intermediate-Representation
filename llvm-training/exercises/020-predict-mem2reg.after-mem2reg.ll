; Teaching-sized expected shape after promoting %slot with mem2reg.
; Exact value names may differ across LLVM versions.
source_filename = "020-predict-mem2reg.input.ll"

define i32 @choose_and_add(i1 %cond, i32 %a, i32 %b) {
entry:
  br i1 %cond, label %then, label %else

then:
  %t = add i32 %a, 1
  br label %merge

else:
  %e = add i32 %b, 2
  br label %merge

merge:
  %slot.0 = phi i32 [ %t, %then ], [ %e, %else ]
  %r = add i32 %slot.0, 10
  ret i32 %r
}
