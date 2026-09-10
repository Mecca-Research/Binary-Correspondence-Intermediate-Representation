; Teaching snapshot for:
;   opt -S -passes=mem2reg mem2reg-diamond-before.ll -o mem2reg-diamond-after.ll

target triple = "x86_64-unknown-linux-gnu"
source_filename = "mem2reg-diamond-before.ll"

define i32 @promote_diamond(i1 %cond, i32 %a, i32 %b) {
entry:
  br i1 %cond, label %then, label %else

then:
  %inc = add i32 %a, 1
  br label %merge

else:
  %dec = sub i32 %b, 1
  br label %merge

merge:
  %slot.0 = phi i32 [ %inc, %then ], [ %dec, %else ]
  %answer = mul i32 %slot.0, 2
  ret i32 %answer
}
