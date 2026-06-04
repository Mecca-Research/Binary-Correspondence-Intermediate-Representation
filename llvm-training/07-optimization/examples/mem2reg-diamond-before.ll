; Run:
;   opt -S -passes=mem2reg mem2reg-diamond-before.ll -o mem2reg-diamond-after.ll
;
; A promotable alloca receives values from two branches. mem2reg replaces the
; stack slot with an SSA phi in the merge block.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "mem2reg-diamond-before.ll"

define i32 @promote_diamond(i1 %cond, i32 %a, i32 %b) {
entry:
  %slot = alloca i32, align 4
  br i1 %cond, label %then, label %else

then:
  %inc = add i32 %a, 1
  store i32 %inc, ptr %slot, align 4
  br label %merge

else:
  %dec = sub i32 %b, 1
  store i32 %dec, ptr %slot, align 4
  br label %merge

merge:
  %loaded = load i32, ptr %slot, align 4
  %answer = mul i32 %loaded, 2
  ret i32 %answer
}
