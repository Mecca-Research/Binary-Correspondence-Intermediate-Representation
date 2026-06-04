; Run:
;   opt -S -passes=simplifycfg simplifycfg-select-before.ll -o simplifycfg-select-after.ll
;
; Both arms of the diamond branch directly to one merge block. simplifycfg can
; replace the branch-only diamond with a select expression.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "simplifycfg-select-before.ll"

define i32 @branch_to_select(i1 %cond, i32 %a, i32 %b) {
entry:
  br i1 %cond, label %then, label %else

then:
  %tv = add i32 %a, 10
  br label %merge

else:
  %fv = add i32 %b, 20
  br label %merge

merge:
  %phi = phi i32 [ %tv, %then ], [ %fv, %else ]
  ret i32 %phi
}
