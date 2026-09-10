; Run:
;   opt -S -passes=simplifycfg simplifycfg-before.ll -o simplifycfg-after.ll
;
; The branch condition is a constant, so one edge and its block can be removed.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "simplifycfg-before.ll"

define i32 @fold_known_branch(i32 %x) {
entry:
  br i1 true, label %then, label %else

then:
  %kept = add i32 %x, 1
  ret i32 %kept

else:
  %unreachable_value = sub i32 %x, 1
  ret i32 %unreachable_value
}
