; Teaching snapshot for:
;   opt -S -passes=simplifycfg simplifycfg-select-before.ll -o simplifycfg-select-after.ll

target triple = "x86_64-unknown-linux-gnu"
source_filename = "simplifycfg-select-before.ll"

define i32 @branch_to_select(i1 %cond, i32 %a, i32 %b) {
entry:
  %tv = add i32 %a, 10
  %fv = add i32 %b, 20
  %phi = select i1 %cond, i32 %tv, i32 %fv
  ret i32 %phi
}
