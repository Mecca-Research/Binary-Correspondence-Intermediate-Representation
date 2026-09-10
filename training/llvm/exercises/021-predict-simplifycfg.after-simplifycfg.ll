; Teaching-sized expected shape after SimplifyCFG folds the diamond.
; Exact value names may differ across LLVM versions.
source_filename = "021-predict-simplifycfg.input.ll"

define i32 @select_like_diamond(i1 %cond, i32 %x, i32 %y) {
entry:
  %tx = add i32 %x, 1
  %ey = add i32 %y, 1
  %r = select i1 %cond, i32 %tx, i32 %ey
  ret i32 %r
}
