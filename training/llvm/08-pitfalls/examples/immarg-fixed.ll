; Fixed form: pass a literal immarg to the intrinsic.
source_filename = "immarg-fixed.c"

declare i32 @llvm.ctlz.i32(i32, i1 immarg)

define i32 @fixed_immarg(i32 %x) {
entry:
  %v = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  ret i32 %v
}
