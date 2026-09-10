; Intrinsic declarations encode overloaded types and immediate-argument constraints.
source_filename = "intrinsic-constraints.c"

declare i32 @llvm.ctpop.i32(i32)
declare i32 @llvm.ctlz.i32(i32, i1 immarg)

define i32 @intrinsic_constraints(i32 %x) {
entry:
  %bits = call i32 @llvm.ctpop.i32(i32 %x)
  %zeros = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  %sum = add i32 %bits, %zeros
  ret i32 %sum
}
