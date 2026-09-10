; LLVM IR equivalent of a tiny arith/func MLIR fragment.
source_filename = "arith-to-llvm-lowered.mlir"

define i32 @add_one(i32 %arg0) {
entry:
  %0 = add i32 %arg0, 1
  ret i32 %0
}
