// A type whose WIDTH is a target decision, not a property of the IR.
//
// `index` is MLIR's target-sized integer -- C's `size_t`, roughly. LLVM IR has no such
// type: every integer there is a fixed width. So the type converter has to pick one, and
// which one it picks is a lowering option rather than anything written here.
//
// Lower it twice and the difference is the whole point:
//
//   mlir-opt index-width-to-llvm.mlir --pass-pipeline='builtin.module(
//     convert-index-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)'
//       -> llvm.func @stride(%arg0: i64, %arg1: i64) -> i64
//
//   ... same pipeline with {index-bitwidth=32} on both conversions
//       -> llvm.func @stride(%arg0: i32, %arg1: i32) -> i32
//
// 18-mlir-lowering-to-llvm/02-typeconverter-and-materialization.md reads both.

func.func @stride(%i: index, %n: index) -> index {
  %c4 = index.constant 4
  %m = index.mul %i, %c4
  %s = index.add %m, %n
  return %s : index
}
