// Two dialects that model things LLVM IR has no native form for, and what the type
// converter does about it.
//
// `complex<f32>` is a TYPE MLIR has and LLVM IR does not. `math.sqrt` and `math.exp` are
// OPERATIONS LLVM IR has only as intrinsics. Both survive lowering, but by different
// routes: one becomes a struct, the other becomes a call.
//
//   mlir-opt complex-and-math-to-llvm.mlir --pass-pipeline='builtin.module(
//     convert-complex-to-llvm,convert-math-to-llvm,convert-func-to-llvm,
//     reconcile-unrealized-casts)' | mlir-translate --mlir-to-llvmir
//
// 18-mlir-lowering-to-llvm/02-typeconverter-and-materialization.md reads the output.

func.func @cmul(%a: complex<f32>, %b: complex<f32>) -> complex<f32> {
  %r = complex.mul %a, %b : complex<f32>
  return %r : complex<f32>
}

func.func @roots(%x: f32) -> f32 {
  %s = math.sqrt %x : f32
  %e = math.exp %s : f32
  return %e : f32
}
