// Rungs two and three: a named linalg operation becomes a linalg.generic, and the generic
// becomes an explicit loop nest -- but only after bufferization.
//
//   mlir-opt linalg-to-loops.mlir --pass-pipeline='builtin.module(
//     func.func(linalg-generalize-named-ops),
//     one-shot-bufferize{bufferize-function-boundaries},
//     func.func(convert-linalg-to-loops))'
//
// The order is not stylistic. convert-linalg-to-loops rewrites operations on BUFFERS;
// run it against tensors and it silently does nothing, which is what
// linalg-to-loops-needs-buffers.mlir exists to demonstrate.

func.func @mm(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>, %out: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  %0 = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x4xf32>)
                     outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
