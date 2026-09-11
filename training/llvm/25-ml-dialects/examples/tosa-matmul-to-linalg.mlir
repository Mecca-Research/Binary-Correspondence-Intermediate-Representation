// Rung one of the ML abstraction ladder: a named framework operation becomes a named
// linear-algebra operation.
//
//   mlir-opt tosa-matmul-to-linalg.mlir --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named))'
//
// tosa.matmul carries its semantics in its NAME -- shapes, batching and the accumulation
// order are all in the operation's definition rather than in the IR. One pass turns it
// into linalg.batch_matmul plus the two things the name was hiding: a tensor.empty for
// the destination and a linalg.fill that zeroes the accumulator.

func.func @mm(%a: tensor<1x2x3xf32>, %b: tensor<1x3x4xf32>) -> tensor<1x2x4xf32> {
  %azp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %bzp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.matmul %a, %b, %azp, %bzp
    : (tensor<1x2x3xf32>, tensor<1x3x4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2x4xf32>
  return %0 : tensor<1x2x4xf32>
}
