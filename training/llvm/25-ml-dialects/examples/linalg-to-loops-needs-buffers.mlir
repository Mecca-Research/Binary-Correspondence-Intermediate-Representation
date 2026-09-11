// A pass that runs, reports success, and changes nothing.
//
//   mlir-opt linalg-to-loops-needs-buffers.mlir \
//     --pass-pipeline='builtin.module(func.func(linalg-generalize-named-ops,convert-linalg-to-loops))'
//
// convert-linalg-to-loops lowers linalg operating on MEMREFS. This module is on tensors,
// so the pass matches nothing, exits zero, and leaves the linalg.generic exactly where it
// was. The registry declares that survivor as the expected outcome, so the claim in
// ../01-the-abstraction-ladder.md is checked rather than asserted.

func.func @mm(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>, %out: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  %0 = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x4xf32>)
                     outs(%out : tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
