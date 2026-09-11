// The step between value semantics and buffer semantics.
//
// `tensor<4xf32>` is a VALUE: it has no address, and `linalg.generic` below returns a
// new one rather than modifying anything. `memref<4xf32>` is a BUFFER: it has an
// address, and an operation writing to it returns nothing at all.
//
// Bufferization is the pass that crosses that line. Run it two ways and compare:
//
//   mlir-opt tensor-to-buffer.mlir \
//     --pass-pipeline='builtin.module(one-shot-bufferize)'
//       -> the signature keeps tensors; bufferization.to_buffer and
//          bufferization.to_tensor appear at the boundary
//
//   mlir-opt tensor-to-buffer.mlir \
//     --pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries=true})'
//       -> the signature converts too, and both boundary ops disappear
//
// 18-mlir-lowering-to-llvm/10-tensors-buffers-and-the-step-between.md reads the output.

func.func @scale(%t: tensor<4xf32>, %k: f32) -> tensor<4xf32> {
  // tensor.empty is a shape request, NOT an allocation: it names the shape of the
  // result so the destination-passing style below has an `outs` to write into.
  %e = tensor.empty() : tensor<4xf32>

  %r = linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
         iterator_types = ["parallel"]}
        ins(%t : tensor<4xf32>) outs(%e : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %m = arith.mulf %in, %k : f32
      linalg.yield %m : f32
  } -> tensor<4xf32>

  return %r : tensor<4xf32>
}
