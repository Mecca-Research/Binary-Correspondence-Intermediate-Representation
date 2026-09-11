// The same linalg.generic as the dense ladder, and a completely different loop nest.
//
//   mlir-opt sparse-matvec-to-loops.mlir --sparse-reinterpret-map --sparsification
//
// Nothing below distinguishes this matvec from a dense one except the TYPE of %A, which
// carries a #sparse_tensor.encoding saying the second level is `compressed`. The
// indexing_maps, the iterator_types and the body are what 01-the-abstraction-ladder.md
// already showed. From that one attribute the compiler derives CSR iteration: an outer
// loop over rows, and an inner loop whose bounds are read out of the positions array so
// it visits only the entries that are stored.

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#matvec = {
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,
    affine_map<(i, j) -> (j)>,
    affine_map<(i, j) -> (i)>
  ],
  iterator_types = ["parallel", "reduction"]
}

func.func @matvec(%A: tensor<8x8xf32, #CSR>, %x: tensor<8xf32>, %y: tensor<8xf32>)
    -> tensor<8xf32> {
  %0 = linalg.generic #matvec
    ins(%A, %x : tensor<8x8xf32, #CSR>, tensor<8xf32>)
    outs(%y : tensor<8xf32>) {
    ^bb0(%a: f32, %xv: f32, %yv: f32):
      %m = arith.mulf %a, %xv : f32
      %s = arith.addf %yv, %m : f32
      linalg.yield %s : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
