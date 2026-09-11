// What a shape computation costs once it stops being a shape computation.
//
// Four operations of the `shape` dialect describe broadcasting two dynamically shaped
// tensors and asking how many elements the result has. None of it is magic; it is
// ordinary index arithmetic that has not been written out yet.
//
//   mlir-opt shape-broadcast-to-std.mlir \
//     --pass-pipeline='builtin.module(func.func(shape-to-shape-lowering),convert-shape-to-std)'
//
// leaves no `shape` operation behind. Run the two passes in the other order and
// shape.reduce, shape.mul and shape.yield are stranded -- see
// ../11-shapes-as-values.md.

func.func @broadcast_extents(%a: tensor<?x?xf32>, %b: tensor<?xf32>) -> index {
  %sa = shape.shape_of %a : tensor<?x?xf32> -> tensor<2xindex>
  %sb = shape.shape_of %b : tensor<?xf32> -> tensor<1xindex>
  %bc = shape.broadcast %sa, %sb : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
  %n = shape.num_elements %bc : tensor<2xindex> -> index
  return %n : index
}
