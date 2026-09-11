// This fixture exists to be INCOMPLETE, and that is its whole claim.
//
// --convert-shape-to-std has no pattern for shape.num_elements, because the dialect
// expresses a reduction over extents with shape.reduce and expects
// --shape-to-shape-lowering to have rewritten it first. Run the converter alone and
// exactly one operation survives:
//
//   mlir-opt shape-one-pass-is-not-enough.mlir --convert-shape-to-std
//
// The registry declares that surviving operation as an expected failure, so the
// sentence in ../11-shapes-as-values.md that says a conversion leaving one operation
// behind means a missing PASS rather than a missing PATTERN is checked rather than
// asserted. shape-broadcast-to-std.mlir is the same computation with both passes.

func.func @count(%a: tensor<?x?xf32>) -> index {
  %s = shape.shape_of %a : tensor<?x?xf32> -> tensor<2xindex>
  %n = shape.num_elements %s : tensor<2xindex> -> index
  return %n : index
}
