// A measurement that cannot measure anything.
//
//   mlir-opt quant-round-trip-folds-away.mlir --canonicalize
//
// Quantizing and dequantizing is lossy by construction -- that loss is the entire reason
// quantization is a subject. In MLIR 23.1.1 the canonicalizer folds dcast(qcast(%x)) to
// %x, so this function becomes the identity and an experiment that measures error this
// way measures exactly zero. The registry pins that fold: if a later MLIR stops doing it,
// this fixture goes red and ../02-quantization-and-bcir.md gets rewritten rather than
// continuing to warn about a toolchain nobody has.

!qty = !quant.uniform<i8:f32, 0.02:-1>

func.func @round_trip(%x: tensor<4xf32>) -> tensor<4xf32> {
  %q = quant.qcast %x : tensor<4xf32> to tensor<4x!qty>
  %d = quant.dcast %q : tensor<4x!qty> to tensor<4xf32>
  return %d : tensor<4xf32>
}
