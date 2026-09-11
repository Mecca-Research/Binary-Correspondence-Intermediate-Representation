// What a quantized type actually is, once the type disappears.
//
//   mlir-opt quant-cast-arithmetic.mlir --pass-pipeline='builtin.module(func.func(lower-quant-ops))'
//
// !quant.uniform<i8:f32, 0.02:-1> is an i8 that MEANS (code - (-1)) * 0.02. qcast and
// dcast are where that meaning is spent: the type carries it until the pass writes it out
// as division, addition and a float-to-int conversion. Note which conversion: arith.fptosi,
// whose out-of-range behaviour ../../23-version-movement/03-conversions-and-shifts.md
// teaches is poison rather than saturation.

!qty = !quant.uniform<i8:f32, 0.02:-1>

func.func @quantize(%x: tensor<4xf32>) -> tensor<4x!qty> {
  %q = quant.qcast %x : tensor<4xf32> to tensor<4x!qty>
  return %q : tensor<4x!qty>
}

func.func @dequantize(%q: tensor<4x!qty>) -> tensor<4xf32> {
  %d = quant.dcast %q : tensor<4x!qty> to tensor<4xf32>
  return %d : tensor<4xf32>
}
