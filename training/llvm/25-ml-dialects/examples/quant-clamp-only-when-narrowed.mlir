// The clamp the specification promises, and the condition under which it is emitted.
//
//   mlir-opt quant-clamp-only-when-narrowed.mlir --pass-pipeline='builtin.module(func.func(lower-quant-ops))'
//
// quant.qcast's own ODS description (QuantOps.td in the installed MLIR) specifies a
// clamp(storedValue, storageMin, storageMax) step. This module declares i8<-100:100> --
// a range NARROWER than i8's own -- and the lowering emits it as arith.maxsi/arith.minsi.
// quant-cast-arithmetic.mlir declares the full i8 range and gets no clamp at all, which
// is the pairing ../02-quantization-and-bcir.md is about.

!narrow = !quant.uniform<i8<-100:100>:f32, 0.02:-1>

func.func @narrow_quantize(%x: tensor<4xf32>) -> tensor<4x!narrow> {
  %q = quant.qcast %x : tensor<4xf32> to tensor<4x!narrow>
  return %q : tensor<4x!narrow>
}
