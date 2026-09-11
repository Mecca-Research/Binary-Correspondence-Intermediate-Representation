// The other direction: MLIR emitting C.
//
// Every other chapter here goes C -> IR. `emitc` goes back, and what it has to do about
// signedness is the interesting part: `arith.addi` is sign-agnostic and wraps, while C's
// `+` on `int32_t` is undefined on overflow. So the conversion routes through unsigned.
//
//   mlir-opt emitc-mlir-to-c.mlir \
//     --pass-pipeline='builtin.module(convert-arith-to-emitc,convert-func-to-emitc)'
//   ... | mlir-translate --mlir-to-cpp
//
// 20-clang-frontend/07-emitc-the-other-direction.md reads the output.

func.func @add(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  return %c : i32
}
