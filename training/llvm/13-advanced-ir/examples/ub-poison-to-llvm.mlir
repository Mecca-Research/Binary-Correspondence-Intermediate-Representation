// MLIR's poison, and the same value seen from two IRs.
//
// The `ub` dialect is two operations wide -- `ub.poison` and `ub.unreachable` -- and it
// exists so that dialects above LLVM can name deferred undefined behaviour without
// depending on the LLVM dialect. `ub.poison` is the direct counterpart of the `poison`
// that 13-advanced-ir/05-poison-undef-freeze.md teaches on the LLVM side.
//
//   mlir-opt ub-poison-to-llvm.mlir --pass-pipeline='builtin.module(
//     convert-ub-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,
//     reconcile-unrealized-casts)' | mlir-translate --mlir-to-llvmir
//
// @poisoned is the plain case. @contaminated is the one worth watching: poison flows
// into an ordinary add, and the add's other operand stops mattering.

func.func @poisoned() -> i32 {
  %p = ub.poison : i32
  return %p : i32
}

func.func @contaminated(%x: i32) -> i32 {
  %p = ub.poison : i32
  %s = arith.addi %x, %p : i32
  return %s : i32
}
