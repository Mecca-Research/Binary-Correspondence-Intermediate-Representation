// MLIR sketch: arith + func dialect before lowering to the LLVM dialect.
module {
  func.func @add_one(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %0 = arith.addi %arg0, %c1 : i32
    return %0 : i32
  }
}
