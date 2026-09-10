// Deterministic stock-dialect fixture for exercising MLIR type conversion.
// The declared training registry pipeline expands memref metadata and converts
// the memref/function boundary completely to the LLVM dialect.
module {
  func.func @load_first(%arg0: memref<?xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %value = memref.load %arg0[%c0] : memref<?xi32>
    return %value : i32
  }
}
