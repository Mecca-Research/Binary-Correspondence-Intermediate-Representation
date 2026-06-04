// MLIR LLVM dialect sketch corresponding to a direct LLVM call.
module {
  llvm.func @callee(i32) -> i32
  llvm.func @caller(%arg0: i32) -> i32 {
    %0 = llvm.call @callee(%arg0) : (i32) -> i32
    llvm.return %0 : i32
  }
}
