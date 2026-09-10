// Input for the standalone training dialect skeleton in this directory.
//
// The `training` dialect is unregistered in this repository -- nothing here
// builds it -- so this file is a syntax-only sketch. Validate it with:
//   mlir-opt --allow-unregistered-dialect training/llvm/18-mlir-lowering-to-llvm/examples/production-dialect/scale.mlir
//
// With the skeleton built (see README.md in this directory), the same file is
// the pass's own test:
//   training-opt --training-to-llvm scale.mlir
//
// Expected after conversion: `training.scale` becomes `llvm.mul` against an
// `llvm.mlir.constant`, while `training.lane_cast` and `training.emit` survive
// unchanged -- the pass is a PARTIAL conversion and says so by declaring those
// two ops legal.

module attributes {training.example = "scale-then-emit"} {
  func.func @scaled(%arg0: i32) -> i32 {
    %0 = "training.scale"(%arg0) {factor = 3 : i32} : (i32) -> i32
    return %0 : i32
  }

  func.func @scaled_rounded(%arg0: i32) -> i32 {
    %0 = "training.scale"(%arg0) {factor = 2 : i32, rounding = 0 : i32} : (i32) -> i32
    return %0 : i32
  }

  func.func @handed_off(%arg0: i32) {
    %0 = "training.scale"(%arg0) {factor = 4 : i32} : (i32) -> i32
    %1 = "training.lane_cast"(%0) : (i32) -> !training.lane<width = 4>
    "training.emit"(%1) : (!training.lane<width = 4>) -> ()
    return
  }
}
