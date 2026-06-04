# SEMVER

LLVM training examples target opaque-pointer LLVM IR accepted by LLVM 15 or
newer. New standalone `.ll` examples must be added to
`llvm-training/examples/README.md` and pass `tools/verify-examples.sh`.

MLIR examples are illustrative unless a future dialect implementation makes them
machine-checkable. Final LLVM IR snapshots produced from MLIR walkthroughs, such
as `14-mlir-bridge/examples/bcir-final.ll`, are checked as normal `.ll` files.
