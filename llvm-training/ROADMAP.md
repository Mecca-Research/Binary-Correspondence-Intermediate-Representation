# ROADMAP

The MLIR bridge now has a staged BCIR-to-LLVM path covering type conversion,
conversion patterns, pass ordering, diagnostics, and an end-to-end LLVM IR
snapshot.

Future work should stay incremental:

- add real `mlir-opt` tests only when the repository carries the corresponding
  dialect implementation;
- keep `14-mlir-bridge/examples/*.mlir` illustrative until then;
- keep final `.ll` snapshots listed in `examples/README.md` and checked by
  `tools/verify-examples.sh`.
