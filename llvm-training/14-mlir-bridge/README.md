# 14 — MLIR Bridge

This section teaches how BCIR-shaped domain facts can remain explicit in MLIR
before lowering into LLVM IR. Use it when a compiler pipeline needs more than a
flat LLVM module: graph operations, schema symbols, layout hints, structured
loops, verifier-owned invariants, or staged type conversion.

## Reading path

1. [`01-what-is-mlir.md`](01-what-is-mlir.md) — modules, operations, regions,
   blocks, attributes, and types.
2. [`02-dialects-and-operations.md`](02-dialects-and-operations.md) — dialect
   design and operation anatomy.
3. [`03-lowering-to-llvm-dialect.md`](03-lowering-to-llvm-dialect.md) — lowering
   pipelines and the LLVM dialect.
4. [`04-bcir-as-custom-dialect.md`](04-bcir-as-custom-dialect.md) — BCIR
   Vertex-Edge-Attribute, HAM, register-binding, and Mixed Stride sketches.
5. [`05-type-conversion-and-materialization.md`](05-type-conversion-and-materialization.md)
   — convert BCIR dialect types into LLVM-compatible descriptors and pointers.
6. [`06-conversion-patterns.md`](06-conversion-patterns.md) — rewrite custom
   operations into `scf`, `cf`, `memref`, `func`, and `llvm` dialect operations.
7. [`07-pass-pipeline-and-diagnostics.md`](07-pass-pipeline-and-diagnostics.md)
   — build, inspect, and debug a staged MLIR-to-LLVM pipeline.
8. [`08-end-to-end-bcir-lowering.md`](08-end-to-end-bcir-lowering.md) — follow a
   small BCIR graph fragment from dialect sketch to final LLVM IR.

## Examples

- [`examples/bcir-dialect-sketch.mlir`](examples/bcir-dialect-sketch.mlir) —
  high-level BCIR dialect sketch.
- [`examples/lowered-llvm-dialect.mlir`](examples/lowered-llvm-dialect.mlir) —
  LLVM dialect shape before textual LLVM IR translation.
- [`examples/bcir-type-conversion.mlir`](examples/bcir-type-conversion.mlir) —
  type conversion from BCIR handles to descriptor pointers and integer IDs.
- [`examples/bcir-conversion-pipeline.mlir`](examples/bcir-conversion-pipeline.mlir)
  — staged rewrite checkpoints for a graph-edge load.
- [`examples/bcir-final.ll`](examples/bcir-final.ll) — final standalone LLVM IR
  emitted after the illustrative lowering path; this file is verified by the
  standalone LLVM example checker.
