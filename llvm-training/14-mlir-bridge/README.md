# MLIR Bridge

This chapter explains how a structured frontend or domain IR can remain in MLIR
long enough to validate domain facts, then lower deliberately to LLVM dialect and
finally textual LLVM IR.

## Lessons

1. [`01-what-is-mlir.md`](01-what-is-mlir.md) — modules, operations, regions,
   blocks, attributes, and types.
2. [`02-dialects-and-operations.md`](02-dialects-and-operations.md) — dialect
   boundaries and operation anatomy.
3. [`03-lowering-to-llvm-dialect.md`](03-lowering-to-llvm-dialect.md) — type
   conversion, conversion targets, LLVM dialect, and `.ll` output.
4. [`04-bcir-as-custom-dialect.md`](04-bcir-as-custom-dialect.md) — how BCIR
   concepts such as vertices, HAM hints, register binding, and Mixed Stride
   graphs can be represented as dialect operations.
5. [`05-vertex-graph-lowering.md`](05-vertex-graph-lowering.md) — a complete
   vertex/edge graph lowering walkthrough that tracks vertex IDs, edge lists,
   register bindings, and metadata hints across source MLIR, LLVM-dialect MLIR,
   and LLVM IR.

## Examples

- [`examples/arith-to-llvm.mlir`](examples/arith-to-llvm.mlir) and
  [`examples/arith-to-llvm-lowered.ll`](examples/arith-to-llvm-lowered.ll) — tiny
  arith/func lowering shape.
- [`examples/bcir-dialect-sketch.mlir`](examples/bcir-dialect-sketch.mlir) —
  source-level BCIR dialect sketch.
- [`examples/lowered-llvm-dialect.mlir`](examples/lowered-llvm-dialect.mlir) —
  illustrative BCIR-to-LLVM-dialect shape.
- [`examples/bcir-vertex-graph.mlir`](examples/bcir-vertex-graph.mlir),
  [`examples/bcir-vertex-graph-lowered-llvm-dialect.mlir`](examples/bcir-vertex-graph-lowered-llvm-dialect.mlir),
  and [`examples/bcir-vertex-graph-lowered.ll`](examples/bcir-vertex-graph-lowered.ll)
  — before/after graph example for lesson 05.

## Cross-links

Use the MLIR bridge together with the BCIR mapping guide when reviewing graph
lowering decisions:

- [`../bcir-mapping/01-vertex-edge-attribute.md`](../bcir-mapping/01-vertex-edge-attribute.md)
  for vertex, edge, and attribute storage choices.
- [`../bcir-mapping/02-register-binding.md`](../bcir-mapping/02-register-binding.md)
  for explicit resource/register table lowering.
- [`../bcir-mapping/10-metadata-and-diagnostics.md`](../bcir-mapping/10-metadata-and-diagnostics.md)
  for metadata that can aid diagnostics without becoming semantics.

## Checks

From the repository root, run:

```sh
./llvm-training/tools/verify-mlir-examples.sh
./llvm-training/tools/verify-examples.sh
./llvm-training/tools/verify-exercises.sh
```
