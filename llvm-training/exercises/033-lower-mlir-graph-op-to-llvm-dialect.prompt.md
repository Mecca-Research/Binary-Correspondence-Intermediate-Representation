# Exercise 033: Lower an MLIR graph op to the LLVM dialect

## Task family

This is an **MLIR bridge review** exercise. Describe how a hypothetical
`bcir.graph.load_vertex` operation lowers to LLVM-dialect operations.

## Source sketch

```mlir
%value = bcir.graph.load_vertex %graph[%vertex]
  { field = "rank" } : (!bcir.graph, index) -> f32
```

## Required review points

Provide a lowering outline that covers descriptor access, index conversion,
field offset computation, LLVM-dialect load formation, and metadata preservation.
Also classify whether vertex IDs, edge lists, register-binding facts, and
metadata hints survive as executable values, survive only as diagnostics, or
intentionally lower away. Use
[`../14-mlir-bridge/05-vertex-graph-lowering.md`](../14-mlir-bridge/05-vertex-graph-lowering.md)
and its vertex-graph examples as reference material.

## Verification command

Markdown review exercise; no MLIR tool is required. The checked-in reference
answer is:

```sh
cat llvm-training/exercises/033-lower-mlir-graph-op-to-llvm-dialect.solution.md
```
