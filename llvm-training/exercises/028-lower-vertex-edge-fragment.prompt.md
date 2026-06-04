# Exercise 028: Lower a BCIR vertex/edge fragment

## Task family

This is a **BCIR lowering** exercise. Lower a graph fragment with vertex values,
edge destination indices, and edge weights to explicit LLVM IR array accesses.

## Source sketch

```text
for edge e in outgoing(vertex):
  dst = edge.dst[e]
  out[dst] += vertex.value[vertex] * edge.weight[e]
```

## Required LLVM constructs

Write a standalone module containing:

- `define void @lower_vertex_edge_fragment(ptr %vertex_values, ptr %edge_dst, ptr %edge_weight, ptr %out, i64 %vertex, i64 %edge)`.
- `getelementptr` loads for the vertex value, edge destination, edge weight, and
  output slot.
- A floating-point multiply and accumulation store.
- Clear alignment on all loads and stores.

## Expected observation

The module should assemble and make all BCIR graph accesses visible as ordinary
LLVM pointer arithmetic and scalar operations.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/028-lower-vertex-edge-fragment.solution.ll
```
