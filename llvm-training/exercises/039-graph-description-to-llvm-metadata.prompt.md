# Exercise 039: Lower a graph description to LLVM metadata

## Task family

This is an **advanced graph-to-LLVM metadata** exercise. Encode a small graph
schema as LLVM named metadata, then attach graph facts to ordinary scalar IR.

## Source sketch

```text
graph pagerank_step {
  vertices: 1024
  edges: 4096
  vertex attribute rank: f32
  edge attributes src: i32, dst: i32, weight: f32
}

out[dst[edge]] += rank[src[edge]] * weight[edge]
```

## Required LLVM constructs

Write a standalone module containing:

- `define void @apply_graph_edge(ptr %rank, ptr %edge_src, ptr %edge_dst, ptr %edge_weight, ptr %out, i64 %edge)`.
- Scalar loads for `src`, `dst`, `rank[src]`, `weight[edge]`, and `out[dst]`.
- A multiply, add, and store for the edge contribution.
- Instruction metadata attachments identifying graph, vertex attribute, edge
  attributes, and update semantics.
- Named metadata that catalogs the graph and attribute descriptions.

## Expected observation

The module should assemble. The graph description should be visible only through
metadata; the executable computation should remain normal LLVM IR.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/039-graph-description-to-llvm-metadata.solution.ll
```
