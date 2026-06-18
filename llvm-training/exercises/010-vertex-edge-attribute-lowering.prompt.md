# Exercise 010: Lower vertex and edge attributes with nested `getelementptr`

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


## BCIR concept being modeled

Model the BCIR pattern where graph-like vertex and edge records carry compact
attribute fields that must be lowered to concrete LLVM struct loads before a
kernel can combine them. Treat a vertex record as `{ i64 id, i32 color, i32 cost }`
and an edge record as `{ i64 src, i64 dst, i32 weight, i32 flags }`.

Write a standalone LLVM IR module that defines:

```llvm
define i32 @bcir.exercise.vertex_edge_score(ptr %vertices, ptr %edges, i64 %edge_index)
```

The function should load the selected edge, use its `src` and `dst` vertex
indices to find the corresponding vertex records, and return:

```text
src.cost + dst.cost + edge.weight
```

## Required LLVM constructs

- Named struct types for the vertex and edge records.
- `getelementptr inbounds` for indexing both arrays and struct fields.
- `load` instructions with reasonable `align` values for `i64` and `i32` fields.
- Integer `add` instructions to combine the loaded attributes.

## Expected verification command

```sh
llvm-as -disable-output llvm-training/exercises/010-vertex-edge-attribute-lowering.solution.ll
```

## Expected observation

The module assembles successfully. The learner should observe that BCIR
attribute projection lowers to a sequence of typed GEPs and scalar loads before
ordinary SSA arithmetic combines the fields.

## Optional runtime reference

Compare this with the schema-oriented field projection style in
`runtime/llvm/bcir_claim_accessors.ll`.
