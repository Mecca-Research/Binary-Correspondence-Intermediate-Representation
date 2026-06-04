# Exercise 039: Lower a BCIR MLIR fragment end to end

## Task family

This is an **MLIR bridge review** exercise. Trace a BCIR source-dialect sketch
through canonical BCIR, LLVM dialect, and final textual LLVM IR carriers.

## Input sketch

```mlir
%root = "bcir.vertex"() {id = 0 : i64, space = "claim"}
  : () -> !bcir.vertex<space = "claim", id_bits = 64>
%child = "bcir.vertex.lookup"(%root) {edge_kind = "contains", ordinal = 0 : i32}
  : (!bcir.vertex<space = "claim", id_bits = 64>) -> !bcir.vertex<space = "blob", id_bits = 64>
%weight = "bcir.attribute"(%child) {name = "weight", storage = "runtime"}
  : (!bcir.vertex<space = "blob", id_bits = 64>) -> f32
```

Assume schema lookup maps `contains` to edge key `3` and `weight` to attribute
key `7`. The runtime-backed ABI provides:

```llvm
declare ptr @bcir_lookup_child(ptr, i64, i32)
declare float @bcir_get_edge_attr_f32(ptr, i64, i32)
```

## Required review points

Describe the canonical BCIR facts, the LLVM-dialect call shape, and the final
textual LLVM IR carriers. State what happens to the custom vertex type.

## Verification command

Markdown review exercise; no MLIR tool is required. The checked-in reference
answer is:

```sh
cat llvm-training/exercises/039-lower-bcir-mlir-end-to-end.solution.md
```
