# 14.6 — Conversion Patterns

Conversion patterns are the executable rules that replace operations from one
set of dialects with operations from another. For BCIR, the highest-value
patterns are graph lookup, edge traversal, attribute load, HAM hint lowering,
and register-binding normalization.

## Pattern inventory

| Source op | Intermediate target | Final LLVM shape |
|---|---|---|
| `bcir.vertex.lookup` | `func.call @bcir_vertex_lookup` or descriptor GEP | `call`, `load`, `getelementptr` |
| `bcir.edge.load` | `memref.load`, `arith`, `scf.if` | descriptor load plus bounds branch |
| `bcir.attribute.read` | typed load or runtime accessor call | `load`, `call`, `!tbaa` metadata |
| `bcir.ham_hint` | `memref.prefetch`/runtime scheduling op | `llvm.prefetch` or erased op |
| `bcir.bind_register` | target-lowering marker | inline asm/ABI hook or erased optional hint |

## Keep legality explicit

A partial conversion should define which dialects are legal at each checkpoint:

1. **BCIR canonicalization**: `bcir`, `func`, `arith`, `scf` are legal.
2. **Descriptor lowering**: selected `bcir` ops become illegal; `memref` and
   runtime `func.call` ops are legal.
3. **LLVM-compatible lowering**: `memref`, `scf`, and `cf` lower away.
4. **LLVM dialect**: only `builtin`, `llvm`, and accepted metadata-like
   attributes remain.

If the target says all unknown operations are legal, a misspelled or unlowered
BCIR op can silently pass through until translation fails much later.

## Example rewrite shape

```mlir
// Before: high-level edge payload read.
%edge = "bcir.edge.lookup"(%graph, %src, %dst) : (!bcir.graph, i64, i64) -> !bcir.edge
%w = "bcir.attribute.read"(%edge) {name = "weight"} : (!bcir.edge) -> i32

// After: descriptor pointer plus field load.
%desc = func.call @bcir_edge_lookup(%graph_ptr, %src, %dst) : (!llvm.ptr, i64, i64) -> !llvm.ptr
%field = llvm.getelementptr %desc[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64, i32)>
%w = llvm.load %field : !llvm.ptr -> i32
```

The rewrite exposes two review questions: does `@bcir_edge_lookup` return a
nonnull pointer on every success path, and is field index `2` stable across all
runtime modules?

## Diagnostics agents should emit

When a conversion fails, report the source operation, expected legality stage,
missing type conversion, and the first surviving illegal op. A useful diagnostic
is concrete:

```text
bcir-to-llvm stage 2 failed: bcir.attribute.read survived descriptor lowering;
expected replacement with @bcir_attr_i32_read or llvm.load of field "weight".
```

That form points an implementer to the rewrite pattern instead of merely saying
"conversion failed".

See [`examples/bcir-conversion-pipeline.mlir`](examples/bcir-conversion-pipeline.mlir).
