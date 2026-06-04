# 14.5 — Type Conversion and Materialization

A BCIR dialect can carry rich types such as `!bcir.vertex`, `!bcir.edge`,
`!bcir.attr`, `!bcir.graph<rank = 3>`, or `!bcir.ham_hint`. LLVM IR cannot keep
those high-level types directly. The MLIR bridge therefore needs a clear type
conversion contract before operation rewrites begin.

## Conversion table

| BCIR dialect type | Lowering target | Runtime meaning |
|---|---|---|
| `!bcir.vertex` | `i64` or `!llvm.ptr` | Stable vertex ID or pointer to a vertex descriptor. |
| `!bcir.edge` | `{ i64, i64, i32 }` descriptor or `!llvm.ptr` | Source ID, destination ID, edge kind, or runtime edge handle. |
| `!bcir.attr<T>` | `T`, descriptor pointer, or call result | Attribute payload loaded from schema storage. |
| `!bcir.graph<...>` | pointer plus rank/stride fields | Graph descriptor with layout and bounds. |
| `!bcir.ham_hint` | metadata, prefetch call, or erased unit value | Optimization guidance only, never required semantics. |

Choose one representation per ABI boundary. Mixing integer IDs in one module and
pointers in another creates the same kind of schema drift that LLVM IR struct
layout mismatches create.

## Materialization rules

MLIR conversion patterns sometimes need a value of the new type before all users
are rewritten. That is materialization. For BCIR, materialization should be
explicit and auditable:

```mlir
// Source dialect value.
%v = "bcir.vertex.lookup"(%id) : (i64) -> !bcir.vertex

// Target materialization shape.
%handle = "bcir.lower.vertex_handle"(%id) : (i64) -> !llvm.ptr
```

A materialization op is temporary. It lets conversion proceed while preserving a
searchable marker for review. The final lowering should remove it by replacing
it with GEP arithmetic, a runtime call, or a verified descriptor load.

## What must be verified

- IDs and pointers must not be interchanged without an explicit lookup or cast
  operation.
- Graph descriptors must define ownership, mutability, alignment, and lifetime.
- Mixed Stride fields must document element-vs-byte units and signedness.
- HAM hints must remain droppable unless their operation name states otherwise.
- Register-binding requests must state whether they are optional preferences or
  hard ABI constraints.

## Review checklist

When reviewing a type converter, search for these failure modes:

1. A custom BCIR type reaches the LLVM dialect boundary unchanged.
2. An `index` value lowers to a fixed integer width without documenting the
   pointer-size assumption.
3. Descriptor pointers lose `readonly`, `noalias`, `nonnull`, or alignment facts
   that later LLVM passes need.
4. Runtime calls are introduced without matching declarations, calling
   conventions, or memory-effect annotations.
5. A hint-only operation lowers into a required control-flow edge.

See [`examples/bcir-type-conversion.mlir`](examples/bcir-type-conversion.mlir)
for a compact before/after sketch.
