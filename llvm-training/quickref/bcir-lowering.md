# Quickref: BCIR Lowering

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


## Lowering layers

1. **Semantic records**: claims, resources, schedules, blobs, batches, graph fragments, and stream packs become named structs, globals, or ABI records.
2. **Accessors/executors**: helper functions unpack fields, bind resources, dispatch runtime operations, and preserve claim/schedule invariants.
3. **Plain LLVM IR**: memory, control flow, atomics, vector operations, calls, and metadata must verify without BCIR-specific magic.

## Agent checklist

- Start from [`../bcir-mapping/README.md`](../bcir-mapping/README.md) and the matching `runtime/llvm/*.ll` schema before inventing a layout.
- Keep runtime ABI boundaries explicit: record pointer, element/access type, alignment, calling convention, and ownership expectations.
- Preserve diagnostic/provenance metadata when it helps review, but do not make core correctness depend on metadata that optimizers may drop.
- Verify standalone `.ll` examples with `llvm-as` and `opt -passes=verify` after lowering changes.

## Common shapes

| BCIR concept | LLVM IR shape |
| --- | --- |
| Vertex/edge/attribute | Structs, arrays, GEP indexing, and typed loads/stores through opaque `ptr`. |
| Register/resource binding | Resource IDs, registry records, lookup helpers, and executor context pointers. |
| Mixed-stride graph | Explicit byte/element offsets, GEPs, integer arithmetic, and alignment metadata. |
| HAM/prefetch hints | Intrinsics or metadata that guide optimization without changing semantics. |
| Runtime op | Declared/defined helper with ABI-stable argument and result types. |

## Pitfalls to re-check

- Duplicate symbols after linking multiple generated modules.
- Schema drift between `%bcir.*` struct declarations and runtime files.
- Incorrect address spaces or stale typed-pointer assumptions.
- Atomic ordering mismatches at runtime call boundaries.
- Pass-pipeline order that drops expected metadata or canonical shapes.

## Deep links

- [`../bcir-mapping/01-vertex-edge-attribute.md`](../bcir-mapping/01-vertex-edge-attribute.md)
- [`../bcir-mapping/05-runtime-abi.md`](../bcir-mapping/05-runtime-abi.md)
- [`../bcir-mapping/06-claim-lowering-pipeline.md`](../bcir-mapping/06-claim-lowering-pipeline.md)
- [`../bcir-mapping/10-metadata-and-diagnostics.md`](../bcir-mapping/10-metadata-and-diagnostics.md)
- `../../runtime/llvm/README.md`
