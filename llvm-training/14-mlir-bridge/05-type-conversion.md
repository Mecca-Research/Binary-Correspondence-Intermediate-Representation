# 14.5 — Type Conversion for a BCIR MLIR Bridge

Type conversion is the contract between a domain dialect and the lower dialects
that can eventually translate to LLVM IR. For BCIR, it decides when graph values
remain first-class MLIR values, when they become descriptor structs, and when
they are reduced to integer IDs or runtime handles.

A good conversion plan is explicit about three questions:

1. **What is the source-level meaning?** A `!bcir.vertex` is not just an integer
   if passes still need its graph space, schema, or identity provenance.
2. **What is the lowered carrier?** The carrier may be `index`, `i64`, `memref`,
   `!llvm.ptr`, an LLVM dialect struct, or a runtime ABI argument list.
3. **What evidence justifies the conversion?** Data layout, ABI version, schema
   version, and target constraints must be known before narrowing, packing, or
   erasing structure.

## Conversion targets by abstraction level

| BCIR / MLIR source type | Mid-level carrier | LLVM-dialect carrier | Textual LLVM IR carrier |
|---|---|---|---|
| `!bcir.vertex<space, id_bits>` | `index`/`i64` ID plus graph descriptor | `i64` and/or `!llvm.ptr` | `i64`, `ptr`, or struct field |
| `!bcir.edge<src, dst>` | pair of vertex IDs, edge ordinal, or adjacency cursor | integer fields or runtime handle pointer | `%EdgeRef = type { i64, i64, i64 }` or `ptr` |
| `!bcir.attr<T>` / attribute read result | `T`, `memref` element, or runtime-call result | LLVM scalar/vector/aggregate value | LLVM first-class value or loaded field |
| `!bcir.hint` | explicit hint op with attributes | metadata, prefetch call operands, or no value | metadata node, intrinsic call, or erased |
| `!bcir.graph<rank>` / mixed-stride graph | `memref`/descriptor with sizes and strides | pointer plus size/stride fields | ABI struct or parallel arguments |
| `index` | target-sized integer | `i32` or `i64` per data layout | `i32`/`i64` |
| `memref<?xT>` | ranked descriptor | unpacked descriptor fields or LLVM struct | `{ ptr, ptr, i64, [N x i64], [N x i64] }`-style descriptor |

Do not use a bare pointer as a universal answer. A pointer can be the right ABI
handle, but it does not by itself preserve rank, bounds, edge direction,
attribute schema, or ownership.

## Index and ID width

MLIR `index` is intentionally target-dependent. If a BCIR vertex ID is declared
as 64 bits, lowering it through `index` and then to `i32` is a semantic narrowing
unless the schema and target prove the high bits are impossible. Prefer carrying
IDs as explicit integer widths (`i64`, `i128`, etc.) when identity stability
matters across modules, serialization, or runtime calls.

Use `index` for loop induction and shape arithmetic that is naturally tied to
the target machine. Use fixed-width integers for persisted BCIR IDs and ABI
fields.

## Descriptor vs handle conversion

A graph value can lower in two common ways:

- **Descriptor conversion**: materialize fields such as vertex base, edge base,
  attribute base, counts, offsets, and strides. This is useful when generated IR
  performs direct loads and GEP arithmetic.
- **Handle conversion**: pass an opaque runtime pointer and use ABI calls for
  lookup, attribute access, validation, and scheduling. This is useful when the
  runtime owns schema evolution, concurrency, or storage layout.

Both choices are valid, but mixing them inside one pipeline needs an explicit
boundary. For example, a pass may lower `bcir.lookup_child` to direct descriptor
loads before the runtime ABI boundary, while keeping `bcir.schedule_prefetch` as
a runtime call because scheduling policy is runtime-owned.

## What survives lowering: vertex identity

| Stage | Representation | What survives | What may be lost |
|---|---|---|---|
| BCIR dialect | `!bcir.vertex<space = "claim", id_bits = 64>` plus `bcir.vertex`/lookup ops | Space, ID width, provenance, verifier-visible identity | Nothing if verifier keeps operands/attributes explicit |
| Canonical BCIR | normalized vertex lookup and fixed ID operands | Stable IDs, graph space, schema references | Source spelling and redundant aliases |
| Mid-level MLIR | `i64` ID plus graph/space descriptor | Numeric identity and descriptor association | Custom type-level identity unless encoded in symbols/attrs |
| LLVM dialect | `i64`, `!llvm.ptr`, or struct fields | ABI-visible ID/handle | Dialect type and most verifier context |
| LLVM IR | `i64`, `ptr`, metadata/debug names | Runtime identity if ABI fields are correct | MLIR operation provenance unless recorded separately |

## What survives lowering: attributes

| Stage | Representation | What survives | What may be lost |
|---|---|---|---|
| BCIR dialect | operation attributes and `bcir.attribute` reads | Names, static payloads, type intent, side-table policy | Nothing if modeled as typed attributes/ops |
| Canonical BCIR | folded static attributes, normalized names | Constant values and resolved schema keys | Original aliases and unused labels |
| Mid-level MLIR | constants, `memref.load`, descriptor fields, or runtime calls | Values needed by computation | Static-vs-runtime distinction if not documented |
| LLVM dialect | constants, loads, call results, metadata | ABI-visible values and optional annotations | Rich attribute schema |
| LLVM IR | constants, loads, calls, metadata | Executable data path and selected metadata | MLIR attribute classes and verifiers |

## Conversion checklist

- State whether every conversion is **semantic**, **ABI-only**, or **hint-only**.
- Verify fixed-width IDs before narrowing to `index` or target-sized integers.
- Preserve graph shape through descriptors when direct memory access remains.
- Preserve ownership and lifetime either through runtime calls, memref-like
  descriptors, or explicit function attributes.
- Keep optional hints separate from correctness-bearing values.
- Add tests or review exercises for every conversion that erases custom types.

## See also

- [`06-conversion-patterns.md`](06-conversion-patterns.md) — how type conversion is consumed by rewrite patterns.
- [`07-pass-pipeline.md`](07-pass-pipeline.md) — where conversion passes fit in a full pipeline.
- [`examples/bcir-canonicalized.mlir`](examples/bcir-canonicalized.mlir) — canonical BCIR after source-level cleanup.
- [`../exercises/034-review-mlir-to-llvm-type-conversion.prompt.md`](../exercises/034-review-mlir-to-llvm-type-conversion.prompt.md) — type-conversion review exercise.
