# 04 — BCIR dialect to LLVM dialect

This lesson maps BCIR domain concepts to LLVM-dialect and LLVM IR shapes. The
exact ABI belongs to the project runtime, but the lowering questions are stable.

## Vertex and edge attributes

BCIR vertex/edge attributes usually fall into three buckets:

| Attribute kind | Lowering choice |
| --- | --- |
| Runtime-needed graph data | LLVM struct fields, arrays, globals, or descriptor tables. |
| Diagnostic/provenance data | LLVM metadata, debug info, or side tables keyed by stable IDs. |
| Planning-only compiler hints | Removed after they have influenced affine/vector/pass selection. |

Do not erase `bcir.graph` until graph ID, vertex IDs, edge IDs, and claim IDs
have been attached to the replacement operations or serialized into a descriptor.

## Register binding

A BCIR register binding or prelock is not the same thing as a physical register
assignment. Lower it to explicit data:

```text
bcir.register_prelock %graph["r7"]
  -> %slot = llvm.getelementptr %resource_table[%r7]
  -> %ptr = llvm.load %slot
  -> call @bcir_use_prelocked(ptr %ptr, i64 %claim)
```

The backend register allocator remains free to assign machine registers, but the
BCIR resource contract survives as an operand, pointer, or table lookup.

## HAM hints

HAM hints may lower to:

- custom metadata attached to a load, store, call, or prefetch;
- `llvm.prefetch` when the hint describes locality and timing;
- a runtime wrapper such as `@bcir_ham_prefetch(ptr, i32 locality, i64 claim)`;
- target-specific intrinsics if the target and ABI policy require them.

Hints should not become required semantics unless the source BCIR operation was
already semantically mandatory.

## GAADMSF operations

Graph-Aware Adaptive Data Movement and Scheduling Framework operations usually
lower to one of three forms:

1. affine/vector loops when the graph fragment has regular structure;
2. runtime calls when scheduling, ownership, or target features are runtime
   decisions;
3. LLVM intrinsic wrappers when the operation maps to a known hardware feature.

Preserve diagnostic metadata on the replacement call or on the core memory
operation, not only on a soon-to-be-erased staging op.

## Custom BCIR types

Custom type lowering must be centralized in the `TypeConverter`:

- graph descriptors become opaque pointers or LLVM structs;
- vertices and edges become IDs, pointers, or descriptor structs;
- register resources become pointers, table indices, or ABI handles;
- claim/proof values become metadata plus optional side-table handles.

The LLVM dialect has explicit pointer and struct types. Make data layout choices
before full conversion, not during translation to textual LLVM IR.

## Preserving claim IDs and diagnostics

Claim IDs and diagnostic metadata may survive as named metadata, instruction
metadata, call operands, or descriptor fields. Choose based on consumer needs:

- optimizer-only hints: metadata is sufficient;
- runtime enforcement: use operands, memory, or calls;
- postmortem diagnostics: metadata plus stable side-table IDs is often best.

A useful rule: if removing metadata would change program behavior, the fact is
not merely metadata and must also be represented in executable IR.

## Normal-form handoff

A successful conversion must establish the first LLVM-side BCIR normal form, not
only satisfy the LLVM dialect conversion target. Preserve operation locations and
stable claim/register IDs on replacement operations or in verifier-visible side
tables, then run the LLVM-side verifier at the translation boundary. See
[BCIR Normal Forms and Verification](../bcir-mapping/11-normal-forms-and-verification.md)
for the handoff invariants and the correlation rules between MLIR conversion
diagnostics and LLVM mapping-drift diagnostics.
