# BCIR Mapping Guide for LLVM IR

This chapter maps common Binary Correspondence Intermediate Representation
(BCIR) concepts to the LLVM IR patterns used in `runtime/llvm/` and in the
training examples. Use it when lowering BCIR-like graph, resource, schedule, or
runtime ABI ideas into standalone LLVM IR that still assembles and verifies.

## Key takeaways

- Lower BCIR concepts in layers: semantic records, accessor/executor helpers, then plain LLVM operations.
- Runtime ABI structs in `runtime/llvm/` are the source of truth for claim, schedule, registry, blob, and executor shapes.
- Metadata should preserve diagnostics and lowering provenance without becoming required for core IR correctness.
- Every BCIR-facing example should still assemble and verify as ordinary opaque-pointer LLVM IR.

## Concepts

| Concept | Page | Standalone example |
|---|---|---|
| Vertex, edge, attribute lowering | [`01-vertex-edge-attribute.md`](01-vertex-edge-attribute.md) | [`examples/vertex-edge-attribute.ll`](examples/vertex-edge-attribute.ll) |
| Register/resource binding | [`02-register-binding.md`](02-register-binding.md) | [`examples/register-binding.ll`](examples/register-binding.ll) |
| Mixed-stride graph indexing | [`03-mixed-stride-graphs.md`](03-mixed-stride-graphs.md) | [`examples/mixed-stride.ll`](examples/mixed-stride.ll) |
| HAM hints and memory guidance | [`04-ham-hints.md`](04-ham-hints.md) | Use the prefetch/profile runtime examples linked from the page |
| Runtime ABI surface | [`05-runtime-abi.md`](05-runtime-abi.md) | Use the schema and executor runtime examples linked from the page |
| Claim lowering pipeline | [`06-claim-lowering-pipeline.md`](06-claim-lowering-pipeline.md) | [`examples/claim-resource-lookup.ll`](examples/claim-resource-lookup.ll), [`examples/bcir-op-runtime-wrapper.ll`](examples/bcir-op-runtime-wrapper.ll) |
| GAADMSF graph/data-movement operations | [`07-gaadmsf-operations.md`](07-gaadmsf-operations.md) | [`examples/graph-fragment-struct-gep.ll`](examples/graph-fragment-struct-gep.ll), [`examples/mixed-stride-byte-offset.ll`](examples/mixed-stride-byte-offset.ll) |
| Dragon Egg runtime-owned operations | [`08-dragon-egg-operations.md`](08-dragon-egg-operations.md) | [`examples/bcir-op-runtime-wrapper.ll`](examples/bcir-op-runtime-wrapper.ll) |
| Runtime call boundaries | [`09-runtime-call-boundaries.md`](09-runtime-call-boundaries.md) | [`examples/bcir-op-runtime-wrapper.ll`](examples/bcir-op-runtime-wrapper.ll), [`examples/claim-resource-lookup.ll`](examples/claim-resource-lookup.ll) |
| Metadata and diagnostics | [`10-metadata-and-diagnostics.md`](10-metadata-and-diagnostics.md) | [`examples/ham-hint-prefetch.ll`](examples/ham-hint-prefetch.ll), [`examples/diagnostic-metadata-preservation.ll`](examples/diagnostic-metadata-preservation.ll) |

## Shared lowering model

BCIR concepts generally lower into LLVM IR through three layers:

1. **Semantic records**: claims, batches, phase ranges, resource records, blob
   views, or stream packs are modeled as LLVM named structs and globals.
2. **Accessor and executor functions**: helpers read packed fields, look up
   resource IDs, execute claims, and step through batches or streams.
3. **Plain LLVM operations**: after the ABI boundary, memory access, arithmetic,
   atomics, vector operations, metadata, and control flow are ordinary LLVM IR
   and must obey the LangRef and verifier rules.

The current runtime seed keeps the ABI compact. `%bcir.claim` stores packed
control bits, fixed read/write resource-ID arrays, a hazard domain, and two
immediates; `%bcir.execctx` carries execution state; schedule structs group
claims into phase and batch ranges; registry/blob structs describe resources and
serialized views.

## Runtime files worth opening first

- [`runtime/llvm/README.md`](../../runtime/llvm/README.md) — high-level runtime
  seed and validation commands.
- [`runtime/llvm/bcir_claim_schema.ll`](../../runtime/llvm/bcir_claim_schema.ll)
  — claim, execution context, opcode/lane/domain metadata, and layout metadata.
- [`runtime/llvm/bcir_claim_accessors.ll`](../../runtime/llvm/bcir_claim_accessors.ll)
  — packed-field accessors for claim control, resource IDs, and immediates.
- [`runtime/llvm/bcir_registry_schema.ll`](../../runtime/llvm/bcir_registry_schema.ll)
  — resource, executable, worklist, blob header, and blob view records.
- [`runtime/llvm/bcir_schedule_schema.ll`](../../runtime/llvm/bcir_schedule_schema.ll)
  — phase, batch, layout, prefetch, tile, and stream-pack records.
- [`runtime/llvm/bcir_ops.ll`](../../runtime/llvm/bcir_ops.ll) — runtime op
  wrappers for loads, stores, vector operations, atomics, gather/scatter, and
  tensor-like kernels.
- [`runtime/llvm/bcir_gem_seed.ll`](../../runtime/llvm/bcir_gem_seed.ll) and
  [`runtime/llvm/bcir_batch_executor.ll`](../../runtime/llvm/bcir_batch_executor.ll)
  — claim, worklist, batch, and stream-pack execution patterns.

## Pitfalls checklist

Before adding or generating BCIR-facing LLVM IR, check these pages:

- [`08-pitfalls/01-nested-instruction-expressions.md`](../08-pitfalls/01-nested-instruction-expressions.md)
- [`08-pitfalls/02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md)
- [`08-pitfalls/04-duplicate-symbols.md`](../08-pitfalls/04-duplicate-symbols.md)
- [`08-pitfalls/05-type-schema-drift.md`](../08-pitfalls/05-type-schema-drift.md)
- [`08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md)
- [`08-pitfalls/09-atomic-ordering-mismatch.md`](../08-pitfalls/09-atomic-ordering-mismatch.md)
- [`08-pitfalls/10-volatile-is-not-atomic.md`](../08-pitfalls/10-volatile-is-not-atomic.md)
- [`08-pitfalls/11-address-space-confusion.md`](../08-pitfalls/11-address-space-confusion.md)
- [`08-pitfalls/12-vectorization-blocked-by-aliasing.md`](../08-pitfalls/12-vectorization-blocked-by-aliasing.md)
- [`08-pitfalls/13-pass-pipeline-ordering-surprise.md`](../08-pitfalls/13-pass-pipeline-ordering-surprise.md)

## Verifying the examples

From the repository root:

```bash
llvm-as llvm-training/bcir-mapping/examples/vertex-edge-attribute.ll -o /dev/null
llvm-as llvm-training/bcir-mapping/examples/register-binding.ll -o /dev/null
llvm-as llvm-training/bcir-mapping/examples/mixed-stride.ll -o /dev/null
llvm-as llvm-training/bcir-mapping/examples/graph-fragment-struct-gep.ll -o /dev/null
llvm-as llvm-training/bcir-mapping/examples/claim-resource-lookup.ll -o /dev/null
llvm-as llvm-training/bcir-mapping/examples/ham-hint-prefetch.ll -o /dev/null
llvm-as llvm-training/bcir-mapping/examples/bcir-op-runtime-wrapper.ll -o /dev/null
llvm-as llvm-training/bcir-mapping/examples/mixed-stride-byte-offset.ll -o /dev/null
llvm-as llvm-training/bcir-mapping/examples/diagnostic-metadata-preservation.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/vertex-edge-attribute.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/register-binding.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/mixed-stride.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/graph-fragment-struct-gep.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/claim-resource-lookup.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/ham-hint-prefetch.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/bcir-op-runtime-wrapper.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/mixed-stride-byte-offset.ll -o /dev/null
opt -passes=verify llvm-training/bcir-mapping/examples/diagnostic-metadata-preservation.ll -o /dev/null
```

The repository-wide helper also covers these files:

```bash
./llvm-training/tools/verify-examples.sh
```
