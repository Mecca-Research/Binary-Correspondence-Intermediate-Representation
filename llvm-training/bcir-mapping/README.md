# BCIR Mapping Guide for LLVM IR

This chapter maps common Binary Correspondence Intermediate Representation
(BCIR) concepts to LLVM IR patterns in the training examples and to the
current executable oracle (`../../bcir/`) plus MLIR law (`../../mlir/`). Use it when lowering BCIR-like graph, resource, schedule, or
runtime ABI ideas into standalone LLVM IR that still assembles and verifies.
For a pattern-oriented dispatcher across BCIR mapping pages, pitfalls, advanced
IR semantics, MLIR bridge notes, backend/JIT notes, and exercises, see the
[`BCIR pattern index`](../indexes/bcir-patterns.md).

## Key takeaways

- Lower BCIR concepts in layers: semantic records, accessor/executor helpers, then plain LLVM operations.
- The BCIR Python model/oracle and MLIR dialect law are the current sources of truth; training-only LLVM structs illustrate lowering patterns rather than defining the project ABI.
- Metadata should preserve diagnostics and lowering provenance without becoming required for core IR correctness.
- Every BCIR-facing example should still assemble and verify as ordinary opaque-pointer LLVM IR.
- Treat BCIR normal forms as stage contracts and use verifier fenceposts to catch mapping drift that generic LLVM verification cannot see.

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
| Normal forms and verification | [`11-normal-forms-and-verification.md`](11-normal-forms-and-verification.md) | [`examples/normal-form-valid.ll`](examples/normal-form-valid.ll), [`examples/normal-form-drift.invalid.ll.txt`](examples/normal-form-drift.invalid.ll.txt), [`examples/normal-form-metadata-loss.invalid.ll.txt`](examples/normal-form-metadata-loss.invalid.ll.txt) |

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

## Current BCIR implementation files worth opening first

- [`../../bcir/model/graph.py`](../../bcir/model/graph.py) — resource, claim,
  phase, and module semantics in the executable oracle.
- [`../../bcir/frontends/rop.py`](../../bcir/frontends/rop.py) and
  [`../../bcir/frontends/map.py`](../../bcir/frontends/map.py) — source-level
  claim/resource front ends.
- [`../../bcir/kbcir/realize.py`](../../bcir/kbcir/realize.py) and
  [`../../bcir/kbcir/calibrate.py`](../../bcir/kbcir/calibrate.py) — realization
  and calibration policy.
- [`../../bcir/gem/streampack.py`](../../bcir/gem/streampack.py) and
  [`../../bcir/gem/execute.py`](../../bcir/gem/execute.py) — StreamPack
  provenance and execution.
- [`../../bcir/lower/llvm.py`](../../bcir/lower/llvm.py) and
  [`../../bcir/lower/jit.py`](../../bcir/lower/jit.py) — current LLVM AOT/JIT
  lowering boundary.
- [`../../mlir/include/BCIR/`](../../mlir/include/BCIR) and
  [`../../mlir/lib/BCIRPasses.cpp`](../../mlir/lib/BCIRPasses.cpp) — compiled
  dialect law and lowering/verification passes.
- [`../../docs/PARITY.md`](../../docs/PARITY.md) — agreement required between
  the executable oracle and MLIR law.

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

## Advanced chapter cross-navigation

| Mapping decision | Continue with |
|---|---|
| Enforce a stage contract or diagnose mapping drift | [`11-normal-forms-and-verification.md`](11-normal-forms-and-verification.md), then [`../17-new-pass-manager/04-adaptive-bcir-pipelines.md`](../17-new-pass-manager/04-adaptive-bcir-pipelines.md) |
| Implement dialect conversion legality and type/materialization rules | [`../18-mlir-lowering-to-llvm/README.md`](../18-mlir-lowering-to-llvm/README.md) |
| Sequence lowering with Transform dialect | [`../18-mlir-lowering-to-llvm/06-transform-dialect-for-bcir.md`](../18-mlir-lowering-to-llvm/06-transform-dialect-for-bcir.md) |
| Choose runtime call versus registered intrinsic versus target pseudo | [`../12-backend-jit/06-custom-bcir-intrinsics.md`](../12-backend-jit/06-custom-bcir-intrinsics.md), [`../19-hardware-aware/01-dragon-egg-gaadmsf-intrinsics.md`](../19-hardware-aware/01-dragon-egg-gaadmsf-intrinsics.md) |
| Carry calibration, pulse/flow, memory, or register policy | [`../19-hardware-aware/README.md`](../19-hardware-aware/README.md) |
| Deploy, replace, and retire generated kernels | [`../12-backend-jit/07-advanced-orc-runtime-integration.md`](../12-backend-jit/07-advanced-orc-runtime-integration.md) |

Run [`../tools/verify-bcir-mapping.sh`](../tools/verify-bcir-mapping.sh) for source-like
mapping fixtures and lowered companions. The repository-wide artifact inventory
is enforced by [`../tools/verify-manifest.sh`](../tools/verify-manifest.sh).
