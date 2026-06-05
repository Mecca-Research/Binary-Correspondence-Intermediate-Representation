# Index: Cross-references to the BCIR project

Real-world examples of LLVM IR concepts (and bugs) live next door:

| Concept | BCIR file | Pitfall |
|---|---|---|
| Cache-line-aligned struct layout | `include/bcir/bcir_ir.hpp` (`BcirClaimV1`) | none |
| LLVM substrate ABI | `runtime/llvm/bcir_master_reference_v2.ll` | metadata-string syntax (fixed in `1f62e86`) |
| Boolean expression construction | `runtime/llvm/bcir_claim_verify.ll` | [`08-pitfalls/01-nested-instruction-expressions.md`](../08-pitfalls/01-nested-instruction-expressions.md) |
| PHI predecessors | `runtime/llvm/bcir_batch_executor.ll` | [`08-pitfalls/02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md) (fixed in `5754354`) |
| Duplicate block labels | `runtime/llvm/bcir_claim_verify.ll` (pre-`1f62e86`) | [`08-pitfalls/03-duplicate-block-labels.md`](../08-pitfalls/03-duplicate-block-labels.md) |
| Cross-module function definition collision | `runtime/llvm/bcir_gem_seed.ll` vs `bcir_worklist.ll` | [`08-pitfalls/04-duplicate-symbols.md`](../08-pitfalls/04-duplicate-symbols.md) |
| Type schema drift | `%bcir.blob.header` in three files | [`08-pitfalls/05-type-schema-drift.md`](../08-pitfalls/05-type-schema-drift.md) |
| `llvm.prefetch` immarg | `runtime/llvm/bcir_prefetch_profiles.ll` | [`08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md) |
| Vertex-Edge-Attribute custom dialect sketch | `llvm-training/14-mlir-bridge/examples/bcir-dialect-sketch.mlir` | [`14-mlir-bridge/04-bcir-as-custom-dialect.md`](../14-mlir-bridge/04-bcir-as-custom-dialect.md) |
| Lowered BCIR-style LLVM dialect sketch | `llvm-training/14-mlir-bridge/examples/lowered-llvm-dialect.mlir` | [`14-mlir-bridge/03-lowering-to-llvm-dialect.md`](../14-mlir-bridge/03-lowering-to-llvm-dialect.md) |
| BCIR graph fragment to struct arrays and GEPs | `llvm-training/bcir-mapping/examples/graph-fragment-struct-gep.ll` | [`bcir-mapping/07-gaadmsf-operations.md`](../bcir-mapping/07-gaadmsf-operations.md) |
| Claim resource lookup to registry loads | `llvm-training/bcir-mapping/examples/claim-resource-lookup.ll` | [`bcir-mapping/06-claim-lowering-pipeline.md`](../bcir-mapping/06-claim-lowering-pipeline.md) |
| HAM hint to metadata and prefetch intrinsic | `llvm-training/bcir-mapping/examples/ham-hint-prefetch.ll` | [`bcir-mapping/10-metadata-and-diagnostics.md`](../bcir-mapping/10-metadata-and-diagnostics.md), [`08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md) |
| BCIR operation to runtime call wrapper | `llvm-training/bcir-mapping/examples/bcir-op-runtime-wrapper.ll` | [`bcir-mapping/08-dragon-egg-operations.md`](../bcir-mapping/08-dragon-egg-operations.md), [`bcir-mapping/09-runtime-call-boundaries.md`](../bcir-mapping/09-runtime-call-boundaries.md) |
| Mixed-stride graph to byte-offset lowering | `llvm-training/bcir-mapping/examples/mixed-stride-byte-offset.ll` | [`bcir-mapping/07-gaadmsf-operations.md`](../bcir-mapping/07-gaadmsf-operations.md) |
| Diagnostic metadata preservation | `llvm-training/bcir-mapping/examples/diagnostic-metadata-preservation.ll` | [`bcir-mapping/10-metadata-and-diagnostics.md`](../bcir-mapping/10-metadata-and-diagnostics.md), [`08-pitfalls/08-stale-debug-locations.md`](../08-pitfalls/08-stale-debug-locations.md) |
| Hardware-aware GEM custom intrinsic fallback | `llvm-training/12-backend-jit/examples/custom-bcir-intrinsic-jit.ll` and `llvm-training/bcir-mapping/examples/hardware-aware-gem-lowering.ll` | [`12-backend-jit/06-custom-bcir-intrinsics.md`](../12-backend-jit/06-custom-bcir-intrinsics.md), [`reference/intrinsics-quickref.md#custom-backend-intrinsics`](../reference/intrinsics-quickref.md#custom-backend-intrinsics) |
| Vertex graph through MLIR and LLVM dialect | `llvm-training/14-mlir-bridge/examples/bcir-vertex-graph.mlir` → `bcir-vertex-graph-lowered-llvm-dialect.mlir` → `bcir-vertex-graph-lowered.ll` | [`14-mlir-bridge/05-vertex-graph-lowering.md`](../14-mlir-bridge/05-vertex-graph-lowering.md) |
| BCIR lowering invariant review | `llvm-training/exercises/038-custom-pass-bcir-invariants.prompt.md` | [`07-optimization/08-deep-optimization-lessons.md`](../07-optimization/08-deep-optimization-lessons.md) |
| Graph description metadata encoding | `llvm-training/exercises/039-graph-description-to-llvm-metadata.prompt.md` | [`bcir-mapping/10-metadata-and-diagnostics.md`](../bcir-mapping/10-metadata-and-diagnostics.md) |
| GAADMSF phi-predecessor lowering debug | `llvm-training/exercises/040-debug-gaadmsf-lowering.prompt.md` | [`bcir-mapping/07-gaadmsf-operations.md`](../bcir-mapping/07-gaadmsf-operations.md), [`08-pitfalls/02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md) |
