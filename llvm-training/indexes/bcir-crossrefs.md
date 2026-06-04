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
