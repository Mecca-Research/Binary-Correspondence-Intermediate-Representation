# Standalone LLVM IR example manifest

Every standalone LLVM IR example in this training pack lives under a chapter
`examples/` directory and must assemble with a modern `llvm-as` (LLVM >= 15,
opaque pointers). This includes checked-in pass-output examples: both
`*-before.ll` inputs and `*-after*.ll` outputs are assembly-checked. Embedded
fenced `llvm` snippets in chapter prose are not part of this manifest unless
they are moved into one of these files.

See [`../EXAMPLES.md`](../EXAMPLES.md) for naming rules for invalid examples,
pass-output examples, chapter-local command documentation, and exercises.

Run the full manifest from the repository root with:

```bash
./llvm-training/tools/verify-examples.sh
```

That script prints per-file `llvm-as` and `opt -passes=verify` status, and it
skips `.ll.txt` files plus filenames containing `invalid`. For quick backend and
interpreter checks, use:

```bash
./llvm-training/tools/smoke-llc.sh
./llvm-training/tools/smoke-lli.sh
```

`smoke-llc.sh` uses a curated portable subset and avoids examples that require
unavailable targets, non-default address-space lowering, target-specific
intrinsics, GC/statepoint tokens, or analysis-only vectorizer artifacts.
`smoke-lli.sh` is narrower still: it runs only examples with a safe `main` or an
explicitly documented runnable entrypoint. Most files in this manifest are
assembly-only because they expose functions that need caller-provided arguments,
show optimization before/after IR, or demonstrate intrinsics/metadata rather
than complete executable programs.

## Standalone `.ll` examples

| File | Expected command |
|---|---|
| `llvm-training/00-foundations/examples/dominance-diamond.ll` | `llvm-as llvm-training/00-foundations/examples/dominance-diamond.ll -o /dev/null` |
| `llvm-training/00-foundations/examples/phi-shape-loop.ll` | `llvm-as llvm-training/00-foundations/examples/phi-shape-loop.ll -o /dev/null` |
| `llvm-training/00-foundations/examples/simple-add.ll` | `llvm-as llvm-training/00-foundations/examples/simple-add.ll -o /dev/null` |
| `llvm-training/00-foundations/examples/ssa-phi.ll` | `llvm-as llvm-training/00-foundations/examples/ssa-phi.ll -o /dev/null` |
| `llvm-training/00-foundations/examples/ssa-renaming-before.ll` | `llvm-as llvm-training/00-foundations/examples/ssa-renaming-before.ll -o /dev/null` |
| `llvm-training/01-syntax/examples/declarations-vs-definitions.ll` | `llvm-as llvm-training/01-syntax/examples/declarations-vs-definitions.ll -o /dev/null` |
| `llvm-training/01-syntax/examples/module-anatomy.ll` | `llvm-as llvm-training/01-syntax/examples/module-anatomy.ll -o /dev/null` |
| `llvm-training/01-syntax/examples/module-flags.ll` | `llvm-as llvm-training/01-syntax/examples/module-flags.ll -o /dev/null` |
| `llvm-training/01-syntax/examples/target-triple-datalayout.ll` | `llvm-as llvm-training/01-syntax/examples/target-triple-datalayout.ll -o /dev/null` |
| `llvm-training/02-types/examples/named-structs.ll` | `llvm-as llvm-training/02-types/examples/named-structs.ll -o /dev/null` |
| `llvm-training/02-types/examples/opaque-pointer-after.ll` | `llvm-as llvm-training/02-types/examples/opaque-pointer-after.ll -o /dev/null` |
| `llvm-training/02-types/examples/packed-structs.ll` | `llvm-as llvm-training/02-types/examples/packed-structs.ll -o /dev/null` |
| `llvm-training/02-types/examples/types-cookbook.ll` | `llvm-as llvm-training/02-types/examples/types-cookbook.ll -o /dev/null` |
| `llvm-training/02-types/examples/vector-types.ll` | `llvm-as llvm-training/02-types/examples/vector-types.ll -o /dev/null` |
| `llvm-training/03-constants/examples/aggregate-constants.ll` | `llvm-as llvm-training/03-constants/examples/aggregate-constants.ll -o /dev/null` |
| `llvm-training/03-constants/examples/constant-expressions.ll` | `llvm-as llvm-training/03-constants/examples/constant-expressions.ll -o /dev/null` |
| `llvm-training/03-constants/examples/constants-cookbook.ll` | `llvm-as llvm-training/03-constants/examples/constants-cookbook.ll -o /dev/null` |
| `llvm-training/03-constants/examples/null-undef-poison-freeze.ll` | `llvm-as llvm-training/03-constants/examples/null-undef-poison-freeze.ll -o /dev/null` |
| `llvm-training/04-memory/examples/aliasing-noalias.ll` | `llvm-as llvm-training/04-memory/examples/aliasing-noalias.ll -o /dev/null` |
| `llvm-training/04-memory/examples/alignment-load-store.ll` | `llvm-as llvm-training/04-memory/examples/alignment-load-store.ll -o /dev/null` |
| `llvm-training/04-memory/examples/gep-store-before-after.ll` | `llvm-as llvm-training/04-memory/examples/gep-store-before-after.ll -o /dev/null` |
| `llvm-training/04-memory/examples/memory-cookbook.ll` | `llvm-as llvm-training/04-memory/examples/memory-cookbook.ll -o /dev/null` |
| `llvm-training/05-control-flow/examples/control-flow-cookbook.ll` | `llvm-as llvm-training/05-control-flow/examples/control-flow-cookbook.ll -o /dev/null` |
| `llvm-training/05-control-flow/examples/indirectbr-table.ll` | `llvm-as llvm-training/05-control-flow/examples/indirectbr-table.ll -o /dev/null` |
| `llvm-training/05-control-flow/examples/switch-lowering.ll` | `llvm-as llvm-training/05-control-flow/examples/switch-lowering.ll -o /dev/null` |
| `llvm-training/05-control-flow/examples/unreachable-error-block.ll` | `llvm-as llvm-training/05-control-flow/examples/unreachable-error-block.ll -o /dev/null` |
| `llvm-training/06-metadata/examples/debug-location-preserved.ll` | `llvm-as llvm-training/06-metadata/examples/debug-location-preserved.ll -o /dev/null` |
| `llvm-training/06-metadata/examples/debug-location.ll` | `llvm-as llvm-training/06-metadata/examples/debug-location.ll -o /dev/null` |
| `llvm-training/06-metadata/examples/loop-metadata.ll` | `llvm-as llvm-training/06-metadata/examples/loop-metadata.ll -o /dev/null` |
| `llvm-training/06-metadata/examples/profile-branch-weights.ll` | `llvm-as llvm-training/06-metadata/examples/profile-branch-weights.ll -o /dev/null` |
| `llvm-training/06-metadata/examples/tbaa-load-store.ll` | `llvm-as llvm-training/06-metadata/examples/tbaa-load-store.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/dead-code-after-adce.ll` | `llvm-as llvm-training/07-optimization/examples/dead-code-after-adce.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/dead-code-before.ll` | `llvm-as llvm-training/07-optimization/examples/dead-code-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/gvn-after.ll` | `llvm-as llvm-training/07-optimization/examples/gvn-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/gvn-before.ll` | `llvm-as llvm-training/07-optimization/examples/gvn-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/gvn-load-after.ll` | `llvm-as llvm-training/07-optimization/examples/gvn-load-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/gvn-load-before.ll` | `llvm-as llvm-training/07-optimization/examples/gvn-load-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/instcombine-after.ll` | `llvm-as llvm-training/07-optimization/examples/instcombine-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/instcombine-before.ll` | `llvm-as llvm-training/07-optimization/examples/instcombine-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/instcombine-canonical-after.ll` | `llvm-as llvm-training/07-optimization/examples/instcombine-canonical-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/instcombine-canonical-before.ll` | `llvm-as llvm-training/07-optimization/examples/instcombine-canonical-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-before.ll` | `llvm-as llvm-training/07-optimization/examples/loop-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-rotate-after.ll` | `llvm-as llvm-training/07-optimization/examples/loop-rotate-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-rotate-before.ll` | `llvm-as llvm-training/07-optimization/examples/loop-rotate-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-rotate-while-after.ll` | `llvm-as llvm-training/07-optimization/examples/loop-rotate-while-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-rotate-while-before.ll` | `llvm-as llvm-training/07-optimization/examples/loop-rotate-while-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-unroll-after.ll` | `llvm-as llvm-training/07-optimization/examples/loop-unroll-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-unroll-before.ll` | `llvm-as llvm-training/07-optimization/examples/loop-unroll-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-unroll-count2-after.ll` | `llvm-as llvm-training/07-optimization/examples/loop-unroll-count2-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-unroll-count2-before.ll` | `llvm-as llvm-training/07-optimization/examples/loop-unroll-count2-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/mem2reg-after.ll` | `llvm-as llvm-training/07-optimization/examples/mem2reg-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/mem2reg-before.ll` | `llvm-as llvm-training/07-optimization/examples/mem2reg-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/mem2reg-diamond-after.ll` | `llvm-as llvm-training/07-optimization/examples/mem2reg-diamond-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/mem2reg-diamond-before.ll` | `llvm-as llvm-training/07-optimization/examples/mem2reg-diamond-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/o2-pipeline-inspection.ll` | `llvm-as llvm-training/07-optimization/examples/o2-pipeline-inspection.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/simplifycfg-after.ll` | `llvm-as llvm-training/07-optimization/examples/simplifycfg-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/simplifycfg-before.ll` | `llvm-as llvm-training/07-optimization/examples/simplifycfg-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/simplifycfg-select-after.ll` | `llvm-as llvm-training/07-optimization/examples/simplifycfg-select-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/simplifycfg-select-before.ll` | `llvm-as llvm-training/07-optimization/examples/simplifycfg-select-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/sroa-struct-after.ll` | `llvm-as llvm-training/07-optimization/examples/sroa-struct-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/sroa-struct-before.ll` | `llvm-as llvm-training/07-optimization/examples/sroa-struct-before.ll -o /dev/null` |
| `llvm-training/08-pitfalls/examples/immarg-fixed.ll` | `llvm-as llvm-training/08-pitfalls/examples/immarg-fixed.ll -o /dev/null` |
| `llvm-training/08-pitfalls/examples/phi-predecessor-fixed.ll` | `llvm-as llvm-training/08-pitfalls/examples/phi-predecessor-fixed.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/interleaved-access-after-vectorize.ll` | `llvm-as llvm-training/09-vectorization/examples/interleaved-access-after-vectorize.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/interleaved-access-before.ll` | `llvm-as llvm-training/09-vectorization/examples/interleaved-access-before.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/masked-load-store-after-vectorize.ll` | `llvm-as llvm-training/09-vectorization/examples/masked-load-store-after-vectorize.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/masked-load-store-before.ll` | `llvm-as llvm-training/09-vectorization/examples/masked-load-store-before.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/not-vectorizable-call.ll` | `llvm-as llvm-training/09-vectorization/examples/not-vectorizable-call.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/not-vectorizable-dependency.ll` | `llvm-as llvm-training/09-vectorization/examples/not-vectorizable-dependency.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/predicated-load-after-vectorize.ll` | `llvm-as llvm-training/09-vectorization/examples/predicated-load-after-vectorize.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/predicated-load-before.ll` | `llvm-as llvm-training/09-vectorization/examples/predicated-load-before.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/slp-scalars-after-slp.ll` | `llvm-as llvm-training/09-vectorization/examples/slp-scalars-after-slp.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/slp-scalars-before.ll` | `llvm-as llvm-training/09-vectorization/examples/slp-scalars-before.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/slp-scalars.ll` | `llvm-as llvm-training/09-vectorization/examples/slp-scalars.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/sum-loop-after-loop-vectorize.ll` | `llvm-as llvm-training/09-vectorization/examples/sum-loop-after-loop-vectorize.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/sum-loop-before.ll` | `llvm-as llvm-training/09-vectorization/examples/sum-loop-before.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/sum-loop.ll` | `llvm-as llvm-training/09-vectorization/examples/sum-loop.ll -o /dev/null` |
| `llvm-training/10-grammar/examples/instruction-forms.ll` | `llvm-as llvm-training/10-grammar/examples/instruction-forms.ll -o /dev/null` |
| `llvm-training/10-grammar/examples/metadata-attachments.ll` | `llvm-as llvm-training/10-grammar/examples/metadata-attachments.ll -o /dev/null` |
| `llvm-training/10-grammar/examples/top-level-entities.ll` | `llvm-as llvm-training/10-grammar/examples/top-level-entities.ll -o /dev/null` |
| `llvm-training/11-concurrency/examples/atomic-counter.ll` | `llvm-as llvm-training/11-concurrency/examples/atomic-counter.ll -o /dev/null` |
| `llvm-training/11-concurrency/examples/atomic-ordering-pairs.ll` | `llvm-as llvm-training/11-concurrency/examples/atomic-ordering-pairs.ll -o /dev/null` |
| `llvm-training/11-concurrency/examples/cmpxchg-loop.ll` | `llvm-as llvm-training/11-concurrency/examples/cmpxchg-loop.ll -o /dev/null` |
| `llvm-training/11-concurrency/examples/fence-patterns.ll` | `llvm-as llvm-training/11-concurrency/examples/fence-patterns.ll -o /dev/null` |
| `llvm-training/11-concurrency/examples/fence.ll` | `llvm-as llvm-training/11-concurrency/examples/fence.ll -o /dev/null` |
| `llvm-training/11-concurrency/examples/volatile-vs-atomic.ll` | `llvm-as llvm-training/11-concurrency/examples/volatile-vs-atomic.ll -o /dev/null` |
| `llvm-training/12-backend-jit/examples/codegen-input.ll` | `llvm-as llvm-training/12-backend-jit/examples/codegen-input.ll -o /dev/null` |
| `llvm-training/12-backend-jit/examples/jit-absolute-symbol.ll` | `llvm-as llvm-training/12-backend-jit/examples/jit-absolute-symbol.ll -o /dev/null` |
| `llvm-training/12-backend-jit/examples/orc-layer-diagnostic.ll` | `llvm-as llvm-training/12-backend-jit/examples/orc-layer-diagnostic.ll -o /dev/null` |
| `llvm-training/12-backend-jit/examples/relocation-symbols.ll` | `llvm-as llvm-training/12-backend-jit/examples/relocation-symbols.ll -o /dev/null` |
| `llvm-training/13-advanced-ir/examples/attributes-callsite.ll` | `llvm-as llvm-training/13-advanced-ir/examples/attributes-callsite.ll -o /dev/null` |
| `llvm-training/13-advanced-ir/examples/fast-math-flags.ll` | `llvm-as llvm-training/13-advanced-ir/examples/fast-math-flags.ll -o /dev/null` |
| `llvm-training/13-advanced-ir/examples/intrinsic-constraints.ll` | `llvm-as llvm-training/13-advanced-ir/examples/intrinsic-constraints.ll -o /dev/null` |
| `llvm-training/13-advanced-ir/examples/memcpy.ll` | `llvm-as llvm-training/13-advanced-ir/examples/memcpy.ll -o /dev/null` |
| `llvm-training/13-advanced-ir/examples/overflow-intrinsic.ll` | `llvm-as llvm-training/13-advanced-ir/examples/overflow-intrinsic.ll -o /dev/null` |
| `llvm-training/13-advanced-ir/examples/poison-freeze-branch.ll` | `llvm-as llvm-training/13-advanced-ir/examples/poison-freeze-branch.ll -o /dev/null` |
| `llvm-training/13-advanced-ir/examples/poison-undef-freeze.ll` | `llvm-as llvm-training/13-advanced-ir/examples/poison-undef-freeze.ll -o /dev/null` |
| `llvm-training/13-advanced-ir/examples/token-outline.ll` | `llvm-as llvm-training/13-advanced-ir/examples/token-outline.ll -o /dev/null` |
| `llvm-training/14-mlir-bridge/examples/arith-to-llvm-lowered.ll` | `llvm-as llvm-training/14-mlir-bridge/examples/arith-to-llvm-lowered.ll -o /dev/null` |
| `llvm-training/14-mlir-bridge/examples/bcir-vertex-graph-lowered.ll` | `llvm-as llvm-training/14-mlir-bridge/examples/bcir-vertex-graph-lowered.ll -o /dev/null` |
| `llvm-training/14-mlir-bridge/examples/memref-descriptor-lowered.ll` | `llvm-as llvm-training/14-mlir-bridge/examples/memref-descriptor-lowered.ll -o /dev/null` |
| `llvm-training/15-binary-analysis/examples/binary-layout-sketch.ll` | `llvm-as llvm-training/15-binary-analysis/examples/binary-layout-sketch.ll -o /dev/null` |
| `llvm-training/15-binary-analysis/examples/constant-time-review.ll` | `llvm-as llvm-training/15-binary-analysis/examples/constant-time-review.ll -o /dev/null` |
| `llvm-training/15-binary-analysis/examples/side-channel-branchy.ll` | `llvm-as llvm-training/15-binary-analysis/examples/side-channel-branchy.ll -o /dev/null` |
| `llvm-training/15-binary-analysis/examples/side-channel-masked.ll` | `llvm-as llvm-training/15-binary-analysis/examples/side-channel-masked.ll -o /dev/null` |
| `llvm-training/bcir-mapping/examples/bcir-op-runtime-wrapper.ll` | `llvm-as llvm-training/bcir-mapping/examples/bcir-op-runtime-wrapper.ll -o /dev/null` |
| `llvm-training/bcir-mapping/examples/claim-resource-lookup.ll` | `llvm-as llvm-training/bcir-mapping/examples/claim-resource-lookup.ll -o /dev/null` |
| `llvm-training/bcir-mapping/examples/diagnostic-metadata-preservation.ll` | `llvm-as llvm-training/bcir-mapping/examples/diagnostic-metadata-preservation.ll -o /dev/null` |
| `llvm-training/bcir-mapping/examples/graph-fragment-struct-gep.ll` | `llvm-as llvm-training/bcir-mapping/examples/graph-fragment-struct-gep.ll -o /dev/null` |
| `llvm-training/bcir-mapping/examples/ham-hint-prefetch.ll` | `llvm-as llvm-training/bcir-mapping/examples/ham-hint-prefetch.ll -o /dev/null` |
| `llvm-training/bcir-mapping/examples/mixed-stride-byte-offset.ll` | `llvm-as llvm-training/bcir-mapping/examples/mixed-stride-byte-offset.ll -o /dev/null` |
| `llvm-training/bcir-mapping/examples/mixed-stride.ll` | `llvm-as llvm-training/bcir-mapping/examples/mixed-stride.ll -o /dev/null` |
| `llvm-training/bcir-mapping/examples/register-binding.ll` | `llvm-as llvm-training/bcir-mapping/examples/register-binding.ll -o /dev/null` |
| `llvm-training/bcir-mapping/examples/vertex-edge-attribute.ll` | `llvm-as llvm-training/bcir-mapping/examples/vertex-edge-attribute.ll -o /dev/null` |

## Non-IR and intentionally excluded artifacts

The following files live next to examples but are not standalone LLVM IR assembly
targets for `verify-examples.sh`:

| File | Kind | Purpose |
|---|---|---|
| `llvm-training/02-types/examples/typed-pointer-before.ll.txt` | Intentionally excluded LLVM IR text fixture | Invalid or legacy IR text fixture; excluded from the known-good manifest. |
| `llvm-training/07-optimization/examples/bolt-layout-demo.c` | C source | Source companion for generating or explaining IR examples. |
| `llvm-training/08-pitfalls/examples/duplicate-symbols.invalid.ll.txt` | Intentionally excluded LLVM IR text fixture | Invalid or legacy IR text fixture; excluded from the known-good manifest. |
| `llvm-training/08-pitfalls/examples/immarg-violation.invalid.ll.txt` | Intentionally excluded LLVM IR text fixture | Invalid or legacy IR text fixture; excluded from the known-good manifest. |
| `llvm-training/08-pitfalls/examples/phi-predecessor-mismatch.invalid.ll.txt` | Intentionally excluded LLVM IR text fixture | Invalid or legacy IR text fixture; excluded from the known-good manifest. |
| `llvm-training/09-vectorization/examples/sum-loop.c` | C source | Source companion for generating or explaining IR examples. |
| `llvm-training/12-backend-jit/examples/lljit-outline.cpp.md` | Markdown sketch | Documentation-only code outline. |
| `llvm-training/12-backend-jit/examples/minimal-instruction.td` | TableGen sketch | TableGen sketch for backend instruction descriptions. |
| `llvm-training/14-mlir-bridge/examples/arith-to-llvm.mlir` | MLIR fragment | MLIR dialect sketch; use MLIR tooling rather than `llvm-as`. |
| `llvm-training/14-mlir-bridge/examples/bcir-dialect-sketch.mlir` | MLIR fragment | MLIR dialect sketch; use MLIR tooling rather than `llvm-as`. |
| `llvm-training/14-mlir-bridge/examples/bcir-vertex-graph.mlir` | MLIR fragment | BCIR vertex graph source sketch; use MLIR tooling rather than `llvm-as`. |
| `llvm-training/14-mlir-bridge/examples/bcir-vertex-graph-lowered-llvm-dialect.mlir` | MLIR fragment | Lowered LLVM-dialect vertex graph sketch; use MLIR tooling rather than `llvm-as`. |
| `llvm-training/14-mlir-bridge/examples/llvm-dialect-call.mlir` | MLIR fragment | MLIR dialect sketch; use MLIR tooling rather than `llvm-as`. |
| `llvm-training/14-mlir-bridge/examples/lowered-llvm-dialect.mlir` | MLIR fragment | MLIR dialect sketch; use MLIR tooling rather than `llvm-as`. |
| `llvm-training/15-binary-analysis/examples/bcsa-feature-sample.csv` | CSV data | Data/sample artifact; not assembled by `llvm-as`. |
| `llvm-training/15-binary-analysis/examples/bcsa-feature-variant-wide.csv` | CSV data | Data/sample artifact; not assembled by `llvm-as`. |
| `llvm-training/15-binary-analysis/examples/dynamic-trace-sample.csv` | CSV data | Data/sample artifact; not assembled by `llvm-as`. |
| `llvm-training/15-binary-analysis/examples/perf-counter-sample.csv` | CSV data | Data/sample artifact; not assembled by `llvm-as`. |
| `llvm-training/15-binary-analysis/examples/side-channel-trace-branchy.csv` | CSV data | Data/sample artifact; not assembled by `llvm-as`. |
| `llvm-training/15-binary-analysis/examples/side-channel-trace-masked.csv` | CSV data | Data/sample artifact; not assembled by `llvm-as`. |
| `llvm-training/examples/README.md` | Markdown sketch | Documentation-only code outline. |
| `llvm-training/examples/broken-example.ll.txt` | Intentionally excluded LLVM IR text fixture | Invalid or legacy IR text fixture; excluded from the known-good manifest. |

Notes:

- `llvm-training/15-binary-analysis/examples/*.csv` trace, counter, and feature schema samples are data artifacts, not `.ll` assembly targets.
- `llvm-training/14-mlir-bridge/examples/*.mlir` files are MLIR fragments; corresponding lowered `.ll` examples are listed in the standalone manifest when present.
- `llvm-training/examples/broken-example.ll.txt` and `llvm-training/08-pitfalls/examples/*.invalid.ll.txt` are intentionally invalid or text-only fixtures and remain excluded from the known-good manifest.
