# Standalone LLVM IR example manifest

Every standalone LLVM IR example in this training pack lives under a chapter
`examples/` directory and must assemble with a modern `llvm-as` (LLVM >= 18,
opaque pointers) -- unless it declares otherwise. An example that teaches a
construct newer than the baseline carries `; REQUIRES: llvm >= N` on its first
lines; `tools/verify-examples.sh` skips it, with the reason printed, on an older
assembler and verifies it normally once the assembler is new enough. The skip has
an owner rather than being a hole: CI's `LLVM training corpus (LLVM 23)` job
installs LLVM 23, so every version-gated example is verified on every run there. This includes checked-in pass-output examples: both
`*-before.ll` inputs and `*-after*.ll` outputs are assembly-checked. Embedded
fenced `llvm` snippets in chapter prose are not part of this manifest unless
they are moved into one of these files.

See [`../EXAMPLES.md`](../EXAMPLES.md) for naming rules for invalid examples,
pass-output examples, chapter-local command documentation, and exercises.

Run the full manifest from the repository root with:

```bash
./training/llvm/tools/verify-examples.sh
```

That script prints per-file `llvm-as` and `opt -passes=verify` status, and it
skips `.ll.txt` files plus filenames containing `invalid`. For quick backend and
interpreter checks, use:

```bash
./training/llvm/tools/smoke-llc.sh
./training/llvm/tools/smoke-lli.sh
```

`smoke-llc.sh` uses a curated portable subset and avoids examples that require
unavailable targets, non-default address-space lowering, target-specific
intrinsics, GC/statepoint tokens, or analysis-only vectorizer artifacts.
`smoke-lli.sh` is narrower still: it runs only examples with a safe `main` or an
explicitly documented runnable entrypoint. The advanced checked-in examples for
stackmaps/patchpoints, operand bundles, EH funclets, coroutines, GC
statepoints, matrix intrinsics, and token/convergence-style IR are manifest
entries because they assemble and verify, but they are intentionally documented
as assembly-only unless a chapter names a target/runtime-specific smoke path.
Most files in this manifest are assembly-only because they expose functions that
need caller-provided arguments, show optimization before/after IR, or demonstrate
intrinsics/metadata rather than complete executable programs. Target-specific
`.ll` sketches, such as RISC-V scalable-vector and AVX-512 mask boundary
examples, stay in the manifest when they assemble and verify as IR; backend smoke
scripts may still skip them if the local target support is unavailable.

## Standalone `.ll` examples

| File | Expected command |
|---|---|
| `training/llvm/00-foundations/examples/dominance-diamond.ll` | `llvm-as training/llvm/00-foundations/examples/dominance-diamond.ll -o /dev/null` |
| `training/llvm/00-foundations/examples/phi-shape-loop.ll` | `llvm-as training/llvm/00-foundations/examples/phi-shape-loop.ll -o /dev/null` |
| `training/llvm/00-foundations/examples/simple-add.ll` | `llvm-as training/llvm/00-foundations/examples/simple-add.ll -o /dev/null` |
| `training/llvm/00-foundations/examples/ssa-phi.ll` | `llvm-as training/llvm/00-foundations/examples/ssa-phi.ll -o /dev/null` |
| `training/llvm/00-foundations/examples/ssa-renaming-before.ll` | `llvm-as training/llvm/00-foundations/examples/ssa-renaming-before.ll -o /dev/null` |
| `training/llvm/01-syntax/examples/declarations-vs-definitions.ll` | `llvm-as training/llvm/01-syntax/examples/declarations-vs-definitions.ll -o /dev/null` |
| `training/llvm/01-syntax/examples/inline-asm.ll` | `llvm-as training/llvm/01-syntax/examples/inline-asm.ll -o /dev/null` |
| `training/llvm/01-syntax/examples/module-anatomy.ll` | `llvm-as training/llvm/01-syntax/examples/module-anatomy.ll -o /dev/null` |
| `training/llvm/01-syntax/examples/module-flags.ll` | `llvm-as training/llvm/01-syntax/examples/module-flags.ll -o /dev/null` |
| `training/llvm/01-syntax/examples/target-triple-datalayout.ll` | `llvm-as training/llvm/01-syntax/examples/target-triple-datalayout.ll -o /dev/null` |
| `training/llvm/02-types/examples/named-structs.ll` | `llvm-as training/llvm/02-types/examples/named-structs.ll -o /dev/null` |
| `training/llvm/02-types/examples/opaque-pointer-after.ll` | `llvm-as training/llvm/02-types/examples/opaque-pointer-after.ll -o /dev/null` |
| `training/llvm/02-types/examples/packed-structs.ll` | `llvm-as training/llvm/02-types/examples/packed-structs.ll -o /dev/null` |
| `training/llvm/02-types/examples/types-cookbook.ll` | `llvm-as training/llvm/02-types/examples/types-cookbook.ll -o /dev/null` |
| `training/llvm/02-types/examples/vector-types.ll` | `llvm-as training/llvm/02-types/examples/vector-types.ll -o /dev/null` |
| `training/llvm/03-constants/examples/aggregate-constants.ll` | `llvm-as training/llvm/03-constants/examples/aggregate-constants.ll -o /dev/null` |
| `training/llvm/03-constants/examples/constant-expressions.ll` | `llvm-as training/llvm/03-constants/examples/constant-expressions.ll -o /dev/null` |
| `training/llvm/03-constants/examples/constants-cookbook.ll` | `llvm-as training/llvm/03-constants/examples/constants-cookbook.ll -o /dev/null` |
| `training/llvm/03-constants/examples/null-undef-poison-freeze.ll` | `llvm-as training/llvm/03-constants/examples/null-undef-poison-freeze.ll -o /dev/null` |
| `training/llvm/04-memory/examples/aliasing-noalias.ll` | `llvm-as training/llvm/04-memory/examples/aliasing-noalias.ll -o /dev/null` |
| `training/llvm/04-memory/examples/alignment-load-store.ll` | `llvm-as training/llvm/04-memory/examples/alignment-load-store.ll -o /dev/null` |
| `training/llvm/04-memory/examples/gep-store-before-after.ll` | `llvm-as training/llvm/04-memory/examples/gep-store-before-after.ll -o /dev/null` |
| `training/llvm/04-memory/examples/memory-cookbook.ll` | `llvm-as training/llvm/04-memory/examples/memory-cookbook.ll -o /dev/null` |
| `training/llvm/05-control-flow/examples/call-and-ret.ll` | `llvm-as training/llvm/05-control-flow/examples/call-and-ret.ll -o /dev/null` |
| `training/llvm/05-control-flow/examples/comparisons-and-select.ll` | `llvm-as training/llvm/05-control-flow/examples/comparisons-and-select.ll -o /dev/null` |
| `training/llvm/05-control-flow/examples/control-flow-cookbook.ll` | `llvm-as training/llvm/05-control-flow/examples/control-flow-cookbook.ll -o /dev/null` |
| `training/llvm/05-control-flow/examples/indirectbr-table.ll` | `llvm-as training/llvm/05-control-flow/examples/indirectbr-table.ll -o /dev/null` |
| `training/llvm/05-control-flow/examples/switch-lowering.ll` | `llvm-as training/llvm/05-control-flow/examples/switch-lowering.ll -o /dev/null` |
| `training/llvm/05-control-flow/examples/unreachable-error-block.ll` | `llvm-as training/llvm/05-control-flow/examples/unreachable-error-block.ll -o /dev/null` |
| `training/llvm/06-metadata/examples/debug-info-optimization.ll` | `llvm-as training/llvm/06-metadata/examples/debug-info-optimization.ll -o /dev/null` |
| `training/llvm/06-metadata/examples/debug-location-preserved.ll` | `llvm-as training/llvm/06-metadata/examples/debug-location-preserved.ll -o /dev/null` |
| `training/llvm/06-metadata/examples/debug-location.ll` | `llvm-as training/llvm/06-metadata/examples/debug-location.ll -o /dev/null` |
| `training/llvm/06-metadata/examples/debug-variable-fragments.ll` | `llvm-as training/llvm/06-metadata/examples/debug-variable-fragments.ll -o /dev/null` |
| `training/llvm/06-metadata/examples/loop-metadata.ll` | `llvm-as training/llvm/06-metadata/examples/loop-metadata.ll -o /dev/null` |
| `training/llvm/06-metadata/examples/profile-branch-weights.ll` | `llvm-as training/llvm/06-metadata/examples/profile-branch-weights.ll -o /dev/null` |
| `training/llvm/06-metadata/examples/profile-entry-count.ll` | `llvm-as training/llvm/06-metadata/examples/profile-entry-count.ll -o /dev/null` |
| `training/llvm/06-metadata/examples/profile-value-indirect-call.ll` | `llvm-as training/llvm/06-metadata/examples/profile-value-indirect-call.ll -o /dev/null` |
| `training/llvm/06-metadata/examples/tbaa-load-store.ll` | `llvm-as training/llvm/06-metadata/examples/tbaa-load-store.ll -o /dev/null` |
| `training/llvm/06-metadata/examples/type-metadata-cfi.ll` | `llvm-as training/llvm/06-metadata/examples/type-metadata-cfi.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/bcir-benchmark-kernel.ll` | `llvm-as training/llvm/07-optimization/examples/bcir-benchmark-kernel.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/bcir-memoryssa-pipeline.ll` | `llvm-as training/llvm/07-optimization/examples/bcir-memoryssa-pipeline.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/bcir-sccp-freeze-after.ll` | `llvm-as training/llvm/07-optimization/examples/bcir-sccp-freeze-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/bcir-sccp-freeze-before.ll` | `llvm-as training/llvm/07-optimization/examples/bcir-sccp-freeze-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/dead-code-after-adce.ll` | `llvm-as training/llvm/07-optimization/examples/dead-code-after-adce.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/dead-code-before.ll` | `llvm-as training/llvm/07-optimization/examples/dead-code-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/gvn-after.ll` | `llvm-as training/llvm/07-optimization/examples/gvn-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/gvn-before.ll` | `llvm-as training/llvm/07-optimization/examples/gvn-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/gvn-load-after.ll` | `llvm-as training/llvm/07-optimization/examples/gvn-load-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/gvn-load-before.ll` | `llvm-as training/llvm/07-optimization/examples/gvn-load-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/instcombine-after.ll` | `llvm-as training/llvm/07-optimization/examples/instcombine-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/instcombine-before.ll` | `llvm-as training/llvm/07-optimization/examples/instcombine-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/instcombine-canonical-after.ll` | `llvm-as training/llvm/07-optimization/examples/instcombine-canonical-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/instcombine-canonical-before.ll` | `llvm-as training/llvm/07-optimization/examples/instcombine-canonical-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/loop-before.ll` | `llvm-as training/llvm/07-optimization/examples/loop-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/loop-rotate-after.ll` | `llvm-as training/llvm/07-optimization/examples/loop-rotate-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/loop-rotate-bcir-after.ll` | `llvm-as training/llvm/07-optimization/examples/loop-rotate-bcir-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/loop-rotate-bcir-before.ll` | `llvm-as training/llvm/07-optimization/examples/loop-rotate-bcir-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/loop-rotate-before.ll` | `llvm-as training/llvm/07-optimization/examples/loop-rotate-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/loop-rotate-while-after.ll` | `llvm-as training/llvm/07-optimization/examples/loop-rotate-while-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/loop-rotate-while-before.ll` | `llvm-as training/llvm/07-optimization/examples/loop-rotate-while-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/loop-unroll-after.ll` | `llvm-as training/llvm/07-optimization/examples/loop-unroll-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/loop-unroll-before.ll` | `llvm-as training/llvm/07-optimization/examples/loop-unroll-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/loop-unroll-count2-after.ll` | `llvm-as training/llvm/07-optimization/examples/loop-unroll-count2-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/loop-unroll-count2-before.ll` | `llvm-as training/llvm/07-optimization/examples/loop-unroll-count2-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/mem2reg-after.ll` | `llvm-as training/llvm/07-optimization/examples/mem2reg-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/mem2reg-before.ll` | `llvm-as training/llvm/07-optimization/examples/mem2reg-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/mem2reg-diamond-after.ll` | `llvm-as training/llvm/07-optimization/examples/mem2reg-diamond-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/mem2reg-diamond-before.ll` | `llvm-as training/llvm/07-optimization/examples/mem2reg-diamond-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/memoryssa-alias-shape.ll` | `llvm-as training/llvm/07-optimization/examples/memoryssa-alias-shape.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/o2-pipeline-inspection.ll` | `llvm-as training/llvm/07-optimization/examples/o2-pipeline-inspection.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/opt-diff-instcombine-before.ll` | `llvm-as training/llvm/07-optimization/examples/opt-diff-instcombine-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/opt-diff-instcombine.after-instcombine.ll` | `llvm-as training/llvm/07-optimization/examples/opt-diff-instcombine.after-instcombine.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/opt-diff-loop-rotate-before.ll` | `llvm-as training/llvm/07-optimization/examples/opt-diff-loop-rotate-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/opt-diff-loop-rotate.after-loop-rotate.ll` | `llvm-as training/llvm/07-optimization/examples/opt-diff-loop-rotate.after-loop-rotate.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/sccp-after.ll` | `llvm-as training/llvm/07-optimization/examples/sccp-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/sccp-before.ll` | `llvm-as training/llvm/07-optimization/examples/sccp-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/simplifycfg-after.ll` | `llvm-as training/llvm/07-optimization/examples/simplifycfg-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/simplifycfg-before.ll` | `llvm-as training/llvm/07-optimization/examples/simplifycfg-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/simplifycfg-select-after.ll` | `llvm-as training/llvm/07-optimization/examples/simplifycfg-select-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/simplifycfg-select-before.ll` | `llvm-as training/llvm/07-optimization/examples/simplifycfg-select-before.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/sroa-struct-after.ll` | `llvm-as training/llvm/07-optimization/examples/sroa-struct-after.ll -o /dev/null` |
| `training/llvm/07-optimization/examples/sroa-struct-before.ll` | `llvm-as training/llvm/07-optimization/examples/sroa-struct-before.ll -o /dev/null` |
| `training/llvm/08-pitfalls/examples/immarg-fixed.ll` | `llvm-as training/llvm/08-pitfalls/examples/immarg-fixed.ll -o /dev/null` |
| `training/llvm/08-pitfalls/examples/phi-predecessor-fixed.ll` | `llvm-as training/llvm/08-pitfalls/examples/phi-predecessor-fixed.ll -o /dev/null` |
| `training/llvm/08-pitfalls/examples/sanitizer-instrumentation.ll` | `llvm-as training/llvm/08-pitfalls/examples/sanitizer-instrumentation.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/bcir-avx512-mask-sketch.ll` | `llvm-as training/llvm/09-vectorization/examples/bcir-avx512-mask-sketch.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/bcir-interleaved-riscv-sketch.ll` | `llvm-as training/llvm/09-vectorization/examples/bcir-interleaved-riscv-sketch.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/interleaved-access-after-vectorize.ll` | `llvm-as training/llvm/09-vectorization/examples/interleaved-access-after-vectorize.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/interleaved-access-before.ll` | `llvm-as training/llvm/09-vectorization/examples/interleaved-access-before.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/masked-load-store-after-vectorize.ll` | `llvm-as training/llvm/09-vectorization/examples/masked-load-store-after-vectorize.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/masked-load-store-before.ll` | `llvm-as training/llvm/09-vectorization/examples/masked-load-store-before.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/not-vectorizable-call.ll` | `llvm-as training/llvm/09-vectorization/examples/not-vectorizable-call.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/not-vectorizable-dependency.ll` | `llvm-as training/llvm/09-vectorization/examples/not-vectorizable-dependency.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/predicated-load-after-vectorize.ll` | `llvm-as training/llvm/09-vectorization/examples/predicated-load-after-vectorize.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/predicated-load-before.ll` | `llvm-as training/llvm/09-vectorization/examples/predicated-load-before.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/slp-scalars-after-slp.ll` | `llvm-as training/llvm/09-vectorization/examples/slp-scalars-after-slp.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/slp-scalars-before.ll` | `llvm-as training/llvm/09-vectorization/examples/slp-scalars-before.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/slp-scalars.ll` | `llvm-as training/llvm/09-vectorization/examples/slp-scalars.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/sum-loop-after-loop-vectorize.ll` | `llvm-as training/llvm/09-vectorization/examples/sum-loop-after-loop-vectorize.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/sum-loop-before.ll` | `llvm-as training/llvm/09-vectorization/examples/sum-loop-before.ll -o /dev/null` |
| `training/llvm/09-vectorization/examples/sum-loop.ll` | `llvm-as training/llvm/09-vectorization/examples/sum-loop.ll -o /dev/null` |
| `training/llvm/10-grammar/examples/instruction-forms.ll` | `llvm-as training/llvm/10-grammar/examples/instruction-forms.ll -o /dev/null` |
| `training/llvm/10-grammar/examples/llvm15-constructs.ll` | `llvm-as training/llvm/10-grammar/examples/llvm15-constructs.ll -o /dev/null` |
| `training/llvm/10-grammar/examples/metadata-attachments.ll` | `llvm-as training/llvm/10-grammar/examples/metadata-attachments.ll -o /dev/null` |
| `training/llvm/10-grammar/examples/top-level-entities.ll` | `llvm-as training/llvm/10-grammar/examples/top-level-entities.ll -o /dev/null` |
| `training/llvm/11-concurrency/examples/atomic-counter.ll` | `llvm-as training/llvm/11-concurrency/examples/atomic-counter.ll -o /dev/null` |
| `training/llvm/11-concurrency/examples/atomic-ordering-pairs.ll` | `llvm-as training/llvm/11-concurrency/examples/atomic-ordering-pairs.ll -o /dev/null` |
| `training/llvm/11-concurrency/examples/cmpxchg-loop.ll` | `llvm-as training/llvm/11-concurrency/examples/cmpxchg-loop.ll -o /dev/null` |
| `training/llvm/11-concurrency/examples/fence-patterns.ll` | `llvm-as training/llvm/11-concurrency/examples/fence-patterns.ll -o /dev/null` |
| `training/llvm/11-concurrency/examples/fence.ll` | `llvm-as training/llvm/11-concurrency/examples/fence.ll -o /dev/null` |
| `training/llvm/11-concurrency/examples/volatile-vs-atomic.ll` | `llvm-as training/llvm/11-concurrency/examples/volatile-vs-atomic.ll -o /dev/null` |
| `training/llvm/12-backend-jit/examples/codegen-input.ll` | `llvm-as training/llvm/12-backend-jit/examples/codegen-input.ll -o /dev/null` |
| `training/llvm/12-backend-jit/examples/custom-bcir-intrinsic-jit.ll` | `llvm-as training/llvm/12-backend-jit/examples/custom-bcir-intrinsic-jit.ll -o /dev/null` |
| `training/llvm/12-backend-jit/examples/jit-absolute-symbol.ll` | `llvm-as training/llvm/12-backend-jit/examples/jit-absolute-symbol.ll -o /dev/null` |
| `training/llvm/12-backend-jit/examples/orc-layer-diagnostic.ll` | `llvm-as training/llvm/12-backend-jit/examples/orc-layer-diagnostic.ll -o /dev/null` |
| `training/llvm/12-backend-jit/examples/relocation-symbols.ll` | `llvm-as training/llvm/12-backend-jit/examples/relocation-symbols.ll -o /dev/null` |
| `training/llvm/12-backend-jit/examples/stackmap-patchpoint.ll` | `llvm-as training/llvm/12-backend-jit/examples/stackmap-patchpoint.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/attributes-callsite.ll` | `llvm-as training/llvm/13-advanced-ir/examples/attributes-callsite.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/bcir-freeze-safe-speculation.ll` | `llvm-as training/llvm/13-advanced-ir/examples/bcir-freeze-safe-speculation.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/coroutine-outline.ll` | `llvm-as training/llvm/13-advanced-ir/examples/coroutine-outline.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/fast-math-flags.ll` | `llvm-as training/llvm/13-advanced-ir/examples/fast-math-flags.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/gc-statepoint-relocate.ll` | `llvm-as training/llvm/13-advanced-ir/examples/gc-statepoint-relocate.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/intrinsic-constraints.ll` | `llvm-as training/llvm/13-advanced-ir/examples/intrinsic-constraints.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/matrix-intrinsics-sketch.ll` | `llvm-as training/llvm/13-advanced-ir/examples/matrix-intrinsics-sketch.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/memcpy.ll` | `llvm-as training/llvm/13-advanced-ir/examples/memcpy.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/operand-bundles-deopt.ll` | `llvm-as training/llvm/13-advanced-ir/examples/operand-bundles-deopt.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/operand-bundles-funclet.ll` | `llvm-as training/llvm/13-advanced-ir/examples/operand-bundles-funclet.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/overflow-intrinsic.ll` | `llvm-as training/llvm/13-advanced-ir/examples/overflow-intrinsic.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/poison-freeze-branch.ll` | `llvm-as training/llvm/13-advanced-ir/examples/poison-freeze-branch.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/poison-undef-freeze.ll` | `llvm-as training/llvm/13-advanced-ir/examples/poison-undef-freeze.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/token-outline.ll` | `llvm-as training/llvm/13-advanced-ir/examples/token-outline.ll -o /dev/null` |
| `training/llvm/13-advanced-ir/examples/ub-poison-to-llvm-lowered.ll` | `llvm-as training/llvm/13-advanced-ir/examples/ub-poison-to-llvm-lowered.ll -o /dev/null` |
| `training/llvm/14-mlir-bridge/examples/arith-to-llvm-lowered.ll` | `llvm-as training/llvm/14-mlir-bridge/examples/arith-to-llvm-lowered.ll -o /dev/null` |
| `training/llvm/14-mlir-bridge/examples/bcir-vertex-graph-lowered.ll` | `llvm-as training/llvm/14-mlir-bridge/examples/bcir-vertex-graph-lowered.ll -o /dev/null` |
| `training/llvm/14-mlir-bridge/examples/memref-descriptor-lowered.ll` | `llvm-as training/llvm/14-mlir-bridge/examples/memref-descriptor-lowered.ll -o /dev/null` |
| `training/llvm/15-binary-analysis/examples/binary-layout-sketch.ll` | `llvm-as training/llvm/15-binary-analysis/examples/binary-layout-sketch.ll -o /dev/null` |
| `training/llvm/15-binary-analysis/examples/constant-time-review.ll` | `llvm-as training/llvm/15-binary-analysis/examples/constant-time-review.ll -o /dev/null` |
| `training/llvm/15-binary-analysis/examples/side-channel-branchy.ll` | `llvm-as training/llvm/15-binary-analysis/examples/side-channel-branchy.ll -o /dev/null` |
| `training/llvm/15-binary-analysis/examples/side-channel-masked.ll` | `llvm-as training/llvm/15-binary-analysis/examples/side-channel-masked.ll -o /dev/null` |
| `training/llvm/16-exception-handling/examples/catchswitch-funclet.ll` | `llvm-as training/llvm/16-exception-handling/examples/catchswitch-funclet.ll -o /dev/null` |
| `training/llvm/16-exception-handling/examples/cleanup-resume.ll` | `llvm-as training/llvm/16-exception-handling/examples/cleanup-resume.ll -o /dev/null` |
| `training/llvm/16-exception-handling/examples/invoke-landingpad.ll` | `llvm-as training/llvm/16-exception-handling/examples/invoke-landingpad.ll -o /dev/null` |
| `training/llvm/17-new-pass-manager/examples/gaadmsf-pipeline-after.ll` | `llvm-as training/llvm/17-new-pass-manager/examples/gaadmsf-pipeline-after.ll -o /dev/null` |
| `training/llvm/17-new-pass-manager/examples/gaadmsf-pipeline-before.ll` | `llvm-as training/llvm/17-new-pass-manager/examples/gaadmsf-pipeline-before.ll -o /dev/null` |
| `training/llvm/17-new-pass-manager/examples/pass-plugin/binding-collision.ll` | `llvm-as training/llvm/17-new-pass-manager/examples/pass-plugin/binding-collision.ll -o /dev/null` |
| `training/llvm/17-new-pass-manager/examples/pass-plugin/binding-ok.ll` | `llvm-as training/llvm/17-new-pass-manager/examples/pass-plugin/binding-ok.ll -o /dev/null` |
| `training/llvm/17-new-pass-manager/examples/pass-plugin/unroll-breaks-binding.ll` | `llvm-as training/llvm/17-new-pass-manager/examples/pass-plugin/unroll-breaks-binding.ll -o /dev/null` |
| `training/llvm/18-mlir-lowering-to-llvm/examples/bcir-register-prelock-ham-hints-lowered.ll` | `llvm-as training/llvm/18-mlir-lowering-to-llvm/examples/bcir-register-prelock-ham-hints-lowered.ll -o /dev/null` |
| `training/llvm/18-mlir-lowering-to-llvm/examples/memref-type-conversion-lowered.ll` | `llvm-as training/llvm/18-mlir-lowering-to-llvm/examples/memref-type-conversion-lowered.ll -o /dev/null` |
| `training/llvm/19-hardware-aware/examples/calibration-governor-metadata.ll` | `llvm-as training/llvm/19-hardware-aware/examples/calibration-governor-metadata.ll -o /dev/null` |
| `training/llvm/19-hardware-aware/examples/dragon-egg-flow-execution.ll` | `llvm-as training/llvm/19-hardware-aware/examples/dragon-egg-flow-execution.ll -o /dev/null` |
| `training/llvm/19-hardware-aware/examples/gaadmsf-programming-pulse.ll` | `llvm-as training/llvm/19-hardware-aware/examples/gaadmsf-programming-pulse.ll -o /dev/null` |
| `training/llvm/19-hardware-aware/examples/riscv-extension-lowering-sketch.ll` | `llvm-as training/llvm/19-hardware-aware/examples/riscv-extension-lowering-sketch.ll -o /dev/null` |
| `training/llvm/20-clang-frontend/examples/abi-boundary.ll` | `llvm-as training/llvm/20-clang-frontend/examples/abi-boundary.ll -o /dev/null` |
| `training/llvm/20-clang-frontend/examples/control-flow-lowering.ll` | `llvm-as training/llvm/20-clang-frontend/examples/control-flow-lowering.ll -o /dev/null` |
| `training/llvm/20-clang-frontend/examples/cxx-object-model.ll` | `llvm-as training/llvm/20-clang-frontend/examples/cxx-object-model.ll -o /dev/null` |
| `training/llvm/20-clang-frontend/examples/struct-layout.ll` | `llvm-as training/llvm/20-clang-frontend/examples/struct-layout.ll -o /dev/null` |
| `training/llvm/bcir-mapping/examples/bcir-op-runtime-wrapper.ll` | `llvm-as training/llvm/bcir-mapping/examples/bcir-op-runtime-wrapper.ll -o /dev/null` |
| `training/llvm/bcir-mapping/examples/claim-resource-lookup.ll` | `llvm-as training/llvm/bcir-mapping/examples/claim-resource-lookup.ll -o /dev/null` |
| `training/llvm/bcir-mapping/examples/diagnostic-metadata-preservation.ll` | `llvm-as training/llvm/bcir-mapping/examples/diagnostic-metadata-preservation.ll -o /dev/null` |
| `training/llvm/bcir-mapping/examples/graph-fragment-struct-gep.ll` | `llvm-as training/llvm/bcir-mapping/examples/graph-fragment-struct-gep.ll -o /dev/null` |
| `training/llvm/bcir-mapping/examples/ham-hint-prefetch.ll` | `llvm-as training/llvm/bcir-mapping/examples/ham-hint-prefetch.ll -o /dev/null` |
| `training/llvm/bcir-mapping/examples/hardware-aware-gem-lowering.ll` | `llvm-as training/llvm/bcir-mapping/examples/hardware-aware-gem-lowering.ll -o /dev/null` |
| `training/llvm/bcir-mapping/examples/mixed-stride-byte-offset.ll` | `llvm-as training/llvm/bcir-mapping/examples/mixed-stride-byte-offset.ll -o /dev/null` |
| `training/llvm/bcir-mapping/examples/mixed-stride.ll` | `llvm-as training/llvm/bcir-mapping/examples/mixed-stride.ll -o /dev/null` |
| `training/llvm/bcir-mapping/examples/normal-form-valid.ll` | `llvm-as training/llvm/bcir-mapping/examples/normal-form-valid.ll -o /dev/null` |
| `training/llvm/bcir-mapping/examples/register-binding.ll` | `llvm-as training/llvm/bcir-mapping/examples/register-binding.ll -o /dev/null` |
| `training/llvm/bcir-mapping/examples/vertex-edge-attribute.ll` | `llvm-as training/llvm/bcir-mapping/examples/vertex-edge-attribute.ll -o /dev/null` |

## Non-IR and intentionally excluded artifacts

The following files live next to examples but are not standalone LLVM IR assembly
targets for `verify-examples.sh`:

| File | Kind | Purpose |
|---|---|---|
| `training/llvm/02-types/examples/typed-pointer-before.ll.txt` | Intentionally excluded LLVM IR text fixture | Invalid or legacy IR text fixture; excluded from the known-good manifest. |
| `training/llvm/07-optimization/examples/bolt-layout-demo.c` | C source | Source companion for generating or explaining IR examples. |
| `training/llvm/08-pitfalls/examples/duplicate-symbols.invalid.ll.txt` | Intentionally excluded LLVM IR text fixture | Invalid or legacy IR text fixture; excluded from the known-good manifest. |
| `training/llvm/08-pitfalls/examples/immarg-violation.invalid.ll.txt` | Intentionally excluded LLVM IR text fixture | Invalid or legacy IR text fixture; excluded from the known-good manifest. |
| `training/llvm/08-pitfalls/examples/phi-predecessor-mismatch.invalid.ll.txt` | Intentionally excluded LLVM IR text fixture | Invalid or legacy IR text fixture; excluded from the known-good manifest. |
| `training/llvm/09-vectorization/examples/sum-loop.c` | C source | Source companion for generating or explaining IR examples. |
| `training/llvm/12-backend-jit/examples/lljit-outline.cpp.md` | Markdown sketch | Documentation-only code outline. |
| `training/llvm/12-backend-jit/examples/minimal-instruction.td` | TableGen sketch | TableGen sketch for backend instruction descriptions. |
| `training/llvm/14-mlir-bridge/examples/arith-to-llvm.mlir` | MLIR fragment | MLIR dialect sketch; use MLIR tooling rather than `llvm-as`. |
| `training/llvm/14-mlir-bridge/examples/bcir-dialect-sketch.mlir` | MLIR fragment | MLIR dialect sketch; use MLIR tooling rather than `llvm-as`. |
| `training/llvm/14-mlir-bridge/examples/bcir-vertex-graph.mlir` | MLIR fragment | BCIR vertex graph source sketch; use MLIR tooling rather than `llvm-as`. |
| `training/llvm/14-mlir-bridge/examples/bcir-vertex-graph-lowered-llvm-dialect.mlir` | MLIR fragment | Lowered LLVM-dialect vertex graph sketch; use MLIR tooling rather than `llvm-as`. |
| `training/llvm/14-mlir-bridge/examples/llvm-dialect-call.mlir` | MLIR fragment | MLIR dialect sketch; use MLIR tooling rather than `llvm-as`. |
| `training/llvm/14-mlir-bridge/examples/lowered-llvm-dialect.mlir` | MLIR fragment | MLIR dialect sketch; use MLIR tooling rather than `llvm-as`. |
| `training/llvm/15-binary-analysis/examples/bcsa-feature-sample.csv` | CSV data | Data/sample artifact; not assembled by `llvm-as`. |
| `training/llvm/15-binary-analysis/examples/bcsa-feature-variant-wide.csv` | CSV data | Data/sample artifact; not assembled by `llvm-as`. |
| `training/llvm/15-binary-analysis/examples/dynamic-trace-sample.csv` | CSV data | Data/sample artifact; not assembled by `llvm-as`. |
| `training/llvm/15-binary-analysis/examples/perf-counter-sample.csv` | CSV data | Data/sample artifact; not assembled by `llvm-as`. |
| `training/llvm/15-binary-analysis/examples/side-channel-trace-branchy.csv` | CSV data | Data/sample artifact; not assembled by `llvm-as`. |
| `training/llvm/15-binary-analysis/examples/side-channel-trace-masked.csv` | CSV data | Data/sample artifact; not assembled by `llvm-as`. |
| `training/llvm/17-new-pass-manager/examples/adaptive-pipeline-sketch.cpp.md` | Markdown sketch | Documentation-only modern-pass-manager driver outline. |
| `training/llvm/17-new-pass-manager/examples/bcir-pass-plugin-skeleton.cpp.md` | Markdown sketch | Documentation-only pass plugin skeleton. |
| `training/llvm/19-hardware-aware/examples/mir-register-hint-sketch.mir.txt` | MIR-shaped text sketch | Documentation-only backend sketch; not LLVM IR and intentionally excluded from `llvm-as` verification. |
| `training/llvm/examples/README.md` | Markdown sketch | Documentation-only code outline. |
| `training/llvm/examples/broken-example.ll.txt` | Intentionally excluded LLVM IR text fixture | Invalid or legacy IR text fixture; excluded from the known-good manifest. |
| `training/llvm/07-optimization/examples/bcir-benchmark-pipeline-notes.md` | Analysis-only Markdown notes | Benchmark/pass-pipeline review artifact; not an LLVM IR input. |
| `training/llvm/12-backend-jit/examples/dynamic-kernel-deployment-sketch.md` | JIT-only deployment sketch | ORC deployment/lifetime design artifact; not assembled or executed by portable checks. |
| `training/llvm/12-backend-jit/examples/orc-irtransformlayer-bcir.cpp.md` | JIT-only C++ sketch | Documentation-only ORC `IRTransformLayer` outline. |
| `training/llvm/12-backend-jit/examples/orc-materialization-unit-gaadmsf.cpp.md` | JIT-only C++ sketch | Documentation-only custom `MaterializationUnit` outline. |
| `training/llvm/12-backend-jit/examples/remote-jitlink-heterogeneous-sketch.md` | Remote-target JIT sketch | Host/transport/target-specific deployment artifact; not a portable execution fixture. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/bcir-conversion-pass-skeleton.cpp.md` | MLIR C++ sketch | Documentation-only dialect-conversion pass outline. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/bcir-graph-to-affine.mlir` | MLIR data/sketch artifact | Parsed by the optional MLIR verifier, never by `llvm-as` or portable `llc` smoke. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/bcir-graph-to-llvm-dialect.mlir` | MLIR data/sketch artifact | Parsed by the optional MLIR verifier, never by `llvm-as` or portable `llc` smoke. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/bcir-graph-to-vector.mlir` | MLIR data/sketch artifact | Parsed by the optional MLIR verifier, never by `llvm-as` or portable `llc` smoke. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/bcir-register-prelock-ham-hints.mlir` | MLIR data/sketch artifact | Parsed by the optional MLIR verifier, never by `llvm-as` or portable `llc` smoke. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/memref-type-conversion.mlir` | Tier 4 registered MLIR conversion fixture | The registry lowers memref/func/arith to LLVM dialect, translates it, assembles it, and verifies it. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/tensor-to-buffer.mlir` | Tier 3 registered MLIR conversion fixture | The registry bufferizes tensors to memrefs across function boundaries and requires both boundary casts to be gone; no LLVM translation, so never `llvm-as`. |
| `training/llvm/13-advanced-ir/examples/ub-poison-to-llvm.mlir` | Tier 4 registered MLIR conversion fixture | The registry lowers ub/arith/func to LLVM dialect, translates, assembles and verifies it. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/bcir-transform-sequence.mlir` | Transform dialect sketch | Optional-MLIR syntax fixture; transform execution depends on registered payload dialects/passes. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/bcir-transform-vectorize-then-lower.mlir` | Transform dialect sketch | Optional-MLIR syntax fixture; transform execution depends on registered payload dialects/passes. |
| `training/llvm/bcir-mapping/examples/bcir-operation.prompt.md` | Mapping prompt | Review-only learner prompt paired with BCIR runtime-boundary examples. |
| `training/llvm/bcir-mapping/examples/claim-resource-lookup.bcir.txt` | BCIR source-like data fixture | Checked structurally by `verify-bcir-mapping.sh`; not accepted by `llvm-as`. |
| `training/llvm/bcir-mapping/examples/diagnostic-metadata.prompt.md` | Mapping prompt | Review-only learner prompt for metadata preservation. |
| `training/llvm/bcir-mapping/examples/graph-fragment.bcir.txt` | BCIR source-like data fixture | Checked structurally by `verify-bcir-mapping.sh`; not accepted by `llvm-as`. |
| `training/llvm/bcir-mapping/examples/ham-hint.prompt.md` | Mapping prompt | Review-only learner prompt for advisory hardware/memory hints. |
| `training/llvm/bcir-mapping/examples/mixed-stride-graph.bcir.txt` | BCIR source-like data fixture | Checked structurally by `verify-bcir-mapping.sh`; not accepted by `llvm-as`. |
| `training/llvm/bcir-mapping/examples/normal-form-drift.invalid.ll.txt` | Semantic-only invalid LLVM IR text fixture | Assembles/verifies generically but intentionally violates BCIR normal-form correspondence. |
| `training/llvm/bcir-mapping/examples/normal-form-metadata-loss.invalid.ll.txt` | Semantic-only invalid LLVM IR text fixture | Assembles/verifies generically but intentionally demonstrates required BCIR metadata loss. |

Notes:

- `training/llvm/15-binary-analysis/examples/*.csv` trace, counter, and feature schema samples are data artifacts, not `.ll` assembly targets.
- `training/llvm/14-mlir-bridge/examples/*.mlir` files are MLIR fragments; corresponding lowered `.ll` examples are listed in the standalone manifest when present.
- `training/llvm/examples/broken-example.ll.txt` and `training/llvm/08-pitfalls/examples/*.invalid.ll.txt` are intentionally invalid or text-only fixtures and remain excluded from the known-good manifest.
| `training/llvm/17-new-pass-manager/examples/pass-plugin/BCIRRegisterBinding.cpp` | Buildable C++ pass plugin | Real out-of-tree New PM plugin source; built and exercised by `tools/build-pass-plugin.sh`, never by `llvm-as`. |
| `training/llvm/17-new-pass-manager/examples/pass-plugin/CMakeLists.txt` | Standalone CMake project | Out-of-tree build for the pass plugin; needs LLVM development headers and is not part of the corpus CMake project. |
| `training/llvm/17-new-pass-manager/examples/pass-plugin/README.md` | Chapter-local example manifest | Build, run, and verification-boundary notes for the pass plugin. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/production-dialect/README.md` | Chapter-local example manifest | Build and verification-boundary notes for the standalone MLIR dialect skeleton. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/production-dialect/CMakeLists.txt` | Standalone CMake project | TableGen wiring, dialect library, and tool for the MLIR skeleton; needs an MLIR development install and no corpus gate builds it. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/production-dialect/TrainingOps.td` | TableGen ODS source | Complete ODS dialect definition; reviewed reference code, not TableGen-processed by any corpus gate. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/production-dialect/TrainingDialect.h` | MLIR C++ header | Hand-written header pulling in TableGen-generated `.inc` files; reviewed reference code. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/production-dialect/TrainingDialect.cpp` | MLIR C++ source | Dialect initialization and hand-written verifier; reviewed reference code. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/production-dialect/TrainingToLLVM.cpp` | MLIR C++ source | Partial dialect-conversion pass to the LLVM dialect; reviewed reference code. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/production-dialect/training-opt.cpp` | MLIR C++ source | `mlir-opt`-style driver showing registry and pass registration; reviewed reference code. |
| `training/llvm/18-mlir-lowering-to-llvm/examples/production-dialect/scale.mlir` | MLIR data/sketch artifact | Tier 1 registered unregistered-dialect sketch; parsed by the optional MLIR verifier with `--allow-unregistered-dialect`. |
| `training/llvm/20-clang-frontend/examples/struct-layout.c` | C source | Frontend-lowering input for aggregates, packing, bit-fields, and unions; compiled and claim-checked by `tools/verify-frontend-lowering.py`. |
| `training/llvm/20-clang-frontend/examples/abi-boundary.c` | C source | Frontend-lowering input for argument classification across two target ABIs. |
| `training/llvm/20-clang-frontend/examples/control-flow-lowering.c` | C source | Frontend-lowering input for short-circuit operators, `switch`, loops, and `volatile`. |
| `training/llvm/20-clang-frontend/examples/cxx-object-model.cpp` | C++ source | Frontend-lowering input for mangling, virtual dispatch, templates, and temporaries. |
| `training/llvm/21-performance-methodology/examples/README.md` | Chapter-local example manifest | Benchmark sample schema, fixture inventory, and synthetic-data provenance. |
| `training/llvm/21-performance-methodology/examples/clean-improvement.json` | Benchmark sample data | Synthetic timing series; graded to `improvement` by `tools/verify-benchmark-analysis.py`. |
| `training/llvm/21-performance-methodology/examples/noise-only.json` | Benchmark sample data | Synthetic timing series; the harness self-check, graded to `no-detectable-difference`. |
| `training/llvm/21-performance-methodology/examples/below-resolution.json` | Benchmark sample data | Synthetic timing series; an effect below the rig's resolution, graded to `no-detectable-difference`. |
| `training/llvm/21-performance-methodology/examples/thermal-drift.json` | Benchmark sample data | Synthetic timing series with a drifting baseline, graded to `inconclusive`. |
| `training/llvm/21-performance-methodology/examples/significant-but-trivial.json` | Benchmark sample data | Synthetic timing series; significant and immaterial, graded to `detectable-but-immaterial`. |
| `training/llvm/21-performance-methodology/examples/wall-clock-indicative.json` | Benchmark sample data | Synthetic wall-class timing series; `--gate` must refuse to gate on it. |
| `training/llvm/23-version-movement/examples/conversions-and-shifts.ll` | Standalone example | The cast and shift opcodes the corpus listed but never ran; every asserted result is LLVM 23's own constant folding. |
| `training/llvm/23-version-movement/examples/ptrtoaddr-vs-ptrtoint.ll` | Standalone example (`REQUIRES: llvm >= 23`) | `ptrtoaddr` against `ptrtoint` on an address space whose pointers are wider than their addresses. Not parseable before LLVM 23. |
| `training/llvm/24-mlir-infrastructure/examples/regions-and-blocks.mlir` | MLIR example (Tier 2) | Op/region/block nesting and block arguments in place of phi nodes; the source for the custom-versus-generic and bytecode round-trip checks. |
| `training/llvm/24-mlir-infrastructure/examples/interfaces-inlining.mlir` | MLIR example (Tier 2) | `--inline` reaching `func.func` through `CallOpInterface`/`CallableOpInterface` alone. |
| `training/llvm/24-mlir-infrastructure/examples/unrealized-casts.mlir` | MLIR example (Tier 2) | A cancelling `unrealized_conversion_cast` pair and a lone one; reconciliation folds the first and leaves the second. |
| `training/llvm/09-vectorization/examples/vector-predication-evl.ll` | Standalone example | The `llvm.vp.*` family: the mask and `%evl` operands, an RVV tail-handling loop built on `llvm.experimental.get.vector.length`, and `llvm.masked.load` alongside for contrast. |
