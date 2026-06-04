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

| File | Expected command |
|---|---|
| `llvm-training/00-foundations/examples/simple-add.ll` | `llvm-as llvm-training/00-foundations/examples/simple-add.ll -o /dev/null` |
| `llvm-training/00-foundations/examples/ssa-phi.ll` | `llvm-as llvm-training/00-foundations/examples/ssa-phi.ll -o /dev/null` |
| `llvm-training/01-syntax/examples/module-anatomy.ll` | `llvm-as llvm-training/01-syntax/examples/module-anatomy.ll -o /dev/null` |
| `llvm-training/02-types/examples/opaque-pointer-after.ll` | `llvm-as llvm-training/02-types/examples/opaque-pointer-after.ll -o /dev/null` |
| `llvm-training/02-types/examples/types-cookbook.ll` | `llvm-as llvm-training/02-types/examples/types-cookbook.ll -o /dev/null` |
| `llvm-training/03-constants/examples/constants-cookbook.ll` | `llvm-as llvm-training/03-constants/examples/constants-cookbook.ll -o /dev/null` |
| `llvm-training/04-memory/examples/memory-cookbook.ll` | `llvm-as llvm-training/04-memory/examples/memory-cookbook.ll -o /dev/null` |
| `llvm-training/05-control-flow/examples/control-flow-cookbook.ll` | `llvm-as llvm-training/05-control-flow/examples/control-flow-cookbook.ll -o /dev/null` |
| `llvm-training/06-metadata/examples/debug-location.ll` | `llvm-as llvm-training/06-metadata/examples/debug-location.ll -o /dev/null` |
| `llvm-training/06-metadata/examples/loop-metadata.ll` | `llvm-as llvm-training/06-metadata/examples/loop-metadata.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/dead-code-after-adce.ll` | `llvm-as llvm-training/07-optimization/examples/dead-code-after-adce.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/dead-code-before.ll` | `llvm-as llvm-training/07-optimization/examples/dead-code-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/gvn-after.ll` | `llvm-as llvm-training/07-optimization/examples/gvn-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/gvn-before.ll` | `llvm-as llvm-training/07-optimization/examples/gvn-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/instcombine-after.ll` | `llvm-as llvm-training/07-optimization/examples/instcombine-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/instcombine-before.ll` | `llvm-as llvm-training/07-optimization/examples/instcombine-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-before.ll` | `llvm-as llvm-training/07-optimization/examples/loop-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-rotate-after.ll` | `llvm-as llvm-training/07-optimization/examples/loop-rotate-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-rotate-before.ll` | `llvm-as llvm-training/07-optimization/examples/loop-rotate-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-unroll-after.ll` | `llvm-as llvm-training/07-optimization/examples/loop-unroll-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/loop-unroll-before.ll` | `llvm-as llvm-training/07-optimization/examples/loop-unroll-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/mem2reg-after.ll` | `llvm-as llvm-training/07-optimization/examples/mem2reg-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/mem2reg-before.ll` | `llvm-as llvm-training/07-optimization/examples/mem2reg-before.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/simplifycfg-after.ll` | `llvm-as llvm-training/07-optimization/examples/simplifycfg-after.ll -o /dev/null` |
| `llvm-training/07-optimization/examples/simplifycfg-before.ll` | `llvm-as llvm-training/07-optimization/examples/simplifycfg-before.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/not-vectorizable-call.ll` | `llvm-as llvm-training/09-vectorization/examples/not-vectorizable-call.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/not-vectorizable-dependency.ll` | `llvm-as llvm-training/09-vectorization/examples/not-vectorizable-dependency.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/slp-scalars-after-slp.ll` | `llvm-as llvm-training/09-vectorization/examples/slp-scalars-after-slp.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/slp-scalars-before.ll` | `llvm-as llvm-training/09-vectorization/examples/slp-scalars-before.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/slp-scalars.ll` | `llvm-as llvm-training/09-vectorization/examples/slp-scalars.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/sum-loop-after-loop-vectorize.ll` | `llvm-as llvm-training/09-vectorization/examples/sum-loop-after-loop-vectorize.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/sum-loop-before.ll` | `llvm-as llvm-training/09-vectorization/examples/sum-loop-before.ll -o /dev/null` |
| `llvm-training/09-vectorization/examples/sum-loop.ll` | `llvm-as llvm-training/09-vectorization/examples/sum-loop.ll -o /dev/null` |
| `llvm-training/11-concurrency/examples/atomic-counter.ll` | `llvm-as llvm-training/11-concurrency/examples/atomic-counter.ll -o /dev/null` |
| `llvm-training/11-concurrency/examples/cmpxchg-loop.ll` | `llvm-as llvm-training/11-concurrency/examples/cmpxchg-loop.ll -o /dev/null` |
| `llvm-training/11-concurrency/examples/fence.ll` | `llvm-as llvm-training/11-concurrency/examples/fence.ll -o /dev/null` |
| `llvm-training/12-backend-jit/examples/codegen-input.ll` | `llvm-as llvm-training/12-backend-jit/examples/codegen-input.ll -o /dev/null` |
| `llvm-training/13-advanced-ir/examples/memcpy.ll` | `llvm-as llvm-training/13-advanced-ir/examples/memcpy.ll -o /dev/null` |
| `llvm-training/13-advanced-ir/examples/overflow-intrinsic.ll` | `llvm-as llvm-training/13-advanced-ir/examples/overflow-intrinsic.ll -o /dev/null` |
| `llvm-training/13-advanced-ir/examples/token-outline.ll` | `llvm-as llvm-training/13-advanced-ir/examples/token-outline.ll -o /dev/null` |
| `llvm-training/14-mlir-bridge/examples/bcir-final.ll` | `llvm-as llvm-training/14-mlir-bridge/examples/bcir-final.ll -o /dev/null` |
| `llvm-training/bcir-mapping/examples/mixed-stride.ll` | `llvm-as llvm-training/bcir-mapping/examples/mixed-stride.ll -o /dev/null` |
| `llvm-training/bcir-mapping/examples/register-binding.ll` | `llvm-as llvm-training/bcir-mapping/examples/register-binding.ll -o /dev/null` |
| `llvm-training/bcir-mapping/examples/vertex-edge-attribute.ll` | `llvm-as llvm-training/bcir-mapping/examples/vertex-edge-attribute.ll -o /dev/null` |
