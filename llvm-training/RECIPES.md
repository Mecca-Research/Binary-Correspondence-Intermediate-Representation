# RECIPES — Task to File Map

Use this when you know the task but not the chapter name. Each row gives the
shortest path through the corpus for an agent or reviewer.

| If you want to... | Start with | Then check |
| --- | --- | --- |
| Explain what LLVM IR is and why BCIR lowers to it | [`00-foundations/01-what-is-llvm-ir.md`](00-foundations/01-what-is-llvm-ir.md) | [`00-foundations/03-ir-vs-asm-vs-other-irs.md`](00-foundations/03-ir-vs-asm-vs-other-irs.md) |
| Write a new function with blocks and SSA values | [`01-syntax/01-modules-functions-blocks.md`](01-syntax/01-modules-functions-blocks.md) | [`00-foundations/02-ssa.md`](00-foundations/02-ssa.md), exercises 001-003 |
| Index into structs, arrays, or nested data | [`02-types/02-composite-types.md`](02-types/02-composite-types.md) | Exercises 005 and 006 |
| Migrate typed-pointer examples to opaque pointers | [`02-types/04-opaque-pointer-migration.md`](02-types/04-opaque-pointer-migration.md) | [`02-types/examples/opaque-pointer-after.ll`](02-types/examples/opaque-pointer-after.ll) |
| Add loads, stores, globals, or address spaces | [`04-memory/02-load-store.md`](04-memory/02-load-store.md) | [`04-memory/03-global-variables.md`](04-memory/03-global-variables.md), [`04-memory/04-address-spaces.md`](04-memory/04-address-spaces.md) |
| Add control flow with branches, switches, or indirect branches | [`05-control-flow/`](05-control-flow/) | Exercises 002 and 003 |
| Attach debug, TBAA, profile, or loop metadata | [`06-metadata/01-metadata-basics.md`](06-metadata/01-metadata-basics.md) | Exercise 009 |
| Pick attributes for functions, parameters, or ABI lowering | [`13-advanced-ir/04-attributes.md`](13-advanced-ir/04-attributes.md) | [`04-memory/02-load-store.md`](04-memory/02-load-store.md) for access alignment |
| Run and debug optimization passes | [`07-optimization/01-pass-model.md`](07-optimization/01-pass-model.md) | [`07-optimization/05-debugging-passes.md`](07-optimization/05-debugging-passes.md) |
| Understand why vectorization did or did not happen | [`09-vectorization/01-loop-vectorizer.md`](09-vectorization/01-loop-vectorizer.md) | [`09-vectorization/02-slp-vectorizer.md`](09-vectorization/02-slp-vectorizer.md), [`09-vectorization/03-vector-predication.md`](09-vectorization/03-vector-predication.md) |
| Lower C++/Rust atomics to LLVM atomics | [`11-concurrency/04-memory-model-mapping.md`](11-concurrency/04-memory-model-mapping.md) | [`11-concurrency/01-atomic-orderings.md`](11-concurrency/01-atomic-orderings.md), exercise 008 |
| Diagnose volatile-vs-atomic confusion | [`11-concurrency/03-volatile-vs-atomic.md`](11-concurrency/03-volatile-vs-atomic.md) | [`08-pitfalls/10-volatile-is-not-atomic.md`](08-pitfalls/10-volatile-is-not-atomic.md) |
| Explain the IR-to-machine-code path | [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) | [`12-backend-jit/04-mc-and-relocations.md`](12-backend-jit/04-mc-and-relocations.md) |
| Debug a JIT missing-symbol failure | [`12-backend-jit/03-orc-jit.md`](12-backend-jit/03-orc-jit.md) | [`12-backend-jit/04-mc-and-relocations.md`](12-backend-jit/04-mc-and-relocations.md), [`08-pitfalls/14-orc-jit-symbol-resolution.md`](08-pitfalls/14-orc-jit-symbol-resolution.md) |
| Read or modify TableGen target descriptions | [`12-backend-jit/02-tablegen.md`](12-backend-jit/02-tablegen.md) | [`12-backend-jit/examples/minimal-instruction.td`](12-backend-jit/examples/minimal-instruction.td) |
| Wrap LLVM intrinsics behind BCIR names | [`13-advanced-ir/01-common-intrinsics.md`](13-advanced-ir/01-common-intrinsics.md) | Exercise 009 and exercise 012 |
| Sketch an MLIR bridge for BCIR concepts | [`14-mlir-bridge/04-bcir-as-custom-dialect.md`](14-mlir-bridge/04-bcir-as-custom-dialect.md) | [`14-mlir-bridge/examples/bcir-dialect-sketch.mlir`](14-mlir-bridge/examples/bcir-dialect-sketch.mlir) |
| Validate all known-good standalone examples | [`tools/verify-examples.sh`](tools/verify-examples.sh) | [`tools/README.md`](tools/README.md), [`EXAMPLES.md`](EXAMPLES.md) |
| Evaluate whether crypto-like IR may leak through timing | [`15-binary-analysis/01-microarchitecture-side-channels.md`](15-binary-analysis/01-microarchitecture-side-channels.md) | [`15-binary-analysis/02-dynamic-traces-and-counters.md`](15-binary-analysis/02-dynamic-traces-and-counters.md), [`12-backend-jit/04-mc-and-relocations.md`](12-backend-jit/04-mc-and-relocations.md) |
| Interpret PGO/LTO/BOLT optimized binaries | [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md) | [`06-metadata/03-profile-and-optimization-metadata.md`](06-metadata/03-profile-and-optimization-metadata.md), [`15-binary-analysis/02-dynamic-traces-and-counters.md`](15-binary-analysis/02-dynamic-traces-and-counters.md) |
| Triage BCSA candidates before dense embeddings | [`15-binary-analysis/03-interpretable-bcsa-features.md`](15-binary-analysis/03-interpretable-bcsa-features.md) | [`15-binary-analysis/examples/bcsa-feature-sample.csv`](15-binary-analysis/examples/bcsa-feature-sample.csv), [`INDEX.md`](INDEX.md) |
| Check whether you have consumed the corpus | [`EVAL.md`](EVAL.md) | [`INDEX.md`](INDEX.md) only after answering |
