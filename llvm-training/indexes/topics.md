# Index: By topic

| Topic | File |
|---|---|
| Curriculum roadmap and intentionally out-of-scope topics | [`ROADMAP.md`](../ROADMAP.md) |
| Corpus self-test and path prompts | [`EVAL.md`](../EVAL.md) |
| What LLVM IR is, big picture | [`00-foundations/01-what-is-llvm-ir.md`](../00-foundations/01-what-is-llvm-ir.md) |
| SSA form, phi nodes | [`00-foundations/02-ssa.md`](../00-foundations/02-ssa.md) |
| LLVM IR vs assembly, vs GIMPLE/CIL/SPIR-V | [`00-foundations/03-ir-vs-asm-vs-other-irs.md`](../00-foundations/03-ir-vs-asm-vs-other-irs.md) |
| Modules, functions, basic blocks | [`01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) |
| Instruction format, operands | [`01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md) |
| Comments (`;`), metadata (`!N`, `!{...}`) | [`01-syntax/03-comments-metadata.md`](../01-syntax/03-comments-metadata.md) |
| Metadata tags (`!dbg`, `!tbaa`, `!prof`, `!range`, `!nonnull`, `!llvm.loop`, `!type`) | [`06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md), [`06-metadata/03-profile-and-optimization-metadata.md`](../06-metadata/03-profile-and-optimization-metadata.md), [`06-metadata/04-type-metadata-cfi.md`](../06-metadata/04-type-metadata-cfi.md) |
| Debug-info nodes (`DI*`) | [`06-metadata/02-debug-info.md`](../06-metadata/02-debug-info.md) |
| Integer types `iN` | [`02-types/01-primitive-types.md`](../02-types/01-primitive-types.md) |
| `float`, `double`, `half`, `bfloat`, `fp128` | [`02-types/01-primitive-types.md`](../02-types/01-primitive-types.md) |
| `void`, `ptr`, `label`, `token`, `metadata` | [`02-types/01-primitive-types.md`](../02-types/01-primitive-types.md) |
| Special types (`token`, `metadata`, `x86_mmx`, `x86_fp80`, `ppc_fp128`) | [`02-types/01-primitive-types.md`](../02-types/01-primitive-types.md) |
| Struct, array, vector | [`02-types/02-composite-types.md`](../02-types/02-composite-types.md) |
| Opaque types, opaque pointers | [`02-types/03-opaque-and-pointer-types.md`](../02-types/03-opaque-and-pointer-types.md) |
| Opaque-pointer migration dispatcher | [`02-types/04-opaque-pointer-migration.md`](../02-types/04-opaque-pointer-migration.md) |
| Opaque pointer migration patterns (`load`, `store`, `getelementptr`, calls) | [`02-types/05-opaque-pointer-migration-patterns.md`](../02-types/05-opaque-pointer-migration-patterns.md) |
| Opaque pointer migration diagnostics and pitfalls | [`02-types/06-opaque-pointer-migration-diagnostics.md`](../02-types/06-opaque-pointer-migration-diagnostics.md) |
| Opaque pointer migration before/after examples | [`02-types/07-opaque-pointer-migration-examples.md`](../02-types/07-opaque-pointer-migration-examples.md) |
| Integer constants (`i32 42`) | [`03-constants/01-integer.md`](../03-constants/01-integer.md) |
| Floating-point constants (`float 3.14`, hex floats) | [`03-constants/02-floating-point.md`](../03-constants/02-floating-point.md) |
| String constants (`c"...\00"`) | [`03-constants/03-strings.md`](../03-constants/03-strings.md) |
| Global vs local constants, linkage, visibility | [`03-constants/04-global-vs-local.md`](../03-constants/04-global-vs-local.md) |
| `alloca` | [`04-memory/01-alloca.md`](../04-memory/01-alloca.md) |
| `load`, `store`, atomic load/store | [`04-memory/02-load-store.md`](../04-memory/02-load-store.md), [`11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md) |
| Global variables, linkage types, TLS | [`04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) |
| `addrspace(N)`, `addrspacecast` | [`04-memory/04-address-spaces.md`](../04-memory/04-address-spaces.md) |
| Atomics and orderings | [`11-concurrency/01-atomic-orderings.md`](../11-concurrency/01-atomic-orderings.md), [`11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md) |
| Atomic orderings (`unordered`, `monotonic`, `acquire`, `release`, `acq_rel`, `seq_cst`) | [`11-concurrency/01-atomic-orderings.md`](../11-concurrency/01-atomic-orderings.md) |
| Atomic instruction syntax (`load atomic`, `store atomic`, `cmpxchg`, `atomicrmw`, `fence`) | [`11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md) |
| Volatile vs atomic | [`11-concurrency/03-volatile-vs-atomic.md`](../11-concurrency/03-volatile-vs-atomic.md) |
| C++/Rust memory model mapping to LLVM atomics | [`11-concurrency/04-memory-model-mapping.md`](../11-concurrency/04-memory-model-mapping.md) |
| Unconditional `br label %X` | [`05-control-flow/01-unconditional-br.md`](../05-control-flow/01-unconditional-br.md) |
| Conditional `br i1 %c, label %t, label %f` | [`05-control-flow/02-conditional-br.md`](../05-control-flow/02-conditional-br.md) |
| `switch` | [`05-control-flow/03-switch.md`](../05-control-flow/03-switch.md) |
| `indirectbr`, `blockaddress` | [`05-control-flow/04-indirectbr.md`](../05-control-flow/04-indirectbr.md) |
| Metadata syntax (`!0`, `!{...}`, `distinct`, named metadata) | [`06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md) |
| Instruction metadata attachments (`!dbg`, `!tbaa`, `!prof`, `!range`, `!nonnull`, `!llvm.loop`) | [`06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md) |
| Debug-info nodes (`DIFile`, `DICompileUnit`, `DISubprogram`, `DILocation`, `DILocalVariable`) | [`06-metadata/02-debug-info.md`](../06-metadata/02-debug-info.md) |
| Reading `!dbg` locations back to source files and lines | [`06-metadata/02-debug-info.md`](../06-metadata/02-debug-info.md) |
| Profile metadata, branch weights, loop metadata | [`06-metadata/03-profile-and-optimization-metadata.md`](../06-metadata/03-profile-and-optimization-metadata.md) |
| Type metadata, CFI-style checks, and vtable address-point testing | [`06-metadata/04-type-metadata-cfi.md`](../06-metadata/04-type-metadata-cfi.md), [`15-binary-analysis/README.md`](../15-binary-analysis/README.md) |
| Optimization passes | [`07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md), [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) |
| `opt` pipelines | [`07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md), [`07-optimization/04-optimization-levels.md`](../07-optimization/04-optimization-levels.md) |
| Debugging pass pipelines, IR diffs, pass-manager traces | [`07-optimization/05-debugging-passes.md`](../07-optimization/05-debugging-passes.md) |
| PGO, LTO, ThinLTO, and BOLT profile-driven pipelines | [`07-optimization/06-pgo-lto-bolt.md`](../07-optimization/06-pgo-lto-bolt.md) |
| Vectorizers dispatcher | [`09-vectorization/README.md`](../09-vectorization/README.md) |
| Vectorization legality and blockers | [`09-vectorization/04-vectorization-legality.md`](../09-vectorization/04-vectorization-legality.md) |
| Loop Vectorizer auto-vectorization | [`09-vectorization/01-loop-vectorizer.md`](../09-vectorization/01-loop-vectorizer.md) |
| SLP Vectorizer / superword-level parallelism | [`09-vectorization/02-slp-vectorizer.md`](../09-vectorization/02-slp-vectorizer.md) |
| Vector predication, masks, scalable-vector tail handling | [`09-vectorization/03-vector-predication.md`](../09-vectorization/03-vector-predication.md) |
| Vectorization diagnostics (`-Rpass`, `-Rpass-missed`, optimization remarks) | [`09-vectorization/05-example-walkthroughs.md`](../09-vectorization/05-example-walkthroughs.md) |
| Vector IR patterns (`<N x T>`, vector loads/stores, `shufflevector`, reductions) | [`09-vectorization/06-recognizing-vector-ir.md`](../09-vectorization/06-recognizing-vector-ir.md) |
| Masked stores, predication legality, and interleaved/deinterleaved accesses | [`09-vectorization/07-masked-and-interleaved-access.md`](../09-vectorization/07-masked-and-interleaved-access.md) |
| Pitfalls overview | [`08-pitfalls/README.md`](../08-pitfalls/README.md) |
| Nested instruction-as-expression syntax errors | [`08-pitfalls/01-nested-instruction-expressions.md`](../08-pitfalls/01-nested-instruction-expressions.md) |
| PHI node predecessor mismatch | [`08-pitfalls/02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md) |
| Duplicate block labels / SSA names | [`08-pitfalls/03-duplicate-block-labels.md`](../08-pitfalls/03-duplicate-block-labels.md) |
| Duplicate symbol definitions across modules | [`08-pitfalls/04-duplicate-symbols.md`](../08-pitfalls/04-duplicate-symbols.md) |
| Type schema drift across modules | [`08-pitfalls/05-type-schema-drift.md`](../08-pitfalls/05-type-schema-drift.md) |
| `immarg` parameter violations on intrinsics | [`08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md) |
| Sanitizer instrumentation, shadow memory, redzone checks, and `!nosanitize` metadata | [`08-pitfalls/16-sanitizer-instrumentation.md`](../08-pitfalls/16-sanitizer-instrumentation.md), [`08-pitfalls/examples/sanitizer-instrumentation.ll`](../08-pitfalls/examples/sanitizer-instrumentation.ll) |
| Advanced IR: common intrinsics (`llvm.memcpy`, overflow, lifetime, prefetch) | [`13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md) |
| Advanced IR: target-specific intrinsics (`llvm.x86.*`, features, portability) | [`13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md) |
| Advanced IR: special types (`token`, `metadata`, `half`, `bfloat`, `x86_amx`, scalable vectors) | [`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) |
| Function, parameter, memory-effect, pointer, and ABI attributes | [`13-advanced-ir/04-attributes.md`](../13-advanced-ir/04-attributes.md) |
| Poison, `undef`, `freeze`, `noundef`, verifier-valid unsafe patterns, and BCIR safe speculation | [`13-advanced-ir/05-poison-undef-freeze.md`](../13-advanced-ir/05-poison-undef-freeze.md), [`13-advanced-ir/examples/bcir-freeze-safe-speculation.ll`](../13-advanced-ir/examples/bcir-freeze-safe-speculation.ll) |
| Fast-math flags, floating-point comparisons, reductions, reassociation, and vectorization | [`13-advanced-ir/06-fast-math-flags.md`](../13-advanced-ir/06-fast-math-flags.md) |
| Exception handling overview: `invoke`, unwind edges, personalities, and EH pads | [`16-exception-handling/README.md`](../16-exception-handling/README.md), [`16-exception-handling/01-eh-overview.md`](../16-exception-handling/01-eh-overview.md) |
| Itanium EH: `landingpad`, cleanup clauses, and `resume` | [`16-exception-handling/02-itanium-landingpad.md`](../16-exception-handling/02-itanium-landingpad.md), [`16-exception-handling/examples/invoke-landingpad.ll`](../16-exception-handling/examples/invoke-landingpad.ll), [`16-exception-handling/examples/cleanup-resume.ll`](../16-exception-handling/examples/cleanup-resume.ll) |
| Windows EH funclets: `catchswitch`, `catchpad`, `cleanuppad`, `catchret`, `cleanupret`, and `"funclet"` bundles | [`16-exception-handling/03-wineh-funclets.md`](../16-exception-handling/03-wineh-funclets.md), [`16-exception-handling/04-cleanups-and-resume.md`](../16-exception-handling/04-cleanups-and-resume.md), [`16-exception-handling/examples/catchswitch-funclet.ll`](../16-exception-handling/examples/catchswitch-funclet.ll) |
| Operand bundles on `call` and `invoke`; deoptimization, funclets, GC liveness, and ARC attached calls | [`13-advanced-ir/07-operand-bundles.md`](../13-advanced-ir/07-operand-bundles.md), [`13-advanced-ir/examples/operand-bundles-deopt.ll`](../13-advanced-ir/examples/operand-bundles-deopt.ll), [`13-advanced-ir/examples/operand-bundles-funclet.ll`](../13-advanced-ir/examples/operand-bundles-funclet.ll) |
| MLIR overview: modules, operations, regions, blocks, attributes, types | [`14-mlir-bridge/01-what-is-mlir.md`](../14-mlir-bridge/01-what-is-mlir.md) |
| MLIR dialect design and operation anatomy | [`14-mlir-bridge/02-dialects-and-operations.md`](../14-mlir-bridge/02-dialects-and-operations.md) |
| MLIR lowering to LLVM dialect / LLVM IR | [`14-mlir-bridge/03-lowering-to-llvm-dialect.md`](../14-mlir-bridge/03-lowering-to-llvm-dialect.md), [`14-mlir-bridge/examples/lowered-llvm-dialect.mlir`](../14-mlir-bridge/examples/lowered-llvm-dialect.mlir) |
| BCIR as an MLIR custom dialect | [`14-mlir-bridge/04-bcir-as-custom-dialect.md`](../14-mlir-bridge/04-bcir-as-custom-dialect.md), [`14-mlir-bridge/examples/bcir-dialect-sketch.mlir`](../14-mlir-bridge/examples/bcir-dialect-sketch.mlir) |
| MLIR vertex/edge graph lowering walkthrough | [`14-mlir-bridge/05-vertex-graph-lowering.md`](../14-mlir-bridge/05-vertex-graph-lowering.md), [`14-mlir-bridge/examples/bcir-vertex-graph-lowered.ll`](../14-mlir-bridge/examples/bcir-vertex-graph-lowered.ll) |
| BCIR mapping guide dispatcher | [`bcir-mapping/README.md`](../bcir-mapping/README.md) |
| BCIR claim lowering pipeline | [`bcir-mapping/06-claim-lowering-pipeline.md`](../bcir-mapping/06-claim-lowering-pipeline.md), [`bcir-mapping/examples/claim-resource-lookup.ll`](../bcir-mapping/examples/claim-resource-lookup.ll) |
| GAADMSF graph/data movement lowering | [`bcir-mapping/07-gaadmsf-operations.md`](../bcir-mapping/07-gaadmsf-operations.md), [`bcir-mapping/examples/graph-fragment-struct-gep.ll`](../bcir-mapping/examples/graph-fragment-struct-gep.ll) |
| Dragon Egg runtime-owned operation wrappers | [`bcir-mapping/08-dragon-egg-operations.md`](../bcir-mapping/08-dragon-egg-operations.md), [`bcir-mapping/examples/bcir-op-runtime-wrapper.ll`](../bcir-mapping/examples/bcir-op-runtime-wrapper.ll) |
| BCIR runtime call boundaries | [`bcir-mapping/09-runtime-call-boundaries.md`](../bcir-mapping/09-runtime-call-boundaries.md) |
| BCIR metadata, HAM hints, and diagnostics | [`bcir-mapping/10-metadata-and-diagnostics.md`](../bcir-mapping/10-metadata-and-diagnostics.md), [`bcir-mapping/examples/ham-hint-prefetch.ll`](../bcir-mapping/examples/ham-hint-prefetch.ll), [`bcir-mapping/examples/diagnostic-metadata-preservation.ll`](../bcir-mapping/examples/diagnostic-metadata-preservation.ll) |
| Formal Textmapper grammar | `10-grammar/llvm-ir.tm` |
| Grammar notes / how to use it | [`10-grammar/README.md`](../10-grammar/README.md) |
| Quick reference dispatcher | [`quickref/README.md`](../quickref/README.md) |
| Advanced IR quick reference | [`quickref/advanced-ir.md`](../quickref/advanced-ir.md) |
| MLIR bridge quick reference | [`quickref/mlir-bridge.md`](../quickref/mlir-bridge.md) |
| Instruction quick reference | [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| Common intrinsics declaration rules | [`reference/intrinsics.md`](../reference/intrinsics.md) |
| Intrinsics quick reference | [`reference/intrinsics-quickref.md`](../reference/intrinsics-quickref.md) |
| Backend/codegen terms | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md), [`reference/glossary.md`](../reference/glossary.md) |
| Backend code generation pipeline | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) |
| SelectionDAG instruction selection | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) |
| GlobalISel instruction selection | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) |
| `MachineInstr` machine-code representation | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) |
| Register allocation | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) |
| MC layer / code emission | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md), [`12-backend-jit/04-mc-and-relocations.md`](../12-backend-jit/04-mc-and-relocations.md) |
| Relocations and JIT symbol dependencies | [`12-backend-jit/04-mc-and-relocations.md`](../12-backend-jit/04-mc-and-relocations.md), [`12-backend-jit/05-orc-layers.md`](../12-backend-jit/05-orc-layers.md) |
| TableGen | [`12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md) |
| TableGen `.td` target descriptions | [`12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md) |
| `llvm-tblgen` generated include files | [`12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md) |
| ORC/LLJIT | [`12-backend-jit/03-orc-jit.md`](../12-backend-jit/03-orc-jit.md), [`12-backend-jit/05-orc-layers.md`](../12-backend-jit/05-orc-layers.md) |
| ORC JIT | [`12-backend-jit/03-orc-jit.md`](../12-backend-jit/03-orc-jit.md), [`12-backend-jit/05-orc-layers.md`](../12-backend-jit/05-orc-layers.md) |
| ORC layers (`ExecutionSession`, `JITDylib`, compile/object/transform layers) | [`12-backend-jit/05-orc-layers.md`](../12-backend-jit/05-orc-layers.md) |
| ORC materialization responsibility | [`12-backend-jit/05-orc-layers.md`](../12-backend-jit/05-orc-layers.md) |
| ORC symbol interning and resolution | [`12-backend-jit/05-orc-layers.md`](../12-backend-jit/05-orc-layers.md), [`08-pitfalls/14-orc-jit-symbol-resolution.md`](../08-pitfalls/14-orc-jit-symbol-resolution.md) |
| JITLink and ORC object linking | [`12-backend-jit/05-orc-layers.md`](../12-backend-jit/05-orc-layers.md), [`12-backend-jit/04-mc-and-relocations.md`](../12-backend-jit/04-mc-and-relocations.md) |
| `LLJIT` | [`12-backend-jit/03-orc-jit.md`](../12-backend-jit/03-orc-jit.md) |
| Glossary | [`reference/glossary.md`](../reference/glossary.md) |
| Microarchitecture side-channel review | [`15-binary-analysis/01-microarchitecture-side-channels.md`](../15-binary-analysis/01-microarchitecture-side-channels.md) |
| Dynamic traces and hardware counters | [`15-binary-analysis/02-dynamic-traces-and-counters.md`](../15-binary-analysis/02-dynamic-traces-and-counters.md) |
| Binary evidence for IR-level CFI hardening | [`06-metadata/04-type-metadata-cfi.md`](../06-metadata/04-type-metadata-cfi.md), [`15-binary-analysis/02-dynamic-traces-and-counters.md`](../15-binary-analysis/02-dynamic-traces-and-counters.md) |
| Interpretable BCSA feature triage | [`15-binary-analysis/03-interpretable-bcsa-features.md`](../15-binary-analysis/03-interpretable-bcsa-features.md) |
| Advanced exercise set: BCIR lowering, MLIR review, backend/JIT, custom pass invariants, graph metadata, GAADMSF debugging | [`exercises/README.md`](../exercises/README.md) |
