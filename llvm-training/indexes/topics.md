# Index: By topic

| Topic | File |
|---|---|
| What LLVM IR is, big picture | [`00-foundations/01-what-is-llvm-ir.md`](../00-foundations/01-what-is-llvm-ir.md) |
| SSA form, phi nodes | [`00-foundations/02-ssa.md`](../00-foundations/02-ssa.md) |
| LLVM IR vs assembly, vs GIMPLE/CIL/SPIR-V | [`00-foundations/03-ir-vs-asm-vs-other-irs.md`](../00-foundations/03-ir-vs-asm-vs-other-irs.md) |
| Modules, functions, basic blocks | [`01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) |
| Instruction format, operands | [`01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md) |
| Comments (`;`), metadata (`!N`, `!{...}`) | [`01-syntax/03-comments-metadata.md`](../01-syntax/03-comments-metadata.md) |
| Metadata tags (`!dbg`, `!tbaa`, `!prof`, `!range`, `!nonnull`, `!llvm.loop`) | [`06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md), [`06-metadata/03-profile-and-optimization-metadata.md`](../06-metadata/03-profile-and-optimization-metadata.md) |
| Debug-info nodes (`DI*`) | [`06-metadata/02-debug-info.md`](../06-metadata/02-debug-info.md) |
| Integer types `iN` | [`02-types/01-primitive-types.md`](../02-types/01-primitive-types.md) |
| `float`, `double`, `half`, `bfloat`, `fp128` | [`02-types/01-primitive-types.md`](../02-types/01-primitive-types.md) |
| `void`, `ptr`, `label`, `token`, `metadata` | [`02-types/01-primitive-types.md`](../02-types/01-primitive-types.md) |
| Special types (`token`, `metadata`, `x86_mmx`, `x86_fp80`, `ppc_fp128`) | [`02-types/01-primitive-types.md`](../02-types/01-primitive-types.md) |
| Struct, array, vector | [`02-types/02-composite-types.md`](../02-types/02-composite-types.md) |
| Opaque types, opaque pointers | [`02-types/03-opaque-and-pointer-types.md`](../02-types/03-opaque-and-pointer-types.md) |
| Opaque-pointer migration | [`02-types/04-opaque-pointer-migration.md`](../02-types/04-opaque-pointer-migration.md) |
| Opaque pointer migration from typed pointers | [`02-types/04-opaque-pointer-migration.md`](../02-types/04-opaque-pointer-migration.md) |
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
| Optimization passes | [`07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md), [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) |
| `opt` pipelines | [`07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md), [`07-optimization/04-optimization-levels.md`](../07-optimization/04-optimization-levels.md) |
| Debugging pass pipelines, IR diffs, pass-manager traces | [`07-optimization/05-debugging-passes.md`](../07-optimization/05-debugging-passes.md) |
| Vectorizers | [`09-vectorization/README.md`](../09-vectorization/README.md) |
| Loop Vectorizer auto-vectorization | [`09-vectorization/01-loop-vectorizer.md`](../09-vectorization/01-loop-vectorizer.md) |
| SLP Vectorizer / superword-level parallelism | [`09-vectorization/02-slp-vectorizer.md`](../09-vectorization/02-slp-vectorizer.md) |
| Vector predication, masks, scalable-vector tail handling | [`09-vectorization/03-vector-predication.md`](../09-vectorization/03-vector-predication.md) |
| Vectorization diagnostics (`-Rpass`, `-Rpass-missed`, optimization remarks) | [`09-vectorization/README.md`](../09-vectorization/README.md) |
| Vector IR patterns (`<N x T>`, vector loads/stores, `shufflevector`, reductions) | [`09-vectorization/README.md`](../09-vectorization/README.md) |
| Pitfalls overview | [`08-pitfalls/README.md`](../08-pitfalls/README.md) |
| Nested instruction-as-expression syntax errors | [`08-pitfalls/01-nested-instruction-expressions.md`](../08-pitfalls/01-nested-instruction-expressions.md) |
| PHI node predecessor mismatch | [`08-pitfalls/02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md) |
| Duplicate block labels / SSA names | [`08-pitfalls/03-duplicate-block-labels.md`](../08-pitfalls/03-duplicate-block-labels.md) |
| Duplicate symbol definitions across modules | [`08-pitfalls/04-duplicate-symbols.md`](../08-pitfalls/04-duplicate-symbols.md) |
| Type schema drift across modules | [`08-pitfalls/05-type-schema-drift.md`](../08-pitfalls/05-type-schema-drift.md) |
| `immarg` parameter violations on intrinsics | [`08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md) |
| Advanced IR: common intrinsics (`llvm.memcpy`, overflow, lifetime, prefetch) | [`13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md) |
| Advanced IR: target-specific intrinsics (`llvm.x86.*`, features, portability) | [`13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md) |
| Advanced IR: special types (`token`, `metadata`, `half`, `bfloat`, `x86_amx`, scalable vectors) | [`13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md) |
| Function, parameter, memory-effect, pointer, and ABI attributes | [`13-advanced-ir/04-attributes.md`](../13-advanced-ir/04-attributes.md) |
| MLIR overview: modules, operations, regions, blocks, attributes, types | [`14-mlir-bridge/01-what-is-mlir.md`](../14-mlir-bridge/01-what-is-mlir.md) |
| MLIR dialect design and operation anatomy | [`14-mlir-bridge/02-dialects-and-operations.md`](../14-mlir-bridge/02-dialects-and-operations.md) |
| MLIR lowering to LLVM dialect / LLVM IR | [`14-mlir-bridge/03-lowering-to-llvm-dialect.md`](../14-mlir-bridge/03-lowering-to-llvm-dialect.md), [`14-mlir-bridge/examples/lowered-llvm-dialect.mlir`](../14-mlir-bridge/examples/lowered-llvm-dialect.mlir) |
| BCIR as an MLIR custom dialect | [`14-mlir-bridge/04-bcir-as-custom-dialect.md`](../14-mlir-bridge/04-bcir-as-custom-dialect.md), [`14-mlir-bridge/examples/bcir-dialect-sketch.mlir`](../14-mlir-bridge/examples/bcir-dialect-sketch.mlir) |
| Formal Textmapper grammar | `10-grammar/llvm-ir.tm` |
| Grammar notes / how to use it | [`10-grammar/README.md`](../10-grammar/README.md) |
| Instruction quick reference | [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| Common intrinsics | [`reference/intrinsics.md`](../reference/intrinsics.md) |
| Intrinsics list | [`reference/intrinsics.md`](../reference/intrinsics.md) |
| Backend/codegen terms | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md), [`reference/glossary.md`](../reference/glossary.md) |
| Backend code generation pipeline | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) |
| SelectionDAG instruction selection | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) |
| GlobalISel instruction selection | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) |
| `MachineInstr` machine-code representation | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) |
| Register allocation | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) |
| MC layer / code emission | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md), [`12-backend-jit/04-mc-and-relocations.md`](../12-backend-jit/04-mc-and-relocations.md) |
| Relocations and JIT symbol dependencies | [`12-backend-jit/04-mc-and-relocations.md`](../12-backend-jit/04-mc-and-relocations.md) |
| TableGen | [`12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md) |
| TableGen `.td` target descriptions | [`12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md) |
| `llvm-tblgen` generated include files | [`12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md) |
| ORC/LLJIT | [`12-backend-jit/03-orc-jit.md`](../12-backend-jit/03-orc-jit.md) |
| ORC JIT | [`12-backend-jit/03-orc-jit.md`](../12-backend-jit/03-orc-jit.md) |
| `LLJIT` | [`12-backend-jit/03-orc-jit.md`](../12-backend-jit/03-orc-jit.md) |
| Glossary | [`reference/glossary.md`](../reference/glossary.md) |
