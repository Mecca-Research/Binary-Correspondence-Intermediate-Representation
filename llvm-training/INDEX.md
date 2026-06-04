# INDEX — Topic / Symbol → File

Agent: this is your entry point. Find your topic, jump to the file.

## By topic

| Topic | File |
|---|---|
| What LLVM IR is, big picture | [`00-foundations/01-what-is-llvm-ir.md`](00-foundations/01-what-is-llvm-ir.md) |
| SSA form, phi nodes | [`00-foundations/02-ssa.md`](00-foundations/02-ssa.md) |
| LLVM IR vs assembly, vs GIMPLE/CIL/SPIR-V | [`00-foundations/03-ir-vs-asm-vs-other-irs.md`](00-foundations/03-ir-vs-asm-vs-other-irs.md) |
| Modules, functions, basic blocks | [`01-syntax/01-modules-functions-blocks.md`](01-syntax/01-modules-functions-blocks.md) |
| Instruction format, operands | [`01-syntax/02-instruction-format.md`](01-syntax/02-instruction-format.md) |
| Comments (`;`), metadata (`!N`, `!{...}`) | [`01-syntax/03-comments-metadata.md`](01-syntax/03-comments-metadata.md) |
| Metadata tags (`!dbg`, `!tbaa`, `!prof`, `!range`, `!nonnull`, `!llvm.loop`) | [`06-metadata/01-metadata-basics.md`](06-metadata/01-metadata-basics.md), [`06-metadata/03-profile-and-optimization-metadata.md`](06-metadata/03-profile-and-optimization-metadata.md) |
| Debug-info nodes (`DI*`) | [`06-metadata/02-debug-info.md`](06-metadata/02-debug-info.md) |
| Integer types `iN` | [`02-types/01-primitive-types.md`](02-types/01-primitive-types.md) |
| `float`, `double`, `half`, `bfloat`, `fp128` | [`02-types/01-primitive-types.md`](02-types/01-primitive-types.md) |
| `void`, `ptr`, `label`, `token`, `metadata` | [`02-types/01-primitive-types.md`](02-types/01-primitive-types.md) |
| Special types (`token`, `metadata`, `x86_mmx`, `x86_fp80`, `ppc_fp128`) | [`02-types/01-primitive-types.md`](02-types/01-primitive-types.md) |
| Struct, array, vector | [`02-types/02-composite-types.md`](02-types/02-composite-types.md) |
| Opaque types, opaque pointers | [`02-types/03-opaque-and-pointer-types.md`](02-types/03-opaque-and-pointer-types.md) |
| Opaque-pointer migration | [`02-types/04-opaque-pointer-migration.md`](02-types/04-opaque-pointer-migration.md) |
| Opaque pointer migration from typed pointers | [`02-types/04-opaque-pointer-migration.md`](02-types/04-opaque-pointer-migration.md) |
| Integer constants (`i32 42`) | [`03-constants/01-integer.md`](03-constants/01-integer.md) |
| Floating-point constants (`float 3.14`, hex floats) | [`03-constants/02-floating-point.md`](03-constants/02-floating-point.md) |
| String constants (`c"...\00"`) | [`03-constants/03-strings.md`](03-constants/03-strings.md) |
| Global vs local constants, linkage, visibility | [`03-constants/04-global-vs-local.md`](03-constants/04-global-vs-local.md) |
| `alloca` | [`04-memory/01-alloca.md`](04-memory/01-alloca.md) |
| `load`, `store`, atomic load/store | [`04-memory/02-load-store.md`](04-memory/02-load-store.md), [`11-concurrency/02-atomic-instructions.md`](11-concurrency/02-atomic-instructions.md) |
| Global variables, linkage types, TLS | [`04-memory/03-global-variables.md`](04-memory/03-global-variables.md) |
| `addrspace(N)`, `addrspacecast` | [`04-memory/04-address-spaces.md`](04-memory/04-address-spaces.md) |
| Atomics and orderings | [`11-concurrency/01-atomic-orderings.md`](11-concurrency/01-atomic-orderings.md), [`11-concurrency/02-atomic-instructions.md`](11-concurrency/02-atomic-instructions.md) |
| Atomic orderings (`unordered`, `monotonic`, `acquire`, `release`, `acq_rel`, `seq_cst`) | [`11-concurrency/01-atomic-orderings.md`](11-concurrency/01-atomic-orderings.md) |
| Atomic instruction syntax (`load atomic`, `store atomic`, `cmpxchg`, `atomicrmw`, `fence`) | [`11-concurrency/02-atomic-instructions.md`](11-concurrency/02-atomic-instructions.md) |
| Volatile vs atomic | [`11-concurrency/03-volatile-vs-atomic.md`](11-concurrency/03-volatile-vs-atomic.md) |
| Unconditional `br label %X` | [`05-control-flow/01-unconditional-br.md`](05-control-flow/01-unconditional-br.md) |
| Conditional `br i1 %c, label %t, label %f` | [`05-control-flow/02-conditional-br.md`](05-control-flow/02-conditional-br.md) |
| `switch` | [`05-control-flow/03-switch.md`](05-control-flow/03-switch.md) |
| `indirectbr`, `blockaddress` | [`05-control-flow/04-indirectbr.md`](05-control-flow/04-indirectbr.md) |
| Metadata syntax (`!0`, `!{...}`, `distinct`, named metadata) | [`06-metadata/01-metadata-basics.md`](06-metadata/01-metadata-basics.md) |
| Instruction metadata attachments (`!dbg`, `!tbaa`, `!prof`, `!range`, `!nonnull`, `!llvm.loop`) | [`06-metadata/01-metadata-basics.md`](06-metadata/01-metadata-basics.md) |
| Debug-info nodes (`DIFile`, `DICompileUnit`, `DISubprogram`, `DILocation`, `DILocalVariable`) | [`06-metadata/02-debug-info.md`](06-metadata/02-debug-info.md) |
| Reading `!dbg` locations back to source files and lines | [`06-metadata/02-debug-info.md`](06-metadata/02-debug-info.md) |
| Profile metadata, branch weights, loop metadata | [`06-metadata/03-profile-and-optimization-metadata.md`](06-metadata/03-profile-and-optimization-metadata.md) |
| Optimization passes | [`07-optimization/01-pass-model.md`](07-optimization/01-pass-model.md), [`07-optimization/03-common-transform-passes.md`](07-optimization/03-common-transform-passes.md) |
| `opt` pipelines | [`07-optimization/01-pass-model.md`](07-optimization/01-pass-model.md), [`07-optimization/04-optimization-levels.md`](07-optimization/04-optimization-levels.md) |
| Vectorizers | [`09-vectorization/README.md`](09-vectorization/README.md) |
| Loop Vectorizer auto-vectorization | [`09-vectorization/README.md`](09-vectorization/README.md) |
| SLP Vectorizer / superword-level parallelism | [`09-vectorization/README.md`](09-vectorization/README.md) |
| Vectorization diagnostics (`-Rpass`, `-Rpass-missed`, optimization remarks) | [`09-vectorization/README.md`](09-vectorization/README.md) |
| Vector IR patterns (`<N x T>`, vector loads/stores, `shufflevector`, reductions) | [`09-vectorization/README.md`](09-vectorization/README.md) |
| Pitfalls overview | [`08-pitfalls/README.md`](08-pitfalls/README.md) |
| Nested instruction-as-expression syntax errors | [`08-pitfalls/01-nested-instruction-expressions.md`](08-pitfalls/01-nested-instruction-expressions.md) |
| PHI node predecessor mismatch | [`08-pitfalls/02-phi-predecessor-mismatch.md`](08-pitfalls/02-phi-predecessor-mismatch.md) |
| Duplicate block labels / SSA names | [`08-pitfalls/03-duplicate-block-labels.md`](08-pitfalls/03-duplicate-block-labels.md) |
| Duplicate symbol definitions across modules | [`08-pitfalls/04-duplicate-symbols.md`](08-pitfalls/04-duplicate-symbols.md) |
| Type schema drift across modules | [`08-pitfalls/05-type-schema-drift.md`](08-pitfalls/05-type-schema-drift.md) |
| `immarg` parameter violations on intrinsics | [`08-pitfalls/06-immarg-violation.md`](08-pitfalls/06-immarg-violation.md) |
| Advanced IR: common intrinsics (`llvm.memcpy`, overflow, lifetime, prefetch) | [`13-advanced-ir/01-common-intrinsics.md`](13-advanced-ir/01-common-intrinsics.md) |
| Advanced IR: target-specific intrinsics (`llvm.x86.*`, features, portability) | [`13-advanced-ir/02-target-specific-intrinsics.md`](13-advanced-ir/02-target-specific-intrinsics.md) |
| Advanced IR: special types (`token`, `metadata`, `half`, `bfloat`, `x86_amx`, scalable vectors) | [`13-advanced-ir/03-special-types-and-tokens.md`](13-advanced-ir/03-special-types-and-tokens.md) |
| MLIR overview: modules, operations, regions, blocks, attributes, types | [`14-mlir-bridge/01-what-is-mlir.md`](14-mlir-bridge/01-what-is-mlir.md) |
| MLIR dialect design and operation anatomy | [`14-mlir-bridge/02-dialects-and-operations.md`](14-mlir-bridge/02-dialects-and-operations.md) |
| MLIR lowering to LLVM dialect / LLVM IR | [`14-mlir-bridge/03-lowering-to-llvm-dialect.md`](14-mlir-bridge/03-lowering-to-llvm-dialect.md), [`14-mlir-bridge/examples/lowered-llvm-dialect.mlir`](14-mlir-bridge/examples/lowered-llvm-dialect.mlir) |
| BCIR as an MLIR custom dialect | [`14-mlir-bridge/04-bcir-as-custom-dialect.md`](14-mlir-bridge/04-bcir-as-custom-dialect.md), [`14-mlir-bridge/examples/bcir-dialect-sketch.mlir`](14-mlir-bridge/examples/bcir-dialect-sketch.mlir) |
| MLIR bridge type conversion and materialization | [`14-mlir-bridge/05-type-conversion-and-materialization.md`](14-mlir-bridge/05-type-conversion-and-materialization.md), [`14-mlir-bridge/examples/bcir-type-conversion.mlir`](14-mlir-bridge/examples/bcir-type-conversion.mlir) |
| MLIR conversion patterns and legality stages | [`14-mlir-bridge/06-conversion-patterns.md`](14-mlir-bridge/06-conversion-patterns.md), [`14-mlir-bridge/examples/bcir-conversion-pipeline.mlir`](14-mlir-bridge/examples/bcir-conversion-pipeline.mlir) |
| MLIR pass pipeline diagnostics and final LLVM IR | [`14-mlir-bridge/07-pass-pipeline-and-diagnostics.md`](14-mlir-bridge/07-pass-pipeline-and-diagnostics.md), [`14-mlir-bridge/08-end-to-end-bcir-lowering.md`](14-mlir-bridge/08-end-to-end-bcir-lowering.md), [`14-mlir-bridge/examples/bcir-final.ll`](14-mlir-bridge/examples/bcir-final.ll) |
| Formal Textmapper grammar | `10-grammar/llvm-ir.tm` |
| Grammar notes / how to use it | [`10-grammar/README.md`](10-grammar/README.md) |
| Instruction quick reference | [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| Common intrinsics | [`reference/intrinsics.md`](reference/intrinsics.md) |
| Intrinsics list | [`reference/intrinsics.md`](reference/intrinsics.md) |
| Backend/codegen terms | [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md), [`reference/glossary.md`](reference/glossary.md) |
| Backend code generation pipeline | [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) |
| SelectionDAG instruction selection | [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) |
| GlobalISel instruction selection | [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) |
| `MachineInstr` machine-code representation | [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) |
| Register allocation | [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) |
| MC layer / code emission | [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) |
| TableGen | [`12-backend-jit/02-tablegen.md`](12-backend-jit/02-tablegen.md) |
| TableGen `.td` target descriptions | [`12-backend-jit/02-tablegen.md`](12-backend-jit/02-tablegen.md) |
| `llvm-tblgen` generated include files | [`12-backend-jit/02-tablegen.md`](12-backend-jit/02-tablegen.md) |
| ORC/LLJIT | [`12-backend-jit/03-orc-jit.md`](12-backend-jit/03-orc-jit.md) |
| ORC JIT | [`12-backend-jit/03-orc-jit.md`](12-backend-jit/03-orc-jit.md) |
| `LLJIT` | [`12-backend-jit/03-orc-jit.md`](12-backend-jit/03-orc-jit.md) |
| Glossary | [`reference/glossary.md`](reference/glossary.md) |

## By instruction (most common)

| Instruction | Read |
|---|---|
| `add`, `sub`, `mul`, `sdiv`, `udiv`, `srem`, `urem` | [`01-syntax/02-instruction-format.md`](01-syntax/02-instruction-format.md), [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `fadd`, `fsub`, `fmul`, `fdiv`, `frem`, `fneg` | [`01-syntax/02-instruction-format.md`](01-syntax/02-instruction-format.md), [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `and`, `or`, `xor`, `shl`, `lshr`, `ashr` | [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `alloca` | [`04-memory/01-alloca.md`](04-memory/01-alloca.md) |
| `load`, `store` | [`04-memory/02-load-store.md`](04-memory/02-load-store.md), [`02-types/04-opaque-pointer-migration.md`](02-types/04-opaque-pointer-migration.md) |
| `getelementptr` (GEP) | [`02-types/02-composite-types.md`](02-types/02-composite-types.md), [`02-types/04-opaque-pointer-migration.md`](02-types/04-opaque-pointer-migration.md), [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `br`, `switch`, `indirectbr`, `ret`, `unreachable` | `05-control-flow/` (all four files) |
| `phi` | [`00-foundations/02-ssa.md`](00-foundations/02-ssa.md) |
| `icmp`, `fcmp` | [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `select` | [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `call`, `invoke`, `callbr` | [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `atomicrmw`, `cmpxchg`, `fence` | [`11-concurrency/02-atomic-instructions.md`](11-concurrency/02-atomic-instructions.md), [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `extractvalue`, `insertvalue`, `extractelement`, `insertelement`, `shufflevector` | [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `trunc`, `zext`, `sext`, `fptrunc`, `fpext`, `fptoui`, `fptosi`, `uitofp`, `sitofp` | [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `bitcast`, `addrspacecast`, `inttoptr`, `ptrtoint` | [`02-types/04-opaque-pointer-migration.md`](02-types/04-opaque-pointer-migration.md), [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `landingpad`, `catchpad`, `cleanuppad`, `catchswitch` | [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |

## By optimizer pass / `opt` flag

| Name or flag | Kind | Read |
|---|---|---|
| `opt -passes=verify` | Utility/checking pipeline | [`07-optimization/01-pass-model.md`](07-optimization/01-pass-model.md), [`07-optimization/02-common-analysis-passes.md`](07-optimization/02-common-analysis-passes.md) |
| `opt -S` | Textual IR output flag | [`07-optimization/01-pass-model.md`](07-optimization/01-pass-model.md), [`07-optimization/04-optimization-levels.md`](07-optimization/04-optimization-levels.md) |
| `-disable-output` | Suppress output for check/print workflows | [`07-optimization/01-pass-model.md`](07-optimization/01-pass-model.md), [`07-optimization/02-common-analysis-passes.md`](07-optimization/02-common-analysis-passes.md) |
| `mem2reg` | Transform | [`07-optimization/03-common-transform-passes.md`](07-optimization/03-common-transform-passes.md) |
| `instcombine` | Transform | [`07-optimization/03-common-transform-passes.md`](07-optimization/03-common-transform-passes.md) |
| `simplifycfg` | Transform | [`07-optimization/03-common-transform-passes.md`](07-optimization/03-common-transform-passes.md) |
| `adce` | Transform | [`07-optimization/03-common-transform-passes.md`](07-optimization/03-common-transform-passes.md) |
| `gvn` | Transform | [`07-optimization/03-common-transform-passes.md`](07-optimization/03-common-transform-passes.md) |
| `loop-unroll` | Transform | [`07-optimization/03-common-transform-passes.md`](07-optimization/03-common-transform-passes.md) |
| alias analysis | Analysis family | [`07-optimization/02-common-analysis-passes.md`](07-optimization/02-common-analysis-passes.md) |
| CFG printing/viewing | Utility/analysis inspection | [`07-optimization/02-common-analysis-passes.md`](07-optimization/02-common-analysis-passes.md) |
| loop analysis | Analysis | [`07-optimization/02-common-analysis-passes.md`](07-optimization/02-common-analysis-passes.md) |
| scalar evolution / SCEV | Analysis | [`07-optimization/02-common-analysis-passes.md`](07-optimization/02-common-analysis-passes.md) |
| `default<O1>`, `default<O2>`, `default<O3>`, `default<Os>`, `default<Oz>` | Predefined new-PM pipelines | [`07-optimization/04-optimization-levels.md`](07-optimization/04-optimization-levels.md) |

## By symbol

| Symbol | Means | See |
|---|---|---|
| `%foo`, `%42` | Local (function-scope) identifier | [`01-syntax/01-modules-functions-blocks.md`](01-syntax/01-modules-functions-blocks.md) |
| `@foo`, `@42` | Global identifier | [`01-syntax/01-modules-functions-blocks.md`](01-syntax/01-modules-functions-blocks.md) |
| `!N`, `!"str"`, `!{...}` | Metadata | [`01-syntax/03-comments-metadata.md`](01-syntax/03-comments-metadata.md), [`06-metadata/01-metadata-basics.md`](06-metadata/01-metadata-basics.md) |
| `#N` | Attribute group ID | [`reference/glossary.md`](reference/glossary.md) |
| `$foo` | Comdat name | [`reference/glossary.md`](reference/glossary.md) |
| `i1`, `i8`, `i32`, `i64`, `iN` | Integer of N bits | [`02-types/01-primitive-types.md`](02-types/01-primitive-types.md) |
| `half`, `bfloat` | 16-bit floating-point formats with different semantics | [`02-types/01-primitive-types.md`](02-types/01-primitive-types.md), [`13-advanced-ir/03-special-types-and-tokens.md`](13-advanced-ir/03-special-types-and-tokens.md) |
| `token` | Opaque control value for EH/coroutine/statepoint-like IR | [`02-types/01-primitive-types.md`](02-types/01-primitive-types.md), [`13-advanced-ir/03-special-types-and-tokens.md`](13-advanced-ir/03-special-types-and-tokens.md) |
| `metadata` | Metadata operand type for debug/analysis intrinsics | [`06-metadata/01-metadata-basics.md`](06-metadata/01-metadata-basics.md), [`13-advanced-ir/03-special-types-and-tokens.md`](13-advanced-ir/03-special-types-and-tokens.md) |
| `x86_amx` | X86 AMX target extension tile type | [`13-advanced-ir/03-special-types-and-tokens.md`](13-advanced-ir/03-special-types-and-tokens.md) |
| `<vscale x N x T>` | Scalable vector type | [`09-vectorization/README.md`](09-vectorization/README.md), [`13-advanced-ir/03-special-types-and-tokens.md`](13-advanced-ir/03-special-types-and-tokens.md) |
| `ptr` | Generic pointer (opaque) | [`02-types/03-opaque-and-pointer-types.md`](02-types/03-opaque-and-pointer-types.md), [`02-types/04-opaque-pointer-migration.md`](02-types/04-opaque-pointer-migration.md) |
| `;` | Comment to end of line | [`01-syntax/03-comments-metadata.md`](01-syntax/03-comments-metadata.md) |
| `c"..."` | C-style char array constant | [`03-constants/03-strings.md`](03-constants/03-strings.md) |

## By intrinsic / special type

| Name | Means | See |
|---|---|---|
| `llvm.memcpy.*` | Non-overlapping memory copy intrinsic | [`13-advanced-ir/01-common-intrinsics.md`](13-advanced-ir/01-common-intrinsics.md), [`reference/intrinsics.md`](reference/intrinsics.md) |
| `llvm.memmove.*` | Overlap-safe memory copy intrinsic | [`13-advanced-ir/01-common-intrinsics.md`](13-advanced-ir/01-common-intrinsics.md), [`reference/intrinsics.md`](reference/intrinsics.md) |
| `llvm.memset.*` | Byte-fill memory intrinsic | [`13-advanced-ir/01-common-intrinsics.md`](13-advanced-ir/01-common-intrinsics.md), [`reference/intrinsics.md`](reference/intrinsics.md) |
| `llvm.uadd.with.overflow.*` | Unsigned checked addition; returns `{T, i1}` | [`13-advanced-ir/01-common-intrinsics.md`](13-advanced-ir/01-common-intrinsics.md), [`13-advanced-ir/examples/overflow-intrinsic.ll`](13-advanced-ir/examples/overflow-intrinsic.ll) |
| `llvm.sadd.with.overflow.*` | Signed checked addition; returns `{T, i1}` | [`13-advanced-ir/01-common-intrinsics.md`](13-advanced-ir/01-common-intrinsics.md), [`13-advanced-ir/examples/overflow-intrinsic.ll`](13-advanced-ir/examples/overflow-intrinsic.ll) |
| `llvm.lifetime.start.*`, `llvm.lifetime.end.*` | Lifetime markers for optimizer-visible object liveness | [`13-advanced-ir/01-common-intrinsics.md`](13-advanced-ir/01-common-intrinsics.md), [`13-advanced-ir/examples/memcpy.ll`](13-advanced-ir/examples/memcpy.ll) |
| `llvm.prefetch` | Target-dependent cache prefetch hint | [`13-advanced-ir/01-common-intrinsics.md`](13-advanced-ir/01-common-intrinsics.md), [`08-pitfalls/06-immarg-violation.md`](08-pitfalls/06-immarg-violation.md) |
| `llvm.x86.*` | X86-specific intrinsic namespace | [`13-advanced-ir/02-target-specific-intrinsics.md`](13-advanced-ir/02-target-specific-intrinsics.md) |
| `llvm.aarch64.*`, `llvm.arm.*`, `llvm.riscv.*`, `llvm.amdgcn.*`, `llvm.nvvm.*` | Other target-specific intrinsic namespaces | [`13-advanced-ir/02-target-specific-intrinsics.md`](13-advanced-ir/02-target-specific-intrinsics.md) |
| `token` | Opaque control value used by advanced intrinsic families | [`13-advanced-ir/03-special-types-and-tokens.md`](13-advanced-ir/03-special-types-and-tokens.md), [`13-advanced-ir/examples/token-outline.ll`](13-advanced-ir/examples/token-outline.ll) |
| `metadata` | Special operand type for debug/analysis intrinsics | [`13-advanced-ir/03-special-types-and-tokens.md`](13-advanced-ir/03-special-types-and-tokens.md), [`06-metadata/01-metadata-basics.md`](06-metadata/01-metadata-basics.md) |
| `half`, `bfloat`, `x86_amx`, `<vscale x N x T>` | Special scalar/target/vector types with portability constraints | [`13-advanced-ir/03-special-types-and-tokens.md`](13-advanced-ir/03-special-types-and-tokens.md) |

## By keyword (where it's introduced)

| Keyword | File |
|---|---|
| `define`, `declare` | [`01-syntax/01-modules-functions-blocks.md`](01-syntax/01-modules-functions-blocks.md) |
| `target datalayout`, `target triple` | [`01-syntax/01-modules-functions-blocks.md`](01-syntax/01-modules-functions-blocks.md) |
| `source_filename` | [`01-syntax/01-modules-functions-blocks.md`](01-syntax/01-modules-functions-blocks.md) |
| `global`, `constant` | [`03-constants/04-global-vs-local.md`](03-constants/04-global-vs-local.md), [`04-memory/03-global-variables.md`](04-memory/03-global-variables.md) |
| `addrspace` | [`04-memory/04-address-spaces.md`](04-memory/04-address-spaces.md) |
| `thread_local` | [`04-memory/03-global-variables.md`](04-memory/03-global-variables.md) |
| `private`, `internal`, `external`, `weak`, `linkonce`, `appending`, `common`, etc. | [`04-memory/03-global-variables.md`](04-memory/03-global-variables.md) |
| `default`, `hidden`, `protected` | [`04-memory/03-global-variables.md`](04-memory/03-global-variables.md) |
| `dllexport`, `dllimport` | [`04-memory/03-global-variables.md`](04-memory/03-global-variables.md) |
| `dso_local`, `dso_preemptable` | [`04-memory/03-global-variables.md`](04-memory/03-global-variables.md) |
| `inbounds` | [`02-types/02-composite-types.md`](02-types/02-composite-types.md) |
| `nsw`, `nuw`, `exact`, `fast`, `nnan`, `ninf`, `nsz`, `arcp`, `contract`, `afn`, `reassoc` | [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `align`, `alignstack` | [`04-memory/02-load-store.md`](04-memory/02-load-store.md) |
| `volatile` | [`11-concurrency/03-volatile-vs-atomic.md`](11-concurrency/03-volatile-vs-atomic.md), [`04-memory/02-load-store.md`](04-memory/02-load-store.md) |
| `atomic`, `syncscope` | [`11-concurrency/02-atomic-instructions.md`](11-concurrency/02-atomic-instructions.md), [`04-memory/02-load-store.md`](04-memory/02-load-store.md) |
| `unordered`, `monotonic`, `acquire`, `release`, `acq_rel`, `seq_cst` | [`11-concurrency/01-atomic-orderings.md`](11-concurrency/01-atomic-orderings.md) |
| `cmpxchg`, `atomicrmw`, `fence` | [`11-concurrency/02-atomic-instructions.md`](11-concurrency/02-atomic-instructions.md), [`reference/instruction-quickref.md`](reference/instruction-quickref.md) |
| `blockaddress`, `indirectbr` | [`05-control-flow/04-indirectbr.md`](05-control-flow/04-indirectbr.md) |
| `distinct` | [`06-metadata/01-metadata-basics.md`](06-metadata/01-metadata-basics.md) |
| `DIFile`, `DICompileUnit`, `DISubprogram`, `DILocation`, `DILocalVariable` | [`06-metadata/02-debug-info.md`](06-metadata/02-debug-info.md) |
| `branch_weights`, `llvm.loop.unroll.*`, `llvm.loop.vectorize.*` | [`06-metadata/03-profile-and-optimization-metadata.md`](06-metadata/03-profile-and-optimization-metadata.md) |
| `-Rpass`, `-Rpass-missed`, `-force-vector-width`, `-force-vector-interleave` | [`09-vectorization/README.md`](09-vectorization/README.md) |
| `loop-vectorize`, `slp-vectorizer`, `default<O3>` pass names | [`09-vectorization/README.md`](09-vectorization/README.md) |
| `class`, `def`, `let`, `multiclass`, `defm` | [`12-backend-jit/02-tablegen.md`](12-backend-jit/02-tablegen.md) |
| `llvm-tblgen`, `-gen-register-info`, `-gen-instr-info`, `-gen-dag-isel` | [`12-backend-jit/02-tablegen.md`](12-backend-jit/02-tablegen.md) |
| `module`, `builtin.module`, `func.func`, `arith.*`, `scf.*`, `cf.*`, `memref.*` | [`14-mlir-bridge/01-what-is-mlir.md`](14-mlir-bridge/01-what-is-mlir.md), [`14-mlir-bridge/03-lowering-to-llvm-dialect.md`](14-mlir-bridge/03-lowering-to-llvm-dialect.md) |
| `llvm.func`, `llvm.load`, `llvm.call`, `llvm.br`, `llvm.return` | [`14-mlir-bridge/03-lowering-to-llvm-dialect.md`](14-mlir-bridge/03-lowering-to-llvm-dialect.md), [`14-mlir-bridge/examples/lowered-llvm-dialect.mlir`](14-mlir-bridge/examples/lowered-llvm-dialect.mlir) |
| `bcir.vertex`, `bcir.edge`, `bcir.attribute`, `bcir.ham_hint`, `bcir.bind_register`, `bcir.mixed_stride.graph` | [`14-mlir-bridge/04-bcir-as-custom-dialect.md`](14-mlir-bridge/04-bcir-as-custom-dialect.md), [`14-mlir-bridge/examples/bcir-dialect-sketch.mlir`](14-mlir-bridge/examples/bcir-dialect-sketch.mlir) |
| `bcir.lower.vertex_handle`, `builtin.unrealized_conversion_cast`, `convert-scf-to-cf`, `convert-func-to-llvm`, `reconcile-unrealized-casts` | [`14-mlir-bridge/05-type-conversion-and-materialization.md`](14-mlir-bridge/05-type-conversion-and-materialization.md), [`14-mlir-bridge/07-pass-pipeline-and-diagnostics.md`](14-mlir-bridge/07-pass-pipeline-and-diagnostics.md) |


## Cross-references to the BCIR project

Real-world examples of LLVM IR concepts (and bugs) live next door:

| Concept | BCIR file | Pitfall |
|---|---|---|
| Cache-line-aligned struct layout | `include/bcir/bcir_ir.hpp` (`BcirClaimV1`) | none |
| LLVM substrate ABI | `runtime/llvm/bcir_master_reference_v2.ll` | metadata-string syntax (fixed in `1f62e86`) |
| Boolean expression construction | `runtime/llvm/bcir_claim_verify.ll` | [`08-pitfalls/01-nested-instruction-expressions.md`](08-pitfalls/01-nested-instruction-expressions.md) |
| PHI predecessors | `runtime/llvm/bcir_batch_executor.ll` | [`08-pitfalls/02-phi-predecessor-mismatch.md`](08-pitfalls/02-phi-predecessor-mismatch.md) (fixed in `5754354`) |
| Duplicate block labels | `runtime/llvm/bcir_claim_verify.ll` (pre-`1f62e86`) | [`08-pitfalls/03-duplicate-block-labels.md`](08-pitfalls/03-duplicate-block-labels.md) |
| Cross-module function definition collision | `runtime/llvm/bcir_gem_seed.ll` vs `bcir_worklist.ll` | [`08-pitfalls/04-duplicate-symbols.md`](08-pitfalls/04-duplicate-symbols.md) |
| Type schema drift | `%bcir.blob.header` in three files | [`08-pitfalls/05-type-schema-drift.md`](08-pitfalls/05-type-schema-drift.md) |
| `llvm.prefetch` immarg | `runtime/llvm/bcir_prefetch_profiles.ll` | [`08-pitfalls/06-immarg-violation.md`](08-pitfalls/06-immarg-violation.md) |
| Vertex-Edge-Attribute custom dialect sketch | `llvm-training/14-mlir-bridge/examples/bcir-dialect-sketch.mlir` | [`14-mlir-bridge/04-bcir-as-custom-dialect.md`](14-mlir-bridge/04-bcir-as-custom-dialect.md) |
| Lowered BCIR-style LLVM dialect sketch | `llvm-training/14-mlir-bridge/examples/lowered-llvm-dialect.mlir` | [`14-mlir-bridge/03-lowering-to-llvm-dialect.md`](14-mlir-bridge/03-lowering-to-llvm-dialect.md) |
| BCIR MLIR type-conversion sketch | `llvm-training/14-mlir-bridge/examples/bcir-type-conversion.mlir` | [`14-mlir-bridge/05-type-conversion-and-materialization.md`](14-mlir-bridge/05-type-conversion-and-materialization.md) |
| BCIR MLIR conversion pipeline sketch | `llvm-training/14-mlir-bridge/examples/bcir-conversion-pipeline.mlir` | [`14-mlir-bridge/06-conversion-patterns.md`](14-mlir-bridge/06-conversion-patterns.md) |
| Final LLVM IR snapshot for BCIR MLIR lowering | `llvm-training/14-mlir-bridge/examples/bcir-final.ll` | [`14-mlir-bridge/08-end-to-end-bcir-lowering.md`](14-mlir-bridge/08-end-to-end-bcir-lowering.md) |
