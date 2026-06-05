# Index: By keyword (where it's introduced)

| Keyword | File |
|---|---|
| `define`, `declare` | [`01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) |
| `target datalayout`, `target triple` | [`01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) |
| `source_filename` | [`01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) |
| `global`, `constant` | [`03-constants/04-global-vs-local.md`](../03-constants/04-global-vs-local.md), [`04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) |
| `addrspace` | [`04-memory/04-address-spaces.md`](../04-memory/04-address-spaces.md) |
| `thread_local` | [`04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) |
| `private`, `internal`, `external`, `weak`, `linkonce`, `appending`, `common`, etc. | [`04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) |
| `default`, `hidden`, `protected` | [`04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) |
| `dllexport`, `dllimport` | [`04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) |
| `dso_local`, `dso_preemptable` | [`04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) |
| `inbounds` | [`02-types/02-composite-types.md`](../02-types/02-composite-types.md) |
| `nsw`, `nuw`, `exact`, `fast`, `nnan`, `ninf`, `nsz`, `arcp`, `contract`, `afn`, `reassoc` | [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| `align`, `alignstack` | [`04-memory/02-load-store.md`](../04-memory/02-load-store.md), [`13-advanced-ir/04-attributes.md`](../13-advanced-ir/04-attributes.md) |
| `volatile` | [`11-concurrency/03-volatile-vs-atomic.md`](../11-concurrency/03-volatile-vs-atomic.md), [`04-memory/02-load-store.md`](../04-memory/02-load-store.md) |
| `atomic`, `syncscope` | [`11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md), [`04-memory/02-load-store.md`](../04-memory/02-load-store.md) |
| `unordered`, `monotonic`, `acquire`, `release`, `acq_rel`, `seq_cst` | [`11-concurrency/01-atomic-orderings.md`](../11-concurrency/01-atomic-orderings.md), [`11-concurrency/04-memory-model-mapping.md`](../11-concurrency/04-memory-model-mapping.md) |
| `cmpxchg`, `atomicrmw`, `fence` | [`11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md), [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| `blockaddress`, `indirectbr` | [`05-control-flow/04-indirectbr.md`](../05-control-flow/04-indirectbr.md) |
| `distinct` | [`06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md) |
| `DIFile`, `DICompileUnit`, `DISubprogram`, `DILocation`, `DILocalVariable` | [`06-metadata/02-debug-info.md`](../06-metadata/02-debug-info.md) |
| `branch_weights`, `llvm.loop.unroll.*`, `llvm.loop.vectorize.*` | [`06-metadata/03-profile-and-optimization-metadata.md`](../06-metadata/03-profile-and-optimization-metadata.md) |
| `!type`, `llvm.type.test`, `llvm.type.checked.load`, type identifier | [`06-metadata/04-type-metadata-cfi.md`](../06-metadata/04-type-metadata-cfi.md) |
| `!nosanitize`, `__asan_report_load*`, `__asan_report_store*`, shadow memory, redzone | [`08-pitfalls/16-sanitizer-instrumentation.md`](../08-pitfalls/16-sanitizer-instrumentation.md), [`08-pitfalls/examples/sanitizer-instrumentation.ll`](../08-pitfalls/examples/sanitizer-instrumentation.ll) |
| `-Rpass`, `-Rpass-missed`, `-force-vector-width`, `-force-vector-interleave` | [`09-vectorization/05-example-walkthroughs.md`](../09-vectorization/05-example-walkthroughs.md) |
| `loop-vectorize`, `slp-vectorizer`, `default<O3>` pass names | [`09-vectorization/01-loop-vectorizer.md`](../09-vectorization/01-loop-vectorizer.md), [`09-vectorization/02-slp-vectorizer.md`](../09-vectorization/02-slp-vectorizer.md), [`09-vectorization/05-example-walkthroughs.md`](../09-vectorization/05-example-walkthroughs.md) |
| `llvm.masked.store`, masked load/store, interleaved access, deinterleave | [`09-vectorization/07-masked-and-interleaved-access.md`](../09-vectorization/07-masked-and-interleaved-access.md) |
| `nounwind`, `noalias`, `readonly`, `readnone`, `writeonly`, `memory(none)`, `memory(read)`, `memory(write)`, `memory(argmem: readwrite)`, `memory(inaccessiblemem: readwrite)`, `dereferenceable`, `sret`, `byval` | [`13-advanced-ir/04-attributes.md`](../13-advanced-ir/04-attributes.md) |
| `undef`, `poison`, `freeze`, `noundef`, `nsw`, `nuw`, `exact`, `inbounds` | [`13-advanced-ir/05-poison-undef-freeze.md`](../13-advanced-ir/05-poison-undef-freeze.md), [`13-advanced-ir/05-poison-undef-freeze.md#bcir-safe-speculation-with-freeze`](../13-advanced-ir/05-poison-undef-freeze.md#bcir-safe-speculation-with-freeze) |
| `nnan`, `ninf`, `nsz`, `arcp`, `contract`, `afn`, `reassoc`, `fast` | [`13-advanced-ir/06-fast-math-flags.md`](../13-advanced-ir/06-fast-math-flags.md) |
| `deopt`, `funclet`, `gc-live`, `clang.arc.attachedcall` | [`13-advanced-ir/07-operand-bundles.md`](../13-advanced-ir/07-operand-bundles.md) |
| `class`, `def`, `let`, `multiclass`, `defm` | [`12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md) |
| `llvm-tblgen`, `-gen-register-info`, `-gen-instr-info`, `-gen-dag-isel` | [`12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md) |
| `module`, `builtin.module`, `func.func`, `arith.*`, `scf.*`, `cf.*`, `memref.*` | [`14-mlir-bridge/01-what-is-mlir.md`](../14-mlir-bridge/01-what-is-mlir.md), [`14-mlir-bridge/03-lowering-to-llvm-dialect.md`](../14-mlir-bridge/03-lowering-to-llvm-dialect.md) |
| `llvm.func`, `llvm.load`, `llvm.call`, `llvm.br`, `llvm.return` | [`14-mlir-bridge/03-lowering-to-llvm-dialect.md`](../14-mlir-bridge/03-lowering-to-llvm-dialect.md), [`14-mlir-bridge/examples/lowered-llvm-dialect.mlir`](../14-mlir-bridge/examples/lowered-llvm-dialect.mlir) |
| `bcir.vertex`, `bcir.edge`, `bcir.attribute`, `bcir.ham_hint`, `bcir.bind_register`, `bcir.mixed_stride.graph` | [`14-mlir-bridge/04-bcir-as-custom-dialect.md`](../14-mlir-bridge/04-bcir-as-custom-dialect.md), [`14-mlir-bridge/examples/bcir-dialect-sketch.mlir`](../14-mlir-bridge/examples/bcir-dialect-sketch.mlir) |
| vertex IDs, edge lists, register bindings, graph-op lowering | [`14-mlir-bridge/05-vertex-graph-lowering.md`](../14-mlir-bridge/05-vertex-graph-lowering.md), [`exercises/033-lower-mlir-graph-op-to-llvm-dialect.prompt.md`](../exercises/033-lower-mlir-graph-op-to-llvm-dialect.prompt.md) |
| BCIR invariants, lowering verifier, custom pass review | [`07-optimization/08-deep-optimization-lessons.md`](../07-optimization/08-deep-optimization-lessons.md), [`exercises/038-custom-pass-bcir-invariants.prompt.md`](../exercises/038-custom-pass-bcir-invariants.prompt.md) |
| GAADMSF, Dragon Egg, runtime-call boundary | [`bcir-mapping/07-gaadmsf-operations.md`](../bcir-mapping/07-gaadmsf-operations.md), [`bcir-mapping/08-dragon-egg-operations.md`](../bcir-mapping/08-dragon-egg-operations.md), [`bcir-mapping/09-runtime-call-boundaries.md`](../bcir-mapping/09-runtime-call-boundaries.md) |
| `llvm.bcir.*`, custom backend intrinsic, ORC fallback | [`12-backend-jit/06-custom-bcir-intrinsics.md`](../12-backend-jit/06-custom-bcir-intrinsics.md), [`reference/intrinsics-quickref.md#custom-backend-intrinsics`](../reference/intrinsics-quickref.md#custom-backend-intrinsics) |
| graph metadata, diagnostic metadata, lowering provenance | [`bcir-mapping/10-metadata-and-diagnostics.md`](../bcir-mapping/10-metadata-and-diagnostics.md), [`exercises/039-graph-description-to-llvm-metadata.prompt.md`](../exercises/039-graph-description-to-llvm-metadata.prompt.md) |
| `branch_weights`, `function_entry_count`, PGO profile metadata | [`06-metadata/03-profile-and-optimization-metadata.md`](../06-metadata/03-profile-and-optimization-metadata.md), [`07-optimization/06-pgo-lto-bolt.md`](../07-optimization/06-pgo-lto-bolt.md) |
| `-fprofile-generate`, `-fprofile-use`, `-flto=thin`, `llvm-bolt` | [`07-optimization/06-pgo-lto-bolt.md`](../07-optimization/06-pgo-lto-bolt.md) |
| `cycles`, `branch_misses`, `l1d_misses`, `llc_misses` | [`15-binary-analysis/02-dynamic-traces-and-counters.md`](../15-binary-analysis/02-dynamic-traces-and-counters.md) |
| CFI, vtable hardening, trap edge, checked indirect call | [`06-metadata/04-type-metadata-cfi.md`](../06-metadata/04-type-metadata-cfi.md), [`15-binary-analysis/README.md`](../15-binary-analysis/README.md) |
| cyclomatic complexity, opcode histogram, structural hash | [`15-binary-analysis/03-interpretable-bcsa-features.md`](../15-binary-analysis/03-interpretable-bcsa-features.md) |
