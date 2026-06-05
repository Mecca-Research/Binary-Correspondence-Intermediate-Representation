# Curriculum — Reading Order

Multiple suggested paths depending on how much time you (or your agent
context) have and which advanced LLVM IR topic you need.

## Path 1: 30-minute fast pass (just enough to not be dangerous)

Read these in order. Skip examples; just the prose.

1. [`00-foundations/01-what-is-llvm-ir.md`](00-foundations/01-what-is-llvm-ir.md) — what is IR, what isn't it
2. [`00-foundations/02-ssa.md`](00-foundations/02-ssa.md) — SSA + phi nodes (the single most
   important concept)
3. [`01-syntax/01-modules-functions-blocks.md`](01-syntax/01-modules-functions-blocks.md) — the hierarchy
4. [`01-syntax/02-instruction-format.md`](01-syntax/02-instruction-format.md) — `%result = op type, operands`
5. [`08-pitfalls/README.md`](08-pitfalls/README.md) — the index of what breaks

You now know enough to read existing IR. You can't write it safely yet.

Practice next: [`exercises/001-add.prompt.md`](exercises/001-add.prompt.md) and
[`exercises/002-if-else-phi.prompt.md`](exercises/002-if-else-phi.prompt.md).

## Path 2: 2-hour working knowledge

After Path 1, add:

6. [`02-types/01-primitive-types.md`](02-types/01-primitive-types.md)
7. [`02-types/02-composite-types.md`](02-types/02-composite-types.md) — include the GEP basics in
   `Accessing struct fields` / aggregate access examples
8. [`02-types/03-opaque-and-pointer-types.md`](02-types/03-opaque-and-pointer-types.md)
9. [`02-types/04-opaque-pointer-migration.md`](02-types/04-opaque-pointer-migration.md) — migration dispatcher
10. [`02-types/05-opaque-pointer-migration-patterns.md`](02-types/05-opaque-pointer-migration-patterns.md) — where typed-pointer facts moved (`load`, `store`, `getelementptr`, calls)
11. [`04-memory/01-alloca.md`](04-memory/01-alloca.md)
12. [`04-memory/02-load-store.md`](04-memory/02-load-store.md) — typed memory operations,
    especially explicit access types with opaque pointers
13. [`05-control-flow/01-unconditional-br.md`](05-control-flow/01-unconditional-br.md)
14. [`05-control-flow/02-conditional-br.md`](05-control-flow/02-conditional-br.md)
15. [`06-metadata/01-metadata-basics.md`](06-metadata/01-metadata-basics.md) — metadata syntax and common attachments
16. [`06-metadata/02-debug-info.md`](06-metadata/02-debug-info.md) — source locations and debug-info nodes
17. [`06-metadata/03-profile-and-optimization-metadata.md`](06-metadata/03-profile-and-optimization-metadata.md) — branch weights and loop hints
18. [`06-metadata/04-type-metadata-cfi.md`](06-metadata/04-type-metadata-cfi.md) — `!type` metadata, `llvm.type.test`, and CFI-style checked dispatch
19. [`reference/instruction-quickref.md`](reference/instruction-quickref.md) — read the sections for
    terminators, comparison, memory, conversion, and other/call instructions
20. All six files in `08-pitfalls/` — each is ≤ 5 minutes

Now you can read and write straightforward IR. Verifier failures should
make sense.

Practice next: [`exercises/003-loop-counter.prompt.md`](exercises/003-loop-counter.prompt.md),
[`exercises/004-global-load-store.prompt.md`](exercises/004-global-load-store.prompt.md), and
[`exercises/005-struct-gep.prompt.md`](exercises/005-struct-gep.prompt.md).


## After basics: metadata and optimization

After Path 2, add this path when you need to understand how IR carries
non-semantic annotations and how `opt` uses or rewrites them:

1. [`06-metadata/01-metadata-basics.md`](06-metadata/01-metadata-basics.md) — metadata basics: `!N`, metadata tuples, named metadata, `distinct`, and instruction attachments
2. [`06-metadata/02-debug-info.md`](06-metadata/02-debug-info.md) — debug info: source locations, `!dbg`, and common `DI*` nodes
3. [`06-metadata/03-profile-and-optimization-metadata.md`](06-metadata/03-profile-and-optimization-metadata.md) — optimization metadata: branch weights, loop hints, TBAA, ranges, and nonnull facts
4. [`06-metadata/04-type-metadata-cfi.md`](06-metadata/04-type-metadata-cfi.md) — type metadata: `!type`, type identifiers, `llvm.type.test`, checked loads, and CFI-style hardening
5. [`07-optimization/01-pass-model.md`](07-optimization/01-pass-model.md) — pass model: analysis vs transform vs utility passes and new pass manager syntax
6. [`07-optimization/03-common-transform-passes.md`](07-optimization/03-common-transform-passes.md) — common transform passes such as `mem2reg`, `instcombine`, `simplifycfg`, `adce`, `gvn`, and `loop-unroll`
7. [`07-optimization/02-common-analysis-passes.md`](07-optimization/02-common-analysis-passes.md) — common analysis passes for aliasing, MemorySSA, CFGs, loops, and scalar evolution
8. [`07-optimization/08-deep-optimization-lessons.md`](07-optimization/08-deep-optimization-lessons.md) — BCIR-focused PassBuilder plugins, MemorySSA, SCCP, LoopRotate, metadata preservation, mapping, loop shape, and poison risks

Practice next: run the commands embedded in
[`07-optimization/examples/mem2reg-before.ll`](07-optimization/examples/mem2reg-before.ll),
[`07-optimization/examples/dead-code-before.ll`](07-optimization/examples/dead-code-before.ll), and
[`07-optimization/examples/loop-before.ll`](07-optimization/examples/loop-before.ll),
[`07-optimization/examples/memoryssa-alias-shape.ll`](07-optimization/examples/memoryssa-alias-shape.ll),
[`07-optimization/examples/sccp-before.ll`](07-optimization/examples/sccp-before.ll), and
[`07-optimization/examples/loop-rotate-bcir-before.ll`](07-optimization/examples/loop-rotate-bcir-before.ll).


## Performance path

After the metadata and optimization path, add this sequence when you need to
reason about optimization strength, pass pipelines, and vectorized IR:

1. [`07-optimization/01-pass-model.md`](07-optimization/01-pass-model.md) — pass pipelines and `opt -passes=...` spelling
2. [`07-optimization/04-optimization-levels.md`](07-optimization/04-optimization-levels.md) — optimization levels: `-O0`, `-O1`, `-O2`, `-O3`, `-Os`, and `-Oz`
3. [`07-optimization/08-deep-optimization-lessons.md`](07-optimization/08-deep-optimization-lessons.md) — deeper optimizer legality and BCIR invariants around plugins, MemorySSA, SCCP, loop rotation, metadata, and poison
4. [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md) — PGO, LTO/ThinLTO, and BOLT profile-driven pipeline effects
5. [`09-vectorization/README.md`](09-vectorization/README.md) — auto-vectorization dispatcher
6. [`09-vectorization/01-loop-vectorizer.md`](09-vectorization/01-loop-vectorizer.md) and [`09-vectorization/02-slp-vectorizer.md`](09-vectorization/02-slp-vectorizer.md) — focused Loop Vectorizer and SLP Vectorizer paths
7. [`09-vectorization/04-vectorization-legality.md`](09-vectorization/04-vectorization-legality.md), [`09-vectorization/05-example-walkthroughs.md`](09-vectorization/05-example-walkthroughs.md), and [`09-vectorization/07-masked-and-interleaved-access.md`](09-vectorization/07-masked-and-interleaved-access.md) — blockers, commands, predication, and interleaved-memory observations
8. [`reference/instruction-quickref.md`](reference/instruction-quickref.md) — vector IR quick reference: vector types, vector loads/stores, `extractelement`, `insertelement`, and `shufflevector`

Practice next: run the commands in
[`09-vectorization/examples/sum-loop.c`](09-vectorization/examples/sum-loop.c),
[`09-vectorization/examples/sum-loop.ll`](09-vectorization/examples/sum-loop.ll), and
[`09-vectorization/examples/slp-scalars.ll`](09-vectorization/examples/slp-scalars.ll).


## Concurrent IR path

After Path 2, add this chapter when reading or writing shared-memory IR:

1. Re-read [`04-memory/02-load-store.md`](04-memory/02-load-store.md) — load/store review, especially explicit access types, `align`, and volatile syntax
2. [`11-concurrency/01-atomic-orderings.md`](11-concurrency/01-atomic-orderings.md) — atomic orderings: not atomic vs `unordered`, `monotonic`, acquire/release, `acq_rel`, and `seq_cst`
3. [`11-concurrency/02-atomic-instructions.md`](11-concurrency/02-atomic-instructions.md) — atomic instructions: `load atomic`, `store atomic`, `cmpxchg`, `atomicrmw`, and `fence` syntax
4. [`11-concurrency/03-volatile-vs-atomic.md`](11-concurrency/03-volatile-vs-atomic.md) — why volatile access behavior and atomic synchronization are orthogonal
5. [`11-concurrency/04-memory-model-mapping.md`](11-concurrency/04-memory-model-mapping.md) — mapping C++ and Rust orderings to LLVM atomics

Practice next: inspect and assemble the examples in
[`11-concurrency/examples/atomic-counter.ll`](11-concurrency/examples/atomic-counter.ll),
[`11-concurrency/examples/cmpxchg-loop.ll`](11-concurrency/examples/cmpxchg-loop.ll), and
[`11-concurrency/examples/fence.ll`](11-concurrency/examples/fence.ll).


## Backend/JIT path

After Path 2, add this path when you need to understand how LLVM IR becomes
target machine code or when embedding LLVM as a JIT compiler:

1. [`00-foundations/03-ir-vs-asm-vs-other-irs.md`](00-foundations/03-ir-vs-asm-vs-other-irs.md) — IR vs assembly and what remains target-independent
2. [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) — codegen pipeline: SelectionDAG, GlobalISel, `MachineInstr`, register allocation, and MC emission
3. [`12-backend-jit/02-tablegen.md`](12-backend-jit/02-tablegen.md) — TableGen syntax, generated backend include files, registers, instructions, patterns, and scheduling data
4. [`12-backend-jit/03-orc-jit.md`](12-backend-jit/03-orc-jit.md) — ORC JIT and `LLJIT`: adding modules, symbol lookup, function pointers, and resource ownership
5. [`12-backend-jit/05-orc-layers.md`](12-backend-jit/05-orc-layers.md) — ORC internals: `ExecutionSession`, `JITDylib`, layers, materialization, symbol interning, and JITLink handoff
6. [`12-backend-jit/04-mc-and-relocations.md`](12-backend-jit/04-mc-and-relocations.md) — MC layer concepts, relocations, and JIT missing-symbol debugging
7. [`12-backend-jit/06-custom-bcir-intrinsics.md`](12-backend-jit/06-custom-bcir-intrinsics.md) — custom backend intrinsics plus `llvm.experimental.stackmap`/`llvm.experimental.patchpoint` contracts for deoptimization and runtime patching
8. Skim [`12-backend-jit/examples/minimal-instruction.td`](12-backend-jit/examples/minimal-instruction.td) and [`12-backend-jit/examples/lljit-outline.cpp.md`](12-backend-jit/examples/lljit-outline.cpp.md) as compact reference outlines.

This path is intentionally advanced: it assumes you can already read LLVM IR and
optimizer output, then follows the handoff into backend data structures, target
descriptions, and runtime code generation.


## Binary analysis and dynamic execution path

After the Backend/JIT path, add this path for security-sensitive code, BCSA, or
performance investigations where final binary behavior matters:

1. [`15-binary-analysis/README.md`](15-binary-analysis/README.md) — overview of static IR plus runtime evidence.
2. [`15-binary-analysis/01-microarchitecture-side-channels.md`](15-binary-analysis/01-microarchitecture-side-channels.md) — cache, branch-prediction, and timing side-channel review.
3. [`06-metadata/04-type-metadata-cfi.md`](06-metadata/04-type-metadata-cfi.md) — connect IR-level CFI guards to post-codegen hardening evidence.
4. [`15-binary-analysis/02-dynamic-traces-and-counters.md`](15-binary-analysis/02-dynamic-traces-and-counters.md) — trace/counter schemas and pairing runtime evidence with IR.
5. [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md) — how profiles, LTO, ThinLTO, and BOLT can reshape the final binary.
6. [`15-binary-analysis/03-interpretable-bcsa-features.md`](15-binary-analysis/03-interpretable-bcsa-features.md) — cheap, explainable BCSA triage before dense embeddings.
7. Inspect [`15-binary-analysis/examples/`](15-binary-analysis/examples/) for the constant-time review IR and tiny trace/counter/feature schemas.

This path explicitly prevents an agent from treating static IR equivalence as a
security or performance verdict.

## Advanced IR constructs path

After Path 2, add this path when unusual IR syntax, target hooks, or special
case constructs appear in generated modules:

1. [`quickref/advanced-ir.md`](quickref/advanced-ir.md), [`reference/intrinsics.md`](reference/intrinsics.md), and [`reference/intrinsics-quickref.md`](reference/intrinsics-quickref.md) — fast checklist plus declaration rules and a focused category quick reference for common, memory/lifetime/debug, stackmap/patchpoint, custom, and target-specific intrinsic families
2. [`13-advanced-ir/01-common-intrinsics.md`](13-advanced-ir/01-common-intrinsics.md) and [`13-advanced-ir/02-target-specific-intrinsics.md`](13-advanced-ir/02-target-specific-intrinsics.md) — common and target-specific intrinsic spelling in standalone modules
3. [`13-advanced-ir/03-special-types-and-tokens.md`](13-advanced-ir/03-special-types-and-tokens.md) — special scalar, token, metadata, target-extension, and scalable-vector types
4. [`13-advanced-ir/04-attributes.md`](13-advanced-ir/04-attributes.md) — function, parameter, memory-effect, pointer, and ABI attributes
5. [`13-advanced-ir/05-poison-undef-freeze.md`](13-advanced-ir/05-poison-undef-freeze.md) — `undef`, poison propagation, `freeze`, vector lanes, `noundef`, and verifier-valid unsafe patterns
6. [`13-advanced-ir/06-fast-math-flags.md`](13-advanced-ir/06-fast-math-flags.md) — `nnan`, `ninf`, `nsz`, `arcp`, `contract`, `afn`, `reassoc`, `fast`, FP comparisons, reductions, and vectorization consequences
7. [`13-advanced-ir/07-operand-bundles.md`](13-advanced-ir/07-operand-bundles.md) — call-site operand bundles for deoptimization, funclets, GC liveness, and ARC attached calls
8. [`04-memory/04-address-spaces.md`](04-memory/04-address-spaces.md) — target-specific address spaces and `addrspacecast`
9. [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) — target-specific operations as they leave IR and become machine-level lowering decisions

Use this path as a lookup-oriented supplement rather than a linear beginner
chapter. It is most useful when reviewing frontend output, GPU IR, intrinsic
heavy code, or backend-adjacent transformations.

## MLIR bridge path

After Path 2, add this path when a frontend or domain IR should preserve
structured information before producing LLVM IR:

1. [`quickref/mlir-bridge.md`](quickref/mlir-bridge.md) and [`14-mlir-bridge/01-what-is-mlir.md`](14-mlir-bridge/01-what-is-mlir.md) — bridge checklist plus MLIR modules, operations, regions, blocks, attributes, and types
2. [`14-mlir-bridge/02-dialects-and-operations.md`](14-mlir-bridge/02-dialects-and-operations.md) — dialect design basics and operation anatomy
3. [`14-mlir-bridge/03-lowering-to-llvm-dialect.md`](14-mlir-bridge/03-lowering-to-llvm-dialect.md) — conversion/lowering pipelines, LLVM dialect, and `.ll` differences
4. [`14-mlir-bridge/04-bcir-as-custom-dialect.md`](14-mlir-bridge/04-bcir-as-custom-dialect.md) — where BCIR Vertex-Edge-Attribute, HAM hints, register binding, and Mixed Stride graphs can live
5. [`14-mlir-bridge/05-vertex-graph-lowering.md`](14-mlir-bridge/05-vertex-graph-lowering.md) — track vertex IDs, edge lists, register bindings, and metadata hints through BCIR dialect, LLVM dialect, and LLVM IR
6. Skim [`14-mlir-bridge/examples/bcir-dialect-sketch.mlir`](14-mlir-bridge/examples/bcir-dialect-sketch.mlir), [`14-mlir-bridge/examples/lowered-llvm-dialect.mlir`](14-mlir-bridge/examples/lowered-llvm-dialect.mlir), and [`14-mlir-bridge/examples/bcir-vertex-graph-lowered.ll`](14-mlir-bridge/examples/bcir-vertex-graph-lowered.ll) as illustrative before/after shapes.

Use this path before the Backend/JIT path when the task starts above LLVM IR,
especially for custom frontend lowering or BCIR-specific graph representations.

## BCIR mapping path

After Path 2, add this path when the task is to lower BCIR-like claims, graph
fragments, runtime operations, or diagnostic hints directly to LLVM IR:

1. [`bcir-mapping/README.md`](bcir-mapping/README.md) — dispatcher for the BCIR-to-LLVM mapping guide and standalone examples.
2. [`bcir-mapping/01-vertex-edge-attribute.md`](bcir-mapping/01-vertex-edge-attribute.md) and [`bcir-mapping/07-gaadmsf-operations.md`](bcir-mapping/07-gaadmsf-operations.md) — graph fragments, struct arrays, GEPs, and graph-aware data movement.
3. [`bcir-mapping/02-register-binding.md`](bcir-mapping/02-register-binding.md) and [`bcir-mapping/06-claim-lowering-pipeline.md`](bcir-mapping/06-claim-lowering-pipeline.md) — claim normalization, resource lookup, and registry loads.
4. [`bcir-mapping/03-mixed-stride-graphs.md`](bcir-mapping/03-mixed-stride-graphs.md) — row/column stride arithmetic and byte-offset lowering.
5. [`bcir-mapping/04-ham-hints.md`](bcir-mapping/04-ham-hints.md) and [`bcir-mapping/10-metadata-and-diagnostics.md`](bcir-mapping/10-metadata-and-diagnostics.md) — HAM hints, prefetch intrinsics, custom metadata, and diagnostic preservation.
6. [`bcir-mapping/05-runtime-abi.md`](bcir-mapping/05-runtime-abi.md), [`bcir-mapping/08-dragon-egg-operations.md`](bcir-mapping/08-dragon-egg-operations.md), and [`bcir-mapping/09-runtime-call-boundaries.md`](bcir-mapping/09-runtime-call-boundaries.md) — ABI structs, Dragon Egg runtime-owned operations, and wrapper calls.
7. Run `./llvm-training/tools/verify-bcir-mapping.sh` and `./llvm-training/tools/verify-examples.sh` after editing any checked source-like `.bcir.txt` or lowered `.ll` output under [`bcir-mapping/examples/`](bcir-mapping/examples/).

Use this path together with the MLIR bridge path when the source representation
starts as a dialect operation rather than a source-like `.bcir.txt` prompt.

## BCIR lowering path

After Path 2, use this path when the task is specifically to turn BCIR-domain
constructs into executable LLVM IR or runtime-call boundaries:

1. [`bcir-mapping/06-claim-lowering-pipeline.md`](bcir-mapping/06-claim-lowering-pipeline.md) — normalize claims before dispatch.
2. [`bcir-mapping/01-vertex-edge-attribute.md`](bcir-mapping/01-vertex-edge-attribute.md), [`bcir-mapping/07-gaadmsf-operations.md`](bcir-mapping/07-gaadmsf-operations.md), and [`bcir-mapping/03-mixed-stride-graphs.md`](bcir-mapping/03-mixed-stride-graphs.md) — lower graph fragments, GAADMSF operations, and mixed strides into structs, GEPs, and byte offsets.
3. [`bcir-mapping/02-register-binding.md`](bcir-mapping/02-register-binding.md) and [`bcir-mapping/05-runtime-abi.md`](bcir-mapping/05-runtime-abi.md) — keep registry/resource ABI layouts synchronized.
4. [`bcir-mapping/04-ham-hints.md`](bcir-mapping/04-ham-hints.md) and [`bcir-mapping/10-metadata-and-diagnostics.md`](bcir-mapping/10-metadata-and-diagnostics.md) — lower hints and preserve diagnostics without making metadata semantically required.
5. Exercises [`028`](exercises/028-lower-vertex-edge-fragment.prompt.md)-[`031`](exercises/031-lower-runtime-call-boundary.prompt.md) apply the core BCIR lowering patterns; exercises [`038`](exercises/038-custom-pass-bcir-invariants.prompt.md)-[`040`](exercises/040-debug-gaadmsf-lowering.prompt.md) add verifier-style invariant design, graph metadata encoding, and GAADMSF debugging.

## MLIR integration path

Use this path when BCIR or another domain IR should remain structured as MLIR
before lowering to LLVM dialect or textual LLVM IR:

1. [`14-mlir-bridge/01-what-is-mlir.md`](14-mlir-bridge/01-what-is-mlir.md) and [`14-mlir-bridge/02-dialects-and-operations.md`](14-mlir-bridge/02-dialects-and-operations.md) — identify modules, operations, regions, blocks, dialects, attributes, and types.
2. [`14-mlir-bridge/04-bcir-as-custom-dialect.md`](14-mlir-bridge/04-bcir-as-custom-dialect.md) — decide which BCIR concepts belong in a custom dialect.
3. [`14-mlir-bridge/05-vertex-graph-lowering.md`](14-mlir-bridge/05-vertex-graph-lowering.md) — follow a complete graph lowering across source MLIR, LLVM-dialect MLIR, and textual LLVM IR.
4. [`14-mlir-bridge/03-lowering-to-llvm-dialect.md`](14-mlir-bridge/03-lowering-to-llvm-dialect.md) — review type conversion and LLVM-dialect boundaries.
5. Exercises [`032`](exercises/032-identify-mlir-dialect-boundaries.prompt.md)-[`034`](exercises/034-review-mlir-to-llvm-type-conversion.prompt.md) — practice dialect-boundary, graph-op lowering, and type-conversion reviews.

## Backend/JIT diagnostics path

Use this path after Path 2 when a problem appears below IR optimization: target
lowering, object emission, ORC ownership, symbol lookup, or relocation handling.

1. [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) — place SelectionDAG/GlobalISel, `MachineInstr`, register allocation, MC, and object emission in order.
2. [`12-backend-jit/02-tablegen.md`](12-backend-jit/02-tablegen.md) — identify generated target facts and avoid editing generated files.
3. [`12-backend-jit/03-orc-jit.md`](12-backend-jit/03-orc-jit.md), [`12-backend-jit/05-orc-layers.md`](12-backend-jit/05-orc-layers.md), and [`12-backend-jit/04-mc-and-relocations.md`](12-backend-jit/04-mc-and-relocations.md) — trace missing symbols from `LLJIT` ownership through layers, JITLink, object symbols, and relocations.
4. Exercises [`035`](exercises/035-diagnose-missing-symbol-relocation.prompt.md)-[`037`](exercises/037-tablegen-to-mcinst-review.prompt.md) — practice backend/JIT failure triage.

## Binary-analysis evidence path

Use this path when static LLVM IR is not enough to explain security,
performance, BCSA, or optimized-binary behavior:

1. [`15-binary-analysis/README.md`](15-binary-analysis/README.md) — choose static, dynamic, and post-codegen evidence.
2. [`15-binary-analysis/01-microarchitecture-side-channels.md`](15-binary-analysis/01-microarchitecture-side-channels.md) — review timing, cache, and branch-predictor leakage.
3. [`15-binary-analysis/02-dynamic-traces-and-counters.md`](15-binary-analysis/02-dynamic-traces-and-counters.md) — interpret trace and hardware-counter schemas.
4. [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md) and [`07-optimization/07-bolt-layout-walkthrough.md`](07-optimization/07-bolt-layout-walkthrough.md) — preserve profile, LTO, and BOLT evidence.
5. [`15-binary-analysis/03-interpretable-bcsa-features.md`](15-binary-analysis/03-interpretable-bcsa-features.md) — extract cheap interpretable BCSA features before dense embeddings.

## Repair exercises path

Use this path when the learner or agent must fix broken IR rather than write a
module from scratch:

1. [`08-pitfalls/README.md`](08-pitfalls/README.md) — identify the likely failure family.
2. [`EXAMPLES.md`](EXAMPLES.md) — confirm invalid fixture naming before adding a broken input.
3. Exercises [`016`](exercises/016-fix-phi-predecessor.prompt.md)-[`019`](exercises/019-fix-atomic-ordering.prompt.md), [`026`](exercises/026-poison-freeze-repair.prompt.md), and [`040`](exercises/040-debug-gaadmsf-lowering.prompt.md) — repair CFG, symbol, intrinsic, atomic-ordering, poison/freeze, and BCIR lowering hazards.
4. Run `./llvm-training/tools/verify-invalid-fixtures.sh` for broken inputs, `./llvm-training/tools/verify-exercises.sh` for fixed `.solution.ll` outputs, and `./llvm-training/tools/verify-bcir-mapping.sh` when a repair touches BCIR mapping fixtures.

## Path 3: Deep dive (one sitting; pick up the rest as needed)

Read everything in numerical order:

```
00-foundations/   →  01-syntax/   →  02-types/  →  03-constants/
                                                    ↓
05-control-flow/  ←  04-memory/  ←─────────────────┘
        ↓
06-metadata/
        ↓
07-optimization/  →  09-vectorization/
        ↓                    ↓
08-pitfalls/      →  10-grammar/  →  11-concurrency/  →  14-mlir-bridge/  →  bcir-mapping/
        ↓                                                                    ↓
   12-backend-jit/  →  15-binary-analysis/  →  reference/
```

Cross-references inside each chapter (`See also:`) let you jump
forward when curiosity strikes; come back via the index.

Practice next: complete all exercises in [`exercises/README.md`](exercises/README.md)
and compare against the standalone `.ll` solutions.

## Path 4: Post-optimization vectorization path

After Path 2, or after reading the optimization metadata chapter, add this
focused path when you need to understand LLVM's vectorized IR and diagnostics:

1. Re-read [`06-metadata/03-profile-and-optimization-metadata.md`](06-metadata/03-profile-and-optimization-metadata.md)
   for loop transformation hints and the limits of metadata.
2. Read [`09-vectorization/README.md`](09-vectorization/README.md) for the
   difference between the Loop Vectorizer and SLP Vectorizer.
3. Read [`09-vectorization/04-vectorization-legality.md`](09-vectorization/04-vectorization-legality.md) for the legality facts that block or allow vectorization.
4. Run the commands from [`09-vectorization/05-example-walkthroughs.md`](09-vectorization/05-example-walkthroughs.md) with [`09-vectorization/examples/sum-loop.c`](09-vectorization/examples/sum-loop.c)
   to compare successful and missed loop-vectorization remarks.
5. Run `opt -S -passes=loop-vectorize` on
   [`09-vectorization/examples/sum-loop.ll`](09-vectorization/examples/sum-loop.ll)
   and inspect vector loop structure, vector loads/stores, and reductions.
6. Run `opt -S -passes=slp-vectorizer` on
   [`09-vectorization/examples/slp-scalars.ll`](09-vectorization/examples/slp-scalars.ll)
   and inspect vector packing, `shufflevector`, and straight-line vector IR.
7. Read [`09-vectorization/07-masked-and-interleaved-access.md`](09-vectorization/07-masked-and-interleaved-access.md), then run `opt -S -passes=loop-vectorize` on
   [`09-vectorization/examples/masked-load-store-before.ll`](09-vectorization/examples/masked-load-store-before.ll) and
   [`09-vectorization/examples/interleaved-access-before.ll`](09-vectorization/examples/interleaved-access-before.ll) with remark flags to inspect masked stores, stride recognition, and target-cost decisions.
8. Repeat with `-force-vector-width` and `-force-vector-interleave` to separate
   legality questions from profitability choices.

This path is intentionally about reading and experimenting with transformed IR,
not about writing a custom LLVM pass pipeline.

## Chapter dependency graph

Mostly linear. The real dependencies:

```
foundations ────────┐
        ↓           ↓
     syntax ───→ types ───→ constants
        ↓           ↓
     memory ←──────┘
        ↓
   control-flow
        ↓
   metadata
        ↓
 optimization ───→ vectorization (when performance/vector IR appears)
        ↓
   pitfalls (read alongside everything above)
        ↓
   grammar (open as reference)
        ↓
   concurrency (when shared memory appears)
        ↓
   MLIR bridge (when source/domain structure must lower into LLVM IR)
        ↓
   backend/JIT (when target lowering or runtime compilation appears)
        ↓
   binary analysis (when dynamic execution, side channels, or BCSA matter)
        ↓
   reference (intrinsics, special types, MLIR terms, and quick lookups)
```

## Roadmap and self-test

- [`ROADMAP.md`](ROADMAP.md) tracks topics intentionally left out or only covered at an introductory level.
- [`EVAL.md`](EVAL.md) contains the corpus self-test and path-specific self-test prompts.
