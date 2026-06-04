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
9. [`02-types/04-opaque-pointer-migration.md`](02-types/04-opaque-pointer-migration.md) — moving from typed pointers to `ptr`
10. [`04-memory/01-alloca.md`](04-memory/01-alloca.md)
11. [`04-memory/02-load-store.md`](04-memory/02-load-store.md) — typed memory operations,
    especially explicit access types with opaque pointers
12. [`05-control-flow/01-unconditional-br.md`](05-control-flow/01-unconditional-br.md)
13. [`05-control-flow/02-conditional-br.md`](05-control-flow/02-conditional-br.md)
14. [`06-metadata/01-metadata-basics.md`](06-metadata/01-metadata-basics.md) — metadata syntax and common attachments
15. [`06-metadata/02-debug-info.md`](06-metadata/02-debug-info.md) — source locations and debug-info nodes
16. [`06-metadata/03-profile-and-optimization-metadata.md`](06-metadata/03-profile-and-optimization-metadata.md) — branch weights and loop hints
17. [`reference/instruction-quickref.md`](reference/instruction-quickref.md) — read the sections for
    terminators, comparison, memory, conversion, and other/call instructions
18. All six files in `08-pitfalls/` — each is ≤ 5 minutes

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
3. [`07-optimization/01-pass-model.md`](07-optimization/01-pass-model.md) — pass model: analysis vs transform vs utility passes and new pass manager syntax
4. [`07-optimization/03-common-transform-passes.md`](07-optimization/03-common-transform-passes.md) — common transform passes such as `mem2reg`, `instcombine`, `simplifycfg`, `adce`, `gvn`, and `loop-unroll`
5. [`07-optimization/02-common-analysis-passes.md`](07-optimization/02-common-analysis-passes.md) — common analysis passes for aliasing, CFGs, loops, and scalar evolution

Practice next: run the commands embedded in
[`07-optimization/examples/mem2reg-before.ll`](07-optimization/examples/mem2reg-before.ll),
[`07-optimization/examples/dead-code-before.ll`](07-optimization/examples/dead-code-before.ll), and
[`07-optimization/examples/loop-before.ll`](07-optimization/examples/loop-before.ll).


## Performance path

After the metadata and optimization path, add this sequence when you need to
reason about optimization strength, pass pipelines, and vectorized IR:

1. [`07-optimization/01-pass-model.md`](07-optimization/01-pass-model.md) — pass pipelines and `opt -passes=...` spelling
2. [`07-optimization/04-optimization-levels.md`](07-optimization/04-optimization-levels.md) — optimization levels: `-O0`, `-O1`, `-O2`, `-O3`, `-Os`, and `-Oz`
3. [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md) — PGO, LTO/ThinLTO, and BOLT profile-driven pipeline effects
4. [`09-vectorization/README.md`](09-vectorization/README.md) — auto-vectorization overview
5. [`09-vectorization/01-loop-vectorizer.md`](09-vectorization/01-loop-vectorizer.md) and [`09-vectorization/02-slp-vectorizer.md`](09-vectorization/02-slp-vectorizer.md) — focused Loop Vectorizer and SLP Vectorizer paths
6. [`reference/instruction-quickref.md`](reference/instruction-quickref.md) — vector IR quick reference: vector types, vector loads/stores, `extractelement`, `insertelement`, and `shufflevector`

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
5. [`12-backend-jit/04-mc-and-relocations.md`](12-backend-jit/04-mc-and-relocations.md) — MC layer concepts, relocations, and JIT missing-symbol debugging
6. Skim [`12-backend-jit/examples/minimal-instruction.td`](12-backend-jit/examples/minimal-instruction.td) and [`12-backend-jit/examples/lljit-outline.cpp.md`](12-backend-jit/examples/lljit-outline.cpp.md) as compact reference outlines.

This path is intentionally advanced: it assumes you can already read LLVM IR and
optimizer output, then follows the handoff into backend data structures, target
descriptions, and runtime code generation.


## Binary analysis and dynamic execution path

After the Backend/JIT path, add this path for security-sensitive code, BCSA, or
performance investigations where final binary behavior matters:

1. [`15-binary-analysis/README.md`](15-binary-analysis/README.md) — overview of static IR plus runtime evidence.
2. [`15-binary-analysis/01-microarchitecture-side-channels.md`](15-binary-analysis/01-microarchitecture-side-channels.md) — cache, branch-prediction, and timing side-channel review.
3. [`15-binary-analysis/02-dynamic-traces-and-counters.md`](15-binary-analysis/02-dynamic-traces-and-counters.md) — trace/counter schemas and pairing runtime evidence with IR.
4. [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md) — how profiles, LTO, ThinLTO, and BOLT can reshape the final binary.
5. [`15-binary-analysis/03-interpretable-bcsa-features.md`](15-binary-analysis/03-interpretable-bcsa-features.md) — cheap, explainable BCSA triage before dense embeddings.
6. Inspect [`15-binary-analysis/examples/`](15-binary-analysis/examples/) for the constant-time review IR and tiny trace/counter/feature schemas.

This path explicitly prevents an agent from treating static IR equivalence as a
security or performance verdict.

## Advanced IR constructs path

After Path 2, add this path when unusual IR syntax, target hooks, or special
case constructs appear in generated modules:

1. [`reference/intrinsics.md`](reference/intrinsics.md) — common intrinsics, overloaded names, `immarg`, memory/lifetime/debug intrinsics, and target-specific intrinsic families
2. [`02-types/01-primitive-types.md`](02-types/01-primitive-types.md) — special types such as `token`, `metadata`, `x86_mmx`, `x86_fp80`, and `ppc_fp128`
3. [`04-memory/04-address-spaces.md`](04-memory/04-address-spaces.md) — target-specific address spaces and `addrspacecast`
4. [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) — target-specific operations as they leave IR and become machine-level lowering decisions

Use this path as a lookup-oriented supplement rather than a linear beginner
chapter. It is most useful when reviewing frontend output, GPU IR, intrinsic
heavy code, or backend-adjacent transformations.

## MLIR bridge path

After Path 2, add this path when a frontend or domain IR should preserve
structured information before producing LLVM IR:

1. [`14-mlir-bridge/01-what-is-mlir.md`](14-mlir-bridge/01-what-is-mlir.md) — MLIR modules, operations, regions, blocks, attributes, and types
2. [`14-mlir-bridge/02-dialects-and-operations.md`](14-mlir-bridge/02-dialects-and-operations.md) — dialect design basics and operation anatomy
3. [`14-mlir-bridge/03-lowering-to-llvm-dialect.md`](14-mlir-bridge/03-lowering-to-llvm-dialect.md) — conversion/lowering pipelines, LLVM dialect, and `.ll` differences
4. [`14-mlir-bridge/04-bcir-as-custom-dialect.md`](14-mlir-bridge/04-bcir-as-custom-dialect.md) — where BCIR Vertex-Edge-Attribute, HAM hints, register binding, and Mixed Stride graphs can live
5. Skim [`14-mlir-bridge/examples/bcir-dialect-sketch.mlir`](14-mlir-bridge/examples/bcir-dialect-sketch.mlir) and [`14-mlir-bridge/examples/lowered-llvm-dialect.mlir`](14-mlir-bridge/examples/lowered-llvm-dialect.mlir) as illustrative before/after shapes.

Use this path before the Backend/JIT path when the task starts above LLVM IR,
especially for custom frontend lowering or BCIR-specific graph representations.

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
08-pitfalls/      →  10-grammar/  →  11-concurrency/  →  12-backend-jit/  →  15-binary-analysis/  →  reference/
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
3. Run the commands in [`09-vectorization/examples/sum-loop.c`](09-vectorization/examples/sum-loop.c)
   to compare successful and missed loop-vectorization remarks.
4. Run `opt -S -passes=loop-vectorize` on
   [`09-vectorization/examples/sum-loop.ll`](09-vectorization/examples/sum-loop.ll)
   and inspect vector loop structure, vector loads/stores, and reductions.
5. Run `opt -S -passes=slp-vectorizer` on
   [`09-vectorization/examples/slp-scalars.ll`](09-vectorization/examples/slp-scalars.ll)
   and inspect vector packing, `shufflevector`, and straight-line vector IR.
6. Repeat with `-force-vector-width` and `-force-vector-interleave` to separate
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

## What's intentionally NOT here yet

If your task touches these, you'll need external references:

- **Custom optimization pass design** — pass-manager internals beyond the introductory `opt` vectorization commands
- **C/C++ frontend internals** — Clang, AST, lowering rules
- **Full benchmarking methodology** — the dynamic-analysis chapters define schemas and review loops, not statistically complete benchmark harnesses
- **Calls / returns / comparisons** — a small dedicated chapter may be worth adding if
  this training set keeps expanding beyond the quick reference

These are roadmap items; PRs welcome.

## Self-test prompts

After each path, the agent should be able to answer (without grepping
LLVM source):

**After Path 1**
- What does SSA stand for and why does it require phi nodes?
- What's the difference between `@foo` and `%foo`?
- Why must a basic block end with a terminator?

**After Path 2**
- What's the type of the pointer returned by `alloca i32`?
- In opaque-pointer IR, where do `load`, `store`, and `getelementptr` spell the memory access or element type?
- Why is `add i32 (load ...), 1` invalid as a single expression?
- When does a `br i1` need two labels, and what's the type of the
  condition?

**After Path 3**
- How do you follow an instruction `!dbg` attachment back to a source
  file, line, and column?
- Why does the grammar treat `Linkage` and `ExternLinkage` as separate
  productions?
- How do `opt -passes=mem2reg`, `opt -passes=instcombine`, and
  `opt -passes='default<O2>'` differ in scope and intent?
- Why might `-O3` be a bad default for a size-sensitive workload?
- What's the difference between `dso_local` and `dso_preemptable`?
- What's the layout convention for `%bcir.claim`-style aggregate types,
  and what breaks when consumers disagree on the field count?
  (See [`08-pitfalls/05-type-schema-drift.md`](08-pitfalls/05-type-schema-drift.md).)

**After Path 4**
- When should you expect the Loop Vectorizer rather than the SLP Vectorizer to act?
- What source or IR facts help LLVM prove a loop has predictable memory access and no unsafe dependencies?
- Which commands show successful vs missed loop-vectorization remarks?
- What IR clues suggest vectorization occurred (`<N x T>`, vector loads/stores, `shufflevector`, reductions)?
- Why can PGO+LTO or BOLT change binary shape without changing source semantics?

**After the backend / JIT path**
- Where do SelectionDAG and GlobalISel fit relative to `MachineInstr`?
- Why does register allocation happen after machine-code SSA optimizations?
- Which backend facts are commonly generated from TableGen `.td` files?
- What ownership objects should you identify before adding modules to an ORC `LLJIT`?

**After the binary-analysis path**
- Why is a secret-dependent branch in IR not the only side-channel signal to review?
- Which hardware counters would you pair with branch/path traces for constant-time review?
- What PGO/LTO/BOLT artifacts should be saved before comparing optimized binaries?
- Which interpretable BCSA features are cheap enough for first-pass triage?
