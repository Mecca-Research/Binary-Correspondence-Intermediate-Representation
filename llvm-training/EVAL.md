# EVAL — Corpus Self-Test

Use this as a closed-book coverage check. If you can answer these without grep,
you have a working map of the corpus. After answering, use the links to verify
or fill gaps.

1. What is the difference between a local `%name` and a global `@name`, and where
   do modules declare target triples? See [`01-syntax/01-modules-functions-blocks.md`](01-syntax/01-modules-functions-blocks.md).
2. Why must a `phi` incoming block list match CFG predecessors exactly? See
   [`00-foundations/02-ssa.md`](00-foundations/02-ssa.md) and
   [`08-pitfalls/02-phi-predecessor-mismatch.md`](08-pitfalls/02-phi-predecessor-mismatch.md).
3. How do you write a GEP for the `value` field inside an array of structs? See
   [`02-types/02-composite-types.md`](02-types/02-composite-types.md) and exercise 006.
4. When should pointer `align` or `dereferenceable` be a parameter attribute
   instead of only a load/store fact? See [`13-advanced-ir/04-attributes.md`](13-advanced-ir/04-attributes.md).
5. Which command verifies standalone `.ll` examples, and why does it keep a
   broken `.ll.txt` sentinel? See [`tools/verify-examples.sh`](tools/verify-examples.sh) and [`tools/README.md`](tools/README.md).
6. How do you distinguish standalone `.ll` examples from `.csv`
   binary-analysis schemas? See [`EXAMPLES.md`](EXAMPLES.md) and
   [`15-binary-analysis/02-dynamic-traces-and-counters.md`](15-binary-analysis/02-dynamic-traces-and-counters.md).
7. What LLVM-version policy governs examples and exercises? See
   [`SEMVER.md`](SEMVER.md).
8. How do `monotonic`, `acquire`, `release`, `acq_rel`, and `seq_cst` map from
   C++/Rust source orderings? See [`11-concurrency/04-memory-model-mapping.md`](11-concurrency/04-memory-model-mapping.md).
9. What is the fastest way to print IR after `mem2reg` but before
   `instcombine`? See [`07-optimization/05-debugging-passes.md`](07-optimization/05-debugging-passes.md).
10. What is the distinction between Loop Vectorizer and SLP Vectorizer input
   patterns? See [`09-vectorization/01-loop-vectorizer.md`](09-vectorization/01-loop-vectorizer.md) and [`09-vectorization/02-slp-vectorizer.md`](09-vectorization/02-slp-vectorizer.md).
11. When should you consult the vector-predication chapter for masks, tails,
    scalable vectors, or BCIR lane-validity lowering? See
    [`09-vectorization/03-vector-predication.md`](09-vectorization/03-vector-predication.md).
12. When a JIT says a symbol is missing, what object-layer artifacts should you
   inspect? See [`12-backend-jit/04-mc-and-relocations.md`](12-backend-jit/04-mc-and-relocations.md).
13. Which ABI attributes must stay synchronized between declarations and
    definitions? See [`13-advanced-ir/04-attributes.md`](13-advanced-ir/04-attributes.md).
14. Why is a branch fed by `add nsw` and `icmp` verifier-valid but unsafe for
    overflowing inputs, and where would `freeze` help? See
    [`13-advanced-ir/05-poison-undef-freeze.md`](13-advanced-ir/05-poison-undef-freeze.md).
15. Which fast-math flags allow NaN/infinity assumptions, signed-zero changes,
    reciprocal transforms, contraction, approximations, and reassociation? See
    [`13-advanced-ir/06-fast-math-flags.md`](13-advanced-ir/06-fast-math-flags.md).
16. Why can a crypto function that looks safe in static IR still leak on a CPU?
    See [`15-binary-analysis/01-microarchitecture-side-channels.md`](15-binary-analysis/01-microarchitecture-side-channels.md).
17. What build artifacts must you preserve to explain a PGO/LTO/BOLT optimized
    binary? See [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md).
18. Which cheap BCSA features should be extracted before dense embeddings? See
    [`15-binary-analysis/03-interpretable-bcsa-features.md`](15-binary-analysis/03-interpretable-bcsa-features.md).
19. Which exercise families now go beyond standalone IR writing, and how should
    intentionally broken repair inputs be named? See
    [`exercises/README.md`](exercises/README.md) and [`EXAMPLES.md`](EXAMPLES.md).
20. What should a learner predict before running `mem2reg`, `simplifycfg`, or
    Loop Vectorizer exercises? See exercises 020, 021, and 022 in
    [`exercises/`](exercises/).
21. Why should language-agnostic review prompts come before optional C++ pass
    skeleton exercises, and why should those skeletons remain outside normal IR
    verification? See [`exercises/README.md`](exercises/README.md).
22. Which BCIR mapping pages cover claim lowering, GAADMSF graph operations,
    Dragon Egg runtime-owned operations, runtime call boundaries, and diagnostic
    metadata preservation? See [`bcir-mapping/README.md`](bcir-mapping/README.md).
23. Which checked examples show graph fragments becoming struct arrays, claim
    resource lookup becoming registry loads, HAM hints becoming `llvm.prefetch`,
    BCIR operations becoming runtime wrappers, mixed strides becoming byte
    offsets, and diagnostic tags becoming custom metadata? See
    [`bcir-mapping/examples/`](bcir-mapping/examples/).

## Suggested scoring

- **21-23**: ready to edit examples, repair broken IR, and review BCIR lowering patches.
- **16-20**: read the linked chapters or exercise-family docs for missed questions and rerun the self-test.
- **0-15**: start from [`START_HERE.md`](START_HERE.md), then use
  [`RECIPES.md`](RECIPES.md) for task-specific paths.

## Path-specific self-test prompts

These prompts mirror the curriculum paths; use them after the numbered corpus self-test above.

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

**After the BCIR mapping path**
- Which claim-lowering stages happen before operation dispatch?
- When should a BCIR operation become plain LLVM IR versus a runtime-call wrapper?
- Why do HAM prefetch operands need immediate constants?
- Why should diagnostic metadata never carry semantics required for execution?
