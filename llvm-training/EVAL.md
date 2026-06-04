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
14. Why can a crypto function that looks safe in static IR still leak on a CPU?
    See [`15-binary-analysis/01-microarchitecture-side-channels.md`](15-binary-analysis/01-microarchitecture-side-channels.md).
15. What build artifacts must you preserve to explain a PGO/LTO/BOLT optimized
    binary? See [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md).
16. Which cheap BCSA features should be extracted before dense embeddings? See
    [`15-binary-analysis/03-interpretable-bcsa-features.md`](15-binary-analysis/03-interpretable-bcsa-features.md).
17. Which exercise families now go beyond standalone IR writing, and how should
    intentionally broken repair inputs be named? See
    [`exercises/README.md`](exercises/README.md) and [`EXAMPLES.md`](EXAMPLES.md).
18. What should a learner predict before running `mem2reg`, `simplifycfg`, or
    Loop Vectorizer exercises? See exercises 020, 021, and 022 in
    [`exercises/`](exercises/).
19. Why should language-agnostic review prompts come before optional C++ pass
    skeleton exercises, and why should those skeletons remain outside normal IR
    verification? See [`exercises/README.md`](exercises/README.md).

## Suggested scoring

- **17-19**: ready to edit examples, repair broken IR, and review BCIR lowering patches.
- **12-16**: read the linked chapters or exercise-family docs for missed questions and rerun the self-test.
- **0-11**: start from [`START_HERE.md`](START_HERE.md), then use
  [`RECIPES.md`](RECIPES.md) for task-specific paths.
