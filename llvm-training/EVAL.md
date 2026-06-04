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
6. How do `monotonic`, `acquire`, `release`, `acq_rel`, and `seq_cst` map from
   C++/Rust source orderings? See [`11-concurrency/04-memory-model-mapping.md`](11-concurrency/04-memory-model-mapping.md).
7. What is the fastest way to print IR after `mem2reg` but before
   `instcombine`? See [`07-optimization/05-debugging-passes.md`](07-optimization/05-debugging-passes.md).
8. What is the distinction between Loop Vectorizer and SLP Vectorizer input
   patterns? See [`09-vectorization/01-loop-vectorizer.md`](09-vectorization/01-loop-vectorizer.md) and [`09-vectorization/02-slp-vectorizer.md`](09-vectorization/02-slp-vectorizer.md).
9. When a JIT says a symbol is missing, what object-layer artifacts should you
   inspect? See [`12-backend-jit/04-mc-and-relocations.md`](12-backend-jit/04-mc-and-relocations.md).
10. Which ABI attributes must stay synchronized between declarations and
    definitions? See [`13-advanced-ir/04-attributes.md`](13-advanced-ir/04-attributes.md).
11. Why can a crypto function that looks safe in static IR still leak on a CPU?
    See [`15-binary-analysis/01-microarchitecture-side-channels.md`](15-binary-analysis/01-microarchitecture-side-channels.md).
12. What build artifacts must you preserve to explain a PGO/LTO/BOLT optimized
    binary? See [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md).
13. Which cheap BCSA features should be extracted before dense embeddings? See
    [`15-binary-analysis/03-interpretable-bcsa-features.md`](15-binary-analysis/03-interpretable-bcsa-features.md).

## Suggested scoring

- **11-13**: ready to edit examples and review BCIR lowering patches.
- **7-10**: read the linked chapters for missed questions and rerun the self-test.
- **0-6**: start from [`START_HERE.md`](START_HERE.md), then use
  [`RECIPES.md`](RECIPES.md) for task-specific paths.
