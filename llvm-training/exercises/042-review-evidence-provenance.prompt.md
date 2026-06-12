# Exercise 042: Review an evidence provenance claim

## Task family

This is an **adversarial evidence review** exercise.

A report says:

> Two binaries both have four arithmetic instructions and two basic blocks.
> Binary A was 7% faster in one CI run, so the binaries are semantically
> equivalent and A is universally faster.

## Required review points

Explain:

- why static feature similarity is not semantic equivalence;
- why one shared-host wall-clock result is host-sensitive rather than a
  deterministic golden value;
- which manifest and provenance fields are needed for the static artifacts;
- what measurement metadata and repetition policy a credible optional timing
  experiment should record;
- how to rewrite the conclusion with appropriately limited confidence.

## Verification command

Markdown review exercise; no LLVM tools are required. Inspect the reference with:

```sh
cat llvm-training/exercises/042-review-evidence-provenance.solution.md
```
