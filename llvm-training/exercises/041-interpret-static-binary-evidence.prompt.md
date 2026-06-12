# Exercise 041: Interpret static binary evidence

## Task family

This is an **evidence review** exercise. Read these manifest-backed artifacts:

- `llvm-training/15-binary-analysis/fixtures/generated/linear-arithmetic.csv`
- `llvm-training/15-binary-analysis/fixtures/generated/branch-and-call.csv`
- `llvm-training/15-binary-analysis/evidence-manifest.json`

## Required review points

Write a short report that:

1. identifies at least three static differences between the fixtures, including
   control flow or call edges;
2. explains what the deterministic classification permits you to reproduce;
3. states why instruction-class or basic-block similarity would not prove
   semantic equivalence;
4. names at least two additional checks needed before making an equivalence
   claim.

Do not infer wall-clock performance, branch-miss rate, or cache behavior from the
static rows.

## Verification command

Markdown review exercise; no LLVM tools are required. Inspect the reference with:

```sh
cat llvm-training/exercises/041-interpret-static-binary-evidence.solution.md
```
