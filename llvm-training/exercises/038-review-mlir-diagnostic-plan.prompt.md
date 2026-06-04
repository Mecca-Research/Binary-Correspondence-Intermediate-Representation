# Exercise 038: Review an MLIR diagnostic plan

## Task family

This is an **MLIR bridge review** exercise. Audit whether a BCIR-to-LLVM bridge
will report useful verifier and conversion failures.

## Candidate plan

A pass lowers `bcir.bind_register` and `bcir.ham_hint` during conversion to the
LLVM dialect. Unsupported HAM hints are dropped silently. Unsupported required
register bindings are also dropped silently because LLVM IR has no portable
physical-register assignment mechanism. Runtime ABI call declarations are emitted
at each call site.

## Required review points

Explain which diagnostics should be errors, which may be remarks, and how the
pass should report ABI or target-feature drift. Include at least one example of a
specific diagnostic message.

## Verification command

Markdown review exercise; no MLIR tool is required. The checked-in reference
answer is:

```sh
cat llvm-training/exercises/038-review-mlir-diagnostic-plan.solution.md
```
