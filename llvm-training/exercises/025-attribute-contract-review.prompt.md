# Exercise 025: Review an LLVM attribute contract

## Task family

This is a **review** exercise. Decide whether a proposed function signature uses
LLVM attributes as a sound contract or over-promises behavior that the function
body and callers cannot guarantee.

## Candidate IR to review

```llvm
declare noalias nonnull ptr @lookup_buffer(ptr nocapture readonly %table, i64 %id)

define i32 @read_first(ptr nonnull readonly dereferenceable(4) %p) {
entry:
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
```

## Required review points

Explain:

- Which attributes are promises by the caller, callee, or both.
- Why `nonnull`, `dereferenceable`, `readonly`, `nocapture`, and `noalias` can be
  optimization-enabling but dangerous if guessed.
- What evidence is needed before preserving these attributes through a BCIR or
  MLIR lowering boundary.

## Expected observation

A good answer separates type facts from semantic promises and recommends keeping
only attributes proven by the source ABI, ownership model, or function body.

## Verification command

Markdown review exercise; no LLVM assembler command is required. The checked-in
reference answer is:

```sh
cat llvm-training/exercises/025-attribute-contract-review.solution.md
```
