# Exercise 029: Lower a HAM prefetch hint

## Task family

This is a **BCIR lowering** exercise. Lower a hierarchical-memory hint to an
LLVM prefetch intrinsic call while preserving the original HAM domain as
non-semantic metadata.

## Required LLVM constructs

Write a standalone module containing:

- A declaration of `@llvm.prefetch.p0`.
- `define void @lower_ham_prefetch_hint(ptr %base, i64 %index, i64 %stride)`.
- Explicit byte-offset arithmetic using `mul` and `getelementptr i8`.
- A prefetch call with immediate locality/cache operands.
- A custom `!bcir.ham.hint` metadata attachment and named metadata catalog.

## Expected observation

The module should assemble. The executable effect is the prefetch intrinsic; the
HAM domain survives as metadata for tools that understand BCIR conventions.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/029-lower-ham-prefetch-hint.solution.ll
```
