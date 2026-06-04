# Exercise 031: Lower a BCIR runtime call boundary

## Task family

This is a **BCIR lowering** exercise. Convert a high-level runtime operation into
an explicit LLVM call boundary with stable ABI types and conservative attributes.

## Required LLVM constructs

Write a standalone module containing:

- A declaration of a runtime function named `@bcir_runtime_commit`.
- `define i32 @lower_runtime_call_boundary(ptr %ctx, ptr %claim, i64 %bytes)`.
- A call that passes only ABI-stable LLVM scalar or pointer types.
- Branching on the status code and returning either the status or zero.

## Expected observation

The module should assemble and keep the runtime boundary explicit instead of
inlining BCIR-specific side effects into undocumented metadata.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/031-lower-runtime-call-boundary.solution.ll
```
