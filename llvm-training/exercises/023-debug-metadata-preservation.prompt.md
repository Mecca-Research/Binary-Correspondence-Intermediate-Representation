# Exercise 023: Preserve debug metadata while simplifying IR

## Task family

This is a **beginner/intermediate IR-writing and review** exercise. You are given
an `alloca`-based function shape and must write a cleaned-up LLVM IR solution
that keeps useful source-level debug locations attached to the surviving
instructions.

## Required LLVM constructs

Write a standalone module containing:

- `define i32 @accumulate_debug(i32 %a, i32 %b)`.
- An `add` and a `mul` that implement `(a + b) * 2`.
- `!dbg` locations on the function, arithmetic instructions, and `ret`.
- Minimal compile-unit, file, subprogram, and module-flag metadata so the module
  assembles under LLVM's opaque-pointer mode.

## Expected observation

The module should assemble, and `llvm-dis` should show that the `!dbg`
attachments remain on the surviving arithmetic and return instructions.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/023-debug-metadata-preservation.solution.ll
```
