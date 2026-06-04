# Exercise 024: Attach profile metadata branch weights

## Task family

This is a **beginner/intermediate IR-writing** exercise. Model a hot/cold branch
using LLVM profile metadata so an optimizer can prefer the hot path.

## Required LLVM constructs

Write a standalone module containing:

- `define i32 @classify_hot_path(i32 %x)`.
- A conditional branch comparing `%x` with zero.
- `!prof` branch-weight metadata on the conditional branch.
- Distinct hot and cold blocks that merge through a `phi` node.

## Expected observation

The module assembles, and the conditional branch carries a `!prof` attachment
with a `!"branch_weights"` metadata node.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/024-profile-metadata-branch-weights.solution.ll
```
