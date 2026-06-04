# Exercise 016: Repair a `phi` predecessor mismatch

## Task family

This is a **repair** exercise. Start from an intentionally broken LLVM IR input,
identify why LLVM rejects it, and make the smallest semantic-preserving edit that
produces a valid standalone module.

## Broken input

Inspect:

```sh
llvm-training/exercises/016-fix-phi-predecessor.invalid.ll.txt
```

The function is meant to return `%x` when `%x > 0` and `0` otherwise.

## Required repair

Fix the `phi` in `@clamp_positive` so every incoming block is an actual CFG
predecessor of the block containing the `phi`, and every predecessor is listed
exactly once.

## Expected diagnostic command

```sh
llvm-as -disable-output llvm-training/exercises/016-fix-phi-predecessor.invalid.ll.txt
```

## Expected diagnostic observation

LLVM should reject the broken input because the incoming block list for the
`phi` node does not match the actual predecessors of `%merge`.

## Verification command

After applying the repair, the checked-in reference solution should assemble:

```sh
llvm-as -disable-output llvm-training/exercises/016-fix-phi-predecessor.solution.ll
```
