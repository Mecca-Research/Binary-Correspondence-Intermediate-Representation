# Exercise 019: Repair an atomic ordering mismatch

## Task family

This is a **repair** exercise for LLVM's memory-ordering rules. The syntax of an
atomic instruction must match the direction of synchronization it can perform.

## Broken input

Inspect:

```sh
llvm-training/exercises/019-fix-atomic-ordering.invalid.ll.txt
```

The function is intended to publish a value to memory for another thread to
observe later.

## Required repair

Replace the invalid ordering with a valid store ordering that preserves publish
semantics. Do not remove `atomic`, and keep the explicit alignment.

## Expected diagnostic command

```sh
llvm-as -disable-output llvm-training/exercises/019-fix-atomic-ordering.invalid.ll.txt
```

## Expected diagnostic observation

LLVM should reject the broken input because `acquire` is not a valid ordering for
an atomic `store`.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/019-fix-atomic-ordering.solution.ll
```
