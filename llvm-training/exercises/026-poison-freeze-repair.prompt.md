# Exercise 026: Repair poison-prone control flow with `freeze`

## Task family

This is a **repair** exercise. The broken input assembles, but it branches on a
value that may be poison because it is derived from an `nsw` operation that can
overflow. Repair the IR by inserting `freeze` before the value controls the CFG.

## Broken input

Inspect:

```sh
llvm-training/exercises/026-poison-freeze-repair.invalid.ll.txt
```

## Required repair

Make the smallest change that ensures the branch condition is not poison. Keep
function behavior otherwise equivalent for non-poison inputs.

## Expected diagnostic command

```sh
llvm-as -disable-output llvm-training/exercises/026-poison-freeze-repair.invalid.ll.txt
```

## Expected diagnostic observation

The broken input may assemble because poison is a semantic problem rather than a
syntax error. A reviewer should still reject it for using a potentially poison
condition in control flow.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/026-poison-freeze-repair.solution.ll
```
