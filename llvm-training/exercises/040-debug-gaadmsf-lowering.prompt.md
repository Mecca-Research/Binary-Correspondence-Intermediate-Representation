# Exercise 040: Debug GAADMSF lowering

## Task family

This is an **advanced repair/debugging** exercise. Diagnose a malformed LLVM IR
lowering of a GAADMSF-style gather/apply/accumulate dataflow fragment.

## Broken input

Inspect:

```sh
llvm-training/exercises/040-debug-gaadmsf-lowering.invalid.ll.txt
```

The broken module tries to merge an accumulator value, but the `phi` node names a
predecessor that does not branch to the merge block.

## Required repair

Produce a standalone module that:

- keeps the gather, apply, and accumulate phases visible as simple scalar IR;
- fixes the control-flow graph so every `phi` incoming block is an actual
  predecessor of the merge block;
- preserves phase metadata on the gather load, apply multiply, and accumulate
  store.

## Expected diagnostic command

```sh
llvm-as llvm-training/exercises/040-debug-gaadmsf-lowering.invalid.ll.txt -o /tmp/040-debug-gaadmsf-lowering.invalid.bc && opt -passes=verify /tmp/040-debug-gaadmsf-lowering.invalid.bc -o /dev/null
```

## Expected diagnostic observation

`llvm-as` should reject the broken fixture during verification because the `phi`
node has an incoming block that is not a predecessor. If the IR is assembled with
a non-verifying path, `opt -passes=verify` should reject it for the same reason.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/040-debug-gaadmsf-lowering.solution.ll
```
