# Exercise 018: Repair an `immarg` intrinsic call

## Task family

This is a **repair** exercise for intrinsic contracts. Some LLVM intrinsics mark
operands as `immarg`, which means the call operand must be an immediate constant,
not an arbitrary SSA value.

## Broken input

Inspect:

```sh
llvm-training/exercises/018-fix-immarg-intrinsic.invalid.ll.txt
```

The module tries to wrap `llvm.prefetch`.

## Required repair

Change the wrapper so the prefetch read/write, locality, and cache-type operands
are immediate integer constants at the call site. It is acceptable to specialize
the wrapper name and remove the now-unused dynamic parameter.

## Expected diagnostic command

```sh
llvm-as -disable-output llvm-training/exercises/018-fix-immarg-intrinsic.invalid.ll.txt
```

## Expected diagnostic observation

LLVM should reject the broken input because an operand declared `immarg` is
supplied by the runtime value `%rw`.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/018-fix-immarg-intrinsic.solution.ll
```
