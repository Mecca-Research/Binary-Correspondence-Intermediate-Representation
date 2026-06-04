# Exercise 017: Repair a duplicate symbol

## Task family

This is a **repair** exercise. Start from an intentionally broken module and
repair the symbol table without changing the bodies' arithmetic behavior.

## Broken input

Inspect:

```sh
llvm-training/exercises/017-fix-duplicate-symbol.invalid.ll.txt
```

The module is meant to expose one function that increments an `i32` and one
function that decrements an `i32`.

## Required repair

Give the two definitions distinct global names. Keep both functions externally
visible and keep their function types unchanged.

## Expected diagnostic command

```sh
llvm-as -disable-output llvm-training/exercises/017-fix-duplicate-symbol.invalid.ll.txt
```

## Expected diagnostic observation

LLVM should reject the broken input because the global symbol `@adjust` is
defined more than once in the same module.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/017-fix-duplicate-symbol.solution.ll
```
