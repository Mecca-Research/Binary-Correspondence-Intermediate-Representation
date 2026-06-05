# Exercise 022: Predict Loop Vectorizer behavior

## Task family

This is an **optimization pass reasoning** exercise. The goal is not to memorize
an exact textual output, because Loop Vectorizer output depends on LLVM version,
target, cost model, vector width, interleave count, and follow-on cleanup passes.
Instead, predict whether the loop has the structural properties that make
vectorization legal and profitable.

## Input

Inspect:

```sh
llvm-training/exercises/022-predict-loop-vectorizer.input.ll
```

## Required prediction

Before running `opt`, answer:

1. What is the induction variable, and is the loop count computable from `%n`?
2. Which memory accesses are consecutive across iterations?
3. Which attributes help the vectorizer reason that `%src` and `%dst` do not
   alias?
4. What vector operations would you expect in a vectorized body?
5. If a vectorizer or BCIR lowering creates lane masks from poison-capable
   arithmetic, where would the [BCIR `freeze` safe-speculation rule][bcir-freeze-rule]
   require stabilization before those masks control selects or predicated lanes?
6. What scalar remainder or early-exit path might still be needed?

## Pass command

Use a forced width to make the teaching observation easier to see:

```sh
opt -S -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 llvm-training/exercises/022-predict-loop-vectorizer.input.ll -o -
```

## Expected observation

When vectorized, the transformed function should contain a vector loop body with
`<4 x i32>` loads, shifts or adds, and stores, plus control flow for iterations
that are not a multiple of the vector width. If your LLVM build decides not to
vectorize, rerun with optimization remarks and explain the reported legality or
profitability reason:

```sh
opt -S -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -pass-remarks-missed=loop-vectorize llvm-training/exercises/022-predict-loop-vectorizer.input.ll -o /dev/null
```

For a stable teaching snapshot of vectorized IR, compare the chapter example:

```sh
llvm-as -disable-output llvm-training/09-vectorization/examples/sum-loop-after-loop-vectorize.ll
```

[bcir-freeze-rule]: ../13-advanced-ir/05-poison-undef-freeze.md#bcir-safe-speculation-with-freeze
