# Exercise 020: Predict `mem2reg`

## Task family

This is an **optimization pass reasoning** exercise. Instead of writing LLVM IR
from scratch, predict how a standard transformation changes an existing module.

## Input

Inspect:

```sh
llvm-training/exercises/020-predict-mem2reg.input.ll
```

## Required prediction

Before running `opt`, write down:

1. Which `alloca` is promotable and why.
2. Which `load` and `store` instructions should disappear.
3. Where `mem2reg` must introduce a `phi` node.
4. Which arithmetic instructions remain unchanged.

## Pass command

```sh
opt -S -passes=mem2reg llvm-training/exercises/020-predict-mem2reg.input.ll -o -
```

## Expected observation

The promoted output should have no `alloca`, `load`, or `store` for `%slot`.
The merge block should receive the values from `%then` and `%else` through a
`phi`. The checked-in teaching snapshot shows the expected shape:

```sh
llvm-as -disable-output llvm-training/exercises/020-predict-mem2reg.after-mem2reg.ll
```
