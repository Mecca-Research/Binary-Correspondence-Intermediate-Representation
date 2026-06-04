# Exercise 021: Predict `simplifycfg`

## Task family

This is an **optimization pass reasoning** exercise focused on CFG shape. The
input contains a tiny if/else diamond whose arms have no side effects and then
join through a `phi`.

## Input

Inspect:

```sh
llvm-training/exercises/021-predict-simplifycfg.input.ll
```

## Required prediction

Before running `opt`, answer:

1. Can the branch diamond be folded without changing observable behavior?
2. Which blocks are likely to disappear?
3. What instruction replaces the merge-block `phi`?
4. Why would side effects in either arm make this prediction less safe?

## Pass command

```sh
opt -S -passes=simplifycfg llvm-training/exercises/021-predict-simplifycfg.input.ll -o -
```

## Expected observation

A typical output computes both side-effect-free arm values in `entry`, chooses
between them with `select`, and returns directly. The checked-in teaching
snapshot shows that expected shape:

```sh
llvm-as -disable-output llvm-training/exercises/021-predict-simplifycfg.after-simplifycfg.ll
```
