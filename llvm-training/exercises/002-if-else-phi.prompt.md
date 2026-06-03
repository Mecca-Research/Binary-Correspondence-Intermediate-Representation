# Exercise 002: If/else with `phi`

Write a standalone LLVM IR module that defines this function:

```llvm
define i32 @select_max(i32 %a, i32 %b)
```

The function should compare `%a` and `%b`, branch to separate `then` and `else`
blocks, and use a `phi` node in a merge block to return the larger signed value.

## Expected behavior

- `@select_max(7, 3)` returns `7`.
- `@select_max(3, 7)` returns `7`.
- Equal values may return either incoming value because they are the same value.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/002-if-else-phi.solution.ll
```
