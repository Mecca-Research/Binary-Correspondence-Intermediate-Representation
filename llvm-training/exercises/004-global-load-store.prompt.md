# Exercise 004: Load and store a global

Write a standalone LLVM IR module that declares a mutable global counter and
defines this function:

```llvm
define i32 @bump_global(i32 %delta)
```

The function should load the current global value, add `%delta`, store the new
value back to the global, and return the new value.

## Expected behavior

- If the global is `0`, `@bump_global(5)` stores and returns `5`.
- A later `@bump_global(2)` stores and returns `7`.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/004-global-load-store.solution.ll
```
