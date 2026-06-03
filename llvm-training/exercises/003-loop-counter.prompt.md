# Exercise 003: Loop counter

Write a standalone LLVM IR module that defines this function:

```llvm
define i32 @count_to_n(i32 %n)
```

The function should use a loop-carried `phi` node for a counter that starts at
`0`, increments by `1`, and exits when the counter is no longer less than `%n`.
Return the final counter value.

## Expected behavior

- `@count_to_n(0)` returns `0`.
- `@count_to_n(4)` returns `4`.
- For a negative `%n`, this version returns `0` because the initial comparison fails.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/003-loop-counter.solution.ll
```
