# Exercise 006: GEP into an array of structs

Write a standalone LLVM IR module that defines this struct and function:

```llvm
%Entry = type { i32, i64 }
define i64 @load_entry_value(ptr %base, i64 %index)
```

`%base` points to the first element of an array of `%Entry`. Use
`getelementptr` to select element `%index`, then field `1`, load the `i64`, and
return it.

## Required LLVM constructs

- A named struct type.
- A two-step or single-step `getelementptr` that indexes an array element and a
  struct field.
- A final `load i64` from the computed field pointer.

## Expected observation

Given an array whose third element is `{ i32 7, i64 123 }`, calling
`@load_entry_value(base, 2)` returns `123`.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/006-array-of-structs-gep.solution.ll
```
