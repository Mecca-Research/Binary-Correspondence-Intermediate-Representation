# Exercise 005: Index a struct field with `getelementptr`

Write a standalone LLVM IR module that defines a two-field struct and this
function:

```llvm
define i64 @get_second(ptr %pair)
```

Use `getelementptr` to compute the address of the second field in a value of
struct type `{ i32, i64 }`, load that field, and return it.

## Expected behavior

Given a pointer to `{ i32 10, i64 99 }`, `@get_second` returns `99`.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/005-struct-gep.solution.ll
```
