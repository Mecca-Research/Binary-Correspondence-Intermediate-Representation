# Exercise 007: Vector reduction intrinsic

Write a standalone LLVM IR module that declares the integer add-reduction
intrinsic for `<4 x i32>` and defines:

```llvm
define i32 @sum4(<4 x i32> %v)
```

The function should call the intrinsic and return the horizontal sum of all four
lanes.

## Required LLVM constructs

- Declaration of `@llvm.vector.reduce.add.v4i32`.
- A vector argument of type `<4 x i32>`.
- A `call` returning the scalar reduction result.

## Expected observation

For `<4 x i32> <1, 2, 3, 4>`, `@sum4` returns `10`.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/007-vector-reduction.solution.ll
```
