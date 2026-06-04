# Exercise 009: Intrinsic call with metadata

Write a standalone LLVM IR module that wraps `llvm.memcpy` and attaches TBAA
metadata to the call:

```llvm
define void @copy16(ptr %dst, ptr %src)
```

The function should copy 16 bytes from `%src` to `%dst` with destination and
source alignment of 8.

## Required LLVM constructs

- Declaration of `@llvm.memcpy.p0.p0.i64`.
- A `call` to the intrinsic with constant length `16` and `isvolatile` false.
- A `!tbaa` attachment on the call.
- Named metadata nodes that make the attachment syntactically valid.

## Expected observation

The module assembles successfully and demonstrates that intrinsic calls can carry
ordinary instruction metadata.

## Verification command

```sh
llvm-as -disable-output llvm-training/exercises/009-intrinsic-metadata.solution.ll
```
