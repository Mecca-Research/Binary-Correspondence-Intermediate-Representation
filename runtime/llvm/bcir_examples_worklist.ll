source_filename = "bcir_examples_worklist.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }

@bcir.example.claims = global [5 x %bcir.claim] [
  %bcir.claim { i64 49, [4 x i32] zeroinitializer, [4 x i32] zeroinitializer, i64 0, [2 x i64] zeroinitializer },
  %bcir.claim { i64 257, [4 x i32] [i32 1, i32 0, i32 0, i32 0], [4 x i32] zeroinitializer, i64 0, [2 x i64] [i64 0, i64 0] },
  %bcir.claim { i64 1072, [4 x i32] zeroinitializer, [4 x i32] [i32 1, i32 0, i32 0, i32 0], i64 0, [2 x i64] [i64 0, i64 1] },
  %bcir.claim { i64 560, [4 x i32] zeroinitializer, [4 x i32] zeroinitializer, i64 2, [2 x i64] zeroinitializer },
  %bcir.claim { i64 562, [4 x i32] zeroinitializer, [4 x i32] zeroinitializer, i64 0, [2 x i64] zeroinitializer }
], align 64
