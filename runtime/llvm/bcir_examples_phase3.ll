source_filename = "bcir_examples_phase3.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }
%bcir.batch = type { i32, i32, i32, i32, i32, i32, i64, i64, i64 }
%bcir.phase.range = type { i32, i32, i64, i64 }
%bcir.prefetch.profile = type { i32, i32, i32, i32, i32, i32, i64 }
%bcir.stream.pack = type { ptr, i64, ptr, i64, ptr, i64, ptr, i64 }

@bcir.example.claims.phase3 = global [4 x %bcir.claim] zeroinitializer, align 64

@bcir.example.batches.phase3 = global [3 x %bcir.batch] [
  %bcir.batch { i32 0, i32 0, i32 1, i32 1, i32 2, i32 8, i64 0, i64 1, i64 0 },
  %bcir.batch { i32 0, i32 1, i32 1, i32 3, i32 2, i32 8, i64 1, i64 1, i64 0 },
  %bcir.batch { i32 0, i32 2, i32 1, i32 2, i32 2, i32 8, i64 2, i64 1, i64 0 }
], align 64

@bcir.example.phases.phase3 = global [3 x %bcir.phase.range] [
  %bcir.phase.range { i32 0, i32 0, i64 0, i64 1 },
  %bcir.phase.range { i32 0, i32 1, i64 1, i64 2 },
  %bcir.phase.range { i32 0, i32 2, i64 2, i64 3 }
], align 64

@bcir.example.prefetch.phase3 = global [1 x %bcir.prefetch.profile] [
  %bcir.prefetch.profile { i32 2, i32 1, i32 3, i32 0, i32 0, i32 0, i64 0 }
], align 64

@bcir.example.stream.pack.phase3 = global %bcir.stream.pack {
  ptr @bcir.example.claims.phase3,
  i64 4,
  ptr @bcir.example.batches.phase3,
  i64 3,
  ptr @bcir.example.phases.phase3,
  i64 3,
  ptr @bcir.example.prefetch.phase3,
  i64 1
}, align 64
