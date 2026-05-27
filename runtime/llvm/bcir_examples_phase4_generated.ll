source_filename = "bcir_examples_phase4_generated.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }
%bcir.res = type { i32, i32, ptr, i64, i64, i64, i64 }
%bcir.batch = type { i32, i32, i32, i32, i32, i32, i64, i64, i64 }
%bcir.phase.range = type { i32, i32, i64, i64 }

@bcir.generated.claims = global [2 x %bcir.claim] [
  %bcir.claim { i64 65537, [4 x i32] [i32 1, i32 0, i32 0, i32 0], [4 x i32] zeroinitializer, i64 0, [2 x i64] [i64 0, i64 8] },
  %bcir.claim { i64 66307, [4 x i32] [i32 1, i32 2, i32 0, i32 0], [4 x i32] [i32 3, i32 0, i32 0, i32 0], i64 0, [2 x i64] [i64 0, i64 8] }
], align 64
