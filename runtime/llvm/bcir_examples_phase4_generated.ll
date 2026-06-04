source_filename = "bcir_examples_phase4_generated.ll"
target triple = "x86_64-unknown-linux-gnu"
target datalayout = ""

%bcir.claim = type { i64, [4 x i32], [4 x i32], i64, [2 x i64] }
%bcir.res = type { i32, i32, ptr, i64, i64, i64, i64 }
%bcir.batch = type { i32, i32, i32, i32, i32, i32, i64, i64, i64 }
%bcir.phase.range = type { i32, i32, i64, i64 }

@bcir.generated.claims = global [2 x %bcir.claim] [
  %bcir.claim { i64 281474976710913, [4 x i32] [i32 1, i32 0, i32 0, i32 0], [4 x i32] [i32 0, i32 0, i32 0, i32 0], i64 65793, [2 x i64] [i64 0, i64 8] },
  %bcir.claim { i64 281474976776451, [4 x i32] [i32 1, i32 2, i32 0, i32 0], [4 x i32] [i32 3, i32 0, i32 0, i32 0], i64 131585, [2 x i64] [i64 0, i64 8] }
], align 64

@bcir.generated.resources = global [3 x %bcir.res] [
  %bcir.res { i32 1, i32 0, ptr null, i64 4096, i64 0, i64 0, i64 0 },
  %bcir.res { i32 2, i32 0, ptr null, i64 4096, i64 0, i64 0, i64 0 },
  %bcir.res { i32 3, i32 0, ptr null, i64 4096, i64 0, i64 0, i64 0 }
], align 64

@bcir.generated.batches = global [2 x %bcir.batch] [
  %bcir.batch { i32 0, i32 0, i32 1, i32 1, i32 0, i32 0, i64 0, i64 1, i64 0 },
  %bcir.batch { i32 0, i32 1, i32 1, i32 3, i32 0, i32 0, i64 1, i64 1, i64 0 }
], align 64

@bcir.generated.phase_ranges = global [2 x %bcir.phase.range] [
  %bcir.phase.range { i32 0, i32 0, i64 0, i64 1 },
  %bcir.phase.range { i32 0, i32 1, i64 1, i64 2 }
], align 64

!bcir.generated.module = !{!900}
!900 = !{!"module", !"vector_add", !"claim_layout", !"BCIR_ClaimV2", !"schedule", !"global_indexed"}
