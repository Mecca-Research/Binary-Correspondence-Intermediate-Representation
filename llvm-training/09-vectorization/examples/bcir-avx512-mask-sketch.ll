; BCIR/AVX-512 mask lowering sketch.
;
; The mask is a semantic input from BCIR lane validity. On AVX-512 this may lower
; to k-register predication; on AVX2-like targets, similar IR may become blends,
; full-width memory operations, or scalar fallback depending on legality and cost.
;
; Try:
;   opt -S -passes=verify bcir-avx512-mask-sketch.ll -o -

source_filename = "bcir-avx512-mask-sketch.ll"
target triple = "x86_64-unknown-linux-gnu"

declare <16 x i32> @llvm.masked.load.v16i32.p0(ptr nocapture, i32 immarg, <16 x i1>, <16 x i32>)
declare void @llvm.masked.store.v16i32.p0(<16 x i32>, ptr nocapture, i32 immarg, <16 x i1>)

define void @bcir_avx512_mask_boundary(ptr %src, ptr %dst, <16 x i1> %active, <16 x i32> %passthru) {
entry:
  %values = call <16 x i32> @llvm.masked.load.v16i32.p0(ptr %src, i32 64, <16 x i1> %active, <16 x i32> %passthru), !bcir.node !0
  %biased = add <16 x i32> %values, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  call void @llvm.masked.store.v16i32.p0(<16 x i32> %biased, ptr %dst, i32 64, <16 x i1> %active), !bcir.node !1
  ret void
}

!0 = !{!"bcir.node", i32 300, !"masked lane load"}
!1 = !{!"bcir.node", i32 301, !"masked lane store"}
