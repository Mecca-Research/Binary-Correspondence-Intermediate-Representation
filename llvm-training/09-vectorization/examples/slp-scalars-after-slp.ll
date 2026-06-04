; Example of the kind of IR produced after SLP Vectorization.
;
; This is a teaching-sized, cleaned-up snapshot rather than a guarantee of
; exact output from every LLVM version or target. Real output may leave some
; scalar code in place or use a different lane count.
;
; Expected observation:
;   SLP packs independent scalar operations into <4 x i32> vector loads, vector
;   adds, vector stores, and a shufflevector for the permuted store order.

source_filename = "slp-scalars-after-slp.ll"
target triple = "x86_64-unknown-linux-gnu"

define void @slp_scalars(ptr noalias nocapture readonly %a,
                         ptr noalias nocapture readonly %b,
                         ptr noalias nocapture writeonly %c) {
entry:
  %wide.a = load <4 x i32>, ptr %a, align 4
  %wide.b = load <4 x i32>, ptr %b, align 4
  %wide.sum = add <4 x i32> %wide.a, %wide.b
  store <4 x i32> %wide.sum, ptr %c, align 4
  ret void
}

define void @slp_shuffle_candidate(ptr noalias nocapture readonly %a,
                                   ptr noalias nocapture readonly %b,
                                   ptr noalias nocapture writeonly %c) {
entry:
  %wide.a = load <4 x i32>, ptr %a, align 4
  %wide.b = load <4 x i32>, ptr %b, align 4
  %wide.sum = add <4 x i32> %wide.a, %wide.b
  %reordered = shufflevector <4 x i32> %wide.sum, <4 x i32> poison, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x i32> %reordered, ptr %c, align 4
  ret void
}
