; After vectorization sketch: deinterleaving with a wide load and shuffles.
;
; This cleaned-up snapshot emphasizes the common shape: load several adjacent
; AoS elements, split even and odd lanes with shufflevector, then store the SoA
; result streams with vector stores.

source_filename = "interleaved-access-after-vectorize.ll"
target triple = "x86_64-unknown-linux-gnu"

define void @deinterleave_i32x2_vector_body(ptr noalias nocapture readonly %pairs,
                                           ptr noalias nocapture writeonly %xs,
                                           ptr noalias nocapture writeonly %ys,
                                           i64 %n) {
entry:
  %has.vector = icmp uge i64 %n, 4
  br i1 %has.vector, label %vector.body, label %exit

vector.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %vector.body ]
  %base.index = shl nuw nsw i64 %i, 1
  %pairs.p = getelementptr inbounds i32, ptr %pairs, i64 %base.index
  %wide = load <8 x i32>, ptr %pairs.p, align 8
  %xs.vec = shufflevector <8 x i32> %wide, <8 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %ys.vec = shufflevector <8 x i32> %wide, <8 x i32> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %xs.p = getelementptr inbounds i32, ptr %xs, i64 %i
  %ys.p = getelementptr inbounds i32, ptr %ys, i64 %i
  store <4 x i32> %xs.vec, ptr %xs.p, align 4
  store <4 x i32> %ys.vec, ptr %ys.p, align 4
  %i.next = add nuw nsw i64 %i, 4
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %vector.body, label %exit

exit:
  ret void
}
