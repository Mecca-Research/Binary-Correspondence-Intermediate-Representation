; After vectorization sketch: fixed-width vector loop with masked store.
;
; This is a cleaned-up teaching snapshot, not byte-for-byte output from one LLVM
; release. It shows the IR features to look for after the Loop Vectorizer turns
; a scalar conditional store into per-lane predication.

source_filename = "masked-load-store-after-vectorize.ll"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.masked.store.v4i32.p0(<4 x i32>, ptr nocapture, i32 immarg, <4 x i1>)

define void @copy_if_greater_vector_body(ptr noalias nocapture readonly %src,
                                        ptr noalias nocapture writeonly %dst,
                                        i64 %n,
                                        i32 %threshold) {
entry:
  %has.vector = icmp uge i64 %n, 4
  br i1 %has.vector, label %vector.body, label %exit

vector.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %vector.body ]
  %src.p = getelementptr inbounds i32, ptr %src, i64 %i
  %values = load <4 x i32>, ptr %src.p, align 16
  %threshold.splat.insert = insertelement <4 x i32> poison, i32 %threshold, i64 0
  %threshold.splat = shufflevector <4 x i32> %threshold.splat.insert, <4 x i32> poison, <4 x i32> zeroinitializer
  %mask = icmp sgt <4 x i32> %values, %threshold.splat
  %dst.p = getelementptr inbounds i32, ptr %dst, i64 %i
  call void @llvm.masked.store.v4i32.p0(<4 x i32> %values, ptr %dst.p, i32 16, <4 x i1> %mask)
  %i.next = add nuw nsw i64 %i, 4
  %more = icmp ult i64 %i.next, %n
  br i1 %more, label %vector.body, label %exit

exit:
  ret void
}
