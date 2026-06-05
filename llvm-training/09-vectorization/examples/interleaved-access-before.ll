; Before vectorization: scalar loop that deinterleaves pairs from an AoS buffer.
;
; Try:
;   opt -S -passes=loop-vectorize \
;     -pass-remarks=loop-vectorize \
;     -pass-remarks-analysis=loop-vectorize \
;     -pass-remarks-missed=loop-vectorize \
;     llvm-training/09-vectorization/examples/interleaved-access-before.ll -o -
;   opt -S -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 \
;     llvm-training/09-vectorization/examples/interleaved-access-before.ll -o -
;
; Expected observation:
;   The logical output streams are unit-stride, but the input stream is stride-2.
;   The Loop Vectorizer may represent this as a wide load plus lane shuffles when
;   the target cost model considers deinterleaving profitable.

source_filename = "interleaved-access-before.ll"
target triple = "x86_64-unknown-linux-gnu"

define void @deinterleave_i32x2(ptr noalias nocapture readonly %pairs,
                               ptr noalias nocapture writeonly %xs,
                               ptr noalias nocapture writeonly %ys,
                               i64 %n) {
entry:
  %has.work = icmp sgt i64 %n, 0
  br i1 %has.work, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %base.index = shl nuw nsw i64 %i, 1
  %x.p = getelementptr inbounds i32, ptr %pairs, i64 %base.index
  %y.index = or i64 %base.index, 1
  %y.p = getelementptr inbounds i32, ptr %pairs, i64 %y.index
  %x = load i32, ptr %x.p, align 8
  %y = load i32, ptr %y.p, align 4
  %xs.p = getelementptr inbounds i32, ptr %xs, i64 %i
  %ys.p = getelementptr inbounds i32, ptr %ys, i64 %i
  store i32 %x, ptr %xs.p, align 4
  store i32 %y, ptr %ys.p, align 4
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.interleave.count", i32 1}
