; Before vectorization: scalar conditional copy with per-lane store predication.
;
; Try:
;   opt -S -passes=loop-vectorize \
;     -pass-remarks=loop-vectorize \
;     -pass-remarks-analysis=loop-vectorize \
;     -pass-remarks-missed=loop-vectorize \
;     llvm-training/09-vectorization/examples/masked-load-store-before.ll -o -
;   opt -S -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 \
;     llvm-training/09-vectorization/examples/masked-load-store-before.ll -o -
;
; Expected observation:
;   The loop has unit-stride loads from %src, conditional stores to %dst, noalias
;   pointers, and loop metadata that requests vectorization. A target that can
;   lower masked stores cheaply is more likely to keep a vector predicated body.

source_filename = "masked-load-store-before.ll"
target triple = "x86_64-unknown-linux-gnu"

define void @copy_if_greater(ptr noalias nocapture readonly %src,
                            ptr noalias nocapture writeonly %dst,
                            i64 %n,
                            i32 %threshold) {
entry:
  %has.work = icmp sgt i64 %n, 0
  br i1 %has.work, label %loop, label %exit

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %src.p = getelementptr inbounds i32, ptr %src, i64 %i
  %value = load i32, ptr %src.p, align 16
  %keep = icmp sgt i32 %value, %threshold
  br i1 %keep, label %store, label %latch

store:
  %dst.p = getelementptr inbounds i32, ptr %dst, i64 %i
  store i32 %value, ptr %dst.p, align 16
  br label %latch

latch:
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
