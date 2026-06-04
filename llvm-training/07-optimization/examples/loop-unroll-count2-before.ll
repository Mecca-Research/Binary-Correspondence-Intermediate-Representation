; Run:
;   opt -S -passes=loop-unroll loop-unroll-count2-before.ll -o loop-unroll-count2-after.ll
;
; Metadata asks for unroll-by-2. Unlike full unrolling, the result still has a
; loop, but each iteration performs two source iterations.

target triple = "x86_64-unknown-linux-gnu"
source_filename = "loop-unroll-count2-before.ll"

define void @scale_eight(ptr nocapture %a) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p, align 4
  %scaled = shl i32 %v, 1
  store i32 %scaled, ptr %p, align 4
  %next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %next, 8
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.count", i32 2}
