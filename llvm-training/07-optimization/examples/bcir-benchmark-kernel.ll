; A target-portable BCIR-flavored benchmark kernel for pipeline experiments.
; The caller owns allocation, initialization, warm-up, repetition, and timing.
;
; Verify:
;   opt -passes=verify -disable-output bcir-benchmark-kernel.ll
;
; Compare pipelines:
;   opt -S -passes='default<O1>' bcir-benchmark-kernel.ll -o /tmp/bcir-o1.ll
;   opt -S -passes='default<O2>' bcir-benchmark-kernel.ll -o /tmp/bcir-o2.ll
;
; The loop computes a weighted sum over nonnegative input elements. Unknown
; metadata is intentional: experiments should check whether BCIR provenance and
; diagnostics survive each candidate pipeline.

source_filename = "bcir-benchmark-kernel.ll"

define i64 @bcir_benchmark_kernel(ptr nocapture readonly %input, i64 %count) !bcir.workload !0 {
entry:
  %empty = icmp eq i64 %count, 0, !bcir.reg !1, !bcir.diag !2
  br i1 %empty, label %exit, label %loop

loop:
  %index = phi i64 [ 0, %entry ], [ %next, %latch ], !bcir.reg !3
  %sum = phi i64 [ 0, %entry ], [ %sum.next, %latch ], !bcir.reg !4
  %element.ptr = getelementptr i32, ptr %input, i64 %index, !bcir.reg !5
  %value = load i32, ptr %element.ptr, align 4, !bcir.reg !6, !bcir.diag !7
  %keep = icmp sge i32 %value, 0
  br i1 %keep, label %accumulate, label %latch, !prof !8

accumulate:
  %wide = sext i32 %value to i64
  %weight = add i64 %index, 1
  %weighted = mul i64 %wide, %weight
  %with.value = add i64 %sum, %weighted
  br label %latch

latch:
  %sum.next = phi i64 [ %with.value, %accumulate ], [ %sum, %loop ]
  %next = add nuw i64 %index, 1
  %done = icmp eq i64 %next, %count
  br i1 %done, label %exit, label %loop, !llvm.loop !9

exit:
  %result = phi i64 [ 0, %entry ], [ %sum.next, %latch ]
  ret i64 %result, !bcir.diag !11
}

!0 = !{!"bcir.kernel", !"weighted-nonnegative-sum"}
!1 = !{!"source-register", i32 0}
!2 = !{!"empty-input-check"}
!3 = !{!"source-register", i32 1}
!4 = !{!"source-register", i32 2}
!5 = !{!"source-register", i32 3}
!6 = !{!"source-register", i32 4}
!7 = !{!"input-load"}
!8 = !{!"branch_weights", i32 96, i32 4}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"kernel-result"}
