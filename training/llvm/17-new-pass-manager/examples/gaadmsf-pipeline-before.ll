; RUN: opt -S -passes='verify,function(require<domtree>),sccp,loop-rotate,verify' %s -o -
;
; A small BCIR/GAADMSF-flavored loop before modern-pass-manager experiments.
; The metadata is illustrative but valid LLVM IR metadata.

target triple = "x86_64-unknown-linux-gnu"

!llvm.module.flags = !{!0}
!bcir.module = !{!1}
!gaadmsf.profile = !{!2}

!0 = !{i32 1, !"bcir.gaadmsf", i32 1}
!1 = !{!"stage", !"register-bound"}
!2 = !{!"hardware", !"gaadmsf-v1", !"ham", i1 true}
!3 = !{!"bcir.reg", i32 0, !"r0"}
!4 = !{!"bcir.reg", i32 1, !"r1"}
!5 = !{!"bcir.diag", !"preserve source register correspondence"}
!6 = !{!"ham", !"prefetch-ok"}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}

define i32 @gaadmsf_accumulate(ptr nocapture readonly %in, ptr nocapture %out, i32 %n) #0 {
entry:
  %has.work = icmp sgt i32 %n, 0, !bcir.reg !3, !bcir.diag !5
  br i1 %has.work, label %loop, label %exit

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ], !bcir.reg !3
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %loop ], !bcir.reg !4
  %gep = getelementptr inbounds i32, ptr %in, i32 %iv
  %val = load i32, ptr %gep, align 4, !bcir.reg !4, !bcir.diag !5, !ham !6
  %is.zero = icmp eq i32 %val, 0
  %add = add i32 %sum, %val, !bcir.reg !4
  %sum.next = select i1 %is.zero, i32 %sum, i32 %add, !bcir.reg !4
  %iv.next = add nuw nsw i32 %iv, 1, !bcir.reg !3
  %done = icmp eq i32 %iv.next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !7

exit:
  %result = phi i32 [ 0, %entry ], [ %sum.next, %loop ]
  store i32 %result, ptr %out, align 4, !bcir.diag !5
  ret i32 %result
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind "bcir.stage"="register-bound" "bcir.ham"="required" "gaadmsf.profile"="v1" }
