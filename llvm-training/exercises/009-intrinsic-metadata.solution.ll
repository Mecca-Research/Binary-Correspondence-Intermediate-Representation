; Exercise 009: Intrinsic call with metadata.
; Verification: llvm-as -disable-output llvm-training/exercises/009-intrinsic-metadata.solution.ll

source_filename = "009-intrinsic-metadata.solution.ll"

declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly, ptr noalias readonly, i64, i1 immarg)

define void @copy16(ptr %dst, ptr %src) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 16, i1 false), !tbaa !2
  ret void
}

!0 = !{!"Simple C/C++ TBAA"}
!1 = !{!"omnipotent char", !0, i64 0}
!2 = !{!1, !1, i64 0}
