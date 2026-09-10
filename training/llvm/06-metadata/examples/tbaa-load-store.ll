; TBAA metadata describes aliasing categories for memory operations.
source_filename = "tbaa-load-store.c"

define i32 @tbaa_demo(ptr %p, ptr %q) {
entry:
  %a = load i32, ptr %p, align 4, !tbaa !3
  store i32 %a, ptr %q, align 4, !tbaa !3
  ret i32 %a
}

!0 = !{!"Simple C/C++ TBAA"}
!1 = !{!"omnipotent char", !0, i64 0}
!2 = !{!"int", !1, i64 0}
!3 = !{!2, !2, i64 0}
