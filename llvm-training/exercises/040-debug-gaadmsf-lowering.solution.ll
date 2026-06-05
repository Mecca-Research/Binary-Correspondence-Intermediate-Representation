source_filename = "040-debug-gaadmsf-lowering.solution.ll"

; Fixed GAADMSF lowering: each phi incoming block is an actual predecessor of
; %merge, and the phase metadata remains attached to the lowered operations.
define void @debug_gaadmsf_lowering(ptr %values, ptr %weights, ptr %out, i64 %index, i1 %active) {
entry:
  br i1 %active, label %gather, label %skip

gather:
  %value_ptr = getelementptr float, ptr %values, i64 %index
  %value = load float, ptr %value_ptr, align 4, !bcir.gaadmsf.phase !0
  %weight_ptr = getelementptr float, ptr %weights, i64 %index
  %weight = load float, ptr %weight_ptr, align 4, !bcir.gaadmsf.phase !1
  %applied = fmul float %value, %weight, !bcir.gaadmsf.phase !2
  br label %merge

skip:
  br label %merge

merge:
  %accum.merge = phi float [ %applied, %gather ], [ 0.000000e+00, %skip ]
  %out_ptr = getelementptr float, ptr %out, i64 %index
  store float %accum.merge, ptr %out_ptr, align 4, !bcir.gaadmsf.phase !3
  ret void
}

!bcir.gaadmsf.phases = !{!0, !1, !2, !3}
!0 = !{!"phase", !"gather", !"input", !"values"}
!1 = !{!"phase", !"gather", !"input", !"weights"}
!2 = !{!"phase", !"apply", !"operation", !"scale"}
!3 = !{!"phase", !"accumulate", !"target", !"out"}
