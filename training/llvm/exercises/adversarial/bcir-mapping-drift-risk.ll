; adversarial-class: assemble-valid-semantically-risky
; Risk: r.left and r.right are distinct BCIR registers. Their current values are
; equal, but folding both to one undocumented LLVM value loses 1:1 provenance.

define i32 @bcir_mapping_drift_risk(i32 %input) {
entry:
  %left = add i32 %input, 0, !bcir.reg !0
  %right = add i32 %input, 0, !bcir.reg !1
  %sum = add i32 %left, %right, !bcir.reg !2
  ret i32 %sum
}

!bcir.mapping = !{!3, !4, !5}
!0 = !{!"bcir.reg", !"r.left"}
!1 = !{!"bcir.reg", !"r.right"}
!2 = !{!"bcir.reg", !"r.sum"}
!3 = !{!"r.left", !"left"}
!4 = !{!"r.right", !"right"}
!5 = !{!"r.sum", !"sum"}
