; adversarial-class: metadata-preservation
; adversarial-preserve-metadata: !bcir.mapping
; Risk: a pass that replaces the load must transfer its BCIR identity or retain
; equivalent evidence in the named metadata side table.

@bcir.value = internal global i32 7, align 4

define i32 @metadata_drop_risk() {
entry:
  %value = load i32, ptr @bcir.value, align 4, !bcir.reg !1
  ret i32 %value
}

!bcir.mapping = !{!0}
!0 = !{!"r.load", !"metadata_drop_risk", !"value"}
!1 = !{!"bcir.reg", !"r.load"}
