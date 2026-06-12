; Required governor state is explicit; metadata carries ignorable policy.

source_filename = "calibration-governor-metadata.ll"
target triple = "unknown-unknown-unknown"

declare i32 @bcir.runtime.calibration.select(i32, i32, i32, ptr)

define i32 @select_calibration(i32 %epoch, i32 %minimum_width,
                               i32 %maximum_width, ptr %state_out) {
entry:
  %status = call i32 @bcir.runtime.calibration.select(
      i32 %epoch, i32 %minimum_width, i32 %maximum_width, ptr %state_out),
      !bcir.calibration !0
  ret i32 %status
}

!0 = !{!"bcir.calibration.v1", !1, !2, !3, !4}
!1 = !{!"preferred-governor", !"error-bounded-adaptive"}
!2 = !{!"confidence-permille", i32 950}
!3 = !{!"fallback", !"runtime-conservative"}
!4 = !{!"units", !"pulse-width:target-ticks", !"epoch:generation"}
