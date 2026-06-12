; GAADMSF programming-pulse intrinsic boundary.
; Assembly/verifier sketch: a BCIR-aware pipeline must register/lower this
; custom intrinsic or rewrite it to the runtime fallback before code generation.

source_filename = "gaadmsf-programming-pulse.ll"
target triple = "unknown-unknown-unknown"

; Inputs: device, graph window, pulse width, calibration epoch, mode immediate.
; Result: explicit context ID consumed by later flow execution.
declare i64 @llvm.bcir.gaadmsf.program.pulse(i32, i64, i32, i32, i32 immarg)
declare i64 @bcir.runtime.gaadmsf.program.pulse(i32, i64, i32, i32, i32)

define i64 @program_pulse_intrinsic(i32 %device, i64 %graph_window,
                                    i32 %pulse_width, i32 %epoch) {
entry:
  %context = call i64 @llvm.bcir.gaadmsf.program.pulse(
      i32 %device, i64 %graph_window, i32 %pulse_width, i32 %epoch, i32 1),
      !bcir.gaa !0, !bcir.memory !1
  ret i64 %context
}

define i64 @program_pulse_fallback(i32 %device, i64 %graph_window,
                                   i32 %pulse_width, i32 %epoch) {
entry:
  %context = call i64 @bcir.runtime.gaadmsf.program.pulse(
      i32 %device, i64 %graph_window, i32 %pulse_width, i32 %epoch, i32 1)
  ret i64 %context
}

!0 = !{!"bcir.gaa.v1", !"affinity-group:graph-window", i32 7, !"advisory"}
!1 = !{!"bcir.memory.v1", !"role:pulse-table", !"prefer:near-device", !"fallback:any"}
