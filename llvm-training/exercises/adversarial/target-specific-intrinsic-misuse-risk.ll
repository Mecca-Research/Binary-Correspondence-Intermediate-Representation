; adversarial-class: target-specific
; Category: target-specific intrinsic misuse.
; This x86 pause intrinsic is a target-gated seed. A lowering must not emit it
; for non-x86 targets or assume equivalent availability without a fallback.

target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.x86.sse2.pause()

define void @target_specific_intrinsic_risk() {
entry:
  call void @llvm.x86.sse2.pause()
  ret void
}
