; Custom BCIR intrinsic/JIT boundary sketch.
; The IR keeps operands in register-shaped vectors and metadata on the call;
; an ORC layer can either require a BCIR-aware backend or rewrite the call to
; @bcir.runtime.gem.v4f32 before materialization.

source_filename = "custom-bcir-intrinsic-jit.ll"
target triple = "x86_64-unknown-linux-gnu"

; Custom intrinsic declaration used by a BCIR-aware backend selector.
declare <4 x float> @llvm.bcir.gem.mixed.stride.v4f32(<4 x float>, <4 x float>, <4 x float>, i32 immarg)

; Fallback runtime ABI with the same register-oriented operand layout.
declare <4 x float> @bcir.runtime.gem.v4f32(<4 x float>, <4 x float>, <4 x float>, i32)

define <4 x float> @jit_entry_custom_intrinsic(<4 x float> %a_regs, <4 x float> %b_regs, <4 x float> %acc_regs) {
entry:
  %tile = call <4 x float> @llvm.bcir.gem.mixed.stride.v4f32(<4 x float> %a_regs, <4 x float> %b_regs, <4 x float> %acc_regs, i32 1), !bcir.memory !0
  ret <4 x float> %tile
}

define <4 x float> @jit_entry_runtime_fallback(<4 x float> %a_regs, <4 x float> %b_regs, <4 x float> %acc_regs) {
entry:
  %tile = call <4 x float> @bcir.runtime.gem.v4f32(<4 x float> %a_regs, <4 x float> %b_regs, <4 x float> %acc_regs, i32 1), !bcir.memory !0
  ret <4 x float> %tile
}

!0 = !{!"bcir.memory", !"jit-policy:select-or-rewrite", !1, !2}
!1 = !{!"operand-layout", !"register:v4f32", !"a,b,acc,result"}
!2 = !{!"hierarchy", !"l1:tile-resident", !"l2:spill-ok", !"dram:streaming"}
