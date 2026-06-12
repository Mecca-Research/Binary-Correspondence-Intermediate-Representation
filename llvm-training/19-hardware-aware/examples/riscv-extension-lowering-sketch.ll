; RISC-V extension lowering sketch.
; This custom llvm.bcir.riscv.* name is not an upstream intrinsic. Register it
; and add target lowering, or rewrite it to the runtime fallback before llc.

source_filename = "riscv-extension-lowering-sketch.ll"
target triple = "riscv64-unknown-linux-gnu"

declare i64 @llvm.bcir.riscv.gaadmsf.accumulate(i64, i64, i32 immarg)
declare i64 @bcir.runtime.gaadmsf.accumulate.i64(i64, i64, i32)

define i64 @riscv_extension_path(i64 %accumulator, i64 %payload) #0 {
entry:
  %next = call i64 @llvm.bcir.riscv.gaadmsf.accumulate(
      i64 %accumulator, i64 %payload, i32 3), !bcir.gaa !0
  ret i64 %next
}

define i64 @portable_runtime_path(i64 %accumulator, i64 %payload) {
entry:
  %next = call i64 @bcir.runtime.gaadmsf.accumulate.i64(
      i64 %accumulator, i64 %payload, i32 3)
  ret i64 %next
}

attributes #0 = { "target-features"="+m,+a,+f,+d,+zicsr,+zifencei,+xgaadmsf" }

!0 = !{!"bcir.gaa.v1", !"role:accumulator", !"prefer-bank:gaa-acc", !"advisory"}
