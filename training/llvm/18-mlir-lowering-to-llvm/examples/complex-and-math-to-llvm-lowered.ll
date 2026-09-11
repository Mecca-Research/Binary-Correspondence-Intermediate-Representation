; REQUIRES: llvm >= 23
; mlir-translate 23 puts `nocreateundeforpoison` on the math intrinsic
; declarations below. That attribute arrived in LLVM 23, so llvm-as 18 refuses the
; whole attribute group -- `unterminated attribute group`, which is what an
; assembler says when it meets a keyword it does not know. The construct is genuinely
; newer than the corpus baseline, so this file declares its floor rather than being
; edited to hide it; SEMVER.md and examples/README.md describe the mechanism and name
; the CI job that owns the skip.
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define { float, float } @cmul({ float, float } %0, { float, float } %1) {
  %3 = extractvalue { float, float } %0, 0
  %4 = extractvalue { float, float } %0, 1
  %5 = extractvalue { float, float } %1, 0
  %6 = extractvalue { float, float } %1, 1
  %7 = fmul float %5, %3
  %8 = fmul float %6, %4
  %9 = fmul float %4, %5
  %10 = fmul float %3, %6
  %11 = fsub float %7, %8
  %12 = fadd float %9, %10
  %13 = insertvalue { float, float } poison, float %11, 0
  %14 = insertvalue { float, float } %13, float %12, 1
  ret { float, float } %14
}

define float @roots(float %0) {
  %2 = call float @llvm.sqrt.f32(float %0)
  %3 = call float @llvm.exp.f32(float %2)
  ret float %3
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #0

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #0

attributes #0 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
