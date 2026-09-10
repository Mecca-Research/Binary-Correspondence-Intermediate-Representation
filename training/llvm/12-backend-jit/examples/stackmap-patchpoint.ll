; Minimal stackmap/patchpoint fixture for JIT and deoptimization lessons.
; It is intended to assemble as IR, but useful stackmap consumption depends on
; the target's emitted stackmap section and on a runtime that understands it.

source_filename = "stackmap-patchpoint.ll"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.experimental.stackmap(i64, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, ptr, i32, ...)
declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)

define i64 @demo_stackmap_and_patchpoint(i64 %x, ptr %runtime_target) {
entry:
  %live_sum = add i64 %x, 7

  ; ID 1001 lets the runtime match this safepoint to its own metadata.
  ; The 16 shadow bytes reserve room after the lowered marker/call site.
  ; The varargs are live values whose machine locations are written to the
  ; target stackmap side table, not interpreted by LLVM IR itself.
  call void (i64, i32, ...) @llvm.experimental.stackmap(
    i64 1001, i32 16, i64 %live_sum, ptr %runtime_target)

  ; ID 2001 describes a runtime-patchable call-shaped site. The runtime target
  ; receives one call argument (%live_sum); trailing values are stackmap-only
  ; live values for deoptimization or relocation metadata.
  %patched = call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(
    i64 2001, i32 32, ptr %runtime_target, i32 1, i64 %live_sum, i64 %x)

  ; A void patchpoint with a null target is a placeholder patch site. The live
  ; value still appears in the side table for a runtime that knows ID 2002.
  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(
    i64 2002, i32 16, ptr null, i32 0, i64 %patched)

  ret i64 %patched
}
