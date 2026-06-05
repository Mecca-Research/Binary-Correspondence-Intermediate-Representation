target triple = "x86_64-unknown-linux-gnu"

; GC statepoint relocation outline.
; Assemble/check with: llvm-as gc-statepoint-relocate.ll -o /dev/null
;
; The statepoint records the managed pointers that are live across the safepoint
; in its "gc-live" operand bundle. A moving collector may update those pointers,
; so uses after the statepoint must go through llvm.experimental.gc.relocate.

source_filename = "gc-statepoint-relocate.ll"

%Obj = type { i64, i64 }

declare void @runtime_safepoint()
declare token @llvm.experimental.gc.statepoint.p0(i64 immarg, i32 immarg, ptr, i32 immarg, i32 immarg, ...)
declare ptr @llvm.experimental.gc.relocate.p0(token, i32 immarg, i32 immarg)

define i64 @load_field_after_statepoint(ptr %obj) gc "statepoint-example" {
entry:
  ; %obj is the base object pointer. %field is a derived interior pointer.
  %field = getelementptr inbounds %Obj, ptr %obj, i32 0, i32 1

  ; The gc-live bundle is the live pointer set for this safepoint:
  ;   index 0: base object pointer %obj
  ;   index 1: derived field pointer %field
  %sp = call token (i64, i32, ptr, i32, i32, ...) @llvm.experimental.gc.statepoint.p0(
      i64 42, i32 0, ptr elementtype(void ()) @runtime_safepoint,
      i32 0, i32 0, i32 0, i32 0) [ "gc-live"(ptr %obj, ptr %field) ]

  ; Relocate both values that remain interesting after the safepoint. The first
  ; index is the base pointer; the second index is the pointer being relocated.
  %obj.relocated = call ptr @llvm.experimental.gc.relocate.p0(token %sp, i32 0, i32 0)
  %field.relocated = call ptr @llvm.experimental.gc.relocate.p0(token %sp, i32 0, i32 1)

  ; Correct post-statepoint use: load through the relocated derived pointer, not
  ; through the original %field SSA value.
  %value = load i64, ptr %field.relocated, align 8

  ; Keep the relocated base visibly live in this teaching example.
  %same.object = icmp eq ptr %obj.relocated, %obj.relocated
  %result = select i1 %same.object, i64 %value, i64 0
  ret i64 %result
}
