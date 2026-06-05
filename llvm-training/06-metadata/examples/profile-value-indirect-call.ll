; Standalone example: value profiling metadata for indirect-call targets.
; Assemble: llvm-as llvm-training/06-metadata/examples/profile-value-indirect-call.ll -o /dev/null

source_filename = "profile-value-indirect-call.c"
target triple = "x86_64-unknown-linux-gnu"

define i32 @call_through_pointer(ptr %callee, i32 %x) !prof !0 {
entry:
  %result = call i32 %callee(i32 %x), !prof !1
  ret i32 %result
}

define i32 @target_a(i32 %x) {
entry:
  %r = add i32 %x, 7
  ret i32 %r
}

define i32 @target_b(i32 %x) {
entry:
  %r = mul i32 %x, 2
  ret i32 %r
}

!0 = !{!"function_entry_count", i64 1600}
; VP kind 0 means indirect-call targets. The value/count pairs use illustrative
; target-name hashes followed by observed counts for the hottest callees.
!1 = !{!"VP", i32 0, i64 1600, i64 7651369219802541373, i64 1030, i64 -4377547752858689819, i64 410}
