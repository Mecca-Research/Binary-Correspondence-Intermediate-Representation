; Standalone example: function_entry_count profile metadata.
; Assemble: llvm-as llvm-training/06-metadata/examples/profile-entry-count.ll -o /dev/null

source_filename = "profile-entry-count.c"
target triple = "x86_64-unknown-linux-gnu"

define i32 @hot_worker(i32 %x) !prof !0 {
entry:
  %r = add i32 %x, 1
  ret i32 %r
}

define i32 @cold_worker(i32 %x) !prof !1 {
entry:
  %r = sub i32 %x, 1
  ret i32 %r
}

!0 = !{!"function_entry_count", i64 10000}
!1 = !{!"function_entry_count", i64 3}
