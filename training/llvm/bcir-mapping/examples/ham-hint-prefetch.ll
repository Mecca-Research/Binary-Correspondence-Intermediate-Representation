; Lowering of ham-hint.prompt.md.
; HAM advisory hints become explicit address math, metadata, and llvm.prefetch.

source_filename = "ham-hint.prompt.md"
target triple = "unknown-unknown-unknown"
target datalayout = ""

declare void @llvm.prefetch.p0(ptr nocapture readonly, i32 immarg, i32 immarg, i32 immarg)

define void @bcir.map.ham.prefetch_read(ptr %base, i64 %byte_offset) {
entry:
  %addr = getelementptr i8, ptr %base, i64 %byte_offset, !bcir.ham !0
  call void @llvm.prefetch.p0(ptr %addr, i32 0, i32 3, i32 1), !bcir.ham !0
  ret void
}

!0 = !{!"bcir.ham", !"prefetch", !"tile:7", i32 3}
