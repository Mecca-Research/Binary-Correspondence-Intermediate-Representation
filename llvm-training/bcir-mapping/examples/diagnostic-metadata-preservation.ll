; Lowering of diagnostic-metadata.prompt.md.
; Custom metadata records diagnostic provenance without changing semantics.

source_filename = "diagnostic-metadata.prompt.md"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define i32 @bcir.map.diag.load_i32(ptr %base, i64 %byte_offset) {
entry:
  %addr = getelementptr i8, ptr %base, i64 %byte_offset, !bcir.diag !0
  %value = load i32, ptr %addr, align 4, !bcir.diag !0
  ret i32 %value, !bcir.diag !1
}

!0 = !{!"bcir.diag", !"claim:c42", !"graph.edge:3", !"rule:resource-bounds"}
!1 = !{!"bcir.diag", !"lowering:load_i32", !"preserved"}
