; Textual LLVM IR equivalent of the runtime-backed BCIR MLIR lowering sketch.
; This file is verifier-friendly and intentionally avoids target-specific
; register constraints for the optional bcir.bind_register hint.
source_filename = "bcir-lowered-llvm-dialect.mlir"
target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @bcir_lookup_child(ptr, i64, i32)
declare void @bcir_prefetch_vertex(ptr, i32)
declare float @bcir_get_edge_attr_f32(ptr, i64, i32)
declare i32 @bcir_consume_weighted_child(ptr, ptr, float)

define i32 @bcir_lowered_claim(ptr %graph) {
entry:
  %child = call ptr @bcir_lookup_child(ptr %graph, i64 0, i32 3)
  call void @bcir_prefetch_vertex(ptr %child, i32 2)
  %weight = call float @bcir_get_edge_attr_f32(ptr %graph, i64 0, i32 7)
  %status = call i32 @bcir_consume_weighted_child(ptr %graph, ptr %child, float %weight)
  ret i32 %status
}
