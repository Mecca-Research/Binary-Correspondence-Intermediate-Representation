; LLVM IR lowered shape for bcir-vertex-graph.mlir.
; Vertex IDs and edge rows are executable data. Optional register preferences
; become binding slots; source-only hints survive only as non-semantic metadata.
source_filename = "bcir-vertex-graph-lowered.mlir"
target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%bcir.vertex.row = type { i64, i32 }
%bcir.edge.row = type { i64, i64, i32 }

@bcir.vertex.table = internal constant [3 x %bcir.vertex.row] [
  %bcir.vertex.row { i64 100, i32 0 },
  %bcir.vertex.row { i64 101, i32 3 },
  %bcir.vertex.row { i64 102, i32 0 }
]

@bcir.edge.table = internal constant [2 x %bcir.edge.row] [
  %bcir.edge.row { i64 100, i64 101, i32 4 },
  %bcir.edge.row { i64 101, i64 102, i32 9 }
]

declare void @bcir.consume.edge(i64, i64, i32, i32)

define void @bcir.walk_vertex_graph() {
entry:
  ; Vertex ID 101 survives as a table field.
  %v1 = getelementptr inbounds [3 x %bcir.vertex.row], ptr @bcir.vertex.table, i64 0, i64 1
  %v1.id.ptr = getelementptr inbounds %bcir.vertex.row, ptr %v1, i32 0, i32 0
  %v1.id = load i64, ptr %v1.id.ptr, align 8, !bcir.vertex !0

  ; The optional register preference "r10" has intentionally lowered away.
  ; The remaining executable binding fact is a portable integer slot.
  %v1.slot.ptr = getelementptr inbounds %bcir.vertex.row, ptr %v1, i32 0, i32 1
  %binding.slot = load i32, ptr %v1.slot.ptr, align 4, !bcir.hint !2

  ; Edge-list row 1 survives as source, destination, and attribute fields.
  %edge1 = getelementptr inbounds [2 x %bcir.edge.row], ptr @bcir.edge.table, i64 0, i64 1
  %edge1.src.ptr = getelementptr inbounds %bcir.edge.row, ptr %edge1, i32 0, i32 0
  %edge1.dst.ptr = getelementptr inbounds %bcir.edge.row, ptr %edge1, i32 0, i32 1
  %edge1.weight.ptr = getelementptr inbounds %bcir.edge.row, ptr %edge1, i32 0, i32 2
  %edge1.src = load i64, ptr %edge1.src.ptr, align 8, !bcir.edge !1
  %edge1.dst = load i64, ptr %edge1.dst.ptr, align 8, !bcir.edge !1
  %edge1.weight = load i32, ptr %edge1.weight.ptr, align 4, !bcir.edge !1

  call void @bcir.consume.edge(i64 %edge1.src, i64 %edge1.dst, i32 %edge1.weight, i32 %binding.slot), !bcir.vertex !0, !bcir.edge !1, !bcir.hint !2
  ret void, !bcir.hint !3
}

!bcir.vertices = !{!0, !4, !5}
!bcir.edges = !{!6, !1}
!bcir.hints = !{!2, !3}

!0 = !{!"vertex", !"id", i64 101, !"ordinal", i32 1}
!1 = !{!"edge", !"id", i64 1, !"src", i64 101, !"dst", i64 102, !"kind", !"data"}
!2 = !{!"register_binding", !"slot", i32 3, !"optional_preference_lowered_away", !"r10"}
!3 = !{!"metadata_hint", !"source_rule", !"claim-edge-hot-path", !"semantic", i1 false}
!4 = !{!"vertex", !"id", i64 100, !"ordinal", i32 0}
!5 = !{!"vertex", !"id", i64 102, !"ordinal", i32 2}
!6 = !{!"edge", !"id", i64 0, !"src", i64 100, !"dst", i64 101, !"kind", !"control"}
