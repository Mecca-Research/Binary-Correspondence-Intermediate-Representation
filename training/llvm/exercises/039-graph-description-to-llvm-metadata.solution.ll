source_filename = "039-graph-description-to-llvm-metadata.solution.ll"

; Lower one graph edge update while preserving the source graph schema as
; non-semantic BCIR metadata.
define void @apply_graph_edge(ptr %rank, ptr %edge_src, ptr %edge_dst, ptr %edge_weight, ptr %out, i64 %edge) {
entry:
  %src_ptr = getelementptr i32, ptr %edge_src, i64 %edge
  %src = load i32, ptr %src_ptr, align 4, !bcir.graph.edge.attr !3
  %dst_ptr = getelementptr i32, ptr %edge_dst, i64 %edge
  %dst = load i32, ptr %dst_ptr, align 4, !bcir.graph.edge.attr !4

  %src_index = zext i32 %src to i64
  %rank_ptr = getelementptr float, ptr %rank, i64 %src_index
  %rank_value = load float, ptr %rank_ptr, align 4, !bcir.graph.vertex.attr !2

  %weight_ptr = getelementptr float, ptr %edge_weight, i64 %edge
  %weight = load float, ptr %weight_ptr, align 4, !bcir.graph.edge.attr !5
  %contribution = fmul float %rank_value, %weight

  %dst_index = zext i32 %dst to i64
  %out_ptr = getelementptr float, ptr %out, i64 %dst_index
  %old = load float, ptr %out_ptr, align 4, !bcir.graph.vertex.attr !6
  %new = fadd float %old, %contribution
  store float %new, ptr %out_ptr, align 4, !bcir.graph.update !7
  ret void
}

!bcir.graphs = !{!0}
!bcir.graph.attrs = !{!2, !3, !4, !5, !6}
!bcir.graph.updates = !{!7}

!0 = !{!"graph", !"pagerank_step", !"vertices", i64 1024, !"edges", i64 4096, !"schema", !1}
!1 = !{!"directed", i1 true, !"edge_index_type", !"i64"}
!2 = !{!"graph", !"pagerank_step", !"kind", !"vertex", !"name", !"rank", !"type", !"f32"}
!3 = !{!"graph", !"pagerank_step", !"kind", !"edge", !"name", !"src", !"type", !"i32"}
!4 = !{!"graph", !"pagerank_step", !"kind", !"edge", !"name", !"dst", !"type", !"i32"}
!5 = !{!"graph", !"pagerank_step", !"kind", !"edge", !"name", !"weight", !"type", !"f32"}
!6 = !{!"graph", !"pagerank_step", !"kind", !"vertex", !"name", !"out", !"type", !"f32"}
!7 = !{!"graph", !"pagerank_step", !"operation", !"accumulate", !"target", !"out"}
