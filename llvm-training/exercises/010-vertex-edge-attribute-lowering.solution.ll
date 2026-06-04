; Exercise 010: Lower vertex and edge attributes with nested getelementptr.
; Verification: llvm-as -disable-output llvm-training/exercises/010-vertex-edge-attribute-lowering.solution.ll

source_filename = "010-vertex-edge-attribute-lowering.solution.ll"

%bcir.vertex.attr = type { i64, i32, i32 }
%bcir.edge.attr = type { i64, i64, i32, i32 }

define i32 @bcir.exercise.vertex_edge_score(ptr %vertices, ptr %edges, i64 %edge_index) {
entry:
  %edge = getelementptr inbounds %bcir.edge.attr, ptr %edges, i64 %edge_index
  %src_p = getelementptr inbounds %bcir.edge.attr, ptr %edge, i32 0, i32 0
  %dst_p = getelementptr inbounds %bcir.edge.attr, ptr %edge, i32 0, i32 1
  %weight_p = getelementptr inbounds %bcir.edge.attr, ptr %edge, i32 0, i32 2
  %src = load i64, ptr %src_p, align 8
  %dst = load i64, ptr %dst_p, align 8
  %weight = load i32, ptr %weight_p, align 4

  %src_vertex = getelementptr inbounds %bcir.vertex.attr, ptr %vertices, i64 %src
  %dst_vertex = getelementptr inbounds %bcir.vertex.attr, ptr %vertices, i64 %dst
  %src_cost_p = getelementptr inbounds %bcir.vertex.attr, ptr %src_vertex, i32 0, i32 2
  %dst_cost_p = getelementptr inbounds %bcir.vertex.attr, ptr %dst_vertex, i32 0, i32 2
  %src_cost = load i32, ptr %src_cost_p, align 4
  %dst_cost = load i32, ptr %dst_cost_p, align 4

  %partial = add i32 %src_cost, %dst_cost
  %score = add i32 %partial, %weight
  ret i32 %score
}
