; BCIR vertex/edge/attribute lowering sketch.
; Standalone example: an edge list carries source, destination, and weight.

source_filename = "vertex-edge-attribute.ll"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.vertex = type { i32, i32, i64 }
%bcir.edge = type { i32, i32, i64 }

@bcir.example.vertices = internal constant [3 x %bcir.vertex] [
  %bcir.vertex { i32 0, i32 1, i64 10 },
  %bcir.vertex { i32 1, i32 2, i64 20 },
  %bcir.vertex { i32 2, i32 4, i64 30 }
], align 8

@bcir.example.edges = internal constant [2 x %bcir.edge] [
  %bcir.edge { i32 0, i32 1, i64 7 },
  %bcir.edge { i32 1, i32 2, i64 11 }
], align 8

define i64 @bcir.example.edge_weight(i64 %edge_index) {
entry:
  %edge_ptr = getelementptr inbounds [2 x %bcir.edge], ptr @bcir.example.edges, i64 0, i64 %edge_index
  %weight_ptr = getelementptr inbounds %bcir.edge, ptr %edge_ptr, i32 0, i32 2
  %weight = load i64, ptr %weight_ptr, align 8
  ret i64 %weight
}

define i64 @bcir.example.source_vertex_attr(i64 %edge_index) {
entry:
  %edge_ptr = getelementptr inbounds [2 x %bcir.edge], ptr @bcir.example.edges, i64 0, i64 %edge_index
  %src_ptr = getelementptr inbounds %bcir.edge, ptr %edge_ptr, i32 0, i32 0
  %src32 = load i32, ptr %src_ptr, align 8
  %src64 = zext i32 %src32 to i64
  %vertex_ptr = getelementptr inbounds [3 x %bcir.vertex], ptr @bcir.example.vertices, i64 0, i64 %src64
  %attr_ptr = getelementptr inbounds %bcir.vertex, ptr %vertex_ptr, i32 0, i32 2
  %attr = load i64, ptr %attr_ptr, align 8
  ret i64 %attr
}
