; Lowering of graph-fragment.bcir.txt.
; Graph vertices and edges become struct arrays; queries become GEP + load.

source_filename = "graph-fragment.bcir.txt"
target triple = "unknown-unknown-unknown"
target datalayout = ""

%bcir.map.vertex = type { i32, i32, i64 }
%bcir.map.edge = type { i32, i32, i64 }

@bcir.map.vertices = internal constant [2 x %bcir.map.vertex] [
  %bcir.map.vertex { i32 0, i32 1, i64 10 },
  %bcir.map.vertex { i32 1, i32 2, i64 20 }
], align 8

@bcir.map.edges = internal constant [1 x %bcir.map.edge] [
  %bcir.map.edge { i32 0, i32 1, i64 7 }
], align 8

define i64 @bcir.map.graph.edge_weight(i64 %edge_index) {
entry:
  %edge_ptr = getelementptr inbounds [1 x %bcir.map.edge], ptr @bcir.map.edges, i64 0, i64 %edge_index
  %weight_ptr = getelementptr inbounds %bcir.map.edge, ptr %edge_ptr, i32 0, i32 2
  %weight = load i64, ptr %weight_ptr, align 8
  ret i64 %weight
}

define i64 @bcir.map.graph.source_attr(i64 %edge_index) {
entry:
  %edge_ptr = getelementptr inbounds [1 x %bcir.map.edge], ptr @bcir.map.edges, i64 0, i64 %edge_index
  %src_ptr = getelementptr inbounds %bcir.map.edge, ptr %edge_ptr, i32 0, i32 0
  %src32 = load i32, ptr %src_ptr, align 8
  %src64 = zext i32 %src32 to i64
  %vertex_ptr = getelementptr inbounds [2 x %bcir.map.vertex], ptr @bcir.map.vertices, i64 0, i64 %src64
  %attr_ptr = getelementptr inbounds %bcir.map.vertex, ptr %vertex_ptr, i32 0, i32 2
  %attr = load i64, ptr %attr_ptr, align 8
  ret i64 %attr
}
