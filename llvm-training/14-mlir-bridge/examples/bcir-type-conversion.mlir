// Illustrative MLIR only: not consumed by llvm-as.
module {
  func.func @lower_vertex(%graph: !bcir.graph, %id: i64) -> !llvm.ptr {
    // Before type conversion: !bcir.vertex is a domain handle.
    %vertex = "bcir.vertex.lookup"(%graph, %id) : (!bcir.graph, i64) -> !bcir.vertex

    // After materialization: the vertex becomes a runtime descriptor pointer.
    %ptr = "bcir.lower.vertex_handle"(%vertex) : (!bcir.vertex) -> !llvm.ptr
    return %ptr : !llvm.ptr
  }
}
