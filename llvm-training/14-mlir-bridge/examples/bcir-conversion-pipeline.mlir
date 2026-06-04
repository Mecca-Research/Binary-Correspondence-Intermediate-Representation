// Illustrative staged rewrite sketch; pass names are representative.
module attributes {bcir.schema = "edge-weight-v1"} {
  func.func @edge_weight(%graph: !bcir.graph, %src: i64, %dst: i64) -> i32 {
    %edge = "bcir.edge.lookup"(%graph, %src, %dst) : (!bcir.graph, i64, i64) -> !bcir.edge
    %weight = "bcir.attribute.read"(%edge) {name = "weight"} : (!bcir.edge) -> i32
    return %weight : i32
  }

  // After bcir-lower-graph-to-descriptors, the body should resemble:
  // %desc = func.call @bcir_edge_lookup(%graph_ptr, %src, %dst)
  // %weight = func.call @bcir_edge_weight(%desc)
}
