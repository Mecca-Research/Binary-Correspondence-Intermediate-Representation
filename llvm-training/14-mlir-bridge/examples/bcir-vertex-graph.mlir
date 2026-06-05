// BCIR source-dialect sketch for a vertex/edge graph.
// The bcir dialect is intentionally unregistered in this training repository;
// run syntax checks with mlir-opt --allow-unregistered-dialect.

module attributes {bcir.schema = "vertex-graph-v1"} {
  "bcir.graph.scope"() ({
  ^entry:
    %graph = "bcir.graph.create"() {
      graph_id = 7 : i64,
      vertex_ids = [100 : i64, 101 : i64, 102 : i64],
      edge_list = [[100 : i64, 101 : i64], [101 : i64, 102 : i64]],
      storage = "compact_tables"
    } : () -> !bcir.graph

    %v0 = "bcir.vertex"(%graph) {
      id = 100 : i64,
      ordinal = 0 : i32,
      diag = "entry vertex"
    } : (!bcir.graph) -> !bcir.vertex

    %v1 = "bcir.vertex"(%graph) {
      id = 101 : i64,
      ordinal = 1 : i32,
      diag = "middle vertex"
    } : (!bcir.graph) -> !bcir.vertex

    %v2 = "bcir.vertex"(%graph) {
      id = 102 : i64,
      ordinal = 2 : i32,
      diag = "exit vertex"
    } : (!bcir.graph) -> !bcir.vertex

    %e0 = "bcir.edge"(%v0, %v1) {
      edge_id = 0 : i64,
      kind = "control",
      weight = 4 : i32
    } : (!bcir.vertex, !bcir.vertex) -> !bcir.edge

    %e1 = "bcir.edge"(%v1, %v2) {
      edge_id = 1 : i64,
      kind = "data",
      weight = 9 : i32
    } : (!bcir.vertex, !bcir.vertex) -> !bcir.edge

    %bound = "bcir.bind_register"(%v1) {
      reg_class = "gpr",
      preference = "r10",
      required = false,
      binding_slot = 3 : i32
    } : (!bcir.vertex) -> !bcir.vertex

    %hint = "bcir.metadata_hint"(%e0) {
      source_rule = "claim-edge-hot-path",
      hotness = 87 : i32,
      preserve_for_diagnostics = true
    } : (!bcir.edge) -> !bcir.hint

    "bcir.consume_graph"(%graph, %bound, %e0, %e1, %hint)
      : (!bcir.graph, !bcir.vertex, !bcir.edge, !bcir.edge, !bcir.hint) -> ()
    "bcir.yield"() : () -> ()
  }) : () -> ()
}
