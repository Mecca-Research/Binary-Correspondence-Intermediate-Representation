// Illustrative MLIR sketch for a possible BCIR dialect.
// This file is documentation, not a buildable dialect test: the `bcir` dialect
// and its custom types/attributes would need real TableGen/C++ definitions.

module attributes {bcir.schema = "claim-v1"} {
  "bcir.graph"() ({
  ^entry:
    %root = "bcir.vertex"() {
      id = 0 : i64,
      space = "claim"
    } : () -> !bcir.vertex<space = "claim", id_bits = 64>

    %child = "bcir.vertex.lookup"(%root) {
      edge_kind = "contains",
      ordinal = 0 : i32
    } : (!bcir.vertex<space = "claim", id_bits = 64>)
        -> !bcir.vertex<space = "blob", id_bits = 64>

    %edge = "bcir.edge"(%root, %child) {
      kind = "contains",
      directed = true
    } : (!bcir.vertex<space = "claim", id_bits = 64>,
         !bcir.vertex<space = "blob", id_bits = 64>)
        -> !bcir.edge<src = "claim", dst = "blob">

    %weight = "bcir.attribute"(%edge) {
      name = "weight",
      storage = "side_table"
    } : (!bcir.edge<src = "claim", dst = "blob">) -> f32

    %hint = "bcir.ham_hint"(%child) {
      policy = "prefetch",
      distance = 2 : i32,
      confidence = 0.875 : f64
    } : (!bcir.vertex<space = "blob", id_bits = 64>) -> !bcir.hint

    %bound = "bcir.bind_register"(%weight) {
      reg_class = "fpr",
      preference = "xmm0",
      required = false
    } : (f32) -> f32

    %graph = "bcir.mixed_stride.graph"(%child) {
      rank = 3 : i32,
      static_strides = [1, 4, -1],
      dynamic_stride_positions = [2],
      layout = "soa"
    } : (!bcir.vertex<space = "blob", id_bits = 64>) -> !bcir.graph<rank = 3>

    "bcir.consume"(%graph, %bound, %hint) : (!bcir.graph<rank = 3>, f32, !bcir.hint) -> ()
    "bcir.yield"() : () -> ()
  }) : () -> ()
}
