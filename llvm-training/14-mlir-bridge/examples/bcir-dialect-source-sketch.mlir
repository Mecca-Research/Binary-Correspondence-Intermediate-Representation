// Source-level BCIR dialect sketch before canonicalization.
// Documentation artifact: custom bcir types/ops require a real dialect.
module attributes {bcir.schema = "claim-v1", bcir.abi = "runtime-v2"} {
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
      storage = "runtime"
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

    "bcir.runtime.consume_weighted_child"(%root, %child, %bound, %hint) {
      abi = "runtime-v2"
    } : (!bcir.vertex<space = "claim", id_bits = 64>,
         !bcir.vertex<space = "blob", id_bits = 64>,
         f32,
         !bcir.hint) -> ()
    "bcir.yield" : () -> ()
  }) : () -> ()
}
