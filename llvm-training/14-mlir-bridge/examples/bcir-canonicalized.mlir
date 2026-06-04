// Canonical BCIR sketch after schema lookup and local cleanup.
// Documentation artifact: the bcir dialect is illustrative.
module attributes {bcir.schema = "claim-v1", bcir.abi = "runtime-v2"} {
  "bcir.graph"() ({
  ^entry:
    %root_id = arith.constant 0 : i64
    %edge_kind_contains = arith.constant 3 : i32
    %attr_weight = arith.constant 7 : i32

    %root = "bcir.vertex.from_id"(%root_id) {
      space = "claim"
    } : (i64) -> !bcir.vertex<space = "claim", id_bits = 64>

    %child = "bcir.edge.target_by_kind"(%root, %edge_kind_contains) {
      dst_space = "blob",
      ordinal = 0 : i32
    } : (!bcir.vertex<space = "claim", id_bits = 64>, i32)
        -> !bcir.vertex<space = "blob", id_bits = 64>

    %weight = "bcir.attribute.by_key"(%root, %edge_kind_contains, %attr_weight) {
      result_name = "weight",
      storage = "runtime"
    } : (!bcir.vertex<space = "claim", id_bits = 64>, i32, i32) -> f32

    %hint = "bcir.ham_hint.canonical"(%child) {
      policy = "prefetch",
      distance = 2 : i32,
      confidence_milli = 875 : i32
    } : (!bcir.vertex<space = "blob", id_bits = 64>) -> !bcir.hint

    %preferred = "bcir.bind_register"(%weight) {
      reg_class = "fpr",
      preference = "xmm0",
      required = false,
      drop_if_unsupported = true
    } : (f32) -> f32

    "bcir.runtime.consume_weighted_child"(%root, %child, %preferred, %hint) {
      callee = @bcir_consume_weighted_child
    } : (!bcir.vertex<space = "claim", id_bits = 64>,
         !bcir.vertex<space = "blob", id_bits = 64>,
         f32,
         !bcir.hint) -> ()
    "bcir.yield" : () -> ()
  }) : () -> ()
}
