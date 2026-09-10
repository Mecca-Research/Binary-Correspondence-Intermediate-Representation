// Illustrative BCIR graph fragment staged through affine loops.
// The `bcir` dialect is unregistered in this repository; validate syntax with
// mlir-opt --allow-unregistered-dialect.

module attributes {bcir.example = "graph-to-affine"} {
  func.func @lower_regular_window(%base: memref<16xf32>, %out: memref<16xf32>) {
    %graph = "bcir.graph_descriptor"() {
      graph_id = 42 : i64,
      vertex_count = 16 : i64,
      claim_id = 9001 : i64
    } : () -> !bcir.graph<rank = 1>

    affine.for %i = 0 to 16 {
      %v = "bcir.vertex_id"(%graph) {dimension = 0 : i32} : (!bcir.graph<rank = 1>) -> index
      %x = affine.load %base[%i] : memref<16xf32>
      %scaled = arith.mulf %x, %x : f32
      affine.store %scaled, %out[%i] : memref<16xf32>
      "bcir.claim_anchor"(%v) {claim_id = 9001 : i64, lowered_as = "affine"} : (index) -> ()
    }

    return
  }
}
