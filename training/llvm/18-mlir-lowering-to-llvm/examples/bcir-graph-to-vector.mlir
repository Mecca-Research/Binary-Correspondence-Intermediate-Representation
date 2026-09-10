// Illustrative BCIR graph edge weights staged through vector operations.
// The `bcir` dialect is unregistered in this repository; validate syntax with
// mlir-opt --allow-unregistered-dialect.

module attributes {bcir.example = "graph-to-vector"} {
  func.func @vectorize_edge_weights(%weights: memref<4xf32>, %out: memref<4xf32>) {
    %edge_pack = "bcir.edge_pack"() {
      graph_id = 7 : i64,
      edge_ids = [10, 11, 12, 13],
      claim_id = 700 : i64
    } : () -> !bcir.edge_pack<width = 4>

    %c0 = arith.constant 0 : index
    %w = vector.load %weights[%c0] : memref<4xf32>, vector<4xf32>
    %bias = arith.constant dense<1.000000e+00> : vector<4xf32>
    %adjusted = arith.addf %w, %bias : vector<4xf32>
    vector.store %adjusted, %out[%c0] : memref<4xf32>, vector<4xf32>
    "bcir.claim_anchor"(%edge_pack) {claim_id = 700 : i64, lowered_as = "vector"} : (!bcir.edge_pack<width = 4>) -> ()
    return
  }
}
