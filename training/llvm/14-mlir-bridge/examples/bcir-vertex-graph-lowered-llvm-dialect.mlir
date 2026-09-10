// LLVM-dialect MLIR sketch after lowering bcir-vertex-graph.mlir.
// Vertex IDs and edge rows are now explicit integer values loaded or materialized
// from ABI tables. The optional "r10" preference has lowered to an integer
// binding slot, not a hardware-register guarantee. Diagnostic hints are
// non-semantic and may become LLVM metadata or a side catalog later.

module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128",
  llvm.target_triple = "x86_64-unknown-linux-gnu"
} {
  llvm.func @bcir.consume.edge(i64, i64, i32, i32)

  llvm.func @bcir.walk_vertex_graph() {
    // These constants stand in for values loaded from lowered vertex/edge tables:
    // edge_id=1, src vertex ID 101, dst vertex ID 102, edge weight 9.
    %src_id = llvm.mlir.constant(101 : i64) : i64
    %dst_id = llvm.mlir.constant(102 : i64) : i64
    %edge_weight = llvm.mlir.constant(9 : i32) : i32

    // The executable remnant of optional register binding is slot 3. The source
    // physical preference "r10" is intentionally gone in portable LLVM dialect.
    %binding_slot = llvm.mlir.constant(3 : i32) : i32

    // A real translation may attach non-semantic !bcir.* metadata to this call.
    llvm.call @bcir.consume.edge(%src_id, %dst_id, %edge_weight, %binding_slot)
      : (i64, i64, i32, i32) -> ()

    llvm.return
  }
}
