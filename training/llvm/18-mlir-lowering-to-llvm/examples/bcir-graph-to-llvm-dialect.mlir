// Illustrative LLVM-dialect boundary after lowering BCIR graph identity to an
// explicit descriptor pointer, vertex ID, runtime call, and metadata anchors.

module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128",
  llvm.target_triple = "x86_64-unknown-linux-gnu"
} {
  llvm.func @bcir_lookup_vertex(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @bcir_visit_vertex(!llvm.ptr, i64)

  llvm.func @lowered_graph(%graph_desc: !llvm.ptr) {
    %vertex_id = llvm.mlir.constant(5 : i64) : i64
    %claim_id = llvm.mlir.constant(9001 : i64) : i64
    %vertex_ptr = llvm.call @bcir_lookup_vertex(%graph_desc, %vertex_id)
      : (!llvm.ptr, i64) -> !llvm.ptr
    llvm.call @bcir_visit_vertex(%vertex_ptr, %claim_id) : (!llvm.ptr, i64) -> ()
    llvm.return
  }
}
