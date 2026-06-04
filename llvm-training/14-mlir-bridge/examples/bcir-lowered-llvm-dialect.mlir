// LLVM-dialect MLIR sketch after runtime-backed BCIR lowering.
// Documentation artifact: illustrates the shape before translation to .ll.
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128",
  llvm.target_triple = "x86_64-unknown-linux-gnu"
} {
  llvm.func @bcir_lookup_child(!llvm.ptr, i64, i32) -> !llvm.ptr
  llvm.func @bcir_prefetch_vertex(!llvm.ptr, i32)
  llvm.func @bcir_get_edge_attr_f32(!llvm.ptr, i64, i32) -> f32
  llvm.func @bcir_consume_weighted_child(!llvm.ptr, !llvm.ptr, f32) -> i32

  llvm.func @bcir_lowered_claim(%graph: !llvm.ptr) -> i32 {
    %root_id = llvm.mlir.constant(0 : i64) : i64
    %edge_kind_contains = llvm.mlir.constant(3 : i32) : i32
    %attr_weight = llvm.mlir.constant(7 : i32) : i32
    %prefetch_distance = llvm.mlir.constant(2 : i32) : i32

    %child = llvm.call @bcir_lookup_child(%graph, %root_id, %edge_kind_contains)
      : (!llvm.ptr, i64, i32) -> !llvm.ptr

    llvm.call @bcir_prefetch_vertex(%child, %prefetch_distance)
      : (!llvm.ptr, i32) -> ()

    %weight = llvm.call @bcir_get_edge_attr_f32(%graph, %root_id, %attr_weight)
      : (!llvm.ptr, i64, i32) -> f32

    // Optional bcir.bind_register(preference = "xmm0") is dropped here because
    // this generic LLVM-dialect path has no portable physical-register contract.
    %status = llvm.call @bcir_consume_weighted_child(%graph, %child, %weight)
      : (!llvm.ptr, !llvm.ptr, f32) -> i32
    llvm.return %status : i32
  }
}
