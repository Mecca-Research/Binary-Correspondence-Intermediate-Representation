// Illustrative LLVM-dialect MLIR after lowering BCIR concepts.
// This is MLIR syntax, not textual LLVM IR. It shows the kind of low-level
// shape a BCIR lowering might target before translation to `.ll`/bitcode.

module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-n8:16:32:64-S128",
  llvm.target_triple = "x86_64-unknown-linux-gnu"
} {
  llvm.func @bcir_prefetch_vertex(!llvm.ptr)
  llvm.func @bcir_lookup_child(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @bcir_load_weight(!llvm.ptr) -> f32
  llvm.func @bcir_consume_mixed_stride(!llvm.ptr, f32, i64, i64, i64)

  llvm.func @lowered_claim(%arg0: !llvm.ptr) {
    %root_id = llvm.mlir.constant(0 : i64) : i64
    %child = llvm.call @bcir_lookup_child(%arg0, %root_id) : (!llvm.ptr, i64) -> !llvm.ptr

    // A lowered HAM hint may become a runtime prefetch call when an address is known.
    llvm.call @bcir_prefetch_vertex(%child) : (!llvm.ptr) -> ()

    %weight = llvm.call @bcir_load_weight(%child) : (!llvm.ptr) -> f32

    // Mixed Stride graph facts are now explicit runtime values/ABI fields.
    %stride0 = llvm.mlir.constant(1 : i64) : i64
    %stride1 = llvm.mlir.constant(4 : i64) : i64
    %stride2 = llvm.mlir.constant(16 : i64) : i64
    llvm.call @bcir_consume_mixed_stride(%child, %weight, %stride0, %stride1, %stride2)
      : (!llvm.ptr, f32, i64, i64, i64) -> ()

    llvm.return
  }
}
