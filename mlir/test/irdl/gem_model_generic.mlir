// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// G32: the GEM model seams in generic syntax -- the attribute-only plan records of the matmul,
// attention, normalization, cache, embedding, convolution, autodiff, contention and layout
// planners, and the matmul's lowering buffers -- round-tripped through the IRDL projection.

"bcir.gem_matmul"() {sym_name = "mm", m = 64 : i64, n = 64 : i64, k = 64 : i64,
  tile_m = 16 : i64, tile_n = 16 : i64, tile_k = 16 : i64, loop_order = "ijk",
  compute_cost = 262144 : i64, mem_cost = 49152 : i64, bottleneck = 262144 : i64} : () -> ()
"bcir.gem_fused_matmul_activation"() {sym_name = "mma", m = 64 : i64, n = 64 : i64,
  k = 64 : i64, tile_m = 16 : i64, tile_n = 16 : i64, tile_k = 16 : i64, loop_order = "ijk",
  activation = "relu"} : () -> ()
"bcir.gem_activation"() {sym_name = "act", kind = "gelu", shape = array<i64: 64, 64>,
  dtype = "f32", compute_cost = 4096 : i64, mem_cost = 32768 : i64, bottleneck = 32768 : i64}
  : () -> ()
"bcir.gem_attention"() {sym_name = "attn", seq_len = 128 : i64, d_k = 64 : i64, dtype = "f32"}
  : () -> ()
"bcir.gem_gqa_attention"() {sym_name = "gqa", seq_len = 128 : i64, d_k = 64 : i64,
  n_heads = 8 : i64, n_kv_heads = 2 : i64, dtype = "f32"} : () -> ()
"bcir.gem_rmsnorm"() {sym_name = "norm", rows = 128 : i64, dim = 64 : i64,
  gamma_len = 64 : i64, dtype = "f32"} : () -> ()
"bcir.gem_rope"() {sym_name = "rope", rows = 128 : i64, dim = 64 : i64, base = 10000 : i64,
  pos_offset = 0 : i64, dtype = "f32"} : () -> ()
"bcir.gem_kv_cache"() {sym_name = "kv", n_layers = 2 : i64, n_kv_heads = 2 : i64,
  d_k = 64 : i64, capacity = 512 : i64, pos = 0 : i64, dtype = "f32"} : () -> ()
"bcir.gem_embedding"() {sym_name = "emb", vocab_size = 32000 : i64, dim = 64 : i64,
  n_ids = 16 : i64, dtype = "f32"} : () -> ()
"bcir.gem_conv"() {sym_name = "conv", in_c = 3 : i64, in_h = 32 : i64, in_w = 32 : i64,
  out_c = 8 : i64, kh = 3 : i64, kw = 3 : i64, stride = 1 : i64} : () -> ()
"bcir.gem_autodiff"() {sym_name = "grad", n_inputs = 2 : i64, opcodes = array<i64: 1, 2>,
  arities = array<i64: 2, 2>, arg_base = array<i64: 0, 2>, args = array<i64: 0, 1, 2, 1>,
  consts = array<i64>, output = 3 : i64} : () -> ()
"bcir.gem_contention"() {sym_name = "cont", stride = 64 : i64, elem_bytes = 4 : i64,
  count = 1024 : i64} : () -> ()
"bcir.gem_layout_pivot"() {sym_name = "pivot", rid = 7 : i64, fields = 4 : i64,
  single_field_records = 1000 : i64, whole_record_records = 10 : i64} : () -> ()
func.func @buffers(%a: memref<64x32xf32>, %b: memref<32x16xf32>, %c: memref<64x16xf32>) {
  "bcir.gem_matmul_buffer"(%a, %b, %c) {tile_m = 16 : i64, tile_n = 16 : i64, tile_k = 16 : i64,
    loop_order = "ijk"} : (memref<64x32xf32>, memref<32x16xf32>, memref<64x16xf32>) -> ()
  return
}

// CHECK: "bcir.gem_matmul"
// CHECK: "bcir.gem_fused_matmul_activation"
// CHECK: "bcir.gem_activation"
// CHECK: "bcir.gem_attention"
// CHECK: "bcir.gem_gqa_attention"
// CHECK: "bcir.gem_rmsnorm"
// CHECK: "bcir.gem_rope"
// CHECK: "bcir.gem_kv_cache"
// CHECK: "bcir.gem_embedding"
// CHECK: "bcir.gem_conv"
// CHECK: "bcir.gem_autodiff"
// CHECK: "bcir.gem_contention"
// CHECK: "bcir.gem_layout_pivot"
// CHECK: "bcir.gem_matmul_buffer"
