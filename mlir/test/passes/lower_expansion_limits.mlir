// RUN: bcir-opt -bcir-lower-gem-matmul -bcir-lower-gem-activation -bcir-lower-gem-conv -bcir-lower-gem-attention -verify-diagnostics -split-input-file %s
//
// A tile/stripe lowering must reject an impractical materialized expansion
// before producing any partial IR.  Larger graphs belong on loop-form lowerings.

bcir.module @matmul_expansion {
  // expected-error @+1 {{bcir-lower-gem-matmul: tile expansion is invalid or exceeds the 65536-block safety limit}}
  bcir.gem.matmul @m { m = 65537 : i64, n = 1 : i64, k = 1 : i64, tile_m = 1 : i64, tile_n = 1 : i64, tile_k = 1 : i64, loop_order = "ijk", compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64 }
}

// -----

bcir.module @activation_expansion {
  // expected-error @+1 {{bcir-lower-gem-activation: stripe expansion exceeds the 65536-block safety limit}}
  bcir.gem.activation @a { kind = "relu", shape = array<i64: 65537>, dtype = "f32", axis_len = 0 : i64, width = 1 : i64, compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64 }
}

// -----

bcir.module @conv_expansion {
  // expected-error @+1 {{bcir-lower-gem-conv: tile expansion is invalid or exceeds the 65536-block safety limit}}
  bcir.gem.conv @c { in_c = 1 : i64, in_h = 65537 : i64, in_w = 1 : i64, out_c = 1 : i64, kh = 1 : i64, kw = 1 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32", out_h = 65537 : i64, out_w = 1 : i64, gemm_m = 65537 : i64, gemm_n = 1 : i64, gemm_k = 1 : i64, strategy = "im2col", tile_m = 1 : i64, tile_n = 1 : i64, tile_k = 1 : i64, loop_order = "ijk", compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64 }
}

// -----

bcir.module @attention_expansion {
  // expected-error @+1 {{bcir-lower-gem-attention: decomposition is invalid or exceeds the 65536-block safety limit}}
  bcir.gem.attention @a { seq_len = 257 : i64, d_k = 1 : i64, dtype = "f32", scores_m = 257 : i64, scores_n = 257 : i64, scores_k = 1 : i64, context_m = 257 : i64, context_n = 1 : i64, context_k = 257 : i64, scores_tile_m = 1 : i64, scores_tile_n = 1 : i64, scores_tile_k = 1 : i64, scores_loop_order = "ijk", context_tile_m = 1 : i64, context_tile_n = 1 : i64, context_tile_k = 1 : i64, context_loop_order = "ijk", scores_compute = 0 : i64, scores_mem = 0 : i64, context_compute = 0 : i64, context_mem = 0 : i64, compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64 }
}
