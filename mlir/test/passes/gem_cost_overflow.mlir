// RUN: bcir-opt -bcir-gem-matmul-cost -bcir-gem-activation-cost -bcir-gem-conv-cost -bcir-gem-attention-cost -verify-diagnostics -split-input-file %s
//
// The carrier-op verifier bounds its directly represented geometry.  The cost
// passes must separately reject larger derived traffic/weight terms that do
// not fit the signed-i64 wire domain.

bcir.module @matmul_traffic_overflow {
  // expected-error @+1 {{bcir-gem-matmul-cost: derived roofline cost exceeds signed 64-bit range}}
  bcir.gem.matmul @m { m = 3037000499 : i64, n = 3037000499 : i64, k = 1 : i64, tile_m = 3037000499 : i64, tile_n = 3037000499 : i64, tile_k = 1 : i64, loop_order = "ijk", compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64 }
}

// -----

bcir.module @activation_traffic_overflow {
  // expected-error @+1 {{bcir-gem-activation-cost: memory term exceeds signed 64-bit range}}
  bcir.gem.activation @a { kind = "relu", shape = array<i64: 9223372036854775807>, dtype = "f32", axis_len = 0 : i64, width = 1 : i64, compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64 }
}

// -----

bcir.module @conv_traffic_overflow {
  // expected-error @+1 {{bcir-gem-conv-cost: derived roofline cost exceeds signed 64-bit range}}
  bcir.gem.conv @c { in_c = 1 : i64, in_h = 3037000499 : i64, in_w = 3037000499 : i64, out_c = 1 : i64, kh = 1 : i64, kw = 1 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32", out_h = 3037000499 : i64, out_w = 3037000499 : i64, gemm_m = 9223372030926249001 : i64, gemm_n = 1 : i64, gemm_k = 1 : i64, strategy = "im2col", tile_m = 9223372030926249001 : i64, tile_n = 1 : i64, tile_k = 1 : i64, loop_order = "ijk", compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64 }
}

// -----

bcir.module @attention_traffic_overflow {
  // expected-error @+1 {{bcir-gem-attention-cost: derived per-gemm roofline cost exceeds signed 64-bit range}}
  bcir.gem.attention @a { seq_len = 3037000499 : i64, d_k = 1 : i64, dtype = "f32", scores_m = 3037000499 : i64, scores_n = 3037000499 : i64, scores_k = 1 : i64, context_m = 3037000499 : i64, context_n = 1 : i64, context_k = 3037000499 : i64, scores_tile_m = 3037000499 : i64, scores_tile_n = 3037000499 : i64, scores_tile_k = 1 : i64, scores_loop_order = "ijk", context_tile_m = 3037000499 : i64, context_tile_n = 1 : i64, context_tile_k = 3037000499 : i64, context_loop_order = "ijk", scores_compute = 0 : i64, scores_mem = 0 : i64, context_compute = 0 : i64, context_mem = 0 : i64, compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64 }
}
