// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// The gem.attention op verifier (hasVerifier; mirrors bcir/kbcir/attention.py::check_attention + the
// realization/cost invariants -- the op-level shape/dtype/realization well-formedness check gem.matmul/
// gem.activation/gem.conv validate inline, NOT a new globally-numbered R-law). Each section carries a
// single violation (mirrors lower_gem_conv_neg.mlir's style). Attention DECOMPOSES into two matmuls with
// a softmax between them, so the checks span the attention shape laws (the two derived gemm dims), the
// quarantine dtype rule (the softmax needs f32), the per-gemm tile/loop realization laws, and the SUMMED
// roofline parity (compute/mem == the two priced matmuls; bottleneck == max; the R17 acc_bound).

// the QUARANTINE dtype rule: attention contains a softmax (a transcendental), so it needs f32, never i32.
bcir.module @bad_dtype {
  // expected-error @+1 {{attention: contains a softmax (a transcendental; the libm edge returns float), so it needs f32, got 'i32'}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "i32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 32 : i64, mem_cost = 128 : i64, bottleneck = 128 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the DERIVED scores Q@K^T gemm dims: (scores_m, scores_n, scores_k) must equal (seq_len, seq_len, d_k).
bcir.module @bad_scores_dims {
  // expected-error @+1 {{attention: declared scores Q@K^T gemm dims (8,8,7) != (seq_len, seq_len, d_k) = (8,8,8)}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 7 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 7 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 32 : i64, mem_cost = 128 : i64, bottleneck = 128 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the DERIVED context A@V gemm dims: (context_m, context_n, context_k) must equal (seq_len, d_k, seq_len).
bcir.module @bad_context_dims {
  // expected-error @+1 {{attention: declared context A@V gemm dims (8,8,7) != (seq_len, d_k, seq_len) = (8,8,8)}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 7 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 7 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 32 : i64, mem_cost = 128 : i64, bottleneck = 128 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the realization: each inner gemm tile must be in [1, its gemm dim] (a scores tile wider than the gemm).
bcir.module @bad_scores_tile {
  // expected-error @+1 {{attention: the scores gemm tile must be in [1, gemm dim] (tiles 9x8x8 vs gemm dims 8x8x8)}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 9 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 32 : i64, mem_cost = 128 : i64, bottleneck = 128 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the SUMMED roofline: compute_cost must equal scores_compute + context_compute (the two priced matmuls).
bcir.module @bad_compute_sum {
  // expected-error @+1 {{attention: compute_cost must equal scores_compute + context_compute = 32 (the SUM of the two priced matmuls; got 99)}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 99 : i64, mem_cost = 128 : i64, bottleneck = 128 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the SUMMED roofline: mem_cost must equal scores_mem + context_mem (the two priced matmuls).
bcir.module @bad_mem_sum {
  // expected-error @+1 {{attention: mem_cost must equal scores_mem + context_mem = 128 (the SUM of the two priced matmuls; got 99)}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 32 : i64, mem_cost = 99 : i64, bottleneck = 99 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the roofline invariant: bottleneck must equal max(compute_cost, mem_cost) (the max,+ parity).
bcir.module @bad_bottleneck {
  // expected-error @+1 {{attention: bottleneck must equal max(compute_cost, mem_cost) = 128 (got 99)}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 32 : i64, mem_cost = 128 : i64, bottleneck = 99 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the R17 accuracy axis: a QUANTIZED realization (quant_bits > 0) must carry the 1-ULP Q8-bridge bound
// on the accuracy axis -- declaring acc_bound 0 for a quantized attention is rejected.
bcir.module @bad_acc_bound {
  // expected-error @+1 {{attention: acc_bound must be 1 (the R17 Q8-bridge bound, 1 ULP for a quantized realization), got 0}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 32 : i64, mem_cost = 128 : i64, bottleneck = 128 : i64, quant_bits = 8 : i32, acc_bound = 0 : i64 }
}
