// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// The gem.conv op verifier (hasVerifier; mirrors bcir/kbcir/conv.py::check_conv + the
// realization/cost invariants -- the op-level shape/dtype/realization well-formedness check
// gem.matmul/gem.activation validate inline, NOT a new globally-numbered R-law). Each section
// carries a single violation (mirrors lower_gem_activation_neg.mlir's style). A conv is a
// STRUCTURED MATMUL, so the checks span the conv shape laws (derived out dims, the im2col gemm
// dims, the kernel fitting the padded input) AND the matmul-style realization laws (strategy +
// tile + loop order, the bottleneck == max roofline parity, the R17 acc_bound).

// the DERIVED output dims: declared out_h/out_w must equal floor((in+2*pad-k)/stride)+1.
bcir.module @bad_out_dims {
  // expected-error @+1 {{conv: declared out 99x3 != the derived floor((in+2*pad-k)/stride)+1 = 3x3}}
  bcir.gem.conv @bad { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64,
                       kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32",
                       out_h = 99 : i64, out_w = 3 : i64,
                       gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 18 : i64,
                       strategy = "im2col", tile_m = 9 : i64, tile_n = 3 : i64, tile_k = 18 : i64,
                       loop_order = "ijk", compute_cost = 16 : i64, mem_cost = 68 : i64,
                       bottleneck = 68 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the equivalent im2col gemm dims: (gemm_m, gemm_n, gemm_k) must equal (out_h*out_w, out_c, in_c*kh*kw).
bcir.module @bad_gemm_dims {
  // expected-error @+1 {{conv: declared im2col gemm dims (9,3,17) != (out_h*out_w, out_c, in_c*kh*kw) = (9,3,18)}}
  bcir.gem.conv @bad { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64,
                       kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32",
                       out_h = 3 : i64, out_w = 3 : i64,
                       gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 17 : i64,
                       strategy = "im2col", tile_m = 9 : i64, tile_n = 3 : i64, tile_k = 17 : i64,
                       loop_order = "ijk", compute_cost = 16 : i64, mem_cost = 68 : i64,
                       bottleneck = 68 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the kernel must fit the padded input frame.
bcir.module @bad_kernel_fit {
  // expected-error @+1 {{conv: kernel 5x5 does not fit the padded input 3x3}}
  bcir.gem.conv @bad { in_c = 1 : i64, in_h = 3 : i64, in_w = 3 : i64, out_c = 1 : i64,
                       kh = 5 : i64, kw = 5 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32",
                       out_h = 1 : i64, out_w = 1 : i64,
                       gemm_m = 1 : i64, gemm_n = 1 : i64, gemm_k = 25 : i64,
                       strategy = "im2col", tile_m = 1 : i64, tile_n = 1 : i64, tile_k = 25 : i64,
                       loop_order = "ijk", compute_cost = 1 : i64, mem_cost = 1 : i64,
                       bottleneck = 1 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the realization strategy must be one of direct|im2col.
bcir.module @bad_strategy {
  // expected-error @+1 {{conv: strategy must be one of direct|im2col (got 'winograd')}}
  bcir.gem.conv @bad { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64,
                       kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32",
                       out_h = 3 : i64, out_w = 3 : i64,
                       gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 18 : i64,
                       strategy = "winograd", tile_m = 9 : i64, tile_n = 3 : i64, tile_k = 18 : i64,
                       loop_order = "ijk", compute_cost = 16 : i64, mem_cost = 68 : i64,
                       bottleneck = 68 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the direct strategy is the UNTILED gemm -- a tile smaller than the whole gemm is rejected.
bcir.module @bad_direct_tiled {
  // expected-error @+1 {{conv: the direct strategy is the UNTILED gemm -- tile must be the whole 9x3x18 (got 4x3x18)}}
  bcir.gem.conv @bad { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64,
                       kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32",
                       out_h = 3 : i64, out_w = 3 : i64,
                       gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 18 : i64,
                       strategy = "direct", tile_m = 4 : i64, tile_n = 3 : i64, tile_k = 18 : i64,
                       loop_order = "ijk", compute_cost = 16 : i64, mem_cost = 68 : i64,
                       bottleneck = 68 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the roofline invariant: bottleneck must equal max(compute_cost, mem_cost) (the max,+ parity).
bcir.module @bad_bottleneck {
  // expected-error @+1 {{conv: bottleneck must equal max(compute_cost, mem_cost) = 68 (got 99)}}
  bcir.gem.conv @bad { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64,
                       kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32",
                       out_h = 3 : i64, out_w = 3 : i64,
                       gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 18 : i64,
                       strategy = "im2col", tile_m = 9 : i64, tile_n = 3 : i64, tile_k = 18 : i64,
                       loop_order = "ijk", compute_cost = 16 : i64, mem_cost = 68 : i64,
                       bottleneck = 99 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the R17 accuracy axis: a QUANTIZED realization (quant_bits > 0) must carry the 1-ULP Q8-bridge
// bound on the accuracy axis -- declaring acc_bound 0 for a quantized conv is rejected.
bcir.module @bad_acc_bound {
  // expected-error @+1 {{conv: acc_bound must be 1 (the R17 Q8-bridge bound, 1 ULP for a quantized realization), got 0}}
  bcir.gem.conv @bad { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64,
                       kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32",
                       out_h = 3 : i64, out_w = 3 : i64,
                       gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 18 : i64,
                       strategy = "im2col", tile_m = 9 : i64, tile_n = 3 : i64, tile_k = 18 : i64,
                       loop_order = "ijk", compute_cost = 16 : i64, mem_cost = 68 : i64,
                       bottleneck = 68 : i64, quant_bits = 8 : i32, acc_bound = 0 : i64 }
}
