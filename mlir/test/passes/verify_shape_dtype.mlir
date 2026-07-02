// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// R22 (shape consistency) + R23 (dtype compatibility) over the gem.* tensor seams -- the D2
// promotion (ML/AI roadmap §8.2): the op-level shape/dtype checks the gem op verifiers already
// enforce per-op are joined by the producer->consumer SEAM laws no op verifier can see. The seam
// is the fusion adjacency contract (-bcir-fuse-matmul-activation): a gem.activation immediately
// following a gem.matmul consumes its full m*n product (R22); one following a gem.conv or
// gem.attention inherits its dtype (R23). Each op below passes its own parse-time verifier --
// only the SEAM is illegal. Oracle twins: verify.verify_shape / verify.verify_ml_spec.

// (1 -- R22 NEGATIVE) the activation epilogue declares 8 elements; the adjacent matmul hands over
// m*n = 16. Both ops are individually well-formed.
bcir.module @r22_shape_seam {
  bcir.gem.matmul @mm0 { m = 4 : i64, n = 4 : i64, k = 4 : i64,
                         tile_m = 2 : i64, tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk",
                         compute_cost = 100 : i64, mem_cost = 200 : i64, bottleneck = 200 : i64,
                         quant_bits = 0 : i32 }
  // expected-error @+1 {{R22: gem.activation relu0 consumes the adjacent gem.matmul @mm0 but declares a shape extent of 8 elements != the matmul's m*n = 16}}
  bcir.gem.activation @relu0 { kind = "relu", shape = array<i64: 8>, dtype = "f32",
                               axis_len = 0 : i64, width = 8 : i64,
                               compute_cost = 2 : i64, mem_cost = 16 : i64, bottleneck = 16 : i64,
                               quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// (2 -- R23 NEGATIVE) the activation epilogue declares dtype i32; the adjacent conv producer is
// f32. Both dtypes are individually legal -- only the handover is incompatible.
bcir.module @r23_dtype_seam {
  bcir.gem.conv @conv0 { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64,
                         kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64,
                         dtype = "f32", out_h = 3 : i64, out_w = 3 : i64,
                         gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 18 : i64,
                         strategy = "im2col", tile_m = 9 : i64, tile_n = 3 : i64, tile_k = 18 : i64,
                         loop_order = "ijk", compute_cost = 16 : i64, mem_cost = 68 : i64,
                         bottleneck = 68 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
  // expected-error @+1 {{R23: gem.activation relu1 consumes the adjacent gem.conv @conv0 but declares dtype 'i32' != the producer's 'f32'}}
  bcir.gem.activation @relu1 { kind = "relu", shape = array<i64: 27>, dtype = "i32",
                               axis_len = 0 : i64, width = 27 : i64,
                               compute_cost = 2 : i64, mem_cost = 54 : i64, bottleneck = 54 : i64,
                               quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// (3 -- POSITIVE) the consistent seam: the activation consumes exactly m*n = 16 f32 elements.
bcir.module @r22_r23_legal_seam {
  bcir.gem.matmul @mm1 { m = 4 : i64, n = 4 : i64, k = 4 : i64,
                         tile_m = 2 : i64, tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk",
                         compute_cost = 100 : i64, mem_cost = 200 : i64, bottleneck = 200 : i64,
                         quant_bits = 0 : i32 }
  bcir.gem.activation @relu2 { kind = "relu", shape = array<i64: 16>, dtype = "f32",
                               axis_len = 0 : i64, width = 16 : i64,
                               compute_cost = 2 : i64, mem_cost = 32 : i64, bottleneck = 32 : i64,
                               quant_bits = 0 : i32, acc_bound = 0 : i64 }
}
