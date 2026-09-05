// RUN: bcir-opt -bcir-lower-gem-conv -verify-diagnostics -split-input-file %s
//
// S0-8 / row 13 of the 2026-07/08 assessment: a verifier-legal convolution overflowed signed
// arithmetic in the GEM lowerer. Every extent the gem.conv verifier bounded fit signed 64-bit --
// out_h*out_w = 2^62, in_c*kh*kw = 1 -- and the ONE tile of this direct realization passed
// checkedTileCount; but the tile's element count rows*cols = M*N = 2^62 * 4 = 2^64 wrapped, and
// -bcir-lower-gem-conv emitted `bcir.gem.block @cv_t0 {base = 0, count = 0}` (a zero-count
// realization of a 2^64-element output; measured on the parent build). The op verifier now
// bounds the output element count M*N and the im2col work M*N*K (the same wire-domain rule
// conv.check_conv applies on the oracle), and the lowerer computes every tile origin and count
// with checked arithmetic behind that bound.

bcir.module @one_tile_output_overflow {
  // expected-error @+1 {{conv: output element count exceeds signed 64-bit range}}
  bcir.gem.conv @cv { in_c = 1 : i64, in_h = 2147483648 : i64, in_w = 2147483648 : i64, out_c = 4 : i64, kh = 1 : i64, kw = 1 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32", out_h = 2147483648 : i64, out_w = 2147483648 : i64, gemm_m = 4611686018427387904 : i64, gemm_n = 4 : i64, gemm_k = 1 : i64, strategy = "direct", tile_m = 4611686018427387904 : i64, tile_n = 4 : i64, tile_k = 1 : i64, loop_order = "ijk", compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// The output fits (M*N = 2^50) but the im2col work M*N*K = 2^64 does not: the roofline the
// cost passes derive from it would wrap. Refused at the same seam.
bcir.module @work_overflow {
  // expected-error @+1 {{conv: im2col work M*N*K exceeds signed 64-bit range}}
  bcir.gem.conv @cv { in_c = 16384 : i64, in_h = 1048576 : i64, in_w = 1048576 : i64, out_c = 1024 : i64, kh = 1 : i64, kw = 1 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32", out_h = 1048576 : i64, out_w = 1048576 : i64, gemm_m = 1099511627776 : i64, gemm_n = 1024 : i64, gemm_k = 16384 : i64, strategy = "direct", tile_m = 1099511627776 : i64, tile_n = 1024 : i64, tile_k = 16384 : i64, loop_order = "ijk", compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}
