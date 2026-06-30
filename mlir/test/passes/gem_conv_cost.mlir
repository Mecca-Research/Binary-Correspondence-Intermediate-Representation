// RUN: bcir-opt -bcir-gem-conv-cost %s | FileCheck %s
//
// The -bcir-gem-conv-cost cost producer: a gem.conv plan op (the G7 K_BCIR-chosen direct|im2col
// realization of a 2-D single-group convolution) has its ROOFLINE COST RECOMPUTED (port of
// bcir/kbcir/conv.py::cost_of) and annotated as kbcir.* attrs. Unlike -bcir-lower-gem-conv (which
// ERASES the conv into its im2col-gemm tile gem.block sequence) this is a COST PRODUCER, not a
// lowering -- the op is NOT erased.
//
// This is the DUAL-RAIL PARITY gate (the law reproduces the oracle's conv.py roofline). A 2-D
// single-group conv is a STRUCTURED MATMUL: the im2col (patch-gather) reshaping turns it into the
// gemm C[out_pix x C_out] = patches[out_pix x (C_in*Kh*Kw)] @ W[(C_in*Kh*Kw) x C_out], so the conv
// is PRICED THROUGH THE EXISTING matmul roofline (NO bespoke conv term). cost_of calls matmul.cost_of
// on the equivalent im2col gemm dims (M = out_h*out_w, N = out_c, K = in_c*kh*kw) at the resolved tile:
//   compute   = ceilDiv(M*N*K, vector_width*(2 if fma else 1))
//   a_reads   = M*K*ceil(N/tile_n)   (the patch matrix re-read per N-tile -- the im2col traffic)
//   b_reads   = K*N*ceil(M/tile_m)   (the weight matrix re-read per M-tile)
//   c_traffic = M*N*(ceil(K/tile_k) + 1)
//   mem       = ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels)
//   bottleneck= max(compute, mem)
// The strategy only changes the tile: im2col prices at the chosen inner matmul tile; direct is the
// UNTILED whole gemm. Re-asserted host-agnostically in bcir/tests/test_conv_law_parity.py (the oracle
// target pinned to x86-avx512: vector_width 16, fma true, mem_unit 1, mem_channels 4 -> thr 32, bw 4).
//
// INFORMS-ONLY: the recomputed roofline informs the plan's cost but NEVER gates legality
// (kbcir.informs_only); the gem.conv op is NOT erased and never a legality verdict.

bcir.module @convs {
  // (1) a (C_in=2, 5x5) input, 3 filters, 3x3, valid (stride 1, pad 0): out_h = out_w = 3 ->
  // im2col gemm M = out_h*out_w = 9, N = out_c = 3, K = in_c*kh*kw = 2*3*3 = 18. The small conv fits
  // the cache budget untiled, so plan_conv chooses im2col at the WHOLE gemm tile (9,3,18):
  //   compute = ceilDiv(9*3*18, 32) = ceilDiv(486, 32) = 16;
  //   a_reads = 9*18*ceil(3/3) = 162; b_reads = 18*3*ceil(9/9) = 54; c_traffic = 9*3*(ceil(18/18)+1) = 54;
  //   mem = ceilDiv(162+54+54, 4) = ceilDiv(270, 4) = 68; bottleneck = max(16, 68) = 68 (memory-bound).
  bcir.gem.conv @conv_small { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64,
                              kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64,
                              dtype = "f32", out_h = 3 : i64, out_w = 3 : i64,
                              gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 18 : i64,
                              strategy = "im2col", tile_m = 9 : i64, tile_n = 3 : i64, tile_k = 18 : i64,
                              loop_order = "ijk", compute_cost = 16 : i64, mem_cost = 68 : i64,
                              bottleneck = 68 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }

  // (2) the SAME conv shape declared as the DIRECT strategy: no patch buffer is materialized, but the
  // direct loop can NOT block the reduction the way the im2col gemm tile does -- it is modeled as the
  // UNTILED gemm (tile = the whole M,N,K = 9,3,18). Same MAC count (a conv's MACs are fixed) -> the
  // SAME compute 16 and -- at this small whole-gemm size -- the SAME mem 68 / bottleneck 68. The
  // load-bearing identity: direct == matmul.cost_of at the untiled whole gemm (conv.cost_of's direct
  // branch). The verifier pins direct -> tile == the whole gemm; the cost pass prices it identically.
  bcir.gem.conv @conv_direct { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64,
                               kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64,
                               dtype = "f32", out_h = 3 : i64, out_w = 3 : i64,
                               gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 18 : i64,
                               strategy = "direct", tile_m = 9 : i64, tile_n = 3 : i64, tile_k = 18 : i64,
                               loop_order = "ijk", compute_cost = 16 : i64, mem_cost = 68 : i64,
                               bottleneck = 68 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }

  // (3) a larger (C_in=8, 16x16) input, 16 filters, 3x3, pad 1 (same-conv): out_h = out_w = 16 ->
  // im2col gemm M = 256, N = 16, K = 8*3*3 = 72. The whole untiled gemm does NOT fit the cache budget,
  // so plan_conv blocks the im2col gemm at the chosen inner matmul tile (64,16,72) -- a REAL tile
  // (tile_m 64 < M 256), so the patch-reuse traffic is the blocked-gemm reuse, NOT the untiled stream:
  //   compute = ceilDiv(256*16*72, 32) = ceilDiv(294912, 32) = 9216;
  //   a_reads = 256*72*ceil(16/16) = 18432; b_reads = 72*16*ceil(256/64) = 72*16*4 = 4608;
  //   c_traffic = 256*16*(ceil(72/72)+1) = 4096*2 = 8192; mem = ceilDiv(18432+4608+8192, 4) =
  //   ceilDiv(31232, 4) = 7808; bottleneck = max(9216, 7808) = 9216 (this one is COMPUTE-bound).
  bcir.gem.conv @conv_tiled { in_c = 8 : i64, in_h = 16 : i64, in_w = 16 : i64, out_c = 16 : i64,
                              kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 1 : i64,
                              dtype = "f32", out_h = 16 : i64, out_w = 16 : i64,
                              gemm_m = 256 : i64, gemm_n = 16 : i64, gemm_k = 72 : i64,
                              strategy = "im2col", tile_m = 64 : i64, tile_n = 16 : i64, tile_k = 72 : i64,
                              loop_order = "ijk", compute_cost = 9216 : i64, mem_cost = 7808 : i64,
                              bottleneck = 9216 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// The carrier ops are NOT erased (a cost producer, not a lowering).
// (1) conv_small: the small structured matmul, memory-bound; bottleneck 68.
// CHECK: bcir.gem.conv @conv_small
// CHECK-SAME: kbcir.bottleneck = 68 : i64
// CHECK-SAME: kbcir.compute_cost = 16 : i64
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.mem_cost = 68 : i64

// (2) conv_direct: the direct stream == the untiled whole gemm; identical compute/mem at this size.
// CHECK: bcir.gem.conv @conv_direct
// CHECK-SAME: kbcir.bottleneck = 68 : i64
// CHECK-SAME: kbcir.compute_cost = 16 : i64
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.mem_cost = 68 : i64

// (3) conv_tiled: the blocked im2col gemm (real tile 64x16x72), COMPUTE-bound; bottleneck 9216.
// CHECK: bcir.gem.conv @conv_tiled
// CHECK-SAME: kbcir.bottleneck = 9216 : i64
// CHECK-SAME: kbcir.compute_cost = 9216 : i64
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.mem_cost = 7808 : i64
