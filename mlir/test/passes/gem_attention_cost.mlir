// RUN: bcir-opt -bcir-gem-attention-cost %s | FileCheck %s
//
// The -bcir-gem-attention-cost cost producer: a gem.attention plan op (the G7 K_BCIR-chosen
// two-matmul realization of single-head scaled-dot-product attention softmax(Q@K^T/sqrt(d_k))@V)
// has its ROOFLINE COST RECOMPUTED (port of bcir/kbcir/attention.py::cost_of) and annotated as
// kbcir.* attrs. Unlike -bcir-lower-gem-attention (which ERASES the attention into its two
// gem.matmul tile sequences + the softmax stripe) this is a COST PRODUCER, not a lowering -- the
// op is NOT erased.
//
// This is the DUAL-RAIL PARITY gate (the law reproduces the oracle's attention.py roofline).
// Single-head scaled-dot-product attention DECOMPOSES into two gem.matmuls (a softmax + an exact
// scale ride between them) -- the scores Q@K^T gemm = (seq, seq, d_k) and the context A@V gemm =
// (seq, d_k, seq). So attention is PRICED THROUGH THE EXISTING matmul roofline for BOTH matmuls (NO
// bespoke attention term): cost_of calls matmul.cost_of on each gemm at its chosen tile and SUMS
// them (the O(seq^2) scale/softmax passes are dominated by, and folded into, the Q@K^T traffic --
// attention.cost_of returns c1+c2, m1+m2, NO softmax term). For each gemm matmul.cost_of is:
//   compute   = ceilDiv(M*N*K, vector_width*(2 if fma else 1))
//   a_reads   = M*K*ceil(N/tile_n)   b_reads = K*N*ceil(M/tile_m)   c_traffic = M*N*(ceil(K/tile_k)+1)
//   mem       = ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels)
// and the attention totals are compute_cost = scores_compute + context_compute, mem_cost =
// scores_mem + context_mem, bottleneck = max(compute_cost, mem_cost). Re-asserted host-agnostically
// in bcir/tests/test_attention_law_parity.py (the oracle target pinned to x86-avx512: vector_width
// 16, fma true, mem_unit 1, mem_channels 4 -> thr 32, bw 4).
//
// INFORMS-ONLY: the recomputed roofline informs the plan's cost but NEVER gates legality
// (kbcir.informs_only); the gem.attention op is NOT erased and never a legality verdict.

bcir.module @attentions {
  // (1) a small dense attention seq_len = d_k = 8: BOTH gemms are 8x8x8 (scores Q@K^T = (8,8,8),
  // context A@V = (8,8,8)), each at the whole untiled tile (8,8,8). matmul.cost_of for an 8x8x8 gemm:
  //   compute = ceilDiv(8*8*8, 32) = ceilDiv(512, 32) = 16;
  //   a_reads = 8*8*ceil(8/8) = 64, b_reads = 8*8*ceil(8/8) = 64, c_traffic = 8*8*(ceil(8/8)+1) = 128;
  //   mem = ceilDiv(64+64+128, 4) = ceilDiv(256, 4) = 64. So scores = context = (16, 64). The SUM:
  //   compute_cost = 16+16 = 32, mem_cost = 64+64 = 128, bottleneck = max(32, 128) = 128 (memory-bound).
  bcir.gem.attention @at_square { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 32 : i64, mem_cost = 128 : i64, bottleneck = 128 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }

  // (2) a wider head seq_len = 4, d_k = 16, QUANTIZED (quant_bits = 8 -> the R17 Q8-bridge bound,
  // acc_bound = 1; the cost roofline is unaffected). The two gemms are ASYMMETRIC: scores Q@K^T =
  // (4,4,16), context A@V = (4,16,4), each whole-tiled. matmul.cost_of:
  //   scores (4,4,16): compute = ceilDiv(4*4*16, 32) = ceilDiv(256, 32) = 8; a_reads = 4*16*ceil(4/4)
  //     = 64, b_reads = 16*4*ceil(4/4) = 64, c_traffic = 4*4*(ceil(16/16)+1) = 32; mem = ceilDiv(160, 4)
  //     = 40. scores = (8, 40).
  //   context (4,16,4): compute = ceilDiv(4*16*4, 32) = ceilDiv(256, 32) = 8; a_reads = 4*4*ceil(16/16)
  //     = 16, b_reads = 4*16*ceil(4/4) = 64, c_traffic = 4*16*(ceil(4/4)+1) = 128; mem = ceilDiv(208, 4)
  //     = 52. context = (8, 52).
  //   The SUM: compute_cost = 8+8 = 16, mem_cost = 40+52 = 92, bottleneck = max(16, 92) = 92.
  bcir.gem.attention @at_wide_q { seq_len = 4 : i64, d_k = 16 : i64, dtype = "f32",
    scores_m = 4 : i64, scores_n = 4 : i64, scores_k = 16 : i64,
    context_m = 4 : i64, context_n = 16 : i64, context_k = 4 : i64,
    scores_tile_m = 4 : i64, scores_tile_n = 4 : i64, scores_tile_k = 16 : i64, scores_loop_order = "ijk",
    context_tile_m = 4 : i64, context_tile_n = 16 : i64, context_tile_k = 4 : i64, context_loop_order = "ijk",
    scores_compute = 8 : i64, scores_mem = 40 : i64, context_compute = 8 : i64, context_mem = 52 : i64,
    compute_cost = 16 : i64, mem_cost = 92 : i64, bottleneck = 92 : i64, quant_bits = 8 : i32, acc_bound = 1 : i64 }

  // (3) a longer sequence seq_len = 16, d_k = 8: scores Q@K^T = (16,16,8), context A@V = (16,8,16),
  // each whole-tiled. matmul.cost_of:
  //   scores (16,16,8): compute = ceilDiv(16*16*8, 32) = ceilDiv(2048, 32) = 64; a_reads = 16*8*ceil(16/16)
  //     = 128, b_reads = 8*16*ceil(16/16) = 128, c_traffic = 16*16*(ceil(8/8)+1) = 512; mem = ceilDiv(768, 4)
  //     = 192. scores = (64, 192).
  //   context (16,8,16): compute = ceilDiv(16*8*16, 32) = ceilDiv(2048, 32) = 64; a_reads = 16*16*ceil(8/8)
  //     = 256, b_reads = 16*8*ceil(16/16) = 128, c_traffic = 16*8*(ceil(16/16)+1) = 256; mem = ceilDiv(640, 4)
  //     = 160. context = (64, 160).
  //   The SUM: compute_cost = 64+64 = 128, mem_cost = 192+160 = 352, bottleneck = max(128, 352) = 352.
  bcir.gem.attention @at_long { seq_len = 16 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 16 : i64, scores_n = 16 : i64, scores_k = 8 : i64,
    context_m = 16 : i64, context_n = 8 : i64, context_k = 16 : i64,
    scores_tile_m = 16 : i64, scores_tile_n = 16 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 16 : i64, context_tile_n = 8 : i64, context_tile_k = 16 : i64, context_loop_order = "ijk",
    scores_compute = 64 : i64, scores_mem = 192 : i64, context_compute = 64 : i64, context_mem = 160 : i64,
    compute_cost = 128 : i64, mem_cost = 352 : i64, bottleneck = 352 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// The carrier ops are NOT erased (a cost producer, not a lowering). The kbcir.* attrs print in
// alphabetical order (bottleneck, compute_cost, context_compute, context_mem, fits_cache,
// informs_only, mem_cost, scores_compute, scores_mem).
//
// (1) at_square: both gemms 8x8x8; the SUMMED roofline is memory-bound (bottleneck 128).
// CHECK: bcir.gem.attention @at_square
// CHECK-SAME: kbcir.bottleneck = 128 : i64
// CHECK-SAME: kbcir.compute_cost = 32 : i64
// CHECK-SAME: kbcir.context_compute = 16 : i64
// CHECK-SAME: kbcir.context_mem = 64 : i64
// CHECK-SAME: kbcir.fits_cache = true
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.mem_cost = 128 : i64
// CHECK-SAME: kbcir.scores_compute = 16 : i64
// CHECK-SAME: kbcir.scores_mem = 64 : i64

// (2) at_wide_q: the asymmetric quantized wide head (scores 4x4x16, context 4x16x4); bottleneck 92.
// CHECK: bcir.gem.attention @at_wide_q
// CHECK-SAME: kbcir.bottleneck = 92 : i64
// CHECK-SAME: kbcir.compute_cost = 16 : i64
// CHECK-SAME: kbcir.context_compute = 8 : i64
// CHECK-SAME: kbcir.context_mem = 52 : i64
// CHECK-SAME: kbcir.fits_cache = true
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.mem_cost = 92 : i64
// CHECK-SAME: kbcir.scores_compute = 8 : i64
// CHECK-SAME: kbcir.scores_mem = 40 : i64

// (3) at_long: the longer sequence (scores 16x16x8, context 16x8x16); memory-bound bottleneck 352.
// CHECK: bcir.gem.attention @at_long
// CHECK-SAME: kbcir.bottleneck = 352 : i64
// CHECK-SAME: kbcir.compute_cost = 128 : i64
// CHECK-SAME: kbcir.context_compute = 64 : i64
// CHECK-SAME: kbcir.context_mem = 160 : i64
// CHECK-SAME: kbcir.fits_cache = true
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.mem_cost = 352 : i64
// CHECK-SAME: kbcir.scores_compute = 64 : i64
// CHECK-SAME: kbcir.scores_mem = 192 : i64
