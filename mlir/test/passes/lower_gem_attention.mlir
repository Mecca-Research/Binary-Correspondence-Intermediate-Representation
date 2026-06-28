// RUN: bcir-opt -bcir-lower-gem-attention %s | FileCheck %s
//
// The -bcir-lower-gem-attention lowering: a gem.attention plan record (the K_BCIR-chosen two-matmul
// realization of single-head scaled-dot-product attention softmax(Q@K^T/sqrt(d_k))@V, G7) is lowered
// into its CONCRETE DECOMPOSITION -- the two underlying gem.matmul tile sequences (the scores Q@K^T
// gemm and the context A@V gemm) with the softmax gem.activation between them. Attention is NOT a new
// primitive: it is a COMPOSITION of ops BCIR already has, so each matmul lowers through the SAME tiled-
// gem.block emission -bcir-lower-gem-matmul / -bcir-lower-gem-conv use, and the softmax through the
// SAME single-stripe gem.block -bcir-lower-gem-activation emits (NO bespoke attention loop). The
// attention op is ERASED (a genuine lowering).
//
// This is the DUAL-RAIL PARITY gate (the law reproduces the oracle's bcir/kbcir/attention.py plan):
// the attention is PRICED THROUGH THE EXISTING matmul roofline for BOTH matmuls -- its (compute, mem)
// is the SUM of the two priced matmul costs (NO bespoke term; the O(seq^2) scale/softmax passes are
// folded into the Q@K^T traffic). The recomputed kbcir.* on the first scores tile equal plan_attention/
// cost_of for the same attention. The parity is re-asserted host-agnostically in bcir/tests/test_
// attention_law_parity.py, which drives bcir-opt with the oracle target pinned to x86-avx512.

bcir.module @attn {
  // (1) seq=8, d_k=8 (avx512-pinned oracle plan). Both gemms are 8x8x8, each chosen UNTILED (whole 8x8x8)
  // -> ONE scores tile + ONE context tile. The softmax stripe runs over the seq x seq = 8x8 = 64-elt score
  // matrix. Dense -> acc_bound 0. Per-gemm cost (16,64) each -> SUM compute 32, mem 128, bottleneck 128.
  bcir.gem.attention @at0 { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 32 : i64, mem_cost = 128 : i64, bottleneck = 128 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }

  // (2) seq=4, d_k=16, QUANTIZED (quant_bits = 8) -> the R17 Q8-bridge bound: acc_bound 1. The scores
  // gemm is 4x4x16, the context gemm 4x16x4 (the asymmetric A@V), each untiled. The softmax stripe runs
  // over 4x4 = 16 elts. Per-gemm (8,40) + (8,52) -> SUM compute 16, mem 92, bottleneck 92.
  bcir.gem.attention @atq { seq_len = 4 : i64, d_k = 16 : i64, dtype = "f32",
    scores_m = 4 : i64, scores_n = 4 : i64, scores_k = 16 : i64,
    context_m = 4 : i64, context_n = 16 : i64, context_k = 4 : i64,
    scores_tile_m = 4 : i64, scores_tile_n = 4 : i64, scores_tile_k = 16 : i64, scores_loop_order = "ijk",
    context_tile_m = 4 : i64, context_tile_n = 16 : i64, context_tile_k = 4 : i64, context_loop_order = "ijk",
    scores_compute = 8 : i64, scores_mem = 40 : i64, context_compute = 8 : i64, context_mem = 52 : i64,
    compute_cost = 16 : i64, mem_cost = 92 : i64, bottleneck = 92 : i64, quant_bits = 8 : i32, acc_bound = 1 : i64 }

  // (3) a MULTI-TILE scores gemm to exercise the tile-sequence enumeration: seq=4, d_k=4, the 4x4x4 scores
  // gemm DELIBERATELY tiled 2x2x4 -> ceil(4/2)=2 i, ceil(4/2)=2 j, 1 k = 4 tiles (ijk order); the context
  // gemm 4x4x4 untiled = 1 tile. Per-gemm (4,20) each -> SUM compute 8, mem 40, bottleneck 40.
  bcir.gem.attention @atm { seq_len = 4 : i64, d_k = 4 : i64, dtype = "f32",
    scores_m = 4 : i64, scores_n = 4 : i64, scores_k = 4 : i64,
    context_m = 4 : i64, context_n = 4 : i64, context_k = 4 : i64,
    scores_tile_m = 2 : i64, scores_tile_n = 2 : i64, scores_tile_k = 4 : i64, scores_loop_order = "ijk",
    context_tile_m = 4 : i64, context_tile_n = 4 : i64, context_tile_k = 4 : i64, context_loop_order = "ijk",
    scores_compute = 4 : i64, scores_mem = 20 : i64, context_compute = 4 : i64, context_mem = 20 : i64,
    compute_cost = 8 : i64, mem_cost = 40 : i64, bottleneck = 40 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// Every plan record must be gone after lowering.
// CHECK-NOT: bcir.gem.attention

// (1) at0: the decomposition -- the scores Q@K^T tile, the softmax stripe, then the context A@V tile.
// The first scores tile @at0_s_t0 carries the recomputed dual-rail plan record: the two tiles' loop
// orders, the per-gemm priced costs, the SUMMED roofline (compute 32 = 16+16, mem 128 = 64+64,
// bottleneck 128), and the dense acc_bound 0. strideA = scores_k 8, strideB = strideD = scores_n 8.
// CHECK: bcir.gem.block @at0_s_t0 {base = 0 : i64, count = 64 : i64, kbcir.acc_bound = 0 : i64, kbcir.bottleneck = 128 : i64, kbcir.compute_cost = 32 : i64, kbcir.context_compute = 16 : i64, kbcir.context_loop_order = "ijk", kbcir.context_mem = 64 : i64, kbcir.d_k = 8 : i64, kbcir.mem_cost = 128 : i64, kbcir.scores_compute = 16 : i64, kbcir.scores_loop_order = "ijk", kbcir.scores_mem = 64 : i64, kbcir.seq_len = 8 : i64, strideA = 8 : i64, strideB = 8 : i64, strideD = 8 : i64}
// The softmax stripe @at0_sm_s0 over the 8x8 = 64-elt score matrix: the SOLE transcendental (it routes
// expf via the trusted c.call.libm: edge, the quarantine; the matmuls are exact). strideA = 1 (unit
// read), strideB = seq_len 8 (the last-axis reduce row length), strideD = 1.
// CHECK: bcir.gem.block @at0_sm_s0 {base = 0 : i64, count = 64 : i64, kbcir.lane_kind = "transcendental", kbcir.libm_edge = "expf", strideA = 1 : i64, strideB = 8 : i64, strideD = 1 : i64}
// The context A@V tile @at0_c_t0 (count = 8x8 = 64). strideA = context_k 8, strideB = strideD = context_n 8.
// CHECK: bcir.gem.block @at0_c_t0 {base = 0 : i64, count = 64 : i64, strideA = 8 : i64, strideB = 8 : i64, strideD = 8 : i64}
// Exactly one scores tile + one context tile.
// CHECK-NOT: bcir.gem.block @at0_s_t1
// CHECK-NOT: bcir.gem.block @at0_c_t1

// (2) atq: QUANTIZED -> acc_bound 1; the asymmetric context gemm 4x16x4 (context A@V tile count = 4x16 =
// 64). The summed roofline compute 16 = 8+8, mem 92 = 40+52, bottleneck 92. The softmax stripe runs over
// 4x4 = 16 elts, strideB = seq_len 4.
// CHECK: bcir.gem.block @atq_s_t0 {base = 0 : i64, count = 16 : i64, kbcir.acc_bound = 1 : i64, kbcir.bottleneck = 92 : i64, kbcir.compute_cost = 16 : i64, kbcir.context_compute = 8 : i64, kbcir.context_loop_order = "ijk", kbcir.context_mem = 52 : i64, kbcir.d_k = 16 : i64, kbcir.mem_cost = 92 : i64, kbcir.scores_compute = 8 : i64, kbcir.scores_loop_order = "ijk", kbcir.scores_mem = 40 : i64, kbcir.seq_len = 4 : i64, strideA = 16 : i64, strideB = 4 : i64, strideD = 4 : i64}
// CHECK: bcir.gem.block @atq_sm_s0 {base = 0 : i64, count = 16 : i64, kbcir.lane_kind = "transcendental", kbcir.libm_edge = "expf", strideA = 1 : i64, strideB = 4 : i64, strideD = 1 : i64}
// CHECK: bcir.gem.block @atq_c_t0 {base = 0 : i64, count = 64 : i64, strideA = 4 : i64, strideB = 16 : i64, strideD = 16 : i64}

// (3) atm: the 4-tile scores gemm in ijk order. t0 (i=0,j=0) base 0; t1 (i=0,j=1) base 0*4+2 = 2; t2
// (i=1,j=0) base 2*4+0 = 8; t3 (i=1,j=1) base 2*4+2 = 10. Each tile count = 2x2 = 4. Then one softmax
// stripe (16 elts) + one context tile.
// CHECK: bcir.gem.block @atm_s_t0 {base = 0 : i64, count = 4 : i64, kbcir.acc_bound = 0 : i64, kbcir.bottleneck = 40 : i64, kbcir.compute_cost = 8 : i64, kbcir.context_compute = 4 : i64, kbcir.context_loop_order = "ijk", kbcir.context_mem = 20 : i64, kbcir.d_k = 4 : i64, kbcir.mem_cost = 40 : i64, kbcir.scores_compute = 4 : i64, kbcir.scores_loop_order = "ijk", kbcir.scores_mem = 20 : i64, kbcir.seq_len = 4 : i64, strideA = 4 : i64, strideB = 4 : i64, strideD = 4 : i64}
// CHECK: bcir.gem.block @atm_s_t1 {base = 2 : i64, count = 4 : i64, strideA = 4 : i64, strideB = 4 : i64, strideD = 4 : i64}
// CHECK: bcir.gem.block @atm_s_t2 {base = 8 : i64, count = 4 : i64, strideA = 4 : i64, strideB = 4 : i64, strideD = 4 : i64}
// CHECK: bcir.gem.block @atm_s_t3 {base = 10 : i64, count = 4 : i64, strideA = 4 : i64, strideB = 4 : i64, strideD = 4 : i64}
// Exactly 4 scores tiles -> no t4.
// CHECK-NOT: bcir.gem.block @atm_s_t4
// CHECK: bcir.gem.block @atm_sm_s0 {base = 0 : i64, count = 16 : i64, kbcir.lane_kind = "transcendental", kbcir.libm_edge = "expf", strideA = 1 : i64, strideB = 4 : i64, strideD = 1 : i64}
// CHECK: bcir.gem.block @atm_c_t0 {base = 0 : i64, count = 16 : i64, strideA = 4 : i64, strideB = 4 : i64, strideD = 4 : i64}
