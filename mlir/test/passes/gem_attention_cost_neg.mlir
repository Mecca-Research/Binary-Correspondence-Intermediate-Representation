// RUN: bcir-opt -bcir-gem-attention-cost -verify-diagnostics -split-input-file %s
//
// The -bcir-gem-attention-cost ANALYTIC PARITY gate (the dual-rail R13 check the gem.attention op
// verifier deliberately omits: the verifier checks only well-formedness -- positive seq_len/d_k, a
// known + preserved dtype that is f32 (the softmax quarantine rule), the DERIVED two gemm dims
// internally consistent with (seq_len, d_k), each gemm tile in [1, gemm dim] with a known loop order,
// all cost terms non-negative, the SUMMED compute/mem == the two declared per-gemm costs, bottleneck
// == max(compute, mem), and the R17 acc_bound -- it does NOT recompute the analytic per-gemm cost from
// the gemm dims/tile). Each section declares a roofline that is OP-LEVEL well-formed (so it parses past
// the verifier: its declared sums are consistent with its declared per-gemm terms and its bottleneck is
// their max) but DISAGREES with the recomputed port of attention.py::cost_of (== matmul.cost_of on each
// gemm, summed) under the pinned x86-avx512 constants (vector_width 16, fma true, mem_unit 1,
// mem_channels 4) -> the pass rejects it. The recomputed roofline is INFORMS-ONLY (it informs the plan's
// cost but never gates legality); these checks only enforce that the declared cost IS the oracle's
// priced roofline, not a legality verdict.

// the scores Q@K^T per-gemm compute must equal ceilDiv(scores_m*scores_n*scores_k, vector_width*fma):
// for the small dense attention (scores gemm 8x8x8) that is ceilDiv(8*8*8, 32) = ceilDiv(512, 32) = 16.
// The declared 99 (with compute_cost = 99+16 = 115 and bottleneck = max(115, 128) = 128 kept op-level
// consistent) parses past the verifier but != the recomputed 16.
bcir.module @bad_scores_compute {
  // expected-error @+1 {{bcir-gem-attention-cost: scores_compute 99 != the recomputed scores Q@K^T roofline compute 16 (ceilDiv(scores_m*scores_n*scores_k, vector_width*fma))}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 99 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 115 : i64, mem_cost = 128 : i64, bottleneck = 128 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the scores Q@K^T per-gemm mem must equal ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels):
// for the scores gemm 8x8x8 (whole-tiled) a_reads = 8*8*ceil(8/8) = 64, b_reads = 64, c_traffic =
// 8*8*(ceil(8/8)+1) = 128, mem = ceilDiv(256, 4) = 64. The declared 999 (with mem_cost = 999+64 = 1063,
// bottleneck = max(32, 1063) = 1063 kept op-level consistent) parses past the verifier but != the 64.
bcir.module @bad_scores_mem {
  // expected-error @+1 {{bcir-gem-attention-cost: scores_mem 999 != the recomputed scores Q@K^T roofline mem 64 (ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels))}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 999 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 32 : i64, mem_cost = 1063 : i64, bottleneck = 1063 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the context A@V per-gemm compute must equal ceilDiv(context_m*context_n*context_k, vector_width*fma):
// for the context gemm 8x8x8 that is ceilDiv(512, 32) = 16. The declared 77 (with compute_cost = 16+77
// = 93, bottleneck = max(93, 128) = 128 kept op-level consistent) parses past the verifier but != 16.
bcir.module @bad_context_compute {
  // expected-error @+1 {{bcir-gem-attention-cost: context_compute 77 != the recomputed context A@V roofline compute 16 (ceilDiv(context_m*context_n*context_k, vector_width*fma))}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 77 : i64, context_mem = 64 : i64,
    compute_cost = 93 : i64, mem_cost = 128 : i64, bottleneck = 128 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the context A@V per-gemm mem must equal ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels):
// for the context gemm 8x8x8 (whole-tiled) mem = 64 (same shape as scores here). The declared 500 (with
// mem_cost = 64+500 = 564, bottleneck = max(32, 564) = 564 kept op-level consistent) parses past the
// verifier but != the recomputed 64.
bcir.module @bad_context_mem {
  // expected-error @+1 {{bcir-gem-attention-cost: context_mem 500 != the recomputed context A@V roofline mem 64 (ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels))}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 500 : i64,
    compute_cost = 32 : i64, mem_cost = 564 : i64, bottleneck = 564 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the TILE is the load-bearing per-gemm knob: tiling changes ONLY the mem traffic (the MAC count --
// compute -- is fixed). The longer attention's context A@V gemm is 16x8x16; declaring a SMALLER tile_m
// (8 instead of the whole 16) re-reads the weight matrix per M-tile: b_reads = 16*8*ceil(16/8) = 256
// (vs 128 untiled), so mem = ceilDiv(256 + 256 + 256, 4) = ceilDiv(768, 4) = 192 -- NOT the untiled 160.
// Declaring the untiled-traffic mem 160 against the declared blocked tile_m 8 is op-level well-formed
// (context_mem 160; mem_cost = 192+160 = 352; bottleneck = max(128, 352) = 352) but != the recomputed
// blocked-tile mem 192 -- the pass catches the tile the declared context_mem priced through being
// inconsistent with the declared context_tile_m 8.
bcir.module @bad_context_tiled_mem {
  // expected-error @+1 {{bcir-gem-attention-cost: context_mem 160 != the recomputed context A@V roofline mem 192 (ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels))}}
  bcir.gem.attention @bad { seq_len = 16 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 16 : i64, scores_n = 16 : i64, scores_k = 8 : i64,
    context_m = 16 : i64, context_n = 8 : i64, context_k = 16 : i64,
    scores_tile_m = 16 : i64, scores_tile_n = 16 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 16 : i64, context_loop_order = "ijk",
    scores_compute = 64 : i64, scores_mem = 192 : i64, context_compute = 64 : i64, context_mem = 160 : i64,
    compute_cost = 128 : i64, mem_cost = 352 : i64, bottleneck = 352 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the SUMMED compute_cost must equal scores_compute + context_compute (attention is a COMPOSITION of two
// gem.matmuls; NO bespoke term). Here both per-gemm terms match recompute (scores/context compute = 16,
// mem = 64) but the declared SUM is wrong. The op verifier ALSO enforces compute_cost == the per-gemm
// sum, so a declared sum that disagrees with the (recompute-matching) per-gemm terms is REJECTED BY THE
// OP VERIFIER FIRST (before the cost pass runs) -- documenting that there is no summed-cost mismatch the
// op verifier admits while every per-gemm term agrees with recompute.
bcir.module @bad_sum_rejected_by_verifier {
  // expected-error @+1 {{attention: compute_cost must equal scores_compute + context_compute = 32 (the SUM of the two priced matmuls; got 999)}}
  bcir.gem.attention @bad { seq_len = 8 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 8 : i64, scores_n = 8 : i64, scores_k = 8 : i64,
    context_m = 8 : i64, context_n = 8 : i64, context_k = 8 : i64,
    scores_tile_m = 8 : i64, scores_tile_n = 8 : i64, scores_tile_k = 8 : i64, scores_loop_order = "ijk",
    context_tile_m = 8 : i64, context_tile_n = 8 : i64, context_tile_k = 8 : i64, context_loop_order = "ijk",
    scores_compute = 16 : i64, scores_mem = 64 : i64, context_compute = 16 : i64, context_mem = 64 : i64,
    compute_cost = 999 : i64, mem_cost = 128 : i64, bottleneck = 999 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}
