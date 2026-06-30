// RUN: bcir-opt -bcir-gem-conv-cost -verify-diagnostics -split-input-file %s
//
// The -bcir-gem-conv-cost ANALYTIC PARITY gate (the dual-rail R13 check the gem.conv op verifier
// deliberately omits: the verifier checks only well-formedness -- positive extents, the kernel fits
// the padded input, a known + preserved dtype, the DERIVED out_h/out_w + im2col gemm dims internally
// consistent, the strategy is direct|im2col, each inner tile in [1, gemm dim] with a known loop order,
// direct -> the untiled whole gemm, bottleneck == max(compute_cost, mem_cost), and the R17 acc_bound --
// it does NOT recompute the analytic cost from the gemm dims/tile). Each section declares a roofline
// that is OP-LEVEL well-formed (so it parses past the verifier) but DISAGREES with the recomputed port
// of conv.py::cost_of (== matmul.cost_of on the im2col gemm dims) under the pinned x86-avx512 constants
// (vector_width 16, fma true, mem_unit 1, mem_channels 4) -> the pass rejects it. The recomputed
// roofline is INFORMS-ONLY (it informs the plan's cost but never gates legality); these checks only
// enforce that the declared cost IS the oracle's priced roofline, not a legality verdict.

// the compute term must equal ceilDiv(gemm_m*gemm_n*gemm_k, vector_width*fma): for the small conv
// (im2col gemm 9x3x18) that is ceilDiv(9*3*18, 32) = ceilDiv(486, 32) = 16. The declared 9999 is
// op-level well-formed (bottleneck 9999 == max(9999, 68)) but != the recomputed 16.
bcir.module @bad_compute {
  // expected-error @+1 {{bcir-gem-conv-cost: compute_cost 9999 != the recomputed roofline compute 16 (ceilDiv(gemm_m*gemm_n*gemm_k, vector_width*fma))}}
  bcir.gem.conv @bad { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64,
                       kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64,
                       dtype = "f32", out_h = 3 : i64, out_w = 3 : i64,
                       gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 18 : i64,
                       strategy = "im2col", tile_m = 9 : i64, tile_n = 3 : i64, tile_k = 18 : i64,
                       loop_order = "ijk", compute_cost = 9999 : i64, mem_cost = 68 : i64,
                       bottleneck = 9999 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the mem term must equal ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels): for the small
// conv (gemm 9x3x18, untiled) a_reads = 9*18*ceil(3/3) = 162, b_reads = 18*3*ceil(9/9) = 54, c_traffic
// = 9*3*(ceil(18/18)+1) = 54, mem = ceilDiv(270, 4) = 68. The declared 5000 is op-level well-formed
// (bottleneck 5000 == max(16, 5000)) but != the recomputed 68.
bcir.module @bad_mem {
  // expected-error @+1 {{bcir-gem-conv-cost: mem_cost 5000 != the recomputed roofline mem 68 (ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels))}}
  bcir.gem.conv @bad { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64,
                       kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64,
                       dtype = "f32", out_h = 3 : i64, out_w = 3 : i64,
                       gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 18 : i64,
                       strategy = "im2col", tile_m = 9 : i64, tile_n = 3 : i64, tile_k = 18 : i64,
                       loop_order = "ijk", compute_cost = 16 : i64, mem_cost = 5000 : i64,
                       bottleneck = 5000 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the TILE is the load-bearing structured-matmul knob: tiling changes ONLY the mem traffic (the MAC
// count -- compute -- is fixed). The larger conv (im2col gemm 256x16x72) blocked at tile (64,16,72)
// prices mem = ceilDiv(18432 + 4608 + 8192, 4) = ceilDiv(31232, 4) = 7808, NOT the untiled-whole-gemm
// mem. Declaring the WRONG mem 6944 (the direct/whole-tile traffic) is op-level well-formed (bottleneck
// 9216 == max(9216, 6944)) but != the recomputed blocked-tile mem 7808 -- the pass catches the tile
// the declared mem priced through being inconsistent with the declared tile_m 64.
bcir.module @bad_tiled_mem {
  // expected-error @+1 {{bcir-gem-conv-cost: mem_cost 6944 != the recomputed roofline mem 7808 (ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels))}}
  bcir.gem.conv @bad { in_c = 8 : i64, in_h = 16 : i64, in_w = 16 : i64, out_c = 16 : i64,
                       kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 1 : i64,
                       dtype = "f32", out_h = 16 : i64, out_w = 16 : i64,
                       gemm_m = 256 : i64, gemm_n = 16 : i64, gemm_k = 72 : i64,
                       strategy = "im2col", tile_m = 64 : i64, tile_n = 16 : i64, tile_k = 72 : i64,
                       loop_order = "ijk", compute_cost = 9216 : i64, mem_cost = 6944 : i64,
                       bottleneck = 9216 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the bottleneck must equal max(compute, mem). As with gem.matmul / gem.activation, a declared compute
// that matches recompute, a declared mem that matches recompute, and a declared bottleneck == their max
// is the op-level invariant the verifier ALSO enforces -- so there is NO independent bottleneck mismatch
// the op verifier admits while the analytic terms agree. The compute / mem sections above exercise the
// two independent roofline terms; the bottleneck is their pinned max (cross-checked redundantly).
// (Documented: the bottleneck check is the belt-and-braces dual of the op verifier's identical invariant.)
