// RUN: bcir-opt -bcir-lower-gem-conv %s | FileCheck %s
//
// The -bcir-lower-gem-conv lowering: a gem.conv plan record (the K_BCIR-chosen direct|im2col
// realization of a 2-D single-group convolution, G7) is lowered into its CONCRETE REALIZATION
// -- one gem.block descriptor per tile of the EQUIVALENT im2col gemm's tiled iteration space,
// emitted in the plan's loop_order. A conv IS a STRUCTURED MATMUL: the im2col reshaping turns
// it into the gemm C[out_pix x C_out] = patches[out_pix x (C_in*Kh*Kw)] @ W[(C_in*Kh*Kw) x C_out],
// so this lowers through the SAME tiled-gem.block emission -bcir-lower-gem-matmul uses on that
// gemm (NO bespoke conv loop). The conv op is ERASED (a genuine lowering).
//
// This is the DUAL-RAIL PARITY gate (the law reproduces the oracle's bcir/kbcir/conv.py plan):
// the strategy/compute/mem/bottleneck on each op equal plan_conv/cost_of for the same conv,
// priced THROUGH THE EXISTING matmul roofline (the im2col gemm's cost; the direct strategy is
// the UNTILED gemm). The parity is re-asserted host-agnostically in bcir/tests/test_conv_law_
// parity.py, which drives bcir-opt with the oracle target pinned to x86-avx512. The pass
// recomputes the host-independent invariant (bottleneck == max(compute, mem)) and annotates the
// realization decision (strategy + the priced roofline + the R17 acc_bound) onto the first tile.
//
// Per-tile descriptor (the im2col gemm strides): base = i0*gemm_n + j0 (row-major gemm origin),
// count = min(tm, M-i0)*min(tn, N-j0), strideA = gemm_k (P row stride), strideB = gemm_n (Wm row
// stride), strideD = gemm_n (output row stride).

bcir.module @convs {
  // (1) im2col: a (C_in=2, 5x5) input, 3 filters, 3x3 kernel, stride 1, pad 0 -> out 3x3. The
  // equivalent im2col gemm is 9x3x18 (M=out_pix=9, N=out_c=3, K=taps=2*3*3=18). DELIBERATELY
  // tiled 4x2x18 -> ceil(9/4)=3 i, ceil(3/2)=2 j, ceil(18/18)=1 k -> 6 tiles. Dense -> acc_bound 0.
  bcir.gem.conv @cv0 { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64,
                       kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32",
                       out_h = 3 : i64, out_w = 3 : i64,
                       gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 18 : i64,
                       strategy = "im2col", tile_m = 4 : i64, tile_n = 2 : i64, tile_k = 18 : i64,
                       loop_order = "ijk", compute_cost = 16 : i64, mem_cost = 68 : i64,
                       bottleneck = 68 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }

  // (2) direct: a (C_in=1, 4x4) input, 2 filters, 2x2 kernel, stride 1, pad 0 -> out 3x3. The
  // im2col gemm is 9x2x4; the DIRECT strategy is the UNTILED gemm (tile = the whole 9x2x4) -> ONE
  // whole-gemm block, count = 9*2 = 18. QUANTIZED (quant_bits = 8) -> the R17 Q8 bridge: acc_bound 1.
  bcir.gem.conv @cvd { in_c = 1 : i64, in_h = 4 : i64, in_w = 4 : i64, out_c = 2 : i64,
                       kh = 2 : i64, kw = 2 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32",
                       out_h = 3 : i64, out_w = 3 : i64,
                       gemm_m = 9 : i64, gemm_n = 2 : i64, gemm_k = 4 : i64,
                       strategy = "direct", tile_m = 9 : i64, tile_n = 2 : i64, tile_k = 4 : i64,
                       loop_order = "ijk", compute_cost = 5 : i64, mem_cost = 20 : i64,
                       bottleneck = 20 : i64, quant_bits = 8 : i32, acc_bound = 1 : i64 }
}

// Both plan records must be gone after lowering.
// CHECK-NOT: bcir.gem.conv

// (1) im2col cv0: 6 tiles @cv0_t0..t5 in ijk order. t0 carries the recomputed plan + the dual-rail
// record (strategy "im2col", the priced roofline, dense acc_bound 0). strideA = gemm_k 18, strideB =
// strideD = gemm_n 3. t0 is the full 4x2 tile (count 8); the ragged boundary tiles clamp.
// CHECK: bcir.gem.block @cv0_t0 {base = 0 : i64, count = 8 : i64, kbcir.acc_bound = 0 : i64, kbcir.bottleneck = 68 : i64, kbcir.compute_cost = 16 : i64, kbcir.gemm_k = 18 : i64, kbcir.gemm_m = 9 : i64, kbcir.gemm_n = 3 : i64, kbcir.loop_order = "ijk", kbcir.mem_cost = 68 : i64, kbcir.strategy = "im2col", strideA = 18 : i64, strideB = 3 : i64, strideD = 3 : i64}
// t1 (i=0, j=1): base = 0*3 + 2 = 2, count = 4*1 = 4 (the ragged last column n=3).
// CHECK: bcir.gem.block @cv0_t1 {base = 2 : i64, count = 4 : i64, strideA = 18 : i64, strideB = 3 : i64, strideD = 3 : i64}
// t4 (i=2, j=0): base = 8*3 + 0 = 24, count = 1*2 = 2 (the ragged last row m=9 -> 1 row).
// CHECK: bcir.gem.block @cv0_t4 {base = 24 : i64, count = 2 : i64, strideA = 18 : i64, strideB = 3 : i64, strideD = 3 : i64}
// t5 (i=2, j=1): base = 8*3 + 2 = 26, count = 1*1 = 1.
// CHECK: bcir.gem.block @cv0_t5 {base = 26 : i64, count = 1 : i64, strideA = 18 : i64, strideB = 3 : i64, strideD = 3 : i64}
// Exactly 6 tiles -> no t6.
// CHECK-NOT: bcir.gem.block @cv0_t6

// (2) direct cvd: ONE whole-gemm block (the untiled stream), count = 9*2 = 18, strategy "direct",
// QUANTIZED acc_bound 1. strideA = gemm_k 4, strideB = strideD = gemm_n 2.
// CHECK: bcir.gem.block @cvd_t0 {base = 0 : i64, count = 18 : i64, kbcir.acc_bound = 1 : i64, kbcir.bottleneck = 20 : i64, kbcir.compute_cost = 5 : i64, kbcir.gemm_k = 4 : i64, kbcir.gemm_m = 9 : i64, kbcir.gemm_n = 2 : i64, kbcir.loop_order = "ijk", kbcir.mem_cost = 20 : i64, kbcir.strategy = "direct", strideA = 4 : i64, strideB = 2 : i64, strideD = 2 : i64}
// CHECK-NOT: bcir.gem.block @cvd_t1
