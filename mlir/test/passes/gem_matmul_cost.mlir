// RUN: bcir-opt -bcir-gem-matmul-cost %s | FileCheck %s
//
// The -bcir-gem-matmul-cost cost producer: a gem.matmul plan op (the B1 K_BCIR-chosen tiling /
// loop-order realization) has its ROOFLINE COST RECOMPUTED (port of bcir/kbcir/matmul.py::cost_of)
// and annotated as kbcir.* attrs. Unlike -bcir-lower-gem-matmul (which ERASES the matmul into its
// tiles) this is a COST PRODUCER, not a lowering -- the op is NOT erased.
//
// This is the DUAL-RAIL PARITY gate (the law reproduces the oracle's matmul.py roofline). The cost
// is priced through the analytic roofline terms cost_of recomputes:
//   compute   = ceilDiv(M*N*K, vector_width*(2 if fma else 1))
//   a_reads   = M*K*ceil(N/tile_n)   b_reads = K*N*ceil(M/tile_m)   c_traffic = M*N*(ceil(K/tile_k)+1)
//   mem       = ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels)
//   bottleneck= max(compute, mem)
//   working_set = tile_m*tile_k + tile_k*tile_n + tile_m*tile_n <= (512*cacheline)/elem_bytes (fits_cache)
// Re-asserted host-agnostically in bcir/tests/test_matmul_law_parity.py (the oracle target pinned to
// x86-avx512: vector_width 16, fma true, mem_unit 1, mem_channels 4, cacheline 64, elem_bytes 4 ->
// thr 32, bw 4, cache budget 8192 elems).
//
// INFORMS-ONLY: the recomputed roofline informs the plan's cost but NEVER gates legality
// (kbcir.informs_only); the gem.matmul op is NOT erased and never a legality verdict.

bcir.module @matmuls {
  // (1) 64x64x64 untiled (tile 64x64x64): compute = ceilDiv(262144, 32) = 8192; a_reads = 64*64*ceil(64/64)
  // = 4096, b_reads = 4096, c_traffic = 64*64*(ceil(64/64)+1) = 8192 -> mem = ceilDiv(16384, 4) = 4096;
  // bottleneck = max(8192, 4096) = 8192. working_set = 64*64*3 = 12288 > 8192 -> does NOT fit the budget.
  bcir.gem.matmul @mm_untiled { m = 64 : i64, n = 64 : i64, k = 64 : i64,
                                tile_m = 64 : i64, tile_n = 64 : i64, tile_k = 64 : i64,
                                loop_order = "ijk",
                                compute_cost = 8192 : i64, mem_cost = 4096 : i64, bottleneck = 8192 : i64 }

  // (2) 64x64x64 tiled 32x32x32: compute = 8192 (shape-only); a_reads = 64*64*ceil(64/32) = 8192, b_reads =
  // 8192, c_traffic = 64*64*(ceil(64/32)+1) = 64*64*3 = 12288 -> mem = ceilDiv(28672, 4) = 7168; bottleneck
  // = max(8192, 7168) = 8192. working_set = 32*32*3 = 3072 <= 8192 -> FITS the modeled cache budget.
  bcir.gem.matmul @mm_tiled { m = 64 : i64, n = 64 : i64, k = 64 : i64,
                              tile_m = 32 : i64, tile_n = 32 : i64, tile_k = 32 : i64,
                              loop_order = "ijk",
                              compute_cost = 8192 : i64, mem_cost = 7168 : i64, bottleneck = 8192 : i64 }
}

// The carrier ops are NOT erased (a cost producer, not a lowering).
// (1) untiled: the larger-traffic, non-cache-fitting tiling; compute-bound bottleneck 8192.
// CHECK: bcir.gem.matmul @mm_untiled
// CHECK-SAME: kbcir.bottleneck = 8192 : i64
// CHECK-SAME: kbcir.compute_cost = 8192 : i64
// CHECK-SAME: kbcir.fits_cache = false
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.mem_cost = 4096 : i64

// (2) tiled 32x32x32: less reuse (more traffic) but the working set fits the cache budget.
// CHECK: bcir.gem.matmul @mm_tiled
// CHECK-SAME: kbcir.bottleneck = 8192 : i64
// CHECK-SAME: kbcir.compute_cost = 8192 : i64
// CHECK-SAME: kbcir.fits_cache = true
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.mem_cost = 7168 : i64
