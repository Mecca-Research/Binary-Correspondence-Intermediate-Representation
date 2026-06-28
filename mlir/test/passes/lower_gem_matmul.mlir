// RUN: bcir-opt -bcir-lower-gem-matmul %s | FileCheck %s
//
// The -bcir-lower-gem-matmul lowering: a gem.matmul plan record (the K_BCIR-chosen
// tiling/loop-order realization, B1) is lowered into its CONCRETE TILED REALIZATION
// -- one gem.block descriptor per tile of the tiled iteration space, emitted in the
// plan's loop_order. The matmul op is ERASED (a genuine lowering).
//
// Here: a 4x4x4 matmul tiled 2x2x2 in loop_order "ijk" -> ceil(4/2)^3 = 2*2*2 = 8
// tiles (the k-loop re-visits the same C tile, accumulating over k -- so consecutive
// pairs share a base). Per-tile descriptor:
//   base    = i0*N + j0          (row-major C origin; N = 4)
//   count   = min(tm,M-i0)*min(tn,N-j0)  (here a full 2x2 = 4 everywhere)
//   strideA = K = 4,  strideB = N = 4,  strideD = N = 4

bcir.module @m {
  bcir.gem.matmul @mm0 { m = 4 : i64, n = 4 : i64, k = 4 : i64,
                         tile_m = 2 : i64, tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk",
                         compute_cost = 100 : i64, mem_cost = 200 : i64, bottleneck = 200 : i64,
                         quant_bits = 0 : i32 }
}

// The plan record must be gone after lowering.
// CHECK-NOT: bcir.gem.matmul

// The 8 tile blocks, named @mm0_t0 .. @mm0_t7, in ijk enumeration order. The CHECK
// lines are sequential (FileCheck matches forward), so they pin both the per-tile
// descriptors AND their order. The first/last block fully exercise base/count/strides;
// the trailing CHECK-NOT @mm0_t8 + the presence of t7 bound the tile count at exactly 8.

// First tile (i=0,j=0): base 0, full 2x2 count 4, strides K=4 / N=4 / N=4.
// CHECK: bcir.gem.block @mm0_t0 {base = 0 : i64, count = 4 : i64, strideA = 4 : i64, strideB = 4 : i64, strideD = 4 : i64}
// Second tile is the k=1 revisit of the SAME C tile (same base 0).
// CHECK: bcir.gem.block @mm0_t1 {base = 0 : i64, count = 4 : i64, strideA = 4 : i64, strideB = 4 : i64, strideD = 4 : i64}
// Third tile (i=0,j=1): base = 0*4 + 2 = 2.
// CHECK: bcir.gem.block @mm0_t2 {base = 2 : i64, count = 4 : i64, strideA = 4 : i64, strideB = 4 : i64, strideD = 4 : i64}
// Fifth tile (i=1,j=0): base = 2*4 + 0 = 8.
// CHECK: bcir.gem.block @mm0_t4 {base = 8 : i64, count = 4 : i64, strideA = 4 : i64, strideB = 4 : i64, strideD = 4 : i64}
// Last tile (i=1,j=1): base = 2*4 + 2 = 10.  (t5/t6 are the k-revisits in between.)
// CHECK: bcir.gem.block @mm0_t7 {base = 10 : i64, count = 4 : i64, strideA = 4 : i64, strideB = 4 : i64, strideD = 4 : i64}

// No 9th tile -> exactly 8 blocks were emitted.
// CHECK-NOT: bcir.gem.block @mm0_t8
