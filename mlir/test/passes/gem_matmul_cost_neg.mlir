// RUN: bcir-opt -bcir-gem-matmul-cost -verify-diagnostics -split-input-file %s
//
// The -bcir-gem-matmul-cost ANALYTIC PARITY gate (the dual-rail R13 check the gem.matmul op verifier
// deliberately omits: the verifier checks only well-formedness -- positive dims, each tile in [1, dim],
// a known loop order, bottleneck == max(compute_cost, mem_cost) -- it does NOT recompute the analytic
// cost from the shape/tile). Each section declares a roofline that is OP-LEVEL well-formed (so it parses
// past the verifier) but DISAGREES with the recomputed port of matmul.py::cost_of under the pinned
// x86-avx512 constants (vector_width 16, fma true, mem_unit 1, mem_channels 4) -> the pass rejects it.
// The recomputed roofline is INFORMS-ONLY (it informs the plan's cost but never gates legality); these
// checks only enforce that the declared cost IS the oracle's priced roofline, not a legality verdict.

// the compute term must equal ceilDiv(M*N*K, vector_width*fma): 64*64*64 / 32 = 8192. The declared
// 9999 is op-level well-formed (bottleneck 9999 == max(9999, 4096)) but != the recomputed 8192.
bcir.module @bad_compute {
  // expected-error @+1 {{bcir-gem-matmul-cost: compute_cost 9999 != the recomputed roofline compute 8192 (ceilDiv(M*N*K, vector_width*fma))}}
  bcir.gem.matmul @bad { m = 64 : i64, n = 64 : i64, k = 64 : i64,
                         tile_m = 64 : i64, tile_n = 64 : i64, tile_k = 64 : i64,
                         loop_order = "ijk",
                         compute_cost = 9999 : i64, mem_cost = 4096 : i64, bottleneck = 9999 : i64 }
}

// -----

// the mem term must equal ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels) = 4096 for the
// 64x64x64 untiled tiling. The declared 5000 is op-level well-formed (bottleneck 8192 == max(8192, 5000))
// but != the recomputed 4096.
bcir.module @bad_mem {
  // expected-error @+1 {{bcir-gem-matmul-cost: mem_cost 5000 != the recomputed roofline mem 4096 (ceilDiv(a_reads + b_reads + c_traffic, mem_unit*mem_channels))}}
  bcir.gem.matmul @bad { m = 64 : i64, n = 64 : i64, k = 64 : i64,
                         tile_m = 64 : i64, tile_n = 64 : i64, tile_k = 64 : i64,
                         loop_order = "ijk",
                         compute_cost = 8192 : i64, mem_cost = 5000 : i64, bottleneck = 8192 : i64 }
}

// -----

// the bottleneck must equal max(compute, mem) = max(8192, 4096) = 8192. Declaring a SMALLER mem (1024)
// with a matching bottleneck (8192 == max(8192, 1024) -- op-level well-formed) flags the mem term first;
// to reach the bottleneck check we keep compute/mem at their recomputed values and declare a too-large
// bottleneck -- but bottleneck != max(8192, 4096) fails the OP verifier, so the bottleneck-mismatch path
// is reached only when the op-level invariant (bottleneck == max) holds yet disagrees with the analytic
// max. That happens when a declared compute matches recompute, a declared mem matches recompute, and the
// declared bottleneck equals their max -- i.e. there is NO independent bottleneck mismatch the verifier
// admits. The compute / mem sections above exercise the two independent roofline terms; the bottleneck is
// their pinned max (cross-checked redundantly). (Documented: the bottleneck check is the belt-and-braces
// dual of the op verifier's identical invariant.)
