// RUN: bcir-opt -bcir-gem-activation-cost -verify-diagnostics -split-input-file %s
//
// The -bcir-gem-activation-cost ANALYTIC PARITY gate (the dual-rail R13 check the gem.activation op
// verifier deliberately omits: the verifier checks only well-formedness -- a known kind, positive
// shape extents, a known + preserved dtype, the quarantine dtype rule, the softmax axis, width in
// [1, count], the R17 acc_bound, and bottleneck == max(compute_cost, mem_cost) -- it does NOT
// recompute the analytic cost from the shape/kind). Each section declares a roofline that is
// OP-LEVEL well-formed (so it parses past the verifier) but DISAGREES with the recomputed port of
// activation.py::cost_of under the pinned x86-avx512 constants (vector_width 16, fma true, mem_unit 1,
// mem_channels 4) -> the pass rejects it. The recomputed roofline is INFORMS-ONLY (it informs the
// plan's cost but never gates legality); these checks only enforce that the declared cost IS the
// oracle's priced roofline, not a legality verdict.

// the compute term must equal ceilDiv(count*op_weight, vector_width*fma): for relu over 64 elements,
// op_weight 1 -> ceilDiv(64*1, 32) = 2. The declared 9999 is op-level well-formed (bottleneck 9999 ==
// max(9999, 32)) but != the recomputed 2.
bcir.module @bad_compute {
  // expected-error @+1 {{bcir-gem-activation-cost: compute_cost 9999 != the recomputed roofline compute 2 (ceilDiv(count*op_weight, vector_width*fma))}}
  bcir.gem.activation @bad { kind = "relu", shape = array<i64: 64>, dtype = "f32",
                             axis_len = 0 : i64, width = 16 : i64,
                             compute_cost = 9999 : i64, mem_cost = 32 : i64, bottleneck = 9999 : i64,
                             quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the mem term must equal ceilDiv(count*streams, mem_unit*mem_channels) = ceilDiv(64*2, 4) = 32 for
// relu over 64 elements (streams = 2, the elementwise read+write). The declared 5000 is op-level
// well-formed (bottleneck 5000 == max(2, 5000)) but != the recomputed 32.
bcir.module @bad_mem {
  // expected-error @+1 {{bcir-gem-activation-cost: mem_cost 5000 != the recomputed roofline mem 32 (ceilDiv(count*streams, mem_unit*mem_channels))}}
  bcir.gem.activation @bad { kind = "relu", shape = array<i64: 64>, dtype = "f32",
                             axis_len = 0 : i64, width = 16 : i64,
                             compute_cost = 2 : i64, mem_cost = 5000 : i64, bottleneck = 5000 : i64,
                             quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the softmax extra stream (streams = 3, not 2) is the load-bearing kind-specific term: softmax over
// a 4x8 tensor (count 32, op_weight 8) prices mem = ceilDiv(32*3, 4) = 24, NOT the elementwise
// ceilDiv(32*2, 4) = 16. Declaring 16 (the elementwise mem) is op-level well-formed (bottleneck 16 ==
// max(8, 16)) but != the recomputed softmax mem 24 -- the pass catches the missing re-read stream.
bcir.module @bad_softmax_mem {
  // expected-error @+1 {{bcir-gem-activation-cost: mem_cost 16 != the recomputed roofline mem 24 (ceilDiv(count*streams, mem_unit*mem_channels))}}
  bcir.gem.activation @bad { kind = "softmax", shape = array<i64: 4, 8>, dtype = "f32",
                             axis_len = 8 : i64, width = 8 : i64,
                             compute_cost = 8 : i64, mem_cost = 16 : i64, bottleneck = 16 : i64,
                             quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the bottleneck must equal max(compute, mem). As with gem.matmul, a declared compute that matches
// recompute, a declared mem that matches recompute, and a declared bottleneck == their max is the
// op-level invariant the verifier ALSO enforces -- so there is NO independent bottleneck mismatch the
// op verifier admits while the analytic terms agree. The compute / mem sections above exercise the two
// independent roofline terms; the bottleneck is their pinned max (cross-checked redundantly).
// (Documented: the bottleneck check is the belt-and-braces dual of the op verifier's identical
// invariant.)
