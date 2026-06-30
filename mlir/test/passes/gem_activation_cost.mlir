// RUN: bcir-opt -bcir-gem-activation-cost %s | FileCheck %s
//
// The -bcir-gem-activation-cost cost producer: a gem.activation plan op (the G1 K_BCIR-chosen
// lane-width realization of an elementwise relu/sigmoid/tanh/gelu/softmax) has its ROOFLINE COST
// RECOMPUTED (port of bcir/kbcir/activation.py::cost_of) and annotated as kbcir.* attrs. Unlike
// -bcir-lower-gem-activation (which ERASES the activation into its lane-width stripes) this is a
// COST PRODUCER, not a lowering -- the op is NOT erased.
//
// This is the DUAL-RAIL PARITY gate (the law reproduces the oracle's activation.py roofline). The
// cost is priced through the analytic roofline terms cost_of recomputes (the SAME shape matmul.py
// uses, specialized to an elementwise op):
//   op_weight = {relu:1, sigmoid:8, tanh:8, gelu:12, softmax:8}[kind]
//   compute   = ceilDiv(count*op_weight, vector_width*(2 if fma else 1))
//   streams   = 3 if softmax else 2   (softmax re-reads the row for reduce-max + normalize)
//   mem       = ceilDiv(count*streams, mem_unit*mem_channels)
//   bottleneck= max(compute, mem)
// (cost_of does NOT depend on the chosen lane width -- the roofline is shape/kind-only.) Re-asserted
// host-agnostically in bcir/tests/test_activation_law_parity.py (the oracle target pinned to
// x86-avx512: vector_width 16, fma true, mem_unit 1, mem_channels 4 -> thr 32, bw 4).
//
// INFORMS-ONLY: the recomputed roofline informs the plan's cost but NEVER gates legality
// (kbcir.informs_only); the gem.activation op is NOT erased and never a legality verdict.

bcir.module @activations {
  // (1) relu over 64 elements: op_weight = 1, count = 64; compute = ceilDiv(64*1, 32) = 2;
  // streams = 2 (1 read + 1 write), mem = ceilDiv(64*2, 4) = 32; bottleneck = max(2, 32) = 32.
  // relu is the EXACT, 0-ULP integer/Q-fixed-clean activation (memory-bound here).
  bcir.gem.activation @relu_dense { kind = "relu", shape = array<i64: 64>, dtype = "f32",
                                    axis_len = 0 : i64, width = 16 : i64,
                                    compute_cost = 2 : i64, mem_cost = 32 : i64, bottleneck = 32 : i64,
                                    quant_bits = 0 : i32, acc_bound = 0 : i64 }

  // (2) gelu over 64 elements: op_weight = 12 (the heaviest -- a cube + a tanh), count = 64;
  // compute = ceilDiv(64*12, 32) = ceilDiv(768, 32) = 24; streams = 2, mem = ceilDiv(128, 4) = 32;
  // bottleneck = max(24, 32) = 32. A transcendental (the tanhf edge) -- the compute term lifts but
  // it is still memory-bound at this size.
  bcir.gem.activation @gelu_dense { kind = "gelu", shape = array<i64: 64>, dtype = "f32",
                                    axis_len = 0 : i64, width = 16 : i64,
                                    compute_cost = 24 : i64, mem_cost = 32 : i64, bottleneck = 32 : i64,
                                    quant_bits = 0 : i32, acc_bound = 0 : i64 }

  // (3) softmax over a 4x8 tensor (last-axis length 8): op_weight = 8 (the exp), count = 32;
  // compute = ceilDiv(32*8, 32) = 8; streams = 3 (softmax re-reads the row: reduce-max + exp/sum +
  // normalize), mem = ceilDiv(32*3, 4) = ceilDiv(96, 4) = 24; bottleneck = max(8, 24) = 24. The
  // extra stream is the softmax-only re-read the elementwise four do not pay.
  bcir.gem.activation @softmax_row { kind = "softmax", shape = array<i64: 4, 8>, dtype = "f32",
                                     axis_len = 8 : i64, width = 8 : i64,
                                     compute_cost = 8 : i64, mem_cost = 24 : i64, bottleneck = 24 : i64,
                                     quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// The carrier ops are NOT erased (a cost producer, not a lowering).
// (1) relu: the exact, memory-bound elementwise activation; bottleneck 32.
// CHECK: bcir.gem.activation @relu_dense
// CHECK-SAME: kbcir.bottleneck = 32 : i64
// CHECK-SAME: kbcir.compute_cost = 2 : i64
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.mem_cost = 32 : i64

// (2) gelu: the heaviest op_weight (a transcendental cube + tanh); compute lifts to 24, still mem-bound.
// CHECK: bcir.gem.activation @gelu_dense
// CHECK-SAME: kbcir.bottleneck = 32 : i64
// CHECK-SAME: kbcir.compute_cost = 24 : i64
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.mem_cost = 32 : i64

// (3) softmax: the last-axis reduce -- 3 streams, the softmax-only re-read; bottleneck 24.
// CHECK: bcir.gem.activation @softmax_row
// CHECK-SAME: kbcir.bottleneck = 24 : i64
// CHECK-SAME: kbcir.compute_cost = 8 : i64
// CHECK-SAME: kbcir.informs_only = true
// CHECK-SAME: kbcir.mem_cost = 24 : i64
