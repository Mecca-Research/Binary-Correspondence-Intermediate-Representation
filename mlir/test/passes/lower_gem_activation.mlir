// RUN: bcir-opt -bcir-lower-gem-activation %s | FileCheck %s
//
// The -bcir-lower-gem-activation lowering: a gem.activation plan record (the K_BCIR-chosen
// lane-width realization of the elementwise tensor-op family relu/sigmoid/tanh/gelu/softmax,
// G1) is lowered into its CONCRETE REALIZATION -- one gem.block descriptor per lane-width
// stripe of the elementwise iteration space, emitted in flat row-major order. The activation
// op is ERASED (a genuine lowering, the activation analog of -bcir-lower-gem-matmul).
//
// This is the DUAL-RAIL PARITY gate (the law reproduces the oracle's bcir/kbcir/activation.py
// plan): the width/compute/mem/bottleneck on each op equal plan_activation/cost_of for the same
// op (pinned for the x86-avx512 lane set, width 16 -- the parity is re-asserted host-agnostically
// in bcir/tests/test_activation_law_parity.py, which drives bcir-opt with the oracle target
// pinned). The pass recomputes the host-independent invariants (bottleneck == max(compute, mem);
// the per-kind op_weight/streams; the R17 acc_bound) and annotates the realization decision +
// the QUARANTINE verdict (relu exact / no libm edge; the transcendentals route expf/tanhf through
// the trusted c.call.libm: edge) onto the first emitted stripe.
//
// Per-stripe descriptor: base = idx*width, count = min(width, count-base), strideA = 1 (unit read),
// strideB = softmax row length else 0, strideD = 1 (unit write).

bcir.module @acts {
  // (1) relu over a 64-element f32 tensor, width 16 -> ceil(64/16) = 4 stripes. relu is the EXACT,
  // 0-ULP integer/Q-fixed-clean activation: lane_kind "exact", NO libm edge, op_weight 1, acc_bound 0.
  bcir.gem.activation @relu0 { kind = "relu", shape = array<i64: 64>, dtype = "f32",
                              axis_len = 0 : i64, width = 16 : i64,
                              compute_cost = 2 : i64, mem_cost = 32 : i64, bottleneck = 32 : i64,
                              quant_bits = 0 : i32, acc_bound = 0 : i64 }

  // (2) relu over an INTEGER tensor -- the exact activation MAY be i32 (the quarantine clean side).
  bcir.gem.activation @relu_i32 { kind = "relu", shape = array<i64: 32>, dtype = "i32",
                                 axis_len = 0 : i64, width = 16 : i64,
                                 compute_cost = 2 : i64, mem_cost = 16 : i64, bottleneck = 16 : i64,
                                 quant_bits = 0 : i32, acc_bound = 0 : i64 }

  // (3) gelu over a 64-element f32 tensor -- a TRANSCENDENTAL: lane_kind "transcendental", routes the
  // tanhf libm edge, op_weight 12 (the heaviest: a cube + a tanh). Dense, so acc_bound 0.
  bcir.gem.activation @gelu0 { kind = "gelu", shape = array<i64: 64>, dtype = "f32",
                              axis_len = 0 : i64, width = 16 : i64,
                              compute_cost = 24 : i64, mem_cost = 32 : i64, bottleneck = 32 : i64,
                              quant_bits = 0 : i32, acc_bound = 0 : i64 }

  // (4) softmax over a 4x8 f32 tensor, axis_len 8 -> count 32, width 16 -> 2 stripes. A transcendental
  // (expf edge), streams = 3 (its reduce re-streams the row); strideB = axis_len 8. QUANTIZED
  // (quant_bits = 8), so the R17 Q8-bridge bound rides the accuracy axis: acc_bound 1 ULP.
  bcir.gem.activation @softmax0 { kind = "softmax", shape = array<i64: 4, 8>, dtype = "f32",
                                 axis_len = 8 : i64, width = 16 : i64,
                                 compute_cost = 8 : i64, mem_cost = 24 : i64, bottleneck = 24 : i64,
                                 quant_bits = 8 : i32, acc_bound = 1 : i64 }
}

// All four plan records must be gone after lowering.
// CHECK-NOT: bcir.gem.activation

// (1) relu: 4 stripes @relu0_s0..s3. s0 carries the recomputed plan + the EXACT quarantine verdict
// (lane_kind "exact", empty libm_edge, op_weight 1, acc_bound 0). strideA/strideD = 1, strideB = 0.
// CHECK: bcir.gem.block @relu0_s0 {base = 0 : i64, count = 16 : i64, kbcir.acc_bound = 0 : i64, kbcir.bottleneck = 32 : i64, kbcir.compute_cost = 2 : i64, kbcir.lane_kind = "exact", kbcir.libm_edge = "", kbcir.mem_cost = 32 : i64, kbcir.op_weight = 1 : i64, kbcir.streams = 2 : i64, kbcir.width = 16 : i64, strideA = 1 : i64, strideB = 0 : i64, strideD = 1 : i64}
// CHECK: bcir.gem.block @relu0_s1 {base = 16 : i64, count = 16 : i64, strideA = 1 : i64, strideB = 0 : i64, strideD = 1 : i64}
// CHECK: bcir.gem.block @relu0_s3 {base = 48 : i64, count = 16 : i64, strideA = 1 : i64, strideB = 0 : i64, strideD = 1 : i64}
// CHECK-NOT: bcir.gem.block @relu0_s4

// (2) relu_i32: the exact activation on an integer dtype lowers clean (the clean quarantine side).
// CHECK: bcir.gem.block @relu_i32_s0 {{.*}}kbcir.lane_kind = "exact"{{.*}}

// (3) gelu: lane_kind "transcendental", the tanhf libm edge, op_weight 12 (the heaviest).
// CHECK: bcir.gem.block @gelu0_s0 {{.*}}kbcir.lane_kind = "transcendental", kbcir.libm_edge = "tanhf", kbcir.mem_cost = 32 : i64, kbcir.op_weight = 12 : i64{{.*}}

// (4) softmax: transcendental (expf edge), streams 3, strideB = axis_len 8, QUANTIZED acc_bound 1.
// CHECK: bcir.gem.block @softmax0_s0 {base = 0 : i64, count = 16 : i64, kbcir.acc_bound = 1 : i64, kbcir.bottleneck = 24 : i64, kbcir.compute_cost = 8 : i64, kbcir.lane_kind = "transcendental", kbcir.libm_edge = "expf", kbcir.mem_cost = 24 : i64, kbcir.op_weight = 8 : i64, kbcir.streams = 3 : i64, kbcir.width = 16 : i64, strideA = 1 : i64, strideB = 8 : i64, strideD = 1 : i64}
// CHECK: bcir.gem.block @softmax0_s1 {base = 16 : i64, count = 16 : i64, strideA = 1 : i64, strideB = 8 : i64, strideD = 1 : i64}
// CHECK-NOT: bcir.gem.block @softmax0_s2
