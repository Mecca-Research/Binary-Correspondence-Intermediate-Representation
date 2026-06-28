// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// The gem.activation op verifier (hasVerifier; mirrors bcir/kbcir/activation.py::check_activation --
// the op-level shape/dtype well-formedness check gem.matmul validates inline, NOT a new globally-numbered
// R-law). Each section carries a single violation (mirrors lower_gem_matmul_buffer_neg.mlir / the
// gem_passes_neg.mlir style). The HEADLINE case is the QUARANTINE rule: a transcendental on an integer
// (exact) dtype is rejected, because its libm edge (expf/tanhf) returns float -- relu may be i32, a
// transcendental may not.

// (THE QUARANTINE RULE) a transcendental on an integer dtype: rejected (the libm edge returns float).
bcir.module @trans_on_int {
  // expected-error @+1 {{sigmoid: transcendental activation needs f32 (the libm edge returns float), got 'i32'}}
  bcir.gem.activation @bad { kind = "sigmoid", shape = array<i64: 8>, dtype = "i32", axis_len = 0 : i64,
                            width = 8 : i64, compute_cost = 8 : i64, mem_cost = 16 : i64,
                            bottleneck = 16 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// an unknown activation kind is rejected.
bcir.module @bad_kind {
  // expected-error @+1 {{unknown activation kind 'softsign' (expected relu|sigmoid|tanh|gelu|softmax)}}
  bcir.gem.activation @bad { kind = "softsign", shape = array<i64: 8>, dtype = "f32", axis_len = 0 : i64,
                            width = 8 : i64, compute_cost = 8 : i64, mem_cost = 16 : i64,
                            bottleneck = 16 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// softmax: a declared axis_len that is not the last shape dim is rejected.
bcir.module @bad_softmax_axis {
  // expected-error @+1 {{softmax: axis_len 4 != last shape dim 8}}
  bcir.gem.activation @bad { kind = "softmax", shape = array<i64: 4, 8>, dtype = "f32", axis_len = 4 : i64,
                            width = 8 : i64, compute_cost = 8 : i64, mem_cost = 24 : i64,
                            bottleneck = 24 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// an elementwise (non-softmax) activation must declare axis_len 0 -- a stray reduction axis is rejected.
bcir.module @bad_elementwise_axis {
  // expected-error @+1 {{tanh: elementwise activation must declare axis_len 0, got 3}}
  bcir.gem.activation @bad { kind = "tanh", shape = array<i64: 8>, dtype = "f32", axis_len = 3 : i64,
                            width = 8 : i64, compute_cost = 8 : i64, mem_cost = 16 : i64,
                            bottleneck = 16 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the roofline invariant: bottleneck must equal max(compute_cost, mem_cost) (the max,+ parity).
bcir.module @bad_bottleneck {
  // expected-error @+1 {{relu: bottleneck must equal max(compute_cost, mem_cost) = 32 (got 99)}}
  bcir.gem.activation @bad { kind = "relu", shape = array<i64: 64>, dtype = "f32", axis_len = 0 : i64,
                            width = 16 : i64, compute_cost = 2 : i64, mem_cost = 32 : i64,
                            bottleneck = 99 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}

// -----

// the R17 accuracy axis: a QUANTIZED realization (quant_bits > 0) must carry the 1-ULP Q8-bridge bound
// on the accuracy axis -- declaring acc_bound 0 for a quantized op is rejected.
bcir.module @bad_acc_bound {
  // expected-error @+1 {{relu: acc_bound must be 1 (the R17 Q8-bridge bound, 1 ULP for a quantized realization), got 0}}
  bcir.gem.activation @bad { kind = "relu", shape = array<i64: 64>, dtype = "f32", axis_len = 0 : i64,
                            width = 16 : i64, compute_cost = 2 : i64, mem_cost = 32 : i64,
                            bottleneck = 32 : i64, quant_bits = 8 : i32, acc_bound = 0 : i64 }
}

// -----

// a width wider than the tensor element count is rejected (mirrors _divisor_widths' cap).
bcir.module @bad_width {
  // expected-error @+1 {{relu: width 128 must be in [1, count 64]}}
  bcir.gem.activation @bad { kind = "relu", shape = array<i64: 64>, dtype = "f32", axis_len = 0 : i64,
                            width = 128 : i64, compute_cost = 2 : i64, mem_cost = 32 : i64,
                            bottleneck = 32 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }
}
