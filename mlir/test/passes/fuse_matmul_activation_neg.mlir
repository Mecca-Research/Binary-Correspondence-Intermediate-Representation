// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// The gem.fused_matmul_activation op verifier (hasVerifier; mirrors bcir/kbcir/fusion.py -- the
// op-level fusion-legality laws, NOT a new globally-numbered R-law). Each section carries a single
// violation. The HEADLINE cases are (a) the softmax SCOPE-OUT (a last-axis row reduction can not be
// an inline epilogue) and (b) the QUARANTINE rule carried through the fused epilogue (a transcendental
// on an integer dtype is rejected -- its libm edge returns float), plus the FusionCertificate
// invariant that an adopted fusion must be a STRICT win (gain = unfused - fused > 0).

// (THE SOFTMAX SCOPE-OUT) a fused softmax: rejected (a row reduction needs the whole row, so no output
// element is final inline -- it can not be an epilogue; mirrors fusion.is_fusible_activation).
bcir.module @fused_softmax {
  // expected-error @+1 {{softmax is non-fusible (a last-axis row reduction needs the whole row before any output element is final, so it can not be an inline epilogue)}}
  bcir.gem.fused_matmul_activation @bad { m = 4 : i64, n = 4 : i64, k = 4 : i64, tile_m = 2 : i64,
                            tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk", kind = "softmax",
                            dtype = "f32", unfused_score = 232 : i64, fused_score = 224 : i64, gain = 8 : i64 }
}

// -----

// (THE QUARANTINE RULE) a transcendental epilogue on an integer dtype: rejected (the libm edge returns
// float). relu may be i32, a transcendental may not -- the quarantine split carries through the fusion.
bcir.module @trans_on_int {
  // expected-error @+1 {{gelu: transcendental activation needs f32 (the libm edge returns float), got 'i32'}}
  bcir.gem.fused_matmul_activation @bad { m = 4 : i64, n = 4 : i64, k = 4 : i64, tile_m = 2 : i64,
                            tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk", kind = "gelu",
                            dtype = "i32", unfused_score = 232 : i64, fused_score = 224 : i64, gain = 8 : i64 }
}

// -----

// an unknown (non-fusible) activation kind is rejected.
bcir.module @bad_kind {
  // expected-error @+1 {{unknown fusible activation kind 'softsign' (expected relu|sigmoid|tanh|gelu)}}
  bcir.gem.fused_matmul_activation @bad { m = 4 : i64, n = 4 : i64, k = 4 : i64, tile_m = 2 : i64,
                            tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk", kind = "softsign",
                            dtype = "f32", unfused_score = 232 : i64, fused_score = 224 : i64, gain = 8 : i64 }
}

// -----

// (THE FUSION-CERTIFICATE INVARIANT) gain must equal unfused_score - fused_score.
bcir.module @bad_gain_arith {
  // expected-error @+1 {{gain must equal unfused_score - fused_score = 8 (got 99)}}
  bcir.gem.fused_matmul_activation @bad { m = 4 : i64, n = 4 : i64, k = 4 : i64, tile_m = 2 : i64,
                            tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk", kind = "relu",
                            dtype = "f32", unfused_score = 232 : i64, fused_score = 224 : i64, gain = 99 : i64 }
}

// -----

// an adopted fusion that did NOT strictly lower the score is unsound evidence: rejected (gain <= 0).
bcir.module @not_a_win {
  // expected-error @+1 {{an adopted fusion must STRICTLY lower the score (gain 0 <= 0: the deforestation discount did not win)}}
  bcir.gem.fused_matmul_activation @bad { m = 4 : i64, n = 4 : i64, k = 4 : i64, tile_m = 2 : i64,
                            tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk", kind = "relu",
                            dtype = "f32", unfused_score = 200 : i64, fused_score = 200 : i64, gain = 0 : i64 }
}

// -----

// the matmul tile half must be well-formed: a tile extent larger than its dim is rejected.
bcir.module @bad_tile {
  // expected-error @+1 {{each tile extent must be in [1, dim] (tiles 8x2x2 vs dims 4x4x4)}}
  bcir.gem.fused_matmul_activation @bad { m = 4 : i64, n = 4 : i64, k = 4 : i64, tile_m = 8 : i64,
                            tile_n = 2 : i64, tile_k = 2 : i64, loop_order = "ijk", kind = "relu",
                            dtype = "f32", unfused_score = 232 : i64, fused_score = 224 : i64, gain = 8 : i64 }
}
