// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// Rung 5 of the open-weight ladder (§7.4): the LLM decode ops on the law rail --
// gem.embedding / gem.rmsnorm / gem.rope -- with the D2 adjacency discipline (R22 shape
// seams, R23 dtype handover) over the decode chain the rung-3 oracle reference composes
// (frontends/models/decode.py). Vacuous when no adjacent pair is present.

// (1 -- NEGATIVE, op law) RoPE with an ODD dim: the rotation pairs channels (2k, 2k+1);
// an odd dim has no legal pairing (the same law DecoderSpec enforces oracle-side).
bcir.module @rope_odd_dim {
  // expected-error @+1 {{rope: dim must be EVEN (the rotation pairs channels 2k/2k+1), got 5}}
  bcir.gem.rope @r { rows = 4 : i64, dim = 5 : i64, dtype = "f32" }
}

// -----

// (2 -- NEGATIVE, R22) the rmsnorm after an embedding lies about the extent: the gather
// produced n_ids*dim = 3*8 = 24 elements, the normalizer declares rows*dim = 4*8 = 32.
bcir.module @rmsnorm_extent_lie {
  bcir.gem.embedding @e { vocab_size = 264 : i64, dim = 8 : i64, n_ids = 3 : i64, dtype = "f32" }
  // expected-error @+1 {{R22: gem.rmsnorm n consumes the adjacent gem.embedding @e but declares an extent of 32 elements != the gather's n_ids*dim = 24}}
  bcir.gem.rmsnorm @n { rows = 4 : i64, dim = 8 : i64, gamma_len = 8 : i64, dtype = "f32" }
}

// -----

// (3 -- NEGATIVE, R22) the attention after a rope lies about the head width: RoPE rotated
// dim = 4 channels per row, the attention declares d_k = 8 -- the rotation would have
// straddled head boundaries.
bcir.module @rope_dk_lie {
  bcir.gem.rope @p { rows = 6 : i64, dim = 4 : i64, dtype = "f32" }
  // expected-error @+1 {{R22: gem.attention a consumes the adjacent gem.rope @p but declares d_k = 8 != the rotated dim = 4}}
  bcir.gem.attention @a { seq_len = 6 : i64, d_k = 8 : i64, dtype = "f32",
    scores_m = 6 : i64, scores_n = 6 : i64, scores_k = 8 : i64,
    context_m = 6 : i64, context_n = 8 : i64, context_k = 6 : i64,
    scores_tile_m = 6 : i64, scores_tile_n = 6 : i64, scores_tile_k = 8 : i64,
    scores_loop_order = "ijk",
    context_tile_m = 6 : i64, context_tile_n = 8 : i64, context_tile_k = 6 : i64,
    context_loop_order = "ijk",
    scores_compute = 9 : i64, scores_mem = 33 : i64,
    context_compute = 9 : i64, context_mem = 33 : i64,
    compute_cost = 18 : i64, mem_cost = 66 : i64, bottleneck = 66 : i64 }
}

// -----

// (4 -- POSITIVE) the legal decode chain the rung-3 reference composes: a truthful
// embedding -> rmsnorm seam (24 == 24, f32 handover) and a rope sized to the head.
bcir.module @legal_decode_chain {
  bcir.gem.embedding @e2 { vocab_size = 264 : i64, dim = 8 : i64, n_ids = 3 : i64, dtype = "f32" }
  bcir.gem.rmsnorm @n2 { rows = 3 : i64, dim = 8 : i64, gamma_len = 8 : i64, dtype = "f32" }
  bcir.gem.rope @p2 { rows = 3 : i64, dim = 4 : i64, pos_offset = 0 : i64, dtype = "f32" }
}
