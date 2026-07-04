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

// -----

// (5 -- NEGATIVE, op law) GQA with ragged head groups: 4 query heads cannot share 3 kv
// heads (whole-group sharing only -- the same law DecoderSpec enforces oracle-side).
bcir.module @gqa_ragged_groups {
  // expected-error @+1 {{gqa_attention: n_heads 4 not divisible by n_kv_heads 3 (GQA shares whole head groups)}}
  bcir.gem.gqa_attention @g { seq_len = 4 : i64, d_k = 2 : i64, n_heads = 4 : i64,
                              n_kv_heads = 3 : i64, dtype = "f32" }
}

// -----

// (6 -- NEGATIVE, op law) an over-full KV cache: pos 9 rows in a capacity-8 cache is the
// paging lie a serving runtime must never tell.
bcir.module @kv_overfull {
  // expected-error @+1 {{kv_cache: pos 9 exceeds capacity 8 (an over-full cache is the paging lie)}}
  bcir.gem.kv_cache @c { n_layers = 2 : i64, n_kv_heads = 2 : i64, d_k = 2 : i64,
                         capacity = 8 : i64, pos = 9 : i64, dtype = "f32" }
}

// -----

// (7 -- NEGATIVE, R22) the gqa_attention after a kv_cache lies about the shared head
// geometry: the cache holds 2 kv lanes, the consumer declares 4.
bcir.module @kv_head_lie {
  bcir.gem.kv_cache @c2 { n_layers = 2 : i64, n_kv_heads = 2 : i64, d_k = 2 : i64, dtype = "f32" }
  // expected-error @+1 {{R22: gem.gqa_attention g2 reads the adjacent gem.kv_cache @c2 but declares n_kv_heads = 4 != the cache's 2}}
  bcir.gem.gqa_attention @g2 { seq_len = 4 : i64, d_k = 2 : i64, n_heads = 4 : i64,
                               n_kv_heads = 4 : i64, dtype = "f32" }
}

// -----

// (8 -- NEGATIVE, R22) the gqa_attention after a rope lies about the head width, exactly
// like plain attention (the rotation would straddle head boundaries).
bcir.module @gqa_rope_dk_lie {
  bcir.gem.rope @p3 { rows = 4 : i64, dim = 2 : i64, dtype = "f32" }
  // expected-error @+1 {{R22: gem.gqa_attention g3 consumes the adjacent gem.rope @p3 but declares d_k = 4 != the rotated dim = 2}}
  bcir.gem.gqa_attention @g3 { seq_len = 4 : i64, d_k = 4 : i64, n_heads = 2 : i64,
                               n_kv_heads = 2 : i64, dtype = "f32" }
}

// -----

// (9 -- POSITIVE) the legal cached GQA pair: the cache and its consumer agree on the kv
// lanes and head width (n_kv_heads == n_heads is plain MHA riding the same op).
bcir.module @legal_gqa_chain {
  bcir.gem.kv_cache @c3 { n_layers = 2 : i64, n_kv_heads = 2 : i64, d_k = 2 : i64,
                          capacity = 16 : i64, pos = 5 : i64, dtype = "f32" }
  bcir.gem.gqa_attention @g4 { seq_len = 5 : i64, d_k = 2 : i64, n_heads = 4 : i64,
                               n_kv_heads = 2 : i64, dtype = "f32" }
}
