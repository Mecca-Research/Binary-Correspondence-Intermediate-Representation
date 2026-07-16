// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// Security regressions: parsed integer attributes are untrusted.  Every
// verifier calculation must reject an unrepresentable derived extent/cost
// instead of invoking signed-overflow UB or accepting wrapped metadata.

bcir.module @resource_product_overflow {
  bcir.registry @R {
    // expected-error @+1 {{shape element count exceeds signed 64-bit range}}
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4611686018427387904, 4>, layout = #bcir.layout<soa> }
  }
}

// -----

bcir.module @claim_negative_offset {
  // expected-error @+1 {{offset must be non-negative (got -1)}}
  bcir.claim @c attributes { claim_id = 1 : i32, phase = @p, op = "copy", reads = [], writes = [], offset = -1 : i64, count = 1 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<none>, bounds = #bcir.bounds<strict> } { %i = bcir.index_range 0 to 1 step 1 }
}

// -----

bcir.module @claim_negative_stride {
  // expected-error @+1 {{stride_k must be positive (got -1)}}
  bcir.claim @c attributes { claim_id = 1 : i32, phase = @p, op = "copy", reads = [], writes = [], count = 1 : i64, stride_k = -1 : i32, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<none>, bounds = #bcir.bounds<strict> } { %i = bcir.index_range 0 to 1 step 1 }
}

// -----

bcir.module @matmul_product_overflow {
  // expected-error @+1 {{matmul m*n*k exceeds signed 64-bit range}}
  bcir.gem.matmul @mm { m = 4611686018427387904 : i64, n = 4 : i64, k = 1 : i64, tile_m = 1 : i64, tile_n = 1 : i64, tile_k = 1 : i64, loop_order = "ijk", compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64 }
}

// -----

bcir.module @activation_product_overflow {
  // expected-error @+1 {{shape element count exceeds signed 64-bit range}}
  bcir.gem.activation @a { kind = "relu", shape = array<i64: 4611686018427387904, 4>, dtype = "f32", axis_len = 0 : i64, width = 1 : i64, compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64 }
}

// -----

bcir.module @conv_padding_overflow {
  // expected-error @+1 {{conv: padded input extent exceeds signed 64-bit range}}
  bcir.gem.conv @c { in_c = 1 : i64, in_h = 9223372036854775807 : i64, in_w = 1 : i64, out_c = 1 : i64, kh = 1 : i64, kw = 1 : i64, stride = 1 : i64, pad = 1 : i64, dtype = "f32", out_h = 1 : i64, out_w = 1 : i64, gemm_m = 1 : i64, gemm_n = 1 : i64, gemm_k = 1 : i64, strategy = "direct", tile_m = 1 : i64, tile_n = 1 : i64, tile_k = 1 : i64, loop_order = "ijk", compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64 }
}

// -----

bcir.module @attention_geometry_overflow {
  // expected-error @+1 {{attention: seq_len*seq_len*d_k exceeds signed 64-bit range}}
  bcir.gem.attention @a { seq_len = 9223372036854775807 : i64, d_k = 1 : i64, dtype = "f32", scores_m = 9223372036854775807 : i64, scores_n = 9223372036854775807 : i64, scores_k = 1 : i64, context_m = 9223372036854775807 : i64, context_n = 1 : i64, context_k = 9223372036854775807 : i64, scores_tile_m = 1 : i64, scores_tile_n = 1 : i64, scores_tile_k = 1 : i64, scores_loop_order = "ijk", context_tile_m = 1 : i64, context_tile_n = 1 : i64, context_tile_k = 1 : i64, context_loop_order = "ijk", scores_compute = 0 : i64, scores_mem = 0 : i64, context_compute = 0 : i64, context_mem = 0 : i64, compute_cost = 0 : i64, mem_cost = 0 : i64, bottleneck = 0 : i64 }
}

// -----

bcir.module @attention_cost_overflow {
  // expected-error @+1 {{attention: summed roofline cost exceeds signed 64-bit range}}
  bcir.gem.attention @a { seq_len = 1 : i64, d_k = 1 : i64, dtype = "f32", scores_m = 1 : i64, scores_n = 1 : i64, scores_k = 1 : i64, context_m = 1 : i64, context_n = 1 : i64, context_k = 1 : i64, scores_tile_m = 1 : i64, scores_tile_n = 1 : i64, scores_tile_k = 1 : i64, scores_loop_order = "ijk", context_tile_m = 1 : i64, context_tile_n = 1 : i64, context_tile_k = 1 : i64, context_loop_order = "ijk", scores_compute = 9223372036854775807 : i64, scores_mem = 0 : i64, context_compute = 1 : i64, context_mem = 0 : i64, compute_cost = 9223372036854775807 : i64, mem_cost = 0 : i64, bottleneck = 9223372036854775807 : i64 }
}

// -----

bcir.module @embedding_product_overflow {
  // expected-error @+1 {{embedding: table/output element count exceeds signed 64-bit range}}
  bcir.gem.embedding @e { vocab_size = 9223372036854775807 : i64, dim = 2 : i64, n_ids = 1 : i64, dtype = "f32" }
}

// -----

bcir.module @rmsnorm_product_overflow {
  // expected-error @+1 {{rmsnorm: rows*dim exceeds signed 64-bit range}}
  bcir.gem.rmsnorm @r { rows = 9223372036854775807 : i64, dim = 2 : i64, gamma_len = 2 : i64, dtype = "f32" }
}

// -----

bcir.module @rope_position_overflow {
  // expected-error @+1 {{rope: tensor/position extent exceeds signed 64-bit range}}
  bcir.gem.rope @r { rows = 1 : i64, dim = 2 : i64, pos_offset = 9223372036854775807 : i64, dtype = "f32" }
}

// -----

bcir.module @gqa_product_overflow {
  // expected-error @+1 {{gqa_attention: query/KV element count exceeds signed 64-bit range}}
  bcir.gem.gqa_attention @g { seq_len = 9223372036854775807 : i64, d_k = 1 : i64, n_heads = 2 : i64, n_kv_heads = 1 : i64, dtype = "f32" }
}

// -----

bcir.module @kv_product_overflow {
  // expected-error @+1 {{kv_cache: bounded cache element count exceeds signed 64-bit range}}
  bcir.gem.kv_cache @k { n_layers = 9223372036854775807 : i64, n_kv_heads = 1 : i64, d_k = 1 : i64, capacity = 2 : i64, pos = 0 : i64, dtype = "f32" }
}

// -----

bcir.module @contention_product_overflow {
  // expected-error @+1 {{contention: derived cost exceeds signed 64-bit range}}
  bcir.gem.contention @c { stride = 2 : i64, elem_bytes = 1 : i64, count = 9223372036854775807 : i64, cacheline = 64 : i64, banks = 1 : i64, line_waste_q8 = 0 : i64, bank_conflict_q8 = 0 : i64, contention_q8 = 0 : i64, contention_cost = 0 : i64 }
}

// -----

bcir.module @layout_product_overflow {
  // expected-error @+1 {{layout_pivot: derived memory cost exceeds signed 64-bit range}}
  bcir.gem.layout_pivot @l { rid = 1 : i64, fields = 2 : i64, single_field_records = 9223372036854775807 : i64, whole_record_records = 0 : i64, cacheline = 64 : i64, elem_bytes = 4 : i64, base_overhead = 1 : i64, streams = 2 : i64, vector_width = 1 : i64, gather_penalty = 2 : i64, soa_cost = 0 : i64, aos_cost = 0 : i64, layout = "soa", gain = 0 : i64 }
}

// -----

bcir.module @fused_product_overflow {
  // expected-error @+1 {{fused matmul m*n*k exceeds signed 64-bit range}}
  bcir.gem.fused_matmul_activation @f { m = 9223372036854775807 : i64, n = 2 : i64, k = 1 : i64, tile_m = 1 : i64, tile_n = 1 : i64, tile_k = 1 : i64, loop_order = "ijk", kind = "relu", dtype = "f32", unfused_score = 2 : i64, fused_score = 1 : i64, gain = 1 : i64 }
}

// -----

bcir.module @negative_capability_geometry {
  // expected-error @+1 {{cacheline must be a positive power of two (got -1)}}
  bcir.target.capability @bad { triple = "x", isa_features = [], lane_widths = array<i64: 1>, cacheline = -1 : i32 }
}

// -----

bcir.module @negative_capability_cost {
  // expected-error @+1 {{warp, gather_penalty, base_overhead, thermal_density, power_density, per_op_heat, and cal_gen must be non-negative}}
  bcir.target.capability @bad { triple = "x", isa_features = [], lane_widths = array<i64: 1>, thermal_density = -1 : i32 }
}

// -----

bcir.module @nonpositive_capability_denominator {
  // expected-error @+1 {{affinity_domains, mem_channels, mem_unit, and elem_bytes must be positive}}
  bcir.target.capability @bad { triple = "x", isa_features = [], lane_widths = array<i64: 1>, mem_channels = 0 : i32 }
}

// -----

bcir.module @nonstr_capability_feature {
  // expected-error @+1 {{isa_features entries must all be strings}}
  bcir.target.capability @bad { triple = "x", isa_features = [7 : i32], lane_widths = array<i64: 1> }
}
