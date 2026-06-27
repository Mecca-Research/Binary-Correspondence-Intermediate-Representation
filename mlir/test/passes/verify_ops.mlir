// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// Parse-time op verifiers (ODS hasVerifier=1): structural well-formedness checked at
// parse/build, independent of the -bcir-verify pass (which carries the cross-op R-laws).

bcir.module @bad_align_not_pow2 {
  bcir.registry @R {
    // expected-error @+1 {{align must be a positive power of two (got 3)}}
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, align = 3 : i32 }
  }
}

// -----

bcir.module @bad_align_zero {
  bcir.registry @R {
    // expected-error @+1 {{align must be a positive power of two (got 0)}}
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 16>, layout = #bcir.layout<soa>, align = 0 : i32 }
  }
}

// -----

bcir.module @bad_shape_zero {
  bcir.registry @R {
    // expected-error @+1 {{shape extents must be positive (got 0)}}
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 0>, layout = #bcir.layout<soa> }
  }
}

// -----

bcir.module @bad_shape_negative {
  bcir.registry @R {
    // expected-error @+1 {{shape extents must be positive (got -4)}}
    bcir.resource @A { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: -4>, layout = #bcir.layout<soa> }
  }
}

// -----

bcir.module @bad_segment_width {
  // expected-error @+1 {{width must be a positive power of two (got 6)}}
  bcir.gem.lane_segment @s { claim = @c, phase = @p, lane = #bcir.lane<u>, width = 6 : i32, opcode = "vector.add", reads = [], writes = [], fence_before = [], fence_after = [] }
}

// -----

bcir.module @bad_segment_stride {
  // expected-error @+1 {{stride_k must be positive (got 0)}}
  bcir.gem.lane_segment @s { claim = @c, phase = @p, lane = #bcir.lane<u>, width = 8 : i32, stride_k = 0 : i32, opcode = "vector.add", reads = [], writes = [], fence_before = [], fence_after = [] }
}

// -----

bcir.module @bad_claim_count {
  // expected-error @+1 {{count must be non-negative (got -5)}}
  bcir.claim @c attributes { claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [], writes = [], count = -5 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<none>, bounds = #bcir.bounds<strict> } { %i = bcir.index_range 0 to 1 step 1 }
}

// -----

bcir.module @bad_claim_stride {
  // expected-error @+1 {{stride_k must be positive (got 0)}}
  bcir.claim @c attributes { claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [], writes = [], count = 1 : i64, stride_k = 0 : i32, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<none>, bounds = #bcir.bounds<strict> } { %i = bcir.index_range 0 to 1 step 1 }
}

// -----

bcir.module @bad_cap_cacheline {
  // expected-error @+1 {{cacheline must be a positive power of two (got 48)}}
  bcir.target.capability @cpu { triple = "x", isa_features = [], lane_widths = array<i64: 1, 8>, cacheline = 48 : i32 }
}

// -----

bcir.module @bad_cap_lane_width {
  // expected-error @+1 {{lane widths must be positive (got 0)}}
  bcir.target.capability @cpu { triple = "x", isa_features = [], lane_widths = array<i64: 1, 0, 8> }
}

// -----

bcir.module @bad_matmul_bottleneck {
  // expected-error @+1 {{bottleneck must equal max(compute_cost, mem_cost) = 20 (got 99)}}
  bcir.gem.matmul @mm { m = 8 : i64, n = 8 : i64, k = 8 : i64, tile_m = 4 : i64, tile_n = 8 : i64, tile_k = 2 : i64, loop_order = "ijk", compute_cost = 10 : i64, mem_cost = 20 : i64, bottleneck = 99 : i64 }
}

// -----

bcir.module @bad_matmul_tile {
  // expected-error @+1 {{each tile extent must be in [1, dim] (tiles 99x8x2 vs dims 8x8x8)}}
  bcir.gem.matmul @mm { m = 8 : i64, n = 8 : i64, k = 8 : i64, tile_m = 99 : i64, tile_n = 8 : i64, tile_k = 2 : i64, loop_order = "ijk", compute_cost = 10 : i64, mem_cost = 20 : i64, bottleneck = 20 : i64 }
}

// -----

bcir.module @bad_matmul_loop_order {
  // expected-error @+1 {{loop_order must be one of ijk|ikj|jik (got 'zzz')}}
  bcir.gem.matmul @mm { m = 8 : i64, n = 8 : i64, k = 8 : i64, tile_m = 4 : i64, tile_n = 8 : i64, tile_k = 2 : i64, loop_order = "zzz", compute_cost = 10 : i64, mem_cost = 20 : i64, bottleneck = 20 : i64 }
}

// -----

bcir.module @bad_matmul_dim {
  // expected-error @+1 {{matmul dims m/n/k must be positive (got 0x8x8)}}
  bcir.gem.matmul @mm { m = 0 : i64, n = 8 : i64, k = 8 : i64, tile_m = 1 : i64, tile_n = 8 : i64, tile_k = 2 : i64, loop_order = "ijk", compute_cost = 10 : i64, mem_cost = 20 : i64, bottleneck = 20 : i64 }
}
