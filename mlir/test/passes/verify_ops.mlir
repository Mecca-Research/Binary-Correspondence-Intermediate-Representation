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
