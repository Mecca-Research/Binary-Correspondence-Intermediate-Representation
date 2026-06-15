// RUN: bcir-opt -bcir-select-realization -bcir-lower-to-llvm -verify-diagnostics -split-input-file %s
//
// The GEM pipeline cross-checks (against the oracle): the min-plus selection
// recomputes the score from cost . weights and rejects a declared selection that
// is not the true argmin; the lowering contract rejects a StreamPack segment
// that does not preserve its claim's lane geometry or resource set (R12). Each
// section carries a single violation (mirrors verify_laws_deep.mlir).

// select-realization: a declared score that is not the recomputed min-plus cost.
bcir.module @bad_score {
  bcir.kbcir.policy @perf { mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1> }
  bcir.kbcir.plan @plan0 {
    %a = bcir.kbcir.path @pA { claim = @c, realization = "u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 128, memory = 4608, fabric = 0, sync = 0, compile = 0, thermal = 640, power = 640, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> } : !bcir.path
    %b = bcir.kbcir.path @pB { claim = @c, realization = "u16", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 64, memory = 3840, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> } : !bcir.path
    // expected-error @+1 {{bcir-select-realization: computed score 7808 != declared score 9999}}
    %s = bcir.kbcir.select @c from [@pA, @pB] { policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>, selected = @pB, score = 9999 : i64 } : !bcir.path
  }
}

// -----

// select-realization: a declared selection that is not the true argmin (pA costs
// 9472 > pB's 7808, so claiming pA without a budget is wrong).
bcir.module @bad_selected {
  bcir.kbcir.policy @perf { mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1> }
  bcir.kbcir.plan @plan0 {
    %a = bcir.kbcir.path @pA { claim = @c, realization = "u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 128, memory = 4608, fabric = 0, sync = 0, compile = 0, thermal = 640, power = 640, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> } : !bcir.path
    %b = bcir.kbcir.path @pB { claim = @c, realization = "u16", lane = #bcir.lane<u>, layout = #bcir.layout<soa>, cost = #bcir.costvec<compute = 64, memory = 3840, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0> } : !bcir.path
    // The declared score is the true min (7808) so only the *selection* is wrong.
    // expected-error @+1 {{bcir-select-realization: computed argmin @pB != declared selected @pA}}
    %s = bcir.kbcir.select @c from [@pA, @pB] { policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>, selected = @pA, score = 7808 : i64 } : !bcir.path
  }
}

// -----

// lower-to-llvm: a StreamPack segment that drops a resource the claim reads (the
// R12 lowering contract: lane geometry + resource set must be preserved).
bcir.module @bad_segment {
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @add attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C], count = 4 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<none>, bounds = #bcir.bounds<masked>
  } { %i = bcir.index_range 0 to 4 step 1 }
  // expected-error @+1 {{bcir-lower-to-llvm: segment resource set not preserved (R12) for claim @add}}
  bcir.gem.lane_segment @seg0 { claim = @add, phase = @p0, lane = #bcir.lane<u>, width = 16 : i32, opcode = "f32.add", reads = [@A], writes = [@C], fence_before = [], fence_after = [] }
}

// -----

// R14 (CIM/PIM dispatch legality): a NON-reduction segment cannot be dispatched to
// processing-in-memory (mirrors gem.cim in the oracle: only reduce.* offloads).
bcir.module @bad_pim_dispatch {
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @add attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C], count = 4 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<none>, bounds = #bcir.bounds<masked>
  } { %i = bcir.index_range 0 to 4 step 1 }
  // expected-error @+1 {{R14: pim dispatch illegal for non-reduction op 'vector.add' (claim @add)}}
  bcir.gem.lane_segment @seg0 { claim = @add, phase = @p0, lane = #bcir.lane<u>, width = 16 : i32, opcode = "vector.add", reads = [@A, @B], writes = [@C], fence_before = [], fence_after = [], dispatch = "pim" }
}

// -----

// R14 (positive): a reduction segment MAY be dispatched to processing-in-memory --
// it lowers clean (no diagnostic).
bcir.module @ok_pim_dispatch {
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @reduce attributes {
    claim_id = 1 : i32, phase = @p0, op = "reduce.gather", reads = [@TABLE], writes = [@ACC], count = 1024 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.gem.lane_segment @seg0 { claim = @reduce, phase = @p0, lane = #bcir.lane<u>, width = 1 : i32, opcode = "reduce.gather", reads = [@TABLE], writes = [@ACC], fence_before = [], fence_after = [], dispatch = "pim" }
}

// -----

// R15 (DVFS clock legality): a clock outside the legal step range [64, 512] is illegal.
bcir.module @bad_clock_range {
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @add attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C], count = 4 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<none>, bounds = #bcir.bounds<masked>
  } { %i = bcir.index_range 0 to 4 step 1 }
  // expected-error @+1 {{R15: clock_q8 1000 out of legal range [64, 512]}}
  bcir.gem.lane_segment @seg0 { claim = @add, phase = @p0, lane = #bcir.lane<u>, width = 16 : i32, opcode = "vector.add", reads = [@A, @B], writes = [@C], fence_before = [], fence_after = [], clock_q8 = 1000 : i32 }
}

// -----

// R15 (DVFS clock legality): a memory-bound (pim) reduction must NOT overclock.
bcir.module @bad_pim_overclock {
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @reduce attributes {
    claim_id = 1 : i32, phase = @p0, op = "reduce.gather", reads = [@TABLE], writes = [@ACC], count = 1024 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  // expected-error @+1 {{R15: pim (memory-bound) segment must not overclock (clock_q8 320 > 256)}}
  bcir.gem.lane_segment @seg0 { claim = @reduce, phase = @p0, lane = #bcir.lane<u>, width = 1 : i32, opcode = "reduce.gather", reads = [@TABLE], writes = [@ACC], fence_before = [], fence_after = [], dispatch = "pim", clock_q8 = 320 : i32 }
}

// -----

// R16 (allocator placement legality): a resource placed in L1 must fit (<= 64 KiB).
// A 1 Mi-element i32 tensor is 4 MiB -- it cannot be L1-resident.
bcir.module @bad_l1_placement {
  bcir.registry @RES {
    // expected-error @+1 {{R16: placement l1 does not fit @BIG (4194304 B > 65536 B)}}
    %b = bcir.resource @BIG { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1048576>, layout = #bcir.layout<soa>, placement = #bcir.mem_tier<l1> } : !bcir.resource
  }
}

// -----

// R16 (positive): a small tensor MAY be L1-resident; R15 (positive): a downclock is
// legal. Both lower clean (no diagnostic).
bcir.module @ok_placement_and_clock {
  bcir.registry @RES {
    %s = bcir.resource @SMALL { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 64>, layout = #bcir.layout<soa>, placement = #bcir.mem_tier<l1> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @add attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@SMALL], writes = [@SMALL], count = 64 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<none>, bounds = #bcir.bounds<masked>
  } { %i = bcir.index_range 0 to 64 step 1 }
  bcir.gem.lane_segment @seg0 { claim = @add, phase = @p0, lane = #bcir.lane<u>, width = 16 : i32, opcode = "vector.add", reads = [@SMALL], writes = [@SMALL], fence_before = [], fence_after = [], clock_q8 = 192 : i32 }
}
