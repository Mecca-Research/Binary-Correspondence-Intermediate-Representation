// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// The deep verifier laws: K_BCIR plan laws R8/R9, GEM stream laws R10/R11, and
// the lowering-contract law R12, as the MLIR-native `-bcir-verify` pass. Each
// section is an independent module with a single law violation (mirrors
// bcir/verify verify_plan / verify_pack / verify_lowering in the oracle).

// R8: a selection drawn from a candidate that has no declared cost path.
bcir.module @r8 {
  bcir.kbcir.plan @plan0 {
    // expected-error @+1 {{R8: candidate path @nope does not resolve to a declared cost path}}
    %s = bcir.kbcir.select @c from [@nope] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      selected = @nope, score = 0 : i64
    } : !bcir.path
  }
}

// -----

// R9: the selected path is not among the candidate set.
bcir.module @r9 {
  bcir.kbcir.plan @plan0 {
    %pa = bcir.kbcir.path @pA {
      claim = @c, realization = "cpu.vector.u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 1, memory = 0, fabric = 0, sync = 0, compile = 0, thermal = 0, power = 0, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    } : !bcir.path
    %pb = bcir.kbcir.path @pB {
      claim = @c, realization = "cpu.vector.u16", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 1, memory = 0, fabric = 0, sync = 0, compile = 0, thermal = 0, power = 0, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    } : !bcir.path
    // expected-error @+1 {{R9: selected path @pB is not among the candidate set}}
    %s = bcir.kbcir.select @c from [@pA] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      selected = @pB, score = 1 : i64
    } : !bcir.path
  }
}

// -----

// R9: the selected path realizes a different claim than the selection plans.
bcir.module @r9_claim {
  bcir.kbcir.plan @plan0 {
    %pa = bcir.kbcir.path @pA {
      claim = @c1, realization = "cpu.vector.u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 1, memory = 0, fabric = 0, sync = 0, compile = 0, thermal = 0, power = 0, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    } : !bcir.path
    // expected-error @+1 {{R9: selected path @pA realizes claim @c1, not @c2}}
    %s = bcir.kbcir.select @c2 from [@pA] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      selected = @pA, score = 1 : i64
    } : !bcir.path
  }
}

// -----

// R10: a hydrated segment that references a claim the module never declared.
bcir.module @r10 {
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // expected-error @+1 {{R10: segment seg0 references unknown claim @ghost}}
  bcir.gem.lane_segment @seg0 {
    claim = @ghost, phase = @p0, lane = #bcir.lane<u>, stride_k = 1 : i32, width = 16 : i32,
    opcode = "f32.add", reads = [], writes = [], fence_before = [], fence_after = []
  }
}

// -----

// R11: the registry has drifted past the pack's generation tags -- the pack is
// stale and must rehydrate, never execute silently.
bcir.module @r11 {
  bcir.registry @RES {
    %a = bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa>, map_gen = 2 : i64, data_gen = 0 : i64 } : !bcir.resource
  }
  bcir.kbcir.plan @plan0 {
    %pa = bcir.kbcir.path @pA {
      claim = @c, realization = "cpu.vector.u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 1, memory = 0, fabric = 0, sync = 0, compile = 0, thermal = 0, power = 0, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    } : !bcir.path
  }
  // expected-error @+1 {{R11: stale StreamPack: map_gen 1 != registry 2 (rehydrate: repack)}}
  %sp = bcir.gem.stream_pack @sp0 attributes {
    source_plan = @plan0, topo_gen = 1 : i64, map_gen = 1 : i64, data_gen = 0 : i64
  } {
    bcir.gem.block @blk0 { base = 0 : i64, count = 4 : i64, strideA = 1 : i64, strideB = 1 : i64, strideD = 1 : i64 }
  } : !bcir.stream
}

// -----

// R8: a constrained selection whose budget never resolves.
bcir.module @r8_budget {
  bcir.kbcir.plan @plan0 {
    %pa = bcir.kbcir.path @pA {
      claim = @c, realization = "cpu.vector.u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 1, memory = 0, fabric = 0, sync = 0, compile = 0, thermal = 0, power = 0, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    } : !bcir.path
    // expected-error @+1 {{R8: budget @nope does not resolve}}
    %s = bcir.kbcir.select @c from [@pA] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      budget = @nope, selected = @pA, score = 1 : i64
    } : !bcir.path
  }
}

// -----

// R9: the constrained rail rejects a selection whose path exceeds its caps --
// under a 700 thermal budget the vec16 path (thermal 1088) is infeasible.
bcir.module @r9_budget {
  bcir.kbcir.budget @cap { dims = ["thermal"], caps = array<i64: 700> }
  bcir.kbcir.plan @plan0 {
    %p16 = bcir.kbcir.path @pHot {
      claim = @c, realization = "cpu.vector.u16", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 64, memory = 3840, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    } : !bcir.path
    // expected-error @+1 {{R9: selected path @pHot violates budget @cap (thermal 1088 > 700)}}
    %s = bcir.kbcir.select @c from [@pHot] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      budget = @cap, selected = @pHot, score = 7808 : i64
    } : !bcir.path
  }
}

// -----

// R9: a scheduled price whose (max,+) makespan books do not balance.
bcir.module @r9_price {
  bcir.kbcir.plan @plan0 {
    %pa = bcir.kbcir.path @pA {
      claim = @c, realization = "cpu.vector.u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 1, memory = 0, fabric = 0, sync = 0, compile = 0, thermal = 0, power = 0, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    } : !bcir.path
  }
  // expected-error @+1 {{R9: inconsistent scheduled price (makespan 5 + overlap_gain 2 != serial 10)}}
  bcir.kbcir.scheduled_price @bad { plan = @plan0, makespan = 5 : i64, serial = 10 : i64, overlap_gain = 2 : i64 }
}

// -----

// R9: a schedule certificate with an unknown dispatch mode.
bcir.module @r9_schedule {
  bcir.kbcir.plan @plan0 {
    %pa = bcir.kbcir.path @pA {
      claim = @c, realization = "cpu.vector.u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 1, memory = 0, fabric = 0, sync = 0, compile = 0, thermal = 0, power = 0, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    } : !bcir.path
  }
  // expected-error @+1 {{R9: unknown schedule mode 'vibes'}}
  bcir.gem.schedule @bad { plan = @plan0, mode = "vibes", makespan = 8 : i64, knee = 2 : i32 }
}

// -----

// R10: a prefetch whose double-buffer contract names an illegal buffer count.
bcir.module @r10_buffers {
  // expected-error @+1 {{R10: prefetch pf0 invalid buffer count (1 or 2)}}
  bcir.gem.prefetch @pf0 { distance = 4 : i32, targets = [@A], hint = "T1", pattern = "double_buffer", buffers = 3 : i32 }
}

// -----

// R12: a lowering contract that neither preserves the BCIR semantic
// (bounds/hazard/precision) nor carries an explicit discharge.
bcir.module @r12 {
  // expected-error @+1 {{R12: lowering contract lc must preserve bounds/hazard/precision or carry an explicit discharge}}
  bcir.target.lower_contract @lc {
    target = @cpu, bcir_op = "vector.add", lane = #bcir.lane<u>,
    stride_class = #bcir.stride_class<unit>, opcode = @vadd,
    legal_if = "avx2", preserves = "bounds"
  }
}
