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

// R8: a calibration table whose streaming baseline is not the Q8 unit.
bcir.module @r8_calibration {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  // expected-error @+1 {{R8: calibration stream_q8 must be 256 (the Q8 baseline)}}
  bcir.kbcir.calibration @bad {
    target = @cpu, cal_gen = 1 : i64, samples = 5 : i64, provenance = "microbench",
    stream_q8 = 128 : i64, strided_q8 = 256 : i64, random_q8 = 8192 : i64, compute_q8 = 256 : i64
  }
}

// -----

// R8: a Bayesian/conformal table whose claimed coverage is not a probability.
bcir.module @r8_conformal {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  // expected-error @+1 {{R8: conformal coverage 1500/1000 out of range (0,1000)}}
  bcir.kbcir.calibration @bad {
    target = @cpu, cal_gen = 1 : i64, samples = 8 : i64, provenance = "bayes",
    stream_q8 = 256 : i64, strided_q8 = 256 : i64, random_q8 = 8192 : i64, compute_q8 = 256 : i64,
    coverage_milli = 1500 : i64, random_delta_q8 = 128 : i64
  }
}

// -----

// R9: an admitted replay certificate that carries regressions -- the gate
// never deploys a gain schedule on a measured loss.
bcir.module @r9_replay {
  bcir.kbcir.policy @latency { mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1> }
  bcir.kbcir.policy @hotbias { mode = #bcir.policy_mode<energy>, weights = array<i64: 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0> }
  // expected-error @+1 {{R9: admitted replay certificate carries 1 regression(s)}}
  bcir.kbcir.replay_certificate @forged {
    candidate = @hotbias, incumbent = @latency, episodes = 3 : i64,
    regressions = 1 : i64, admitted = true
  }
}

// -----

// R9: a portfolio whose certification state does not cover its policies.
bcir.module @r9_portfolio {
  bcir.kbcir.policy @latency { mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1> }
  // expected-error @+1 {{R9: portfolio policies/gens/certified arity mismatch}}
  bcir.kbcir.portfolio @gains { policies = [@latency], gens = array<i64: 1, 1>, certified = array<i64: 1> }
}

// -----

// R13: a promoted gain schedule (gen 2) with no admitting replay certificate
// -- rule swaps are never silent.
bcir.module @r13_promotion {
  bcir.kbcir.policy @latency { mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1> }
  // expected-error @+1 {{R13: promoted policy @latency (gen 2) has no admitting replay certificate}}
  bcir.kbcir.portfolio @gains { policies = [@latency], gens = array<i64: 2>, certified = array<i64: 1> }
}

// -----

// R13: a capability claiming calibration without its matching frozen table.
bcir.module @r13_table {
  // expected-error @+1 {{R13: capability @cpu claims cal_gen 2 without a matching calibration certificate}}
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>,
    cal_gen = 2 : i64
  }
  bcir.kbcir.calibration @cal_cpu {
    target = @cpu, cal_gen = 1 : i64, samples = 0 : i64, provenance = "stale",
    stream_q8 = 256 : i64, strided_q8 = 256 : i64, random_q8 = 8192 : i64, compute_q8 = 256 : i64
  }
}

// -----

// R13: a regret ledger whose books do not balance (worst exceeds total).
bcir.module @r13_ledger {
  bcir.kbcir.policy @latency { mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1> }
  // expected-error @+1 {{R13: regret ledger books do not balance}}
  bcir.kbcir.regret_ledger @bad { rule = @latency, episodes = 3 : i64, total_regret = 5 : i64, worst_regret = 9 : i64, gen = 1 : i64 }
}

// -----

// R13: a retune verdict the MDL evidence does not justify (data_fit <=
// complexity) -- the dashboard cannot recommend a swap the bits do not pay for.
bcir.module @r13_unjustified_retune {
  bcir.kbcir.policy @latency { mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1> }
  // expected-error @+1 {{R13: regret verdict 'retune' inconsistent with the MDL evidence (data_fit 100 vs complexity 800)}}
  bcir.kbcir.regret_ledger @bad { rule = @latency, episodes = 3 : i64, total_regret = 50 : i64, worst_regret = 20 : i64, gen = 1 : i64, data_fit_milli = 100 : i64, complexity_milli = 800 : i64, verdict = "retune" }
}

// -----

// R9: a soft_select whose free energy exceeds the hard minimum (softmin <= min
// is violated).
bcir.module @r9_soft_bound {
  // expected-error @+1 {{R9: soft_select free_energy 8000 exceeds the hard minimum 7808}}
  bcir.kbcir.soft_select @bad {
    claim = @add, temperature_milli = 3000000 : i64, score = 7808 : i64,
    free_energy = 8000 : i64, selected = @add_cpu_u16
  }
}

// -----

// R9: a soft_select at temperature 0 that does not recover the tropical score.
bcir.module @r9_soft_zero {
  // expected-error @+1 {{R9: at temperature 0 soft_select free_energy 7000 must equal the tropical score 7808}}
  bcir.kbcir.soft_select @bad {
    claim = @add, temperature_milli = 0 : i64, score = 7808 : i64,
    free_energy = 7000 : i64, selected = @add_cpu_u16
  }
}

// -----

// R13: a learned MoE gate proposed for deployment whose replay gate did not
// pass (regressions > 0) -- the network proposes a route, the verifier disposes.
bcir.module @r13_moe_gate {
  bcir.kbcir.policy @latency { mode = #bcir.policy_mode<latency>, weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1> }
  bcir.kbcir.portfolio @gains { policies = [@latency], gens = array<i64: 1>, certified = array<i64: 1> }
  // expected-error @+1 {{R13: admitted MoE gate carries 2 regression(s)}}
  bcir.kbcir.moe_gate @gate0 {
    portfolio = @gains, num_experts = 4 : i64, hidden = 8 : i64, gate_gen = 1 : i64,
    fingerprint = 123456 : i64, episodes = 5 : i64, regressions = 2 : i64, admitted = true
  }
}

// -----

// R13: a search accelerator deployed as admitted but whose learned ordering
// changed the optimum (mismatches > 0) -- ordering may change work, never result.
bcir.module @r13_search_accel {
  // expected-error @+1 {{R13: admitted search accelerator changed the optimum (1 mismatch(es))}}
  bcir.kbcir.search_accel @accel0 {
    order = "learned", checked = 12 : i64, mismatches = 1 : i64, admitted = true
  }
}

// -----

// R13: a deployed plan whose provenance manifest did not reproduce on replay --
// a plan that cannot be reproduced from its commit hash is not a closed branch.
bcir.module @r13_manifest {
  // expected-error @+1 {{R13: deployed plan manifest did not reproduce on replay}}
  bcir.kbcir.provenance_manifest @man0 {
    digest = 7777777 : i64, score = 7808 : i64, n_artifacts = 0 : i64, reproduced = false
  }
}

// -----

// R9: an equality-saturation extraction that RAISED the cost -- a rewrite is
// legal only if it does not increase the selected cost.
bcir.module @r9_egraph {
  // expected-error @+1 {{R9: egraph rewrite raised the cost (optimized 7 > original 5)}}
  bcir.egraph.extract @eg0 {
    original_cost = 5 : i64, optimized_cost = 7 : i64, iterations = 3 : i64,
    enodes = 9 : i64, saturated = true
  }
}

// -----

// R13: a learned component running inference on the L0 hot path -- the hot path
// carries decisions, not models.
bcir.module @r13_l0 {
  // expected-error @+1 {{R13: component rogue at L0 runs learned inference}}
  bcir.kbcir.amortization @am0 {
    component = "rogue", tier = "L0", inference_cost = 12 : i64, gain = 9999 : i64,
    budget = 0 : i64
  }
}

// -----

// R13: a learned component that does not pay for itself (gain < inference_cost).
bcir.module @r13_amortize {
  // expected-error @+1 {{R13: component wasteful fails amortization}}
  bcir.kbcir.amortization @am1 {
    component = "wasteful", tier = "L1", inference_cost = 5000 : i64, gain = 10 : i64,
    budget = 100000 : i64
  }
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
