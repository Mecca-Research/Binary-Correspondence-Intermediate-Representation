// RUN: bcir-opt %s | FileCheck %s
//
// Canonical BCIR-MLIR v0.1 textual IR -- vector add through the full
// correspondence stack with the CT1 target-open container + memory hierarchy.
// This is the normative *pretty* shape the ODS dialect family must accept. The
// oracle realizes the same program today:
//   python -m bcir.run vector_add --target x86_avx512        (see docs/PARITY.md)
// NOT parseable on this host (no bcir-opt yet).

bcir.module @full_vec_add_ct1 attributes {
  version = "0.1", target_model = "registry-first",
  execution_model = "gem", cost_model = "k_bcir"
} {
  // ---- H: the open target container (any ISA is data) ----
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx2", "avx512f", "fma"],
    lane_widths = array<i64: 1, 8, 16>, warp = 0 : i32, cacheline = 64 : i32,
    gather_penalty = 8 : i32, affinity_domains = 16 : i32, cal_gen = 1 : i64
  }
  bcir.target.capability @gpu {
    triple = "nvptx64-warp", isa_features = ["ptx", "warp"],
    lane_widths = array<i64: 1, 32>, warp = 32 : i32, cacheline = 128 : i32,
    gather_penalty = 16 : i32, affinity_domains = 128 : i32
  }

  // ---- memory hierarchy: Q16 bandwidth/latency factors vs DRAM ----
  bcir.mem.tier @dram { tier = #bcir.mem_tier<dram>, latency_cyc = 200 : i64, bw_factor = 65536 : i64, lat_factor = 65536 : i64 }
  bcir.mem.tier @hbm  { tier = #bcir.mem_tier<hbm>,  latency_cyc = 160 : i64, bw_factor = 16384 : i64, lat_factor = 49152 : i64 }
  bcir.mem.tier @cxl  { tier = #bcir.mem_tier<cxl>,  latency_cyc = 350 : i64, bw_factor = 98304 : i64, lat_factor = 131072 : i64 }

  // ---- BCIR-1/2: registry-first resources (no raw pointers); C resides in HBM ----
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, align = 64 : i32, access = #bcir.access<flat>, priority = 0 : i32, map_gen = 1 : i64, data_gen = 4 : i64 }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, align = 64 : i32, access = #bcir.access<flat>, priority = 0 : i32, map_gen = 1 : i64, data_gen = 4 : i64 }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<hbm>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, align = 64 : i32, access = #bcir.access<flat>, priority = 80 : i32, map_gen = 1 : i64, data_gen = 0 : i64 }
  }

  // ---- BCIR-0: phase DAG + semantic claim ----
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @add attributes {
    claim_id = 1000 : i32, phase = @p0, op = "vector.add",
    reads = [@A, @B], writes = [@C], offset = 0 : i64, count = 1024 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<hbm>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    precision = #bcir.precision<f32, exact = true, tol = 0>, cost_class = #bcir.cost_class<bandwidth>
  } {
    %i = bcir.index_range 0 to 1024 step 1
    %a = bcir.load @A[%i] { lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict> } : f32
    %b = bcir.load @B[%i] { lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict> } : f32
    %c = bcir.compute "fadd"(%a, %b) : (f32, f32) -> f32
    bcir.store %c, @C[%i] { lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32, domain = #bcir.domain<hbm>, hazard = #bcir.hazard<unique>, verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict> } : f32
  }

  // ---- BCIR-3: K_BCIR plan (candidate paths -> min-plus selection) ----
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.kbcir.plan @plan0 {
    bcir.kbcir.path @add_cpu_u8 {
      claim = @add, realization = "cpu.vector.u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 128, memory = 4608, fabric = 0, sync = 0, compile = 0, thermal = 640, power = 640, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    }
    bcir.kbcir.path @add_cpu_u16 {
      claim = @add, realization = "cpu.vector.u16", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 64, memory = 3840, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    }
    // Cool Theta selects vec16 (score 7808); a hot-Theta replan picks vec8.
    %sel = bcir.kbcir.select @add from [@add_cpu_u8, @add_cpu_u16] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      selected = @add_cpu_u16, score = 7808 : i64
    }
    // The temperature dial (kbcir.softdp): the soft log-sum-exp twin at T=0
    // recovers the tropical select exactly -- free_energy == score == 7808.
    bcir.kbcir.soft_select @sel_soft {
      claim = @add, temperature_milli = 0 : i64, score = 7808 : i64,
      free_energy = 7808 : i64, selected = @add_cpu_u16
    }
    // Constrained (RCSP) rail: a 700 thermal/power cap makes vec16 (1088)
    // infeasible -- the budgeted optimum is vec8 at score 9472 (the oracle:
    // optimize_constrained(..., Budget.of(thermal=700)); a point no PERF
    // weight vector can select, since PERF's thermal weight is 0).
    %selc = bcir.kbcir.select @add from [@add_cpu_u8, @add_cpu_u16] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      budget = @thermal_cap, selected = @add_cpu_u8, score = 9472 : i64
    }
  }
  bcir.kbcir.budget @thermal_cap { dims = ["thermal", "power"], caps = array<i64: 700, 700> }
  // M(pi,Theta): the (max,+) wave-overlap price. One claim => the degenerate
  // case: makespan == serial Sigma score, overlap gain 0 (gem.overlap oracle).
  bcir.kbcir.scheduled_price @overlap_price { plan = @plan0, makespan = 7808 : i64, serial = 7808 : i64, overlap_gain = 0 : i64 }
  // Duration-aware schedule certificate (gem.schedule oracle): EFT dispatch of
  // the K_BCIR durations under the target's bandwidth knee.
  bcir.gem.schedule @sched0 { plan = @plan0, mode = "eft", makespan = 7808 : i64, knee = 4 : i32 }

  // ---- LangRef Sec. 13: learning placement, as certified data ----
  // L1: the physics-anchored frozen cost table (ratio-1 reference: reproduces
  // the seeded constants; gather_penalty = 8192/256 = 32, base_overhead = 4).
  bcir.kbcir.calibration @cal_cpu {
    target = @cpu, cal_gen = 1 : i64, samples = 0 : i64,
    provenance = "reference (ratio-1; reproduces the seeded constants exactly)",
    stream_q8 = 256 : i64, strided_q8 = 256 : i64, random_q8 = 8192 : i64, compute_q8 = 256 : i64
  }
  // A Bayesian/conformal table (kbcir.bayescal): the same point estimate plus a
  // certified +/- delta on the random ratio at 90% coverage (split conformal).
  bcir.kbcir.calibration @cal_cpu_bayes {
    target = @cpu, cal_gen = 1 : i64, samples = 8 : i64,
    provenance = "bayes (conjugate VI + split conformal cov=0.9)",
    stream_q8 = 256 : i64, strided_q8 = 256 : i64, random_q8 = 8192 : i64, compute_q8 = 256 : i64,
    coverage_milli = 900 : i64, random_delta_q8 = 256 : i64
  }
  // L2: the certified gain-schedule portfolio + the replay gate that admitted it.
  bcir.kbcir.portfolio @gains { policies = [@perf], gens = array<i64: 1>, certified = array<i64: 1> }
  bcir.kbcir.replay_certificate @perf_cert {
    candidate = @perf, incumbent = @perf, episodes = 3 : i64, regressions = 0 : i64, admitted = true
  }
  // L2: the learned MoE gate (kbcir.moegate) -- a GNN over the claim graph routes
  // among the @gains experts; deployed frozen, behind an admitting replay gate.
  bcir.kbcir.moe_gate @gate0 {
    portfolio = @gains, num_experts = 4 : i64, hidden = 8 : i64, gate_gen = 1 : i64,
    fingerprint = 4242424242 : i64, episodes = 3 : i64, regressions = 0 : i64, admitted = true
  }
  // L2: the propose-verify search accelerator -- a learned candidate ordering
  // certified to reproduce the exact optimum (zero mismatches) over the test set.
  bcir.kbcir.search_accel @accel0 {
    order = "learned", checked = 4 : i64, mismatches = 0 : i64, admitted = true
  }
  // The provenance manifest: the commit hash of this plan (score 7808). Manifest
  // equality => identical plan; replay reproduced it (the version-DAG spine).
  bcir.kbcir.provenance_manifest @manifest0 {
    digest = 7777777 : i64, score = 7808 : i64, n_artifacts = 0 : i64, reproduced = true
  }
  // The building-blocks engine: equality saturation extracted an equivalent form
  // no costlier than the original (a rewrite never raises the selected cost).
  bcir.egraph.extract @egraph0 {
    original_cost = 5 : i64, optimized_cost = 3 : i64, iterations = 3 : i64,
    enodes = 11 : i64, saturated = true
  }
  // The L1 cost throttle: the learned gate pays for itself within the L1 budget
  // (gain >= inference_cost); the hot path (L0) runs zero learned inference.
  bcir.kbcir.amortization @throttle_gate {
    component = "moe_gate", tier = "L1", inference_cost = 300 : i64, gain = 6000 : i64,
    budget = 100000 : i64
  }
  bcir.kbcir.amortization @throttle_hot {
    component = "lane_promote", tier = "L0", inference_cost = 0 : i64, gain = 0 : i64,
    budget = 0 : i64
  }
  // L3: the regret ledger (the boundary dashboard) -- the seeded portfolio is
  // hindsight-optimal on the standard episodes, so the books read zero and the
  // MDL evidence (data_fit 0 <= complexity) yields a principled "keep" verdict.
  bcir.kbcir.regret_ledger @regret_perf { rule = @perf, episodes = 3 : i64, total_regret = 0 : i64, worst_regret = 0 : i64, gen = 1 : i64, data_fit_milli = 0 : i64, complexity_milli = 549 : i64, verdict = "keep" }

  // ---- BCIR-4: GEM StreamPack (hydrated, prefetch + provenance) ----
  bcir.gem.stream_pack @sp0 attributes {
    source_plan = @plan0, topo_gen = 1 : i64, map_gen = 1 : i64, data_gen = 4 : i64
  } {
    bcir.gem.prefetch @pf0 { distance = 4 : i32, targets = [@A, @B], hint = "T0", pattern = "linear" }
    bcir.gem.block @blk0 { base = 0 : i64, count = 1024 : i64, strideA = 1 : i64, strideB = 1 : i64, strideD = 1 : i64 }
    bcir.gem.lane_segment @seg0 {
      claim = @add, phase = @p0, lane = #bcir.lane<u>, stride_k = 1 : i32, width = 16 : i32,
      opcode = "f32.add", reads = [@A, @B], writes = [@C], prefetch = @pf0, fence_before = [], fence_after = []
    }
    bcir.trace.note @add_trace { src_hash = 0 : i64, trace_hash = 0 : i64 }
  }

  // ---- M1: verifier obligations as IR ----
  bcir.verify.registry_symbols @vr_symbols { registry = @RES, resources = [@A, @B, @C], rids = array<i64: 10, 11, 12> }
  bcir.verify.plan_selection @vr_plan { plan = @plan0, claim = @add, candidates = [@add_cpu_u8, @add_cpu_u16], selected = @add_cpu_u16, semiring = #bcir.semiring<min_plus>, score = 7808 : i64 }
  bcir.verify.stream_provenance @vr_stream { stream_pack = @sp0, source_plan = @plan0, claims = [@add], segments = [@seg0], trace_notes = [@add_trace] }
  bcir.verify.generation_tags @vr_gen { stream_pack = @sp0, topo_gen = 1 : i64, map_gen = 1 : i64, data_gen = 4 : i64, required_map_gen = 1 : i64, required_data_gen = 4 : i64 }
  bcir.verify.policy_provenance @vr_policy { portfolio = @gains, certificates = [@perf_cert], calibrations = [@cal_cpu] }
}

// CHECK-LABEL: bcir.module @full_vec_add_ct1
// CHECK: bcir.target.capability @cpu
// CHECK: bcir.mem.tier @hbm
// CHECK: bcir.claim @add
// CHECK: bcir.kbcir.select @add
// CHECK: bcir.kbcir.soft_select @sel_soft
// CHECK: bcir.kbcir.budget @thermal_cap
// CHECK: bcir.kbcir.scheduled_price @overlap_price
// CHECK: bcir.gem.schedule @sched0
// CHECK: bcir.kbcir.calibration @cal_cpu
// CHECK: bcir.kbcir.calibration @cal_cpu_bayes
// CHECK: bcir.kbcir.portfolio @gains
// CHECK: bcir.kbcir.replay_certificate @perf_cert
// CHECK: bcir.kbcir.moe_gate @gate0
// CHECK: bcir.kbcir.search_accel @accel0
// CHECK: bcir.kbcir.provenance_manifest @manifest0
// CHECK: bcir.egraph.extract @egraph0
// CHECK: bcir.kbcir.amortization @throttle_gate
// CHECK: bcir.kbcir.regret_ledger @regret_perf
// CHECK: bcir.gem.stream_pack @sp0
// CHECK: bcir.verify.plan_selection @vr_plan
// CHECK: bcir.verify.policy_provenance @vr_policy
