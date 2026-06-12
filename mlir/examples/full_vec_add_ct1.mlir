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
    gather_penalty = 8 : i32, affinity_domains = 16 : i32
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
    %A = bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, align = 64 : i32, access = #bcir.access<flat>, priority = 0 : i32, map_gen = 1 : i64, data_gen = 4 : i64 } : !bcir.resource
    %B = bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, align = 64 : i32, access = #bcir.access<flat>, priority = 0 : i32, map_gen = 1 : i64, data_gen = 4 : i64 } : !bcir.resource
    %C = bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<hbm>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, align = 64 : i32, access = #bcir.access<flat>, priority = 80 : i32, map_gen = 1 : i64, data_gen = 0 : i64 } : !bcir.resource
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
    %u8 = bcir.kbcir.path @add_cpu_u8 {
      claim = @add, realization = "cpu.vector.u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 128, memory = 4608, fabric = 0, sync = 0, compile = 0, thermal = 640, power = 640, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    } : !bcir.path
    %u16 = bcir.kbcir.path @add_cpu_u16 {
      claim = @add, realization = "cpu.vector.u16", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 64, memory = 3840, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    } : !bcir.path
    // Cool Theta selects vec16 (score 7808); a hot-Theta replan picks vec8.
    %sel = bcir.kbcir.select @add from [@add_cpu_u8, @add_cpu_u16] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      selected = @add_cpu_u16, score = 7808 : i64
    } : !bcir.path
    // Constrained (RCSP) rail: a 700 thermal/power cap makes vec16 (1088)
    // infeasible -- the budgeted optimum is vec8 at score 9472 (the oracle:
    // optimize_constrained(..., Budget.of(thermal=700)); a point no PERF
    // weight vector can select, since PERF's thermal weight is 0).
    %selc = bcir.kbcir.select @add from [@add_cpu_u8, @add_cpu_u16] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      budget = @thermal_cap, selected = @add_cpu_u8, score = 9472 : i64
    } : !bcir.path
  }
  bcir.kbcir.budget @thermal_cap { dims = ["thermal", "power"], caps = array<i64: 700, 700> }
  // M(pi,Theta): the (max,+) wave-overlap price. One claim => the degenerate
  // case: makespan == serial Sigma score, overlap gain 0 (gem.overlap oracle).
  bcir.kbcir.scheduled_price @overlap_price { plan = @plan0, makespan = 7808 : i64, serial = 7808 : i64, overlap_gain = 0 : i64 }

  // ---- BCIR-4: GEM StreamPack (hydrated, prefetch + provenance) ----
  %sp = bcir.gem.stream_pack @sp0 attributes {
    source_plan = @plan0, topo_gen = 1 : i64, map_gen = 1 : i64, data_gen = 4 : i64
  } {
    bcir.gem.prefetch @pf0 { distance = 4 : i32, targets = [@A, @B], hint = "T0", pattern = "linear" }
    bcir.gem.block @blk0 { base = 0 : i64, count = 1024 : i64, strideA = 1 : i64, strideB = 1 : i64, strideD = 1 : i64 }
    bcir.gem.lane_segment @seg0 {
      claim = @add, phase = @p0, lane = #bcir.lane<u>, stride_k = 1 : i32, width = 16 : i32,
      opcode = "f32.add", reads = [@A, @B], writes = [@C], prefetch = @pf0, fence_before = [], fence_after = []
    }
    bcir.trace.note @add_trace { src_hash = 0 : i64, trace_hash = 0 : i64 }
  } : !bcir.stream

  // ---- M1: verifier obligations as IR ----
  bcir.verify.registry_symbols @vr_symbols { registry = @RES, resources = [@A, @B, @C], rids = array<i64: 10, 11, 12> }
  bcir.verify.plan_selection @vr_plan { plan = @plan0, claim = @add, candidates = [@add_cpu_u8, @add_cpu_u16], selected = @add_cpu_u16, semiring = #bcir.semiring<min_plus>, score = 7808 : i64 }
  bcir.verify.stream_provenance @vr_stream { stream_pack = @sp0, source_plan = @plan0, claims = [@add], segments = [@seg0], trace_notes = [@add_trace] }
  bcir.verify.generation_tags @vr_gen { stream_pack = @sp0, topo_gen = 1 : i64, map_gen = 1 : i64, data_gen = 4 : i64, required_map_gen = 1 : i64, required_data_gen = 4 : i64 }
}

// CHECK-LABEL: bcir.module @full_vec_add_ct1
// CHECK: bcir.target.capability @cpu
// CHECK: bcir.mem.tier @hbm
// CHECK: bcir.claim @add
// CHECK: bcir.kbcir.select @add
// CHECK: bcir.kbcir.budget @thermal_cap
// CHECK: bcir.kbcir.scheduled_price @overlap_price
// CHECK: bcir.gem.stream_pack @sp0
// CHECK: bcir.verify.plan_selection @vr_plan
