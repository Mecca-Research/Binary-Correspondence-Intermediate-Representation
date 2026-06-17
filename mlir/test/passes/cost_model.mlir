// RUN: bcir-opt -bcir-cost-model %s | FileCheck %s
//
// The K_BCIR cost algebra on the MLIR rail (C++23 port of bcir/kbcir/cost.py).
// -bcir-cost-model recomputes each claim's candidate cost vectors + the argmin score
// FROM FIRST PRINCIPLES -- the claim + the target.capability seeds + the constexpr
// memory-tier table -- so the law no longer depends on the emitter-baked path costs.
// It reproduces the oracle exactly across three op-classes:
//   * elementwise (vector.add)  -> vec16 @ 7808  (compute 64, memory 3840)
//   * gather      (histogram)   -> 528384        (op-class 0; gather_penalty 32)
//   * tile        (linalg.matmul) -> 126976      (op-class 2)

bcir.module @cost_model {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx2", "avx512f", "fma"],
    lane_widths = array<i64: 1, 8, 16>, cacheline = 64 : i32, gather_penalty = 32 : i32
    // mem_unit / base_overhead / thermal_density / power_density / per_op_heat /
    // elem_bytes default to the cost.py CPU seeds (1 / 4 / 64 / 64 / 1 / 4).
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @TABLE { rid = 30 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, access = #bcir.access<flat> }
    bcir.resource @OUT { rid = 31 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @A2 { rid = 40 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 128, 128>, layout = #bcir.layout<soa> }
    bcir.resource @B2 { rid = 41 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 128, 128>, layout = #bcir.layout<soa> }
    bcir.resource @C2 { rid = 42 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 128, 128>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @add attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @hist attributes {
    claim_id = 2 : i32, phase = @p0, op = "histogram.scatter", reads = [@TABLE], writes = [@OUT],
    count = 1024 : i64, lane = #bcir.lane<ggg>, stride_class = #bcir.stride_class<random>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @tile attributes {
    claim_id = 3 : i32, phase = @p0, op = "linalg.matmul", reads = [@A2, @B2], writes = [@C2],
    count = 16384 : i64, lane = #bcir.lane<t>, stride_class = #bcir.stride_class<tile>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 16384 step 1 }
}

// CHECK-LABEL: bcir.module @cost_model
// elementwise: 3 candidates (scalar/vec8/vec16); vec16 wins at 7808.
// CHECK-DAG: kbcir.cm_candidates = 3
// CHECK-DAG: kbcir.cm_min_cost = #bcir.costvec<compute = 64, memory = 3840, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088
// CHECK-DAG: kbcir.cm_min_score = 7808
// gather: op-class 0 (compute 0), gather_penalty 32 -> 528384.
// CHECK-DAG: kbcir.cm_min_score = 528384
// tile: op-class 2 (T_MACC) -> 126976.
// CHECK-DAG: kbcir.cm_min_score = 126976
