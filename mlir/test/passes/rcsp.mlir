// RUN: bcir-opt -bcir-rcsp %s | FileCheck %s
//
// The constrained-selection (RCSP) pass: the deterministic optimizer core ported
// from bcir/kbcir/rcsp.py to C++ (C++23). Over a claim's candidate paths it runs
// the budget-feasible min-plus argmin via label dominance and the Pareto front over
// (score, thermal, power) -- reproducing the oracle's worked example: the
// unconstrained optimum vec16 @ 7808, the 700-capped optimum vec8 @ 9472 (a point no
// PERF weight vector can select), and the {vec16, vec8} Pareto front of size 2
// (scalar is dominated: same thermal/power, strictly worse score).

bcir.module @rcsp_pipeline {
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }

  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @add attributes {
    claim_id = 1000 : i32, phase = @p0, op = "vector.add",
    reads = [@A, @B], writes = [@C], count = 1024 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }

  bcir.kbcir.plan @plan0 {
    bcir.kbcir.path @add_cpu_s1 {
      claim = @add, realization = "cpu.scalar", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 1024, memory = 15360, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    }
    bcir.kbcir.path @add_cpu_u8 {
      claim = @add, realization = "cpu.vector.u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 128, memory = 4608, fabric = 0, sync = 0, compile = 0, thermal = 640, power = 640, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    }
    bcir.kbcir.path @add_cpu_u16 {
      claim = @add, realization = "cpu.vector.u16", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 64, memory = 3840, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    }
    // Unconstrained: the min-plus argmin is vec16 @ 7808; the front is {vec16, vec8}.
    %sel = bcir.kbcir.select @add from [@add_cpu_s1, @add_cpu_u8, @add_cpu_u16] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      selected = @add_cpu_u16, score = 7808 : i64
    } : !bcir.path
    // Constrained: a 700 thermal/power cap makes scalar + vec16 (1088) infeasible; the
    // budgeted optimum is vec8 @ 9472.
    %selc = bcir.kbcir.select @add from [@add_cpu_s1, @add_cpu_u8, @add_cpu_u16] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      budget = @thermal_cap, selected = @add_cpu_u8, score = 9472 : i64
    } : !bcir.path
  }
  bcir.kbcir.budget @thermal_cap { dims = ["thermal", "power"], caps = array<i64: 700, 700> }
}

// CHECK-LABEL: bcir.module @rcsp_pipeline
// unconstrained: vec16 @ 7808, Pareto front {vec16, vec8} (scalar dominated).
// CHECK-DAG: kbcir.computed_selected = @add_cpu_u16
// CHECK-DAG: kbcir.computed_score = 7808
// constrained: vec8 @ 9472 under the 700 cap.
// CHECK-DAG: kbcir.computed_selected = @add_cpu_u8
// CHECK-DAG: kbcir.computed_score = 9472
// both selects price the same 2-point Pareto front.
// CHECK-DAG: kbcir.pareto_size = 2
// CHECK-DAG: kbcir.pareto_size = 2
