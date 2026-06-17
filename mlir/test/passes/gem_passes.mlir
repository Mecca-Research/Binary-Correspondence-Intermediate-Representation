// RUN: bcir-opt -bcir-classify-lanes -bcir-select-realization -bcir-batch -bcir-schedule -bcir-lower-to-llvm %s | FileCheck %s
//
// The GEM pipeline passes (LangRef Milestone 4..7): classify -> select -> batch
// -> schedule -> lower, MLIR-native, cross-checked against the bcir/ oracle
// (docs/PARITY.md). On the canonical vector-add plan the min-plus selection
// recomputes the oracle score exactly (7808 cool; 9472 under the thermal cap).

bcir.module @gem_pipeline {
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
    bcir.kbcir.path @add_cpu_u8 {
      claim = @add, realization = "cpu.vector.u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 128, memory = 4608, fabric = 0, sync = 0, compile = 0, thermal = 640, power = 640, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    }
    bcir.kbcir.path @add_cpu_u16 {
      claim = @add, realization = "cpu.vector.u16", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 64, memory = 3840, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    }
    // Cool Theta (no thermal weight): the min-plus argmin is vec16 at 7808.
    %sel = bcir.kbcir.select @add from [@add_cpu_u8, @add_cpu_u16] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      selected = @add_cpu_u16, score = 7808 : i64
    }
    // Constrained rail: a 700 thermal/power cap makes vec16 infeasible; the
    // budgeted argmin is vec8 at 9472 (a point no PERF weight selects).
    %selc = bcir.kbcir.select @add from [@add_cpu_u8, @add_cpu_u16] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      budget = @thermal_cap, selected = @add_cpu_u8, score = 9472 : i64
    }
  }
  bcir.kbcir.budget @thermal_cap { dims = ["thermal", "power"], caps = array<i64: 700, 700> }

  bcir.gem.lane_segment @seg0 {
    claim = @add, phase = @p0, lane = #bcir.lane<u>, stride_k = 1 : i32, width = 16 : i32,
    opcode = "f32.add", reads = [@A, @B], writes = [@C], fence_before = [], fence_after = []
  }
}

// classify: the unit-stride claim streams on the U lane.
// CHECK-DAG: kbcir.classified_lane = "u"
// batch + schedule: a single claim is batch 0, exec order 0.
// CHECK-DAG: kbcir.batch = 0
// CHECK-DAG: kbcir.exec_order = 0
// select-realization: the recomputed min-plus optimum -- the oracle's 7808/9472.
// CHECK-DAG: kbcir.computed_score = 7808
// CHECK-DAG: kbcir.computed_selected = @add_cpu_u16
// CHECK-DAG: kbcir.computed_score = 9472
// CHECK-DAG: kbcir.computed_selected = @add_cpu_u8
// lower: the segment preserves the claim's lane + resource set (R12).
// CHECK-DAG: kbcir.lowered = true
