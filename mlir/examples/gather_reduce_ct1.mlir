// RUN: bcir-opt %s | FileCheck %s
// RUN: bcir-opt -bcir-verify %s 2>&1 | FileCheck %s --check-prefix=VERIFY --allow-empty
//
// Phase-23+ parity: a reducible-permutation gather (`reduce.gather`) lowered two
// ways -- a blocked (sequential) realization and a gather (random) realization --
// with the K_BCIR min-plus selection picking the cheaper *blocked* form (the
// gather avoided, the gather_penalty priced out). Mirrors the oracle:
//   python -m bcir.run gather_reduce --bench-reduce   (docs/PARITY.md)
// The reduction writes a single accumulator, so R7 sizes its write extent at one
// element (verified clean below).

bcir.module @gather_reduce_ct1 attributes {
  version = "0.1", target_model = "registry-first",
  execution_model = "gem", cost_model = "k_bcir"
} {
  bcir.registry @RES {
    bcir.resource @TABLE { rid = 50 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa>, align = 64 : i32, access = #bcir.access<flat>, priority = 0 : i32, map_gen = 1 : i64, data_gen = 0 : i64 }
    bcir.resource @ACC { rid = 51 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1>, layout = #bcir.layout<soa>, align = 64 : i32, access = #bcir.access<flat>, priority = 0 : i32, map_gen = 1 : i64, data_gen = 0 : i64 }
  }

  bcir.phase @p0 { id = 0 : i32, deps = [] }
  // The access is a permutation, but + is commutative, so the claim is realizable
  // unit-stride (the blocked sum) -- that is its canonical lane; the gather is the
  // de-optimized alternative. A single accumulator write (R7: extent 1, not count).
  bcir.claim @reduce attributes {
    claim_id = 5000 : i32, phase = @p0, op = "reduce.gather",
    reads = [@TABLE], writes = [@ACC], offset = 0 : i64, count = 1024 : i64,
    lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>, stride_k = 1 : i32,
    domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }

  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.kbcir.plan @plan0 {
    // blocked: sequential access, no gather_penalty (the cost model's choice).
    bcir.kbcir.path @blocked {
      claim = @reduce, realization = "cpu.blocked", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 64, memory = 1024, fabric = 0, sync = 0, compile = 0, thermal = 0, power = 0, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    }
    // gather: random access pays gather_penalty (8x the streamed traffic here).
    bcir.kbcir.path @gather {
      claim = @reduce, realization = "cpu.gather", lane = #bcir.lane<ggg>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 64, memory = 8192, fabric = 0, sync = 0, compile = 0, thermal = 0, power = 0, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    }
    // min-plus selects blocked (2*64 + 2*1024 = 2176) over gather (16512).
    %sel = bcir.kbcir.select @reduce from [@blocked, @gather] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      selected = @blocked, score = 2176 : i64
    }
  }
}

// CHECK-LABEL: bcir.module @gather_reduce_ct1
// CHECK: bcir.claim @reduce
// CHECK: op = "reduce.gather"
// CHECK: bcir.kbcir.path @blocked
// CHECK: bcir.kbcir.path @gather
// CHECK: bcir.kbcir.select @reduce
// CHECK-SAME: selected = @blocked
// VERIFY-NOT: error
