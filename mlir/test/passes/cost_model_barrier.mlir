// RUN: bcir-opt -bcir-cost-model %s | FileCheck %s
//
// ASM3b: a `barriered`-hazard claim is a FIRST-CLASS ORDERING EDGE -- the producer->consumer
// deforestation discount must NOT cross it. This is the R13-parity twin of the oracle test
// bcir/tests/test_barrier_ordering.py (test_deforestation_skipped_when_consumer_is_barriered):
// BCIRCostModel.h::fusedColumns skips the x0.75 deforest factor when the consumer is Barriered
// (or a shared operand was produced by a Barriered producer), byte-for-byte with
// realize.fused_candidates. Same module as cost_model_fusion.mlir, but @c2 is `barriered`:
//   * @c1  (A+B -> T)          : first occurrence, full price                 -> 7808
//   * @c2  (T+C -> O) BARRIERED : consumes c1's T, but the ordering fence FORBIDS fusion ->
//                                NO deforestation credit (memory stays 3840), full price -> 7808
//                                (vs 5888 with the discount in cost_model_fusion.mlir)
//   * @dup (A+B -> E)          : duplicates c1's (op, operand-versions) => CSE -> 5100 (unchanged:
//                                CSE is a same-value match, not a cross-barrier fusion)

bcir.module @cost_model_barrier {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @T { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @O { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @E { rid = 15 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@T],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@T, @C], writes = [@O],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<barriered>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @dup attributes {
    claim_id = 3 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@E],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// CHECK-LABEL: bcir.module @cost_model_barrier
// c1: full price, no fusion credit (vec16 @ 7808, memory 3840).
// CHECK: bcir.claim @c1
// CHECK-SAME: kbcir.cm_min_cost = #bcir.costvec<compute = 64, memory = 3840
// CHECK-SAME: kbcir.cm_min_score = 7808
// c2: BARRIERED ordering edge -> deforestation SKIPPED. Full memory 3840, full score 7808 (the
// fence forbids the credit -- this is the ASM3b R13 parity vs cost_model_fusion.mlir's 2880/5888).
// CHECK: bcir.claim @c2
// CHECK-SAME: kbcir.cm_min_cost = #bcir.costvec<compute = 64, memory = 3840
// CHECK-SAME: kbcir.cm_min_score = 7808
// dup: CSE is a same-value match (not a cross-barrier fusion) -- still credited (3840 -> 2550).
// CHECK: bcir.claim @dup
// CHECK-SAME: kbcir.cm_fusion = "cse"
// CHECK-SAME: kbcir.cm_min_cost = #bcir.costvec<compute = 0, memory = 2550
// CHECK-SAME: kbcir.cm_min_score = 5100
//
// The ASM3b invariant, pinned globally: NO claim in this module carries a deforestation credit --
// @c2's producer->consumer fusion is FORBIDDEN by the barrier (in cost_model_fusion.mlir, the same
// @c2 without the barrier IS tagged kbcir.cm_fusion = "deforest"). Only CSE (a same-value match)
// survives. If the C++ guard regressed and re-applied the discount across the fence, this fails.
// CHECK-NOT: kbcir.cm_fusion = "deforest"
