// RUN: bcir-opt -bcir-cost-model %s | FileCheck %s
//
// G1 / S1-A (assessment row 7): the CSE credit requires the COMPLETE semantic identity and
// EXCLUDES the claim kinds that are never a common subexpression. The parent build keyed CSE on
// the op string and the read versions alone, and checked the barrier guard afterwards, so a
// duplicate over a different count or offset, an atomic, a barriered and a volatile claim all
// took the copy credit. BCIRCostModel.h::cseEligible / cseIdentity mirror realize.cse_eligible /
// cse_identity byte-for-byte (R13 parity; the oracle twin is bcir/tests/test_fusion.py).
//
//   * @c1  (A+B -> T, 1024)          : the seed, full price                       -> 7808
//   * @dup (A+B -> E, 1024)          : an exact duplicate => CSE (copy-priced)     -> 5100
//   * @cnt (A+B -> F, count 512)     : a different COUNT: not the same value       -> 3904 (its own full price)
//   * @off (A+B -> G, offset 8)      : a different OFFSET: not the same elements   -> 7808
//   * @atom (A+B -> H, atomic)       : atomic semantics are never a copy           -> 7808
//   * @vol (A+B -> I, volatile)      : a volatile access is never elided           -> 7808
//   * @v0 (C+D -> J, volatile)       : an INELIGIBLE claim never SEEDS a match ...
//   * @plain (C+D -> K)              : ... so the same value after it pays full    -> 7808
//   * @plain2 (C+D -> L)             : ... and only the eligible seed credits it   -> 5100
//   * @dyn (A+B -> M, dynamic)       : a dynamic bound is a different contract     -> 7808

bcir.module @cost_model_cse_neg {
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
    bcir.resource @E { rid = 13 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @F { rid = 14 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @G { rid = 15 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @H { rid = 16 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @I { rid = 17 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @C { rid = 18 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @D { rid = 19 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @J { rid = 20 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @K { rid = 21 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @L { rid = 22 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @M { rid = 23 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@T],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @dup attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@E],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @cnt attributes {
    claim_id = 3 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@F],
    count = 512 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32
  } { %i = bcir.index_range 0 to 512 step 1 }
  bcir.claim @off attributes {
    claim_id = 4 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@G],
    count = 1024 : i64, offset = 8 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @atom attributes {
    claim_id = 5 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@H],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<atomic>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @vol attributes {
    claim_id = 6 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@I],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<barriered>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32, is_volatile = true
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @v0 attributes {
    claim_id = 7 : i32, phase = @p0, op = "vector.add", reads = [@C, @D], writes = [@J],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<barriered>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32, is_volatile = true
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @plain attributes {
    claim_id = 8 : i32, phase = @p0, op = "vector.add", reads = [@C, @D], writes = [@K],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @plain2 attributes {
    claim_id = 9 : i32, phase = @p0, op = "vector.add", reads = [@C, @D], writes = [@L],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @dyn attributes {
    claim_id = 10 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@M],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>, opcode = 3 : i32, dynamic = true
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// CHECK-LABEL: bcir.module @cost_model_cse_neg
// c1: the seed pays full (vec16 @ 7808, memory 3840).
// CHECK: bcir.claim @c1
// CHECK-NOT: kbcir.cm_fusion
// CHECK: kbcir.cm_min_cost = #bcir.costvec<compute = 64, memory = 3840
// CHECK-SAME: kbcir.cm_min_score = 7808
// dup: the exact duplicate IS a copy (compute zeroed, memory 3840 -> 2550) -- the positive control.
// CHECK: bcir.claim @dup
// CHECK-SAME: kbcir.cm_fusion = "cse"
// CHECK-SAME: kbcir.cm_min_cost = #bcir.costvec<compute = 0, memory = 2550
// CHECK-SAME: kbcir.cm_min_score = 5100
// cnt: a different count is a different value -- its own full price, no credit.
// CHECK: bcir.claim @cnt
// CHECK-NOT: kbcir.cm_fusion
// CHECK: kbcir.cm_min_cost = #bcir.costvec<compute = 32, memory = 1920
// CHECK-SAME: kbcir.cm_min_score = 3904
// off: a different offset touches different elements -- full price, no credit.
// CHECK: bcir.claim @off
// CHECK-NOT: kbcir.cm_fusion
// CHECK: kbcir.cm_min_cost = #bcir.costvec<compute = 64, memory = 3840
// CHECK-SAME: kbcir.cm_min_score = 7808
// atom: atomic semantics are never a copy.
// CHECK: bcir.claim @atom
// CHECK-NOT: kbcir.cm_fusion
// CHECK: kbcir.cm_min_cost = #bcir.costvec<compute = 64, memory = 3840
// CHECK-SAME: kbcir.cm_min_score = 7808
// vol: a volatile access is never elided.
// CHECK: bcir.claim @vol
// CHECK-NOT: kbcir.cm_fusion
// CHECK: kbcir.cm_min_cost = #bcir.costvec<compute = 64, memory = 3840
// CHECK-SAME: kbcir.cm_min_score = 7808
// v0: an ineligible claim never seeds a match ...
// CHECK: bcir.claim @v0
// CHECK-NOT: kbcir.cm_fusion
// CHECK: kbcir.cm_min_score = 7808
// plain: ... so the same value after it pays full ...
// CHECK: bcir.claim @plain attributes
// CHECK-NOT: kbcir.cm_fusion
// CHECK: kbcir.cm_min_cost = #bcir.costvec<compute = 64, memory = 3840
// CHECK-SAME: kbcir.cm_min_score = 7808
// plain2: ... and the eligible seed credits its duplicate.
// CHECK: bcir.claim @plain2
// CHECK-SAME: kbcir.cm_fusion = "cse"
// CHECK-SAME: kbcir.cm_min_score = 5100
// dyn: a dynamic bound is a different contract -- no credit.
// CHECK: bcir.claim @dyn
// CHECK-NOT: kbcir.cm_fusion
// CHECK: kbcir.cm_min_score = 7808
