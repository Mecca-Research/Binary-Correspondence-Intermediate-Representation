// RUN: bcir-opt -bcir-replay %s | FileCheck %s
//
// -bcir-replay ports bcir/kbcir/proof.replay: a recheck of the proof-carrying decision record a
// deployed plan carries. It recomputes a fresh plan from the IR (the cost machinery of -bcir-plan
// / -bcir-explain) and diffs it against the declared kbcir.explain_* record -- the module total and,
// per claim, the chosen width + edge score. @good carries the faithful record (-bcir-explain's
// 7808 / w16 / 7808) -> it reproduces. @tampered's claim declares a wrong edge score (9999) -> the
// replay flags exactly which field diverged. The proof a shipped plan is bit-for-bit re-derivable.

// CHECK-LABEL: bcir.module @good
// CHECK-SAME: kbcir.replay_reproduced = true
bcir.module @good attributes { kbcir.explain_total = 7808 : i64 } {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    %a = bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %b = bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %c = bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    kbcir.explain_chosen = 16 : i64, kbcir.explain_score = 7808 : i64
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// @tampered: the claim's recorded edge score (9999) does not match the fresh plan (7808).
// CHECK-LABEL: bcir.module @tampered
// CHECK-SAME: kbcir.replay_mismatches = ["claim 1: replay (w16/7808) != recorded (w16/9999)"]
// CHECK-SAME: kbcir.replay_reproduced = false
bcir.module @tampered attributes { kbcir.explain_total = 7808 : i64 } {
  bcir.target.capability @cpu {
    triple = "x86_64-avx512", isa_features = ["avx512f"], lane_widths = array<i64: 1, 8, 16>
  }
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.registry @RES {
    %a = bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %b = bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
    %c = bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> } : !bcir.resource
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@A, @B], writes = [@C],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>,
    kbcir.explain_chosen = 16 : i64, kbcir.explain_score = 9999 : i64
  } { %i = bcir.index_range 0 to 1024 step 1 }
}
