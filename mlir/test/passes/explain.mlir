// RUN: bcir-opt -bcir-explain %s | FileCheck %s
//
// -bcir-explain: the proof-carrying decision record (proof.explain) as IR annotations. One
// elementwise vector.add on an AVX-512 capability: the optimizer weighs three candidates
// (lane widths 1, 8, 16) and chooses width 16 for the pinned 7808. The record makes the
// rationale first-class -- the candidate widths it weighed (explain_widths), their scalarized
// costs (explain_candidates), the chosen width (explain_chosen) and edge score
// (explain_score) -- and the module its total (explain_total). The why behind -bcir-plan.

bcir.module @explain {
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
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// CHECK-LABEL: bcir.module @explain
// CHECK: kbcir.explain_total = 7808
// the per-claim decision record: the candidates weighed + the chosen width/score.
// CHECK: bcir.claim @c1 attributes
// CHECK-SAME: kbcir.explain_candidates = array<i64:
// CHECK-SAME: kbcir.explain_chosen = 16 : i64
// CHECK-SAME: kbcir.explain_score = 7808 : i64
// CHECK-SAME: kbcir.explain_widths = array<i64: 1, 8, 16>
