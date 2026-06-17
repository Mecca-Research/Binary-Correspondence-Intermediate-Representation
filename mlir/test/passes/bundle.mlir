// RUN: bcir-opt -bcir-bundle %s | FileCheck %s
//
// -bcir-bundle: the law-rail analysis of kbcir.bundle. It detects clusters of mutually-
// independent same-phase claims that share a read operand -- the bundles whose joint
// intra-phase reorder can recover a fusion discount the pairwise plan misses. Here c1 and
// c3 both read @r1 (and write disjoint outputs, so they are independent), while c2 shares
// nothing -- so the pass finds exactly one bundle {c1, c3} on @r1 and leaves c2 unbundled.
// Read-only: it annotates the structure, it does not reorder or re-price (that is next).

bcir.module @bundle {
  bcir.registry @RES {
    bcir.resource @r1 { rid = 1 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r2 { rid = 2 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r3 { rid = 3 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r4 { rid = 4 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r5 { rid = 5 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r6 { rid = 6 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r7 { rid = 7 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
    bcir.resource @r8 { rid = 8 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 1024>, layout = #bcir.layout<soa> }
  }
  bcir.phase @p0 { id = 0 : i32, deps = [] }
  bcir.claim @c1 attributes {
    claim_id = 1 : i32, phase = @p0, op = "vector.add", reads = [@r1, @r2], writes = [@r3],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c2 attributes {
    claim_id = 2 : i32, phase = @p0, op = "vector.add", reads = [@r4, @r5], writes = [@r6],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
  bcir.claim @c3 attributes {
    claim_id = 3 : i32, phase = @p0, op = "vector.add", reads = [@r1, @r7], writes = [@r8],
    count = 1024 : i64, lane = #bcir.lane<u>, stride_class = #bcir.stride_class<unit>,
    stride_k = 1 : i32, domain = #bcir.domain<ram>, hazard = #bcir.hazard<unique>,
    verify = #bcir.verify<bounds>, bounds = #bcir.bounds<strict>
  } { %i = bcir.index_range 0 to 1024 step 1 }
}

// CHECK-LABEL: bcir.module @bundle
// exactly one bundle detected.
// CHECK: kbcir.bundle_count = 1
// c1 and c3 are the bundle (shared read @r1); one CHECK-DAG per claim (non-overlapping).
// CHECK-DAG: bcir.claim @c1 attributes {{.*}}kbcir.bundle = 0 : i64, kbcir.bundle_shared = @r1
// CHECK-DAG: bcir.claim @c3 attributes {{.*}}kbcir.bundle = 0 : i64, kbcir.bundle_shared = @r1
