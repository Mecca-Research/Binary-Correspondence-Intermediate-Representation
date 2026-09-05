// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// R11 over the per-resource generation vector (StreamPack v4, S0-2). The pack's header
// map_gen/data_gen are the registry's MAXIMA and nothing more: a resource that moved while
// another still held the maximum, and a resource declared after hydration, were invisible
// to them. The vector -- (rid, map_gen, data_gen) triples in rid order, taken at hydration --
// is what R11 reads now; the first case is a pack whose vector matches a registry whose
// generations differ per resource, and it is accepted.

bcir.module @vector_ok {
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa>, map_gen = 3 : i64, data_gen = 2 : i64 }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa>, map_gen = 1 : i64, data_gen = 0 : i64 }
  }
  bcir.kbcir.plan @plan0 {
    ^bb0:
  }
  bcir.gem.stream_pack @sp0 attributes {
    source_plan = @plan0, topo_gen = 1 : i64, map_gen = 3 : i64, data_gen = 2 : i64,
    generations = array<i64: 10, 3, 2, 11, 1, 0>
  } {
    bcir.gem.block @blk0 { base = 0 : i64, count = 4 : i64, strideA = 1 : i64, strideB = 1 : i64, strideD = 1 : i64 }
  }
}

// -----

// B moved from map_gen 1 to 2 after hydration; A still holds the maximum 3, so the header
// maxima agree with the registry -- the law the maxima could not see.
bcir.module @moved_under_the_maxima {
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa>, map_gen = 3 : i64, data_gen = 0 : i64 }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa>, map_gen = 2 : i64, data_gen = 0 : i64 }
  }
  bcir.kbcir.plan @plan0 {
    ^bb0:
  }
  // expected-error @+1 {{R11: stale StreamPack: resource @B (rid 11) map_gen 1 != registry 2 (rehydrate: repack)}}
  bcir.gem.stream_pack @sp0 attributes {
    source_plan = @plan0, topo_gen = 1 : i64, map_gen = 3 : i64, data_gen = 0 : i64,
    generations = array<i64: 10, 3, 0, 11, 1, 0>
  } {
    bcir.gem.block @blk0 { base = 0 : i64, count = 4 : i64, strideA = 1 : i64, strideB = 1 : i64, strideD = 1 : i64 }
  }
}

// -----

// A data generation that moved under the maximum: the plan may be invalid -- replan.
bcir.module @data_moved_under_the_maxima {
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa>, map_gen = 0 : i64, data_gen = 5 : i64 }
    bcir.resource @B { rid = 11 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa>, map_gen = 0 : i64, data_gen = 4 : i64 }
  }
  bcir.kbcir.plan @plan0 {
    ^bb0:
  }
  // expected-error @+1 {{R11: stale StreamPack: resource @B (rid 11) data_gen 1 != registry 4 (rehydrate: replan)}}
  bcir.gem.stream_pack @sp0 attributes {
    source_plan = @plan0, topo_gen = 1 : i64, map_gen = 0 : i64, data_gen = 5 : i64,
    generations = array<i64: 10, 0, 5, 11, 0, 1>
  } {
    bcir.gem.block @blk0 { base = 0 : i64, count = 4 : i64, strideA = 1 : i64, strideB = 1 : i64, strideD = 1 : i64 }
  }
}

// -----

// The vector names a resource the registry does not declare.
bcir.module @undeclared_rid {
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa>, map_gen = 1 : i64, data_gen = 0 : i64 }
  }
  bcir.kbcir.plan @plan0 {
    ^bb0:
  }
  // expected-error @+1 {{R11: stale StreamPack: generation vector names rid 99, which the registry does not declare (rehydrate: repack)}}
  bcir.gem.stream_pack @sp0 attributes {
    source_plan = @plan0, topo_gen = 1 : i64, map_gen = 1 : i64, data_gen = 0 : i64,
    generations = array<i64: 10, 1, 0, 99, 0, 0>
  } {
    bcir.gem.block @blk0 { base = 0 : i64, count = 4 : i64, strideA = 1 : i64, strideB = 1 : i64, strideD = 1 : i64 }
  }
}

// -----

// A resource declared after hydration has no entry: the pack does not know it exists.
bcir.module @declared_after_hydration {
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa>, map_gen = 1 : i64, data_gen = 0 : i64 }
    bcir.resource @C { rid = 12 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa>, map_gen = 0 : i64, data_gen = 0 : i64 }
  }
  bcir.kbcir.plan @plan0 {
    ^bb0:
  }
  // expected-error @+1 {{R11: stale StreamPack: resource @C (rid 12) was declared after hydration (no generation vector entry; rehydrate: repack)}}
  bcir.gem.stream_pack @sp0 attributes {
    source_plan = @plan0, topo_gen = 1 : i64, map_gen = 1 : i64, data_gen = 0 : i64,
    generations = array<i64: 10, 1, 0>
  } {
    bcir.gem.block @blk0 { base = 0 : i64, count = 4 : i64, strideA = 1 : i64, strideB = 1 : i64, strideD = 1 : i64 }
  }
}

// -----

// A pack with no vector in a scope that declares resources: a v1-v3 artifact (or one with
// its vector stripped) is refused, never judged by its maxima -- which agree here.
bcir.module @no_vector {
  bcir.registry @RES {
    bcir.resource @A { rid = 10 : i32, domain_kind = #bcir.domain<ram>, shape = array<i64: 4>, layout = #bcir.layout<soa>, map_gen = 1 : i64, data_gen = 0 : i64 }
  }
  bcir.kbcir.plan @plan0 {
    ^bb0:
  }
  // expected-error @+1 {{R11: stale StreamPack: no per-resource generation vector for a registry of 1 resource(s)}}
  bcir.gem.stream_pack @sp0 attributes {
    source_plan = @plan0, topo_gen = 1 : i64, map_gen = 1 : i64, data_gen = 0 : i64
  } {
    bcir.gem.block @blk0 { base = 0 : i64, count = 4 : i64, strideA = 1 : i64, strideB = 1 : i64, strideD = 1 : i64 }
  }
}

// -----

// The op verifier's shape rules (the codecs' predicate): rids strictly ascending ...
bcir.module @unsorted_vector {
  bcir.kbcir.plan @plan0 {
    ^bb0:
  }
  // expected-error @+1 {{generations: rids must be strictly ascending (rid 10 after 11)}}
  bcir.gem.stream_pack @sp0 attributes {
    source_plan = @plan0, topo_gen = 1 : i64, map_gen = 1 : i64, data_gen = 0 : i64,
    generations = array<i64: 11, 1, 0, 10, 0, 0>
  } {
    bcir.gem.block @blk0 { base = 0 : i64, count = 4 : i64, strideA = 1 : i64, strideB = 1 : i64, strideD = 1 : i64 }
  }
}

// -----

// ... the header maxima are the vector's ...
bcir.module @maxima_mismatch {
  bcir.kbcir.plan @plan0 {
    ^bb0:
  }
  // expected-error @+1 {{map_gen/data_gen (3, 0) must be the generation vector's maxima (1, 0)}}
  bcir.gem.stream_pack @sp0 attributes {
    source_plan = @plan0, topo_gen = 1 : i64, map_gen = 3 : i64, data_gen = 0 : i64,
    generations = array<i64: 10, 1, 0>
  } {
    bcir.gem.block @blk0 { base = 0 : i64, count = 4 : i64, strideA = 1 : i64, strideB = 1 : i64, strideD = 1 : i64 }
  }
}

// -----

// ... and the vector holds whole triples.
bcir.module @not_triples {
  bcir.kbcir.plan @plan0 {
    ^bb0:
  }
  // expected-error @+1 {{generations must hold (rid, map_gen, data_gen) triples (got 4 values)}}
  bcir.gem.stream_pack @sp0 attributes {
    source_plan = @plan0, topo_gen = 1 : i64, map_gen = 1 : i64, data_gen = 0 : i64,
    generations = array<i64: 10, 1, 0, 11>
  } {
    bcir.gem.block @blk0 { base = 0 : i64, count = 4 : i64, strideA = 1 : i64, strideB = 1 : i64, strideD = 1 : i64 }
  }
}
