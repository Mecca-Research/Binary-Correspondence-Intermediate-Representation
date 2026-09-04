// RUN: bcir-opt -bcir-select-realization %s | FileCheck %s
//
// S0-5 (module scope) on the selection pass: two modules each declare a path named @p for
// their own claim, with DIFFERENT costs, and each selects it. Before this fixture the pass
// built `pathByName` with a root-global walk, so the second module's @p overwrote the
// first's and @a's selection was priced with @b's cost -- a false "computed score 7808 !=
// declared score 9472". Each module now prices its own path (the one scope predicate of
// BCIRPassSupport.h, shared with -bcir-verify and the GEM passes).
//
// The scores are the oracle's on the canonical vector-add fixture (docs/PARITY.md): under
// the latency policy weights <2,2,1,1,0,0,0,1,0,0,1,1>, the u8 path prices 2*128 + 2*4608 =
// 9472 and the u16 path 2*64 + 2*3840 = 7808.
//
// CHECK-LABEL: bcir.module @a
// CHECK: kbcir.computed_score = 9472
// CHECK-LABEL: bcir.module @b
// CHECK: kbcir.computed_score = 7808

bcir.module @a {
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.kbcir.plan @plan0 {
    bcir.kbcir.path @p {
      claim = @c, realization = "cpu.vector.u8", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 128, memory = 4608, fabric = 0, sync = 0, compile = 0, thermal = 640, power = 640, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    }
    %sel = bcir.kbcir.select @c from [@p] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      selected = @p, score = 9472 : i64
    } : !bcir.path
  }
}

bcir.module @b {
  bcir.kbcir.policy @perf {
    mode = #bcir.policy_mode<latency>,
    weights = array<i64: 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1>
  }
  bcir.kbcir.plan @plan0 {
    bcir.kbcir.path @p {
      claim = @c, realization = "cpu.vector.u16", lane = #bcir.lane<u>, layout = #bcir.layout<soa>,
      cost = #bcir.costvec<compute = 64, memory = 3840, fabric = 0, sync = 0, compile = 0, thermal = 1088, power = 1088, reliability = 0, security = 0, accuracy = 0, contention = 0, verification = 0>
    }
    %sel = bcir.kbcir.select @c from [@p] {
      policy = #bcir.policy_mode<latency>, semiring = #bcir.semiring<min_plus>,
      selected = @p, score = 7808 : i64
    } : !bcir.path
  }
}
