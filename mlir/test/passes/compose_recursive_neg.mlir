// RUN: bcir-opt -bcir-compose -verify-diagnostics %s
//
// -bcir-verify normally rejects this through R18.  The compose pass is also a
// public standalone pipeline and must fail deterministically rather than
// recursively pricing or traversing effects until stack exhaustion.

// expected-error @+1 {{bcir-compose: recursive call graph or nesting deeper than 64 is unsupported}}
bcir.module @recursive_compose {
  bcir.target.capability @cpu { triple = "x86_64", isa_features = [], lane_widths = array<i64: 1> }
  bcir.kbcir.policy @perf { mode = #bcir.policy_mode<latency>, weights = array<i64: 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0> }
  bcir.kbcir.func @loop {
    bcir.kbcir.call @loop { }
  }
}
