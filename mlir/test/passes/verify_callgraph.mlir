// RUN: bcir-opt -bcir-verify -verify-diagnostics -split-input-file %s
//
// R18 (compositional call-graph integrity): every kbcir.call must resolve to a kbcir.func,
// and the call graph must be acyclic -- no recursion. This is the law-rail form of
// compose.plan_composite's two rejections: an undefined callee (KeyError) and a recursive
// call (RecursionError, for bounded compile time) over the kbcir.func/call/cond op family.

// A well-formed acyclic call graph: @caller -> @leaf, and @leaf calls nothing. Every callee
// resolves and the graph terminates -- no diagnostic.
bcir.module @r18_ok {
  bcir.kbcir.func @leaf attributes { formals = [@f1, @f2, @f3] } {
  }
  bcir.kbcir.func @caller {
    bcir.kbcir.call @leaf { formals = [@f1, @f2, @f3], actuals = [@a, @b, @c] }
  }
}

// -----

// A call to a function that is not defined anywhere -- rejected (callee does not resolve).
bcir.module @r18_undefined {
  bcir.kbcir.func @caller {
    // expected-error @+1 {{R18: call to undefined function @ghost}}
    bcir.kbcir.call @ghost { actuals = [@a] }
  }
}

// -----

// Direct recursion: @rec calls itself -- rejected (unbounded compile time).
bcir.module @r18_self_recursion {
  // expected-error @+1 {{R18: recursive call cycle through @rec}}
  bcir.kbcir.func @rec {
    bcir.kbcir.call @rec { }
  }
}

// -----

// Mutual recursion through a kbcir.cond branch: @ping -> @pong -> @ping -- rejected (the
// call inside the cond still counts as an edge of its enclosing func).
bcir.module @r18_mutual_recursion {
  // expected-error @+1 {{R18: recursive call cycle through @ping}}
  bcir.kbcir.func @ping {
    bcir.kbcir.cond "hot" then {
      bcir.kbcir.call @pong { }
    } else {
      bcir.kbcir.call @pong { }
    }
  }
  bcir.kbcir.func @pong {
    bcir.kbcir.call @ping { }
  }
}
