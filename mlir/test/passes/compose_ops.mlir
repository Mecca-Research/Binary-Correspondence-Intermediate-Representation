// RUN: bcir-opt %s | FileCheck %s
//
// The compositional func / if op family round-trips (parse -> print -> parse). It is the
// law-rail vocabulary for compose.py's region tree: a named kbcir.func with formal
// resources, an inter-procedural kbcir.call that remaps actuals onto those formals, and a
// kbcir.cond control-flow region with then/else branches and a probability weight. The
// compositional cost stays the oracle's; this is the IR these programs lower through.

bcir.module @compose {
  // define once: a function over three formals, whose body calls another function.
  bcir.kbcir.func @ew attributes { formals = [@f1, @f2, @f3] } {
    bcir.kbcir.call @inner { formals = [@f1, @f2], actuals = [@f1, @f2] }
  }
  // control flow: the hot branch calls @ew once, the cold branch twice -- one branch runs,
  // so the bound is the worst case and the expected cost weights by 600/1000.
  bcir.kbcir.cond "hot" attributes { prob_then_milli = 600 : i64 } then {
    bcir.kbcir.call @ew { formals = [@f1, @f2, @f3], actuals = [@a, @b, @c] }
  } else {
    bcir.kbcir.call @ew { formals = [@f1, @f2, @f3], actuals = [@x, @y, @z] }
    bcir.kbcir.call @ew { formals = [@f1, @f2, @f3], actuals = [@x, @y, @z] }
  }
}

// CHECK-LABEL: bcir.module @compose
// CHECK: bcir.kbcir.func @ew
// CHECK-SAME: formals = [@f1, @f2, @f3]
// CHECK: bcir.kbcir.call @inner
// CHECK: bcir.kbcir.cond "hot"
// CHECK-SAME: prob_then_milli = 600
// CHECK: bcir.kbcir.call @ew
