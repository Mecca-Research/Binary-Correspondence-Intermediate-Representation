// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// The bcir.gem.autodiff op verifier (hasVerifier; the CLOSED-SET LAW, mirrors bcir/kbcir/autodiff.py:
// CLOSED_SET / _ARITY -- the op-level structural well-formedness check the sibling gem ops validate inline,
// NOT a new globally-numbered R-law). Each section carries a SINGLE violation (mirrors the
// lower_gem_activation_neg.mlir / lower_gem_conv_neg.mlir style). The three HEADLINE laws -- the slice's core
// value -- are the three closed-set guards: a FOREIGN opcode, a WRONG-ARITY node, and a FORWARD/out-of-range
// operand index. Each base DAG is the canonical f(x,y)=x*y+x with exactly one field corrupted.

// (1 -- CLOSED VOCABULARY) a FOREIGN opcode (code 9 is outside {const..select} = codes 0..8): rejected. This is
// the law the closure proof underwrites -- the serialized DAG may use ONLY the closed vocabulary.
bcir.module @foreign_opcode {
  // expected-error @+1 {{autodiff: node 2 opcode 15 is not in the closed set {const,var,neg,add,sub,mul,div,dot,select,exp,log,sqrt,tanh,sin,cos} (codes 0..14)}}
  bcir.gem.autodiff @bad { n_inputs = 2 : i64, opcodes = array<i64: 1, 1, 15, 3>,
                           arities = array<i64: 0, 0, 2, 2>, arg_base = array<i64: 0, 0, 0, 2>,
                           args = array<i64: 0, 1, 2, 0>, consts = array<i64: 0, 1, -1, -1>,
                           output = 3 : i64 }
}

// -----

// (2 -- ARITY) a wrong-arity node: node 2 is `mul` (code 5, _ARITY 2) but declares arity 1 -- rejected (the
// per-op arity must mirror autodiff._ARITY exactly). The args/arg_base are made consistent with the bad arity
// so this is purely the arity law that trips.
bcir.module @wrong_arity {
  // expected-error @+1 {{autodiff: node 2 (mul) arity must be 2 (mirrors autodiff._ARITY), got 1}}
  bcir.gem.autodiff @bad { n_inputs = 2 : i64, opcodes = array<i64: 1, 1, 5, 3>,
                           arities = array<i64: 0, 0, 1, 2>, arg_base = array<i64: 0, 0, 0, 1>,
                           args = array<i64: 0, 2, 0>, consts = array<i64: 0, 1, -1, -1>,
                           output = 3 : i64 }
}

// -----

// (3 -- DAG / INDEX BOUNDS) a FORWARD operand reference: node 2 (mul) reads operand 3 -- a node that does NOT
// come strictly earlier (3 >= 2) -- rejected. Every operand index must reference a strictly earlier node
// (< the node's own position), which is what guarantees acyclicity + topological order.
bcir.module @forward_ref {
  // expected-error @+1 {{autodiff: node 2 (mul) operand #1 = 3 must reference a strictly earlier node (in [0, 2)) -- forward/out-of-range refs break acyclicity/topological order}}
  bcir.gem.autodiff @bad { n_inputs = 2 : i64, opcodes = array<i64: 1, 1, 5, 3>,
                           arities = array<i64: 0, 0, 2, 2>, arg_base = array<i64: 0, 0, 0, 2>,
                           args = array<i64: 0, 3, 2, 0>, consts = array<i64: 0, 1, -1, -1>,
                           output = 3 : i64 }
}

// -----

// (3b -- INDEX BOUNDS, the var input slot) a `var` leaf whose input slot is out of [0, n_inputs): node 1 is a
// var with slot 5 but n_inputs = 2 -- rejected (a var must reference a real function input).
bcir.module @bad_var_slot {
  // expected-error @+1 {{autodiff: node 1 (var) input slot 5 must be in [0, n_inputs 2)}}
  bcir.gem.autodiff @bad { n_inputs = 2 : i64, opcodes = array<i64: 1, 1, 5, 3>,
                           arities = array<i64: 0, 0, 2, 2>, arg_base = array<i64: 0, 0, 0, 2>,
                           args = array<i64: 0, 1, 2, 0>, consts = array<i64: 0, 5, -1, -1>,
                           output = 3 : i64 }
}

// -----

// (3c -- INDEX BOUNDS, the output) an out-of-range output node index: output 7 but the DAG has 4 nodes -- rejected.
bcir.module @bad_output {
  // expected-error @+1 {{autodiff: output node index 7 must be in [0, node_count 4)}}
  bcir.gem.autodiff @bad { n_inputs = 2 : i64, opcodes = array<i64: 1, 1, 5, 3>,
                           arities = array<i64: 0, 0, 2, 2>, arg_base = array<i64: 0, 0, 0, 2>,
                           args = array<i64: 0, 1, 2, 0>, consts = array<i64: 0, 1, -1, -1>,
                           output = 7 : i64 }
}

// -----

// (4 -- PAYLOAD CONSISTENCY) a non-leaf op carrying a stray const payload: node 2 (mul) must carry the -1
// sentinel (it has no leaf payload), but declares 4 -- rejected.
bcir.module @stray_payload {
  // expected-error @+1 {{autodiff: node 2 (mul) is a non-leaf op and must carry the -1 const sentinel, got 4}}
  bcir.gem.autodiff @bad { n_inputs = 2 : i64, opcodes = array<i64: 1, 1, 5, 3>,
                           arities = array<i64: 0, 0, 2, 2>, arg_base = array<i64: 0, 0, 0, 2>,
                           args = array<i64: 0, 1, 2, 0>, consts = array<i64: 0, 1, 4, -1>,
                           output = 3 : i64 }
}

// -----

// (5 -- DAG / INDEX BOUNDS, CONTIGUITY) a NON-CONTIGUOUS arg_base: node 2 starts at base 1 (not the prefix-sum
// 0), leaving an UNREAD HOLE at args[0]. sum(arities) still == len(args), so the cardinality check passes -- but
// a hole could smuggle an out-of-range / forward operand index past the in-slice bounds + strictly-earlier
// checks (a real soundness hole the prefix-sum contract closes). Every faithful Tape serialization emits
// operands node-by-node, so arg_base IS the prefix-sum; this non-contiguous encoding is rejected.
bcir.module @noncontiguous_arg_base {
  // expected-error @+1 {{autodiff: node 2 arg_base 1 must equal the running prefix-sum of arities 0 (the per-node operand slices must contiguously partition args -- no overlap, no gap)}}
  bcir.gem.autodiff @bad { n_inputs = 2 : i64, opcodes = array<i64: 1, 1, 5, 3>,
                           arities = array<i64: 0, 0, 2, 2>, arg_base = array<i64: 0, 0, 1, 2>,
                           args = array<i64: 99, 0, 1, 0>, consts = array<i64: 0, 1, -1, -1>,
                           output = 3 : i64 }
}
