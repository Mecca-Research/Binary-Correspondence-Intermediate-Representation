// RUN: bcir-opt %s | bcir-opt | FileCheck %s
//
// The bcir.gem.autodiff op (B3 Phase 2a): the law-rail record of the autodiff oracle's CLOSED-SET,
// content-addressed forward DAG (bcir/kbcir/autodiff.py). Slice 5 proved differentiating any expression in
// the closed vocabulary {const,var,neg,add,sub,mul,div,dot,select} stays in the SAME set; this op serializes
// one such forward Tape (the hash-consed Node list) so it round-trips through bcir-opt and is checked by the
// closed-set verifier (closed vocabulary + per-op arity + strictly-earlier DAG index bounds). No lowering and
// no cost pass yet (Phase 2a is the op + verifier + round-trip; that is the next slice). The op auto-registers
// via the dialect .td; this round-trips it (parse -> print -> parse -> print) byte-identically.
//
// THE ENCODING (parallel per-node arrays, the node count = opcodes.size()): node i is described by the i-th
// entry of each array; operands reference STRICTLY EARLIER nodes (the Tape interns operands before users, so
// the array is already topological). opcode codes: 0=const 1=var 2=neg 3=add 4=sub 5=mul 6=div 7=dot 8=select.
// consts[i] = a const's integer value / a var's input slot / -1 for a non-leaf. The DAGs below are DUMPED from
// real Tapes (autodiff.Tape) -- a faithful node list, not hand-rolled.

bcir.module @autodiff {
  // (1) f(x,y) = x*y + x  -- the canonical small DAG: two `var` leaves, a `mul` (binary), an `add` (binary),
  // and the structural SHARING of x (node 0 feeds both the mul and the add -- one node referenced twice, the
  // content-addressing the whole B3 thesis rests on). Dumped from:
  //   t=Tape(); x=t.var('x'); y=t.var('y'); out=t.add(t.mul(x,y), x)
  // node 0 var x  | node 1 var y | node 2 mul(0,1) | node 3 add(2,0); output = 3.
  bcir.gem.autodiff @f_xy_plus_x { n_inputs = 2 : i64, opcodes = array<i64: 1, 1, 5, 3>,
                                   arities = array<i64: 0, 0, 2, 2>, arg_base = array<i64: 0, 0, 0, 2>,
                                   args = array<i64: 0, 1, 2, 0>, consts = array<i64: 0, 1, -1, -1>,
                                   output = 3 : i64 }

  // (2) a RICH DAG exercising EVERY closed-set op kind -- const, var, neg (arity 1), sub/div/add (arity 2),
  // the VARIADIC dot (arity 4 = 2 u-ids then 2 v-ids), and select (arity 3). Dumped from:
  //   t=Tape(); a=t.var('a'); b=t.var('b'); c=t.var('c'); na=t.neg(a); ab=t.sub(a,b); k=t.const(3);
  //   d=t.div(ab,k); dp=t.dot((a,b),(b,c)); sel=t.select(c,d,dp); out=t.add(sel,na)
  // node 0 var a | 1 var b | 2 var c | 3 neg(0) | 4 sub(0,1) | 5 const(3) | 6 div(4,5) |
  // 7 dot(0,1,1,2) | 8 select(2,6,7) | 9 add(8,3); output = 9. The const carries integer payload 3.
  bcir.gem.autodiff @rich { n_inputs = 3 : i64, opcodes = array<i64: 1, 1, 1, 2, 4, 0, 6, 7, 8, 3>,
                            arities = array<i64: 0, 0, 0, 1, 2, 0, 2, 4, 3, 2>,
                            arg_base = array<i64: 0, 0, 0, 0, 1, 3, 3, 5, 9, 12>,
                            args = array<i64: 0, 0, 1, 4, 5, 0, 1, 1, 2, 2, 6, 7, 8, 3>,
                            consts = array<i64: 0, 1, 2, -1, -1, 3, -1, -1, -1, -1>, output = 9 : i64 }
}

// (1) f(x,y) = x*y + x: the canonical shared-x DAG prints back identically. MLIR prints attrs alphabetically
// (arg_base, args, arities, consts, n_inputs, opcodes, output), so the CHECK-SAME order follows the printer.
// CHECK: bcir.gem.autodiff @f_xy_plus_x
// CHECK-SAME: arg_base = array<i64: 0, 0, 0, 2>
// CHECK-SAME: args = array<i64: 0, 1, 2, 0>
// CHECK-SAME: arities = array<i64: 0, 0, 2, 2>
// CHECK-SAME: consts = array<i64: 0, 1, -1, -1>
// CHECK-SAME: n_inputs = 2 : i64
// CHECK-SAME: opcodes = array<i64: 1, 1, 5, 3>
// CHECK-SAME: output = 3 : i64

// (2) the rich DAG (every op kind + the variadic dot) round-trips byte-identically (attrs alphabetical).
// CHECK: bcir.gem.autodiff @rich
// CHECK-SAME: arg_base = array<i64: 0, 0, 0, 0, 1, 3, 3, 5, 9, 12>
// CHECK-SAME: args = array<i64: 0, 0, 1, 4, 5, 0, 1, 1, 2, 2, 6, 7, 8, 3>
// CHECK-SAME: arities = array<i64: 0, 0, 0, 1, 2, 0, 2, 4, 3, 2>
// CHECK-SAME: consts = array<i64: 0, 1, 2, -1, -1, 3, -1, -1, -1, -1>
// CHECK-SAME: n_inputs = 3 : i64
// CHECK-SAME: opcodes = array<i64: 1, 1, 1, 2, 4, 0, 6, 7, 8, 3>
// CHECK-SAME: output = 9 : i64
