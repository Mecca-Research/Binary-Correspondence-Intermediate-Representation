// RUN: mlir-opt --irdl-file=%S/../../irdl/bcir.irdl.mlir %s | FileCheck %s
//
// The K_BCIR runtime-state op (kbcir.theta) + the compositional func/call/cond family
// (compose.Function/Call/Cond) in generic syntax, round-tripped through the IRDL
// projection on stock mlir-opt. These four ODS ops were previously missing from the
// projection (now 84/84 op coverage by name).

// CHECK: bcir.kbcir_theta
"bcir.kbcir_theta"() {sym_name = "theta", thermal = 50 : i32, power = 50 : i32,
  mem_pressure = 0 : i32, contention = 0 : i32} : () -> ()

// CHECK: bcir.kbcir_func
"bcir.kbcir_func"() ({}) {sym_name = "axpy", formals = ["A", "B"]} : () -> ()

// CHECK: bcir.kbcir_cond
"bcir.kbcir_cond"() ({}, {}) {pred = "hot", prob_then_milli = 500 : i64} : () -> ()

// CHECK: bcir.kbcir_call
"bcir.kbcir_call"() {callee = "axpy", formals = ["A"], actuals = ["X"]} : () -> ()
