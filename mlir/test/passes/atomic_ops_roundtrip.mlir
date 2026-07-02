// RUN: bcir-opt %s | bcir-opt | FileCheck %s
//
// Round-trip: the first-class atomic ops (§5.14 Phase 2) parse and re-print stably -- the kind
// string, the optional #bcir.mem_ordering, and the CAS `weak` flag all survive a print->parse cycle.

// CHECK-LABEL: func.func @rt_rmw
func.func @rt_rmw(%addr: i64, %v: i32) -> i32 {
  // CHECK: bcir.atomic_rmw "add" %arg1, %arg0 : i32, i64 -> i32
  %a = bcir.atomic_rmw "add" %v, %addr : i32, i64 -> i32
  // CHECK: bcir.atomic_rmw "xor" %arg1, %arg0 {ordering = #bcir.mem_ordering<acq_rel>} : i32, i64 -> i32
  %b = bcir.atomic_rmw "xor" %v, %addr {ordering = #bcir.mem_ordering<acq_rel>} : i32, i64 -> i32
  return %b : i32
}

// CHECK-LABEL: func.func @rt_cas
func.func @rt_cas(%addr: i64, %e: i32, %d: i32) -> i32 {
  // CHECK: bcir.atomic_cas %arg1, %arg2, %arg0 : i32, i32, i64 -> i32
  %a = bcir.atomic_cas %e, %d, %addr : i32, i32, i64 -> i32
  // CHECK: bcir.atomic_cas weak %arg1, %arg2, %arg0 {ordering = #bcir.mem_ordering<release>} : i32, i32, i64 -> i32
  %b = bcir.atomic_cas weak %e, %d, %addr {ordering = #bcir.mem_ordering<release>} : i32, i32, i64 -> i32
  return %b : i32
}
