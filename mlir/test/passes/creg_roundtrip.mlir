// RUN: bcir-opt %s | bcir-opt | FileCheck %s
//
// bcir.creg_read / bcir.creg_write parse -> print -> parse byte-identically (D1.3).

// CHECK-LABEL: func.func @creg_roundtrip
func.func @creg_roundtrip(%v: i64) -> i64 {
  // CHECK: %[[R:.*]] = bcir.creg_read <cr3> -> i64
  %r = bcir.creg_read <cr3> -> i64
  // CHECK: bcir.creg_write <cr4>, %arg0 : i64
  bcir.creg_write <cr4>, %v : i64
  // CHECK: return %[[R]] : i64
  return %r : i64
}
