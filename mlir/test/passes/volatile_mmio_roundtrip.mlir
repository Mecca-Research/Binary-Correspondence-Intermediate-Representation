// RUN: bcir-opt %s | bcir-opt | FileCheck %s
//
// bcir.volatile_load / bcir.volatile_store parse -> print -> parse byte-identically (first-class MMIO
// on the law rail, SEG8.2 / D1.2).

// CHECK-LABEL: func.func @mmio_roundtrip
func.func @mmio_roundtrip(%val: i32, %addr: i64) -> i32 {
  // CHECK: %[[V:.*]] = bcir.volatile_load %arg1 : i64 -> i32
  %v = bcir.volatile_load %addr : i64 -> i32
  // CHECK: bcir.volatile_store %arg0, %arg1 : i32, i64
  bcir.volatile_store %val, %addr : i32, i64
  // CHECK: return %[[V]] : i32
  return %v : i32
}

// CHECK-LABEL: func.func @mmio_roundtrip_byte
func.func @mmio_roundtrip_byte(%val: i8, %addr: i64) {
  // CHECK: bcir.volatile_store %arg0, %arg1 : i8, i64
  bcir.volatile_store %val, %addr : i8, i64
  return
}
