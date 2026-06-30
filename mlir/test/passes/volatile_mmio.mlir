// RUN: bcir-opt -convert-bcir-to-llvm %s | FileCheck %s
//
// bcir.volatile_load / bcir.volatile_store -> llvm.inttoptr + a VOLATILE llvm.load / llvm.store
// (first-class MMIO on the law rail, SEG8.2 / D1.2). Mirrors the cfront MMIO emit
// `*(volatile T *)(intaddr)` (bcir/frontends/cfront/emit.py): the resolved integer register address is
// inttoptr'd to an opaque !llvm.ptr and the access is marked `volatile` so a device read/write is never
// elided, duplicated, or reordered with other volatile accesses (the MMIO guarantee).

// (1) a device-register READ: inttoptr the i64 address, then a VOLATILE load of the i32 register.
// CHECK-LABEL: func.func @mmio_read
func.func @mmio_read(%addr: i64) -> i32 {
  // CHECK: %[[P:.*]] = llvm.inttoptr %arg0 : i64 to !llvm.ptr
  // CHECK: %[[V:.*]] = llvm.load volatile %[[P]] : !llvm.ptr -> i32
  %v = bcir.volatile_load %addr : i64 -> i32
  // CHECK: return %[[V]] : i32
  return %v : i32
}

// (2) a device-register WRITE: inttoptr the address, then a VOLATILE store of the value (no result).
// CHECK-LABEL: func.func @mmio_write
func.func @mmio_write(%val: i32, %addr: i64) {
  // CHECK: %[[P:.*]] = llvm.inttoptr %arg1 : i64 to !llvm.ptr
  // CHECK: llvm.store volatile %arg0, %[[P]] : i32, !llvm.ptr
  bcir.volatile_store %val, %addr : i32, i64
  return
}

// (3) a narrow (i8) register read -- the natural width rides the result type.
// CHECK-LABEL: func.func @mmio_read_byte
func.func @mmio_read_byte(%addr: i64) -> i8 {
  // CHECK: llvm.load volatile %{{.*}} : !llvm.ptr -> i8
  %v = bcir.volatile_load %addr : i64 -> i8
  return %v : i8
}
