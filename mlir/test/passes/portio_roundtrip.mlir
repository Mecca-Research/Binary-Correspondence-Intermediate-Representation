// RUN: bcir-opt %s | bcir-opt | FileCheck %s
//
// The bcir.portio op (ASM2, SEG8.2): the law-rail twin of the cfront c.portio.in.{b,w,l}: /
// c.portio.out.{b,w,l}: claim (bcir/frontends/cfront/lower.py::_portio + emit.py::_portio_stmt).
// x86 port-mapped I/O (the `in`/`out` instructions) is a TRUSTED OPAQUE EFFECT EDGE -- volatile +
// ordered (barriered), off the legality value-path, reusing the bcir.asm -> llvm.inline_asm
// mechanism (SEG8.1). The direction is an enum (`in` read / `out` void write); the width is the
// access width in bits {8,16,32} (the GCC b/w/l modifier). This is the op + verifier + round-trip
// slice; the LLVM lowering is portio.mlir. The op auto-registers via the dialect .td; this
// round-trips it (parse -> print -> parse -> print) byte-identically.

// (1) an `in.b` READ: ONE operand (the port), ONE result (the read value). Round-trips identically.
// CHECK-LABEL: func.func @portio_in_b
func.func @portio_in_b(%port: i16) -> i8 {
  // CHECK: %{{.*}} = bcir.portio <in> width 8 (%arg0) : (i16) -> i8
  %r = bcir.portio <in> width 8 (%port) : (i16) -> i8
  return %r : i8
}

// (2) an `in.l` READ: 32-bit width, ONE operand (port), ONE result (the i32 read value).
// CHECK-LABEL: func.func @portio_in_l
func.func @portio_in_l(%port: i16) -> i32 {
  // CHECK: %{{.*}} = bcir.portio <in> width 32 (%arg0) : (i16) -> i32
  %r = bcir.portio <in> width 32 (%port) : (i16) -> i32
  return %r : i32
}

// (3) an `out.b` void WRITE: TWO operands (value, then port -- the Linux out(value, port) order),
// ZERO results. Round-trips identically.
// CHECK-LABEL: func.func @portio_out_b
func.func @portio_out_b(%val: i8, %port: i16) {
  // CHECK: bcir.portio <out> width 8 (%arg0, %arg1) : (i8, i16) -> ()
  bcir.portio <out> width 8 (%val, %port) : (i8, i16) -> ()
  return
}

// (4) an `out.l` void WRITE: 32-bit width, TWO operands (i32 value, i16 port), ZERO results.
// CHECK-LABEL: func.func @portio_out_l
func.func @portio_out_l(%val: i32, %port: i16) {
  // CHECK: bcir.portio <out> width 32 (%arg0, %arg1) : (i32, i16) -> ()
  bcir.portio <out> width 32 (%val, %port) : (i32, i16) -> ()
  return
}
