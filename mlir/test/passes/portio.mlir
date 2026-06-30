// RUN: bcir-opt -convert-bcir-to-llvm %s | FileCheck %s
//
// bcir.portio -> llvm.inline_asm (ASM2 lowering, SEG8.2): the x86 `in`/`out` instruction behind a
// volatile `call asm`, reusing the bcir.asm -> llvm.inline_asm mechanism (SEG8.1). The template
// strings + constraint strings are BYTE-IDENTICAL to the cfront rail's _PORTIO_IN_ASM /
// _PORTIO_OUT_ASM (bcir/frontends/cfront/emit.py): the accumulator is %b0/%w0/%k0 (al/ax/eax) and
// the port is %w1 (the 16-bit dx, the `Nd` immediate-or-dx constraint). An `in` READ writes the
// accumulator (constraint "=a,Nd", one result); an `out` WRITE reads it (constraint "a,Nd", no
// result). has_side_effects is ALWAYS set (port I/O is volatile, like the cfront __volatile__). The
// `in` form passes ONLY the port as a call operand (the "=a" output is the RESULT); the `out` form
// passes value-then-port.

// (1) in.b -> "inb %w1, %b0", "=a,Nd", RETURNS an i8, the port (%arg0) is the sole call operand.
// CHECK-LABEL: func.func @portio_in_b
func.func @portio_in_b(%port: i16) -> i8 {
  // CHECK: %[[R:.*]] = llvm.inline_asm has_side_effects "inb %w1, %b0", "=a,Nd" %arg0 : (i16) -> i8
  %r = bcir.portio <in> width 8 (%port) : (i16) -> i8
  // CHECK: return %[[R]] : i8
  return %r : i8
}

// (2) in.w -> "inw %w1, %w0", returns i16.
// CHECK-LABEL: func.func @portio_in_w
func.func @portio_in_w(%port: i16) -> i16 {
  // CHECK: llvm.inline_asm has_side_effects "inw %w1, %w0", "=a,Nd" %arg0 : (i16) -> i16
  %r = bcir.portio <in> width 16 (%port) : (i16) -> i16
  return %r : i16
}

// (3) in.l -> "inl %w1, %k0", returns i32.
// CHECK-LABEL: func.func @portio_in_l
func.func @portio_in_l(%port: i16) -> i32 {
  // CHECK: llvm.inline_asm has_side_effects "inl %w1, %k0", "=a,Nd" %arg0 : (i16) -> i32
  %r = bcir.portio <in> width 32 (%port) : (i16) -> i32
  return %r : i32
}

// (4) out.b -> "outb %b0, %w1", "a,Nd", NO result, operands are value (%arg0) then port (%arg1).
// CHECK-LABEL: func.func @portio_out_b
func.func @portio_out_b(%val: i8, %port: i16) {
  // CHECK: llvm.inline_asm has_side_effects "outb %b0, %w1", "a,Nd" %arg0, %arg1 : (i8, i16) -> ()
  // CHECK-NOT: = llvm.inline_asm
  bcir.portio <out> width 8 (%val, %port) : (i8, i16) -> ()
  return
}

// (5) out.w -> "outw %w0, %w1", no result.
// CHECK-LABEL: func.func @portio_out_w
func.func @portio_out_w(%val: i16, %port: i16) {
  // CHECK: llvm.inline_asm has_side_effects "outw %w0, %w1", "a,Nd" %arg0, %arg1 : (i16, i16) -> ()
  bcir.portio <out> width 16 (%val, %port) : (i16, i16) -> ()
  return
}

// (6) out.l -> "outl %k0, %w1", no result.
// CHECK-LABEL: func.func @portio_out_l
func.func @portio_out_l(%val: i32, %port: i16) {
  // CHECK: llvm.inline_asm has_side_effects "outl %k0, %w1", "a,Nd" %arg0, %arg1 : (i32, i16) -> ()
  bcir.portio <out> width 32 (%val, %port) : (i32, i16) -> ()
  return
}
