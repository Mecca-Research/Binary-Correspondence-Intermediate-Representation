// RUN: bcir-opt -convert-bcir-to-llvm %s | FileCheck %s
//
// bcir.portio -> llvm.inline_asm (ASM2 lowering, SEG8.2): the x86 `in`/`out` instruction behind a
// volatile `call asm`, reusing the bcir.asm -> llvm.inline_asm mechanism (SEG8.1). The template +
// constraint strings are the LLVM-IR-correct forms (VERIFIED to assemble through llc to `inb %dx,%al`
// etc.): the accumulator is ${0:b}/${0:w}/${0:k} (al/ax/eax) and the port is ${1:w} (the 16-bit dx,
// the N{dx} immediate-or-dx constraint). The cfront rail carries the GCC spellings (`%w1`/`%b0`),
// correct on the C->clang path but rejected by llc here ("invalid register name" / "couldn't allocate
// output register"); clang's frontend does the `%b0`->`${0:b}` / `=a`->`={ax}` translation, and the
// frontend-less MLIR rail emits the qualified forms directly. An `in` READ writes the accumulator
// (constraint "={ax},N{dx}", one result); an `out` WRITE reads it (constraint "{ax},N{dx}", no
// result). has_side_effects is ALWAYS set (port I/O is volatile, like the cfront __volatile__). The
// `in` form passes ONLY the port as a call operand (the "={ax}" output is the RESULT); the `out` form
// passes value-then-port.

// (1) in.b -> "inb ${1:w}, ${0:b}", "={ax},N{dx}", RETURNS an i8, the port (%arg0) is the sole call
// operand. Assembles to `inb %dx, %al`.
// CHECK-LABEL: func.func @portio_in_b
func.func @portio_in_b(%port: i16) -> i8 {
  // CHECK: %[[R:.*]] = llvm.inline_asm has_side_effects "inb ${1:w}, ${0:b}", "={ax},N{dx}" %arg0 : (i16) -> i8
  %r = bcir.portio <in> width 8 (%port) : (i16) -> i8
  // CHECK: return %[[R]] : i8
  return %r : i8
}

// (2) in.w -> "inw ${1:w}, ${0:w}", returns i16. Assembles to `inw %dx, %ax`.
// CHECK-LABEL: func.func @portio_in_w
func.func @portio_in_w(%port: i16) -> i16 {
  // CHECK: llvm.inline_asm has_side_effects "inw ${1:w}, ${0:w}", "={ax},N{dx}" %arg0 : (i16) -> i16
  %r = bcir.portio <in> width 16 (%port) : (i16) -> i16
  return %r : i16
}

// (3) in.l -> "inl ${1:w}, ${0:k}", returns i32. Assembles to `inl %dx, %eax`.
// CHECK-LABEL: func.func @portio_in_l
func.func @portio_in_l(%port: i16) -> i32 {
  // CHECK: llvm.inline_asm has_side_effects "inl ${1:w}, ${0:k}", "={ax},N{dx}" %arg0 : (i16) -> i32
  %r = bcir.portio <in> width 32 (%port) : (i16) -> i32
  return %r : i32
}

// (4) out.b -> "outb ${0:b}, ${1:w}", "{ax},N{dx}", NO result, operands are value (%arg0) then port
// (%arg1). Assembles to `outb %al, %dx`.
// CHECK-LABEL: func.func @portio_out_b
func.func @portio_out_b(%val: i8, %port: i16) {
  // CHECK: llvm.inline_asm has_side_effects "outb ${0:b}, ${1:w}", "{ax},N{dx}" %arg0, %arg1 : (i8, i16) -> ()
  // CHECK-NOT: = llvm.inline_asm
  bcir.portio <out> width 8 (%val, %port) : (i8, i16) -> ()
  return
}

// (5) out.w -> "outw ${0:w}, ${1:w}", no result. Assembles to `outw %ax, %dx`.
// CHECK-LABEL: func.func @portio_out_w
func.func @portio_out_w(%val: i16, %port: i16) {
  // CHECK: llvm.inline_asm has_side_effects "outw ${0:w}, ${1:w}", "{ax},N{dx}" %arg0, %arg1 : (i16, i16) -> ()
  bcir.portio <out> width 16 (%val, %port) : (i16, i16) -> ()
  return
}

// (6) out.l -> "outl ${0:k}, ${1:w}", no result. Assembles to `outl %eax, %dx`.
// CHECK-LABEL: func.func @portio_out_l
func.func @portio_out_l(%val: i32, %port: i16) {
  // CHECK: llvm.inline_asm has_side_effects "outl ${0:k}, ${1:w}", "{ax},N{dx}" %arg0, %arg1 : (i32, i16) -> ()
  bcir.portio <out> width 32 (%val, %port) : (i32, i16) -> ()
  return
}
