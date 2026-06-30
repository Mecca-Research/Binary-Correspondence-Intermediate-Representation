// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// The bcir.portio op verifier (hasVerifier; the ASM2 structural well-formedness check, mirrors
// bcir/frontends/cfront/lower.py::_portio + emit.py::_portio_stmt -- the op-level check the sibling
// law ops run inline, NOT a new globally-numbered R-law). The (direction, width) pin the operand/
// result arity + access width: width in {8,16,32}; `in` = 1 operand (port) + 1 result; `out` = 2
// operands (value, port) + 0 results; the value type is the i{width} matching the op width. Each
// section carries a SINGLE violation (mirrors inline_asm_verify_neg.mlir).

// (1 -- BAD WIDTH) the width must be one of 8/16/32 bits (b/w/l). 24 is not a port-I/O access width.
func.func @bad_width(%port: i16) -> i32 {
  // expected-error @+1 {{'bcir.portio' op width must be one of 8/16/32 bits (b=8, w=16, l=32), got 24}}
  %r = bcir.portio <in> width 24 (%port) : (i16) -> i32
  return %r : i32
}

// -----

// (2 -- IN OPERAND COUNT) an `in` read takes exactly ONE operand (the port). Here two operands are
// given (the result/width are kept consistent so it is purely the operand-count law that trips).
func.func @in_two_operands(%port: i16, %x: i16) -> i16 {
  // expected-error @+1 {{'bcir.portio' op an 'in' port read takes exactly 1 operand (the port), got 2}}
  %r = bcir.portio <in> width 16 (%port, %x) : (i16, i16) -> i16
  return %r : i16
}

// -----

// (3 -- OUT WITH A RESULT) an `out` write is VOID -- it produces no result. Here it declares one
// result (the operand count 2 is consistent so it is purely the result-count law that trips).
func.func @out_with_result(%val: i8, %port: i16) -> i8 {
  // expected-error @+1 {{'bcir.portio' op an 'out' port write produces no result (a void write), got 1 results}}
  %r = bcir.portio <out> width 8 (%val, %port) : (i8, i16) -> i8
  return %r : i8
}

// -----

// (4 -- VALUE/WIDTH MISMATCH) the port value type must be the i{width} matching the op width. Here
// the op width is 16 but the read result is i32 (all the arity laws are satisfied so it is purely
// the value-width law that trips).
func.func @value_width_mismatch(%port: i16) -> i32 {
  // expected-error @+1 {{'bcir.portio' op the port value type must be the i16 matching the op width (got 'i32')}}
  %r = bcir.portio <in> width 16 (%port) : (i16) -> i32
  return %r : i32
}

// -----

// (5 -- NON-i16 PORT) the I/O port operand must be an i16 (x86 in/out address the port via the 16-bit
// dx register -- the lowering hardcodes the `Nd` constraint + the `%w1` modifier). Here the port is an
// i32 (all the arity + value-width laws are satisfied, so it is purely the port-type law that trips);
// without it a non-i16 port would pass verify() yet lower to a semantically-wrong port (the symmetric
// analog to the bcir.asm out-constraint-prefix law).
func.func @non_i16_port(%port: i32) -> i8 {
  // expected-error @+1 {{'bcir.portio' op the I/O port operand must be an i16 (the 16-bit dx / Nd port; x86 in/out address the port via dx, got 'i32')}}
  %r = bcir.portio <in> width 8 (%port) : (i32) -> i8
  return %r : i8
}
