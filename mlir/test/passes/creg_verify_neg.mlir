// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// The control-register value is 64-bit in long mode -- the `I64` ODS constraint on the creg_read result
// and the creg_write value. A non-i64 width is rejected. (Enforced by the assemblyFormat type parser /
// the generated verifier, mirroring the volatile-MMIO address-type negatives.)

// (1 -- READ, NON-i64 RESULT) the creg_read result is an i32 -- rejected (control registers are 64-bit).
func.func @rd_i32() -> i32 {
  // expected-error @+1 {{'bcir.creg_read' op result #0 must be 64-bit signless integer, but got 'i32'}}
  %v = bcir.creg_read <cr3> -> i32
  return %v : i32
}

// -----

// (2 -- WRITE, NON-i64 VALUE) the creg_write value is an i32 -- rejected.
func.func @wr_i32(%v: i32) {
  // expected-error @+1 {{'bcir.creg_write' op operand #0 must be 64-bit signless integer, but got 'i32'}}
  bcir.creg_write <cr3>, %v : i32
  return
}
