// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// Op-level structural negatives for the first-class atomic ops (§5.14 Phase 2) -- the same
// well-formedness discipline as the MMIO accessors (NOT a new globally-numbered R-law; the ordering
// LAW is the R5 extension on the claim rail, verify_volatile.mlir).

// (1 -- RMW, UNKNOWN KIND) the kind must be one of the cfront atomic families.
func.func @rmw_bad_kind(%addr: i64, %v: i32) -> i32 {
  // expected-error @+1 {{unknown atomic_rmw kind 'nand' (one of add|sub|xor|exchange, the cfront atomic families)}}
  %old = bcir.atomic_rmw "nand" %v, %addr : i32, i64 -> i32
  return %old : i32
}

// -----

// (2 -- RMW, NON-INTEGER VALUE) the cfront atomic families are integer-typed.
func.func @rmw_float_value(%addr: i64, %v: f32) -> f32 {
  // expected-error @+1 {{the RMW value must be an integer (the cfront atomic families are integer-typed; got 'f32')}}
  %old = bcir.atomic_rmw "add" %v, %addr : f32, i64 -> f32
  return %old : f32
}

// -----

// (3 -- RMW, RESULT/VALUE TYPE MISMATCH) fetch semantics: the result is the OLD value.
func.func @rmw_result_mismatch(%addr: i64, %v: i32) -> i64 {
  // expected-error @+1 {{the RMW result must have the value type (fetch/old-value semantics; got 'i64' vs 'i32')}}
  %old = bcir.atomic_rmw "add" %v, %addr : i32, i64 -> i64
  return %old : i64
}

// -----

// (4 -- RMW, NARROW ADDRESS) the address discipline mirrors the MMIO accessors (>= 32 bits).
func.func @rmw_narrow_addr(%addr: i16, %v: i32) -> i32 {
  // expected-error @+1 {{the object address must be an integer of at least pointer width (>= 32 bits; got 'i16')}}
  %old = bcir.atomic_rmw "add" %v, %addr : i32, i16 -> i32
  return %old : i32
}

// -----

// (5 -- CAS, EXPECTED/DESIRED MISMATCH) the comparand and replacement must agree.
func.func @cas_type_mismatch(%addr: i64, %e: i32, %d: i64) -> i32 {
  // expected-error @+1 {{the CAS expected/desired operands must have the same type (got 'i32' vs 'i64')}}
  %old = bcir.atomic_cas %e, %d, %addr : i32, i64, i64 -> i32
  return %old : i32
}

// -----

// (6 -- CAS, RESULT TYPE MISMATCH) old-value semantics: the result has the comparand type.
func.func @cas_result_mismatch(%addr: i64, %e: i32, %d: i32) -> i64 {
  // expected-error @+1 {{the CAS result must have the comparand type (old-value semantics; got 'i64' vs 'i32')}}
  %old = bcir.atomic_cas %e, %d, %addr : i32, i32, i64 -> i64
  return %old : i64
}
