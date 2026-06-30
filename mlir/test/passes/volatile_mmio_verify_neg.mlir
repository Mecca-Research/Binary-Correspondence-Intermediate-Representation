// RUN: bcir-opt -verify-diagnostics -split-input-file %s
//
// The device-register ADDRESS of a first-class MMIO access must be a SIGNLESS INTEGER (it is inttoptr'd
// to the device pointer in the lowering) -- the op's `AnySignlessInteger:$addr` ODS constraint. A
// non-integer address (e.g. an f32) is rejected. Mirrors the split-input negative style of the sibling
// asm/portio op tests.

// (1 -- LOAD, NON-INTEGER ADDRESS) the volatile_load address is an f32 -- rejected by the
// `AnySignlessInteger:$addr` constraint (enforced at parse time by the assemblyFormat type parser,
// before the verifier -- the generated verifier enforces the same constraint for programmatic builds).
func.func @load_float_addr(%addr: f32) -> i32 {
  // expected-error @+1 {{custom op 'bcir.volatile_load' invalid kind of type specified}}
  %v = bcir.volatile_load %addr : f32 -> i32
  return %v : i32
}

// -----

// (2 -- STORE, NON-INTEGER ADDRESS) the volatile_store address is an f32 -- same signless-integer law.
func.func @store_float_addr(%val: i32, %addr: f32) {
  // expected-error @+1 {{custom op 'bcir.volatile_store' invalid kind of type specified}}
  bcir.volatile_store %val, %addr : i32, f32
  return
}

// -----

// (3 -- NARROW ADDRESS) an i8 address is too narrow -- it would be silently zero-extended into the
// device pointer by the inttoptr lowering. The verifier requires >= pointer-width (>= 32 bits).
func.func @load_narrow_addr(%addr: i8) -> i32 {
  // expected-error @+1 {{the device-register address must be an integer of at least pointer width (>= 32 bits; got 'i8')}}
  %v = bcir.volatile_load %addr : i8 -> i32
  return %v : i32
}

// -----

// (4 -- VECTOR VALUE) a vector device-register value is not a scalar natural-width register -- rejected.
func.func @load_vector_value(%addr: i64) -> vector<4xi32> {
  // expected-error @+1 {{the device-register value must be a scalar integer or float (got 'vector<4xi32>')}}
  %v = bcir.volatile_load %addr : i64 -> vector<4xi32>
  return %v : vector<4xi32>
}

// -----

// (5 -- INDEX VALUE) an `index`-typed device register is meaningless -- rejected (not int/float).
func.func @store_index_value(%val: index, %addr: i64) {
  // expected-error @+1 {{the device-register value must be a scalar integer or float (got 'index')}}
  bcir.volatile_store %val, %addr : index, i64
  return
}
