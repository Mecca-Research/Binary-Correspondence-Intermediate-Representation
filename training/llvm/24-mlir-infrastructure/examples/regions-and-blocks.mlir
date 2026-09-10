// Regions, blocks and block arguments -- the structure every MLIR op is built from.
//
// An operation holds regions; a region holds blocks; a block takes arguments and ends in
// a terminator. There are no phi nodes anywhere in MLIR: where LLVM would merge values
// with a phi, MLIR passes them as block arguments, which is the same information with the
// predecessor edge made explicit.

func.func @sum_to(%n: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  cf.br ^loop(%zero, %zero : i32, i32)

// ^loop takes its live values as ARGUMENTS. In LLVM these two would be phi nodes at the
// top of the block; here each incoming edge names what it passes.
^loop(%i: i32, %acc: i32):
  %done = arith.cmpi sge, %i, %n : i32
  cf.cond_br %done, ^exit(%acc : i32), ^body(%i, %acc : i32, i32)

^body(%bi: i32, %bacc: i32):
  %acc2 = arith.addi %bacc, %bi : i32
  %i2 = arith.addi %bi, %one : i32
  cf.br ^loop(%i2, %acc2 : i32, i32)

^exit(%result: i32):
  return %result : i32
}

// An op with a region NESTED inside it. scf.if owns two regions, and each one is a block
// list whose terminator yields the region's results. Nesting is why MLIR needs no
// separate notion of a function body: a body is just a region.
func.func @pick(%c: i1, %a: i32, %b: i32) -> i32 {
  %r = scf.if %c -> i32 {
    scf.yield %a : i32
  } else {
    scf.yield %b : i32
  }
  return %r : i32
}
