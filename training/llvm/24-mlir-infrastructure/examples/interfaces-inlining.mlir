// Interfaces: how a generic pass reaches a dialect it has never heard of.
//
// The MLIR inliner contains no reference to the func dialect. It works here because
// func.func implements CallableOpInterface (it has a body a call can be inlined into) and
// func.call implements CallOpInterface (it names a callee). Any dialect that implements
// the same two interfaces gets the same inliner, for free.
//
// `mlir-opt --inline` folds @callee into @caller and the call disappears.

func.func private @callee(%a: i32) -> i32 {
  %r = arith.addi %a, %a : i32
  return %r : i32
}

func.func @caller(%x: i32) -> i32 {
  %y = call @callee(%x) : (i32) -> i32
  return %y : i32
}
