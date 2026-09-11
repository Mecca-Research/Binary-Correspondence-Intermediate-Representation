// builtin.unrealized_conversion_cast: the operation a partial lowering leaves behind.
//
// When a conversion rewrites one op but not its neighbour, the two disagree about types.
// The dialect-conversion framework bridges the gap with this cast, which is legal in any
// dialect and means "these types will agree once the rest of the lowering lands".
//
// --reconcile-unrealized-casts removes cast PAIRS that cancel. A cast with no partner is
// left in place, and that is the signal: your lowering is not finished.

// A cancelling pair: i32 -> i64 -> i32. Reconciliation folds both away and @g becomes the
// identity on its argument.
func.func @g(%a: i32) -> i32 {
  %x = builtin.unrealized_conversion_cast %a : i32 to i64
  %y = builtin.unrealized_conversion_cast %x : i64 to i32
  return %y : i32
}

// A lone cast has nothing to cancel against, so it SURVIVES reconciliation. Finding one
// of these in your output means an op on the other side of it was never converted.
func.func @f(%a: i32) -> i64 {
  %c = builtin.unrealized_conversion_cast %a : i32 to i64
  return %c : i64
}
