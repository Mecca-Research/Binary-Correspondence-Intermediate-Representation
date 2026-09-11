// A witness is a proof obligation, and you choose how it is discharged.
//
// shape.cstr_broadcastable does not test anything. It produces a `!shape.witness`, a
// value whose meaning is "someone has established this", and shape.assuming is the
// region that may rely on it. Two passes in the same release discharge it in opposite
// ways:
//
//   --convert-shape-constraints   emits a runtime cf.assert, then relies on it
//   --remove-shape-constraints    relies on nothing
//
// Both replace the witness with shape.const_witness true -- that operation is NOT what
// tells them apart, and a gate forbidding it would catch neither. What tells them apart
// is that the first pass earns the discharge: it emits shape.is_broadcastable and a
// cf.assert first. The registered pipeline is that one, and the gate REQUIRES both of
// those, so swapping the passes fails the fixture rather than quietly shipping an
// unchecked assumption.
//
// Note the two passes are not nested alike: --convert-shape-constraints runs on the
// module, --remove-shape-constraints on each func.func.

func.func @guarded_broadcast(%a: tensor<?xf32>, %b: tensor<?xf32>) -> !shape.shape {
  %sa = shape.shape_of %a : tensor<?xf32> -> !shape.shape
  %sb = shape.shape_of %b : tensor<?xf32> -> !shape.shape
  %w = shape.cstr_broadcastable %sa, %sb : !shape.shape, !shape.shape
  %r = shape.assuming %w -> !shape.shape {
    %bc = shape.broadcast %sa, %sb : !shape.shape, !shape.shape -> !shape.shape
    shape.assuming_yield %bc : !shape.shape
  }
  return %r : !shape.shape
}
