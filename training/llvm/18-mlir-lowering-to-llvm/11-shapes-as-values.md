# 11 — Shapes as values, and what they cost

[`10-tensors-buffers-and-the-step-between.md`](10-tensors-buffers-and-the-step-between.md)
treats a shape as part of a type: `tensor<?x?xf32>` says two dimensions, sizes unknown.
The `shape` dialect makes the other move. A shape becomes an ordinary SSA *value*, shape
arithmetic becomes ordinary IR, and a shape constraint becomes a value you have to
discharge. Everything a compiler already knows how to do to IR — fold it, CSE it, hoist
it out of a loop, prove things about it — then applies to shape computation for free.

The bill arrives at lowering time, and it is worth seeing before you read `linalg` or
`tosa`, where broadcasting is implied by an operation's name rather than written down.

## Four operations, and what they expand into

[`examples/shape-broadcast-to-std.mlir`](examples/shape-broadcast-to-std.mlir) broadcasts
a `tensor<?x?xf32>` against a `tensor<?xf32>` and asks how many elements the result has:

```mlir
%sa = shape.shape_of %a : tensor<?x?xf32> -> tensor<2xindex>
%sb = shape.shape_of %b : tensor<?xf32> -> tensor<1xindex>
%bc = shape.broadcast %sa, %sb : tensor<2xindex>, tensor<1xindex> -> tensor<2xindex>
%n  = shape.num_elements %bc : tensor<2xindex> -> index
```

`--convert-shape-to-std` writes out what those four lines actually mean. The
rank-alignment step comes first — the shorter shape is right-aligned against the longer
one, and the difference is the number of leading dimensions it does not have:

```mlir
%0 = arith.maxui %dim_7, %dim_6 : index   // the result rank
%1 = arith.subi %0, %dim_6 : index        // leading dims missing from the first shape
%2 = arith.subi %0, %dim_7 : index        // ... and from the second
```

Then the rule everyone knows as "a dimension of 1 broadcasts" appears as an actual
comparison, inside a generator that builds the result shape one extent at a time:

```mlir
%extracted = tensor.extract %cast[%8] : tensor<2xindex>
%9 = arith.cmpi eq, %extracted, %c1_9 : index
%10 = arith.select %9, %c1_9, %extracted : index
```

Four operations of `shape` become roughly forty of `tensor`, `arith` and `scf`. That is
the point of the exercise: broadcasting is not a notational convenience the compiler gets
for free, it is a loop, and it is a loop that runs whenever the shapes are not static. A
frontend that emits dynamic shapes everywhere has bought all of this per operation.

## A pass that lowers a dialect into itself

Run only `--convert-shape-to-std` and one operation survives: `shape.num_elements`.
[`examples/shape-one-pass-is-not-enough.mlir`](examples/shape-one-pass-is-not-enough.mlir)
is that single-pass run, registered as a fixture whose declared outcome *is* the leftover
operation, so this paragraph is checked rather than asserted. It is not in the
conversion's pattern set, and the reason is structural rather than an oversight. `shape.num_elements` is a *reduction* over the
extents, and the dialect already has a way to say that — so a separate pass,
`--shape-to-shape-lowering`, rewrites it inside the `shape` dialect first:

```mlir
%3 = shape.reduce(%2, %c1) : tensor<2xindex> -> index {
  ^bb0(...):
    %4 = shape.mul %arg3, %arg4 : index, index -> index
    shape.yield %4 : index
}
```

and `--convert-shape-to-std` knows how to convert *that*. The registered pipeline is
therefore the two of them in order, and it leaves no `shape` operation behind at all.

`shape.reduce`, `shape.mul` and `shape.yield` are bridge operations in exactly the sense
[`03-rewritepattern-and-conversiontarget.md`](03-rewritepattern-and-conversiontarget.md)
warns about: legal for one phase, illegal for the next, with a lifetime that must be
explicit. Here the lifetime is one pass wide, and running the passes in the other order
proves it — the converter runs first, finds nothing it recognises in `shape.num_elements`,
and the normalizer then strands `shape.reduce`, `shape.mul` and `shape.yield` in the
output with no pass left to take them anywhere.

**The rule:** when a conversion leaves one operation behind, ask whether you are missing
a pass rather than a pattern. A dialect large enough to need normalizing usually ships the
normalizer separately, and pipeline order is then part of the contract.

## A witness is a proof obligation

`shape.cstr_broadcastable` does not test whether two shapes are broadcastable. It produces
a `!shape.witness` — a value whose entire meaning is *someone has established this* — and
`shape.assuming` is the region permitted to rely on it.
[`examples/shape-constraints-to-assert.mlir`](examples/shape-constraints-to-assert.mlir)
is that shape:

```mlir
%w = shape.cstr_broadcastable %sa, %sb : !shape.shape, !shape.shape
%r = shape.assuming %w -> !shape.shape {
  %bc = shape.broadcast %sa, %sb : !shape.shape, !shape.shape -> !shape.shape
  shape.assuming_yield %bc : !shape.shape
}
```

Separating the obligation from its discharge is what makes the design useful, because the
same release discharges it two opposite ways. `--convert-shape-constraints` turns the
witness into a runtime check — `shape.is_broadcastable` produces a plain `i1`, and the
assertion is a real one:

```mlir
%2 = shape.is_broadcastable %0, %1 : !shape.shape, !shape.shape
cf.assert %2, "required broadcastable shapes"
%3 = shape.broadcast %0, %1 : !shape.shape, !shape.shape -> !shape.shape
```

`--remove-shape-constraints` discharges the same obligation with a promise instead, and
the interesting part is what the two have in common. *Both* passes replace the witness
with `shape.const_witness true`:

```mlir
%0 = shape.const_witness true
%3 = shape.assuming %0 -> (!shape.shape) { ... }
```

So `shape.const_witness` does not distinguish them — a reviewer grepping for it would
clear the unsafe pipeline and the safe one alike. The difference is that
`--convert-shape-constraints` *earns* the discharge first, with `shape.is_broadcastable`
and the assertion. Once both outputs are canonicalized the witness and its region fold
away in each, and what remains of the difference is exactly two lines:

```mlir
%2 = shape.is_broadcastable %0, %1 : !shape.shape, !shape.shape
cf.assert %2, "required broadcastable shapes"
```

Those two lines are the entire safety posture of the program. Neither pass is wrong: the
first is what you want when the shapes come from outside, the second is what you want
after a frontend has already proved the constraint and you are paying for a check that
cannot fail. Choosing by accident, because one pass name looked like the other in a
pipeline string, is how an unchecked assumption ships.

So this corpus encodes the distinction in the gate rather than in a warning — and it
encodes it as a **requirement**, not a prohibition. The registered pipeline requires
`shape.is_broadcastable` and `cf.assert` with its message; the unsafe pass produces
neither. The first draft of this fixture forbade `shape.const_witness` and believed that
was the discriminating check; it was inert, because the safe pass emits that operation
too. A prohibition on residue only catches what the other pass *leaves*, and here the two
passes leave the same thing — the safe one merely does more before it.

One practical difference while you are wiring a pipeline: `--convert-shape-constraints`
runs on the module, `--remove-shape-constraints` on each `func.func`, so they are not
textually interchangeable in a `--pass-pipeline` string. That is a real inconvenience
standing in for an unreal safety: nesting is not a safeguard.

## Why this comes before `linalg` and `tosa`

In those dialects, broadcasting is implied by an operation's *name* — the shape rules are
in the operation's definition, not in the IR. That is far more convenient and it hides
everything above: the rank alignment, the per-dimension comparison, the loop, and the
question of who proved the shapes were compatible in the first place. Reading `linalg`
having already seen `--convert-shape-to-std` output is reading it with the cost model
intact.

## Pitfalls checklist

- Do not read `shape.cstr_broadcastable` as a check. It is an obligation; a pass decides
  whether it becomes a check or an assumption, and the two are one pipeline entry apart.
- Do not assume one conversion pass empties a dialect. `shape.num_elements` needs the
  in-dialect normalizer first, and the residue is the diagnostic.
- Do not put `--remove-shape-constraints` in a pipeline whose inputs are untrusted. It is
  a promise, not a proof.
- Do not treat dynamic shapes as free because they are only in the type. Every `?` you
  keep is index arithmetic somebody eventually emits.
- Do not reach for `shape` when the shapes are static. The dialect exists for the case
  where they are not; `tensor<4x8xf32>` needs none of this.

## Checks

Both examples are Tier 3 fixtures in
[`../autograder/mlir-examples.json`](../autograder/mlir-examples.json), graded by
[`../tools/verify-mlir-examples.sh`](../tools/verify-mlir-examples.sh).

[`examples/shape-broadcast-to-std.mlir`](examples/shape-broadcast-to-std.mlir) must lower
to `tensor.dim`, `tensor.generate`, `scf.if` and `arith.maxui`, and must contain none of
`shape.shape_of`, `shape.broadcast`, `shape.num_elements`, `shape.reduce`, `shape.mul` or
`shape.yield` — the last three being the bridge operations, so the forbid list is what
pins the pass order rather than a comment claiming it matters.

[`examples/shape-constraints-to-assert.mlir`](examples/shape-constraints-to-assert.mlir)
must lower to `shape.is_broadcastable` and `cf.assert` carrying the message
`required broadcastable shapes`, and must contain no `shape.cstr_broadcastable`. The
require half is the one that carries the safety claim here, for the reason above; the
forbid half only establishes that the conversion ran at all.

[`examples/shape-one-pass-is-not-enough.mlir`](examples/shape-one-pass-is-not-enough.mlir)
is registered with the single converter and an `illegal-ops-remaining` outcome naming
`shape.num_elements`. An expected failure is a claim like any other: if a future MLIR
teaches `--convert-shape-to-std` to handle the reduction directly, that fixture goes red
and this chapter gets rewritten, rather than continuing to describe a toolchain nobody
has.
