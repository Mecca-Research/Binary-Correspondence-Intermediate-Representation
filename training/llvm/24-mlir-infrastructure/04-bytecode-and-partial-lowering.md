# Bytecode, and the cast a partial lowering leaves behind

Two things that look unrelated and are not: MLIR's binary serialization, and the operation
that appears when a conversion is only half done. Both are about a module in a state
between two others — on disk, or mid-lowering.

## `builtin.unrealized_conversion_cast`

A dialect conversion rewrites operations one at a time. Partway through, a converted op
produces a value of the new type while its unconverted neighbour still expects the old
one. Rather than forbid that intermediate state, MLIR gives it a name:

```mlir
%c = builtin.unrealized_conversion_cast %a : i32 to i64
```

It means: *these types will agree once the rest of the lowering lands.* It is legal in any
dialect, verifies fine, and lowers to nothing.

`--reconcile-unrealized-casts` cleans them up, and **how it fails is the useful part**:

```mlir
// a cancelling pair -- i32 -> i64 -> i32
func.func @g(%a: i32) -> i32 {
  %x = builtin.unrealized_conversion_cast %a : i32 to i64
  %y = builtin.unrealized_conversion_cast %x : i64 to i32
  return %y : i32
}
```

becomes, after reconciliation:

```mlir
func.func @g(%arg0: i32) -> i32 {
  return %arg0 : i32
}
```

Both casts are gone: they cancelled. But a **lone** cast has nothing to cancel against and
survives untouched:

```mlir
func.func @f(%a: i32) -> i64 {
  %c = builtin.unrealized_conversion_cast %a : i32 to i64   // still here afterwards
  return %c : i64
}
```

**A surviving `unrealized_conversion_cast` in your output is a diagnosis, not a nuisance.**
It says an operation on one side of it was never converted. The usual causes: a pattern
missing from the `RewritePatternSet`, an op you forgot to mark illegal in the
`ConversionTarget`, or a type the `TypeConverter` has no rule for.

That is why the corpus's own lowering registry lists
`builtin.unrealized_conversion_cast` among the strings a fully lowered artifact must not
contain — see the `forbid_lowered` entries in
[`../autograder/mlir-examples.json`](../autograder/mlir-examples.json).

Runnable: [`examples/unrealized-casts.mlir`](examples/unrealized-casts.mlir).

## Bytecode

MLIR serializes to a binary format as well as to text:

```console
$ mlir-opt m.mlir --emit-bytecode -o m.mlirbc
$ head -c 4 m.mlirbc | od -c
0000000   M   L 357   R
```

The magic is `ML\xefR`. Reading it back produces the same module the text does — the same
module, not an equivalent one, and the gate checks that by comparing the printed output
from both paths.

**Bytecode is not automatically smaller**, and this chapter's own three examples disagree
with each other about it — measured rather than asserted, by the gate, on whatever MLIR is
running it:

<!-- generated: serialization-sizes -->
| example | text | bytecode | bytecode is |
| --- | ---: | ---: | --- |
| [`examples/regions-and-blocks.mlir`](examples/regions-and-blocks.mlir) | 717 | 438 | **smaller** by 279 bytes |
| [`examples/interfaces-inlining.mlir`](examples/interfaces-inlining.mlir) | 233 | 272 | **larger** by 39 bytes |
| [`examples/unrealized-casts.mlir`](examples/unrealized-casts.mlir) | 318 | 237 | **smaller** by 81 bytes |

Measured on this chapter's own 3 examples by the gate, with source locations stripped from both sides first: bytecode is smaller on 2 of them and larger on 1.
<!-- /generated -->

The direction is not a property of the format, it is a property of the module. Bytecode
pays a fixed cost first — a header, a dialect table, a string table — and earns it back by
naming each operation, attribute and type once however often it is used. The smallest
example here has almost nothing to amortise those tables over and comes out larger; the
other two are big enough that the tables start paying.

So the real wins are structural rather than byte-shaving: shared name tables, no re-parsing
of the textual grammar, and lazy loading of nested regions a reader may never look at.
Quoting a size *ratio* from any of these modules would be exactly the kind of measurement
[`../21-performance-methodology/`](../21-performance-methodology) exists to refuse — which
is why the table above reports bytes on named files and stops there.

### Why the measurement strips locations first

Both columns are measured with `builtin.module(strip-debuginfo)` applied, and skipping that
step gets the comparison wrong twice over.

**Text and bytecode disagree about what they remember.** `mlir-opt`'s default textual print
emits no `loc(...)` at all; the bytecode keeps every location the parser attached. Compare
them as they come out of the tool and you are comparing a lossy encoding against a faithful
one — bytecode is carrying information the text column threw away, and being charged bytes
for it. (Ask for the locations with `--mlir-print-debuginfo` and the text form is the one
that balloons.)

**A location holds the source file's path**, so the byte count depends on where the file
lives. Rename the file — same module, same bytes in it — and the bytecode changes size by
exactly the number of characters you added:

```console
$ cp m.mlir m<N more characters>.mlir
$ mlir-opt m.mlir --emit-bytecode -o - | wc -c
$ mlir-opt m<N more characters>.mlir --emit-bytecode -o - | wc -c
```

For N of 1, 10, 40, 100 and 200 the second count exceeds the first by exactly N. One byte
per byte, no more, which is the string table doing its job: 25 locations in that module all
cite the same path, and it is stored once. The absolute numbers are deliberately not quoted
here, because they are a measurement of somebody's checkout directory — that is the whole
point.

Both facts are pinned by the gate rather than left as a note: if bytecode stops embedding
the path, or starts charging more than its length for it, or the text form starts printing
locations, the check fails and says this section needs rewriting.

What bytecode has that text does not is a **versioning story**: the format carries a
version, dialects can implement read and write hooks per version, and a newer reader can
accept an older file deliberately rather than accidentally. Text has no such contract —
today's `mlir-opt` parses yesterday's text only for as long as nobody changes a custom
assembly format.

So: text for reading, diffing and teaching; bytecode for storing an artifact you intend to
read back with a different build.

## Pitfalls checklist

- Do not delete a surviving `unrealized_conversion_cast` to make output look finished. It
  is telling you a conversion is incomplete; deleting it hides the bug.
- Do not expect `--reconcile-unrealized-casts` to remove a lone cast. It removes pairs
  that cancel, and leaving the rest is the feature.
- Do not assume bytecode is smaller. Measure it on a module the size of yours.
- Do not compare a default text print against bytecode and call the difference an encoding
  win. One of them is carrying locations and the other is not.
- Do not treat textual MLIR as a stable interchange format across versions. That is what
  the bytecode version is for.

## Checks

[`../tools/verify-mlir-infrastructure.py`](../tools/verify-mlir-infrastructure.py) runs
reconciliation over the example and requires both halves — the pair in `@g` gone, the lone
cast in `@f` still present — then emits bytecode, checks the magic, reads it back and
requires the round-trip to reproduce the text form exactly.

It also measures the size table above rather than trusting it, and fails if the block has
drifted from what the local `mlir-opt` produces (`--update` rewrites it). Three further
checks guard the claims around it: that bytecode still embeds the source path (the same
module under two filenames must produce two sizes), that stripping locations still removes
that dependence (one size under both names), and that the default text print still emits no
`loc(...)`. If every example ever serialized the same direction, the gate says so too —
"not automatically smaller" would then be a sentence with nothing behind it.
