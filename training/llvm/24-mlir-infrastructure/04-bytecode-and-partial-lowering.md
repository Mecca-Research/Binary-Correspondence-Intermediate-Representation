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

**Bytecode is not automatically smaller.** For the small module used here it is *larger*:
241 bytes of bytecode against 93 of text. The wins are on real modules and are structural
rather than byte-shaving — a string table shared across the module, no re-parsing of the
textual grammar, and lazy loading of nested regions a reader may never look at. Quoting a
size ratio from a toy module would be exactly the kind of measurement
[`../21-performance-methodology/`](../21-performance-methodology) exists to refuse.

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
- Do not treat textual MLIR as a stable interchange format across versions. That is what
  the bytecode version is for.

## Checks

[`../tools/verify-mlir-infrastructure.py`](../tools/verify-mlir-infrastructure.py) runs
reconciliation over the example and requires both halves — the pair in `@g` gone, the lone
cast in `@f` still present — then emits bytecode, checks the magic, reads it back and
requires the round-trip to reproduce the text form exactly.
