# 03 — Sparsity is a property of the type

[`01-the-abstraction-ladder.md`](01-the-abstraction-ladder.md) ends with a `linalg.generic`
becoming a loop nest, and makes the point that `indexing_maps` and `iterator_types` are a
loop nest written as data. This chapter is the sharpest consequence of that design, and
the reason it is worth the abstraction:

**Change the type and the loop changes, with the operation untouched.**

## The same generic, twice

[`examples/sparse-matvec-to-loops.mlir`](examples/sparse-matvec-to-loops.mlir) is a
matrix-vector product. Its `indexing_maps`, its `iterator_types` and its body are
ordinary — a multiply, an add, a `linalg.yield`. Exactly one thing marks it:

```mlir
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

func.func @matvec(%A: tensor<8x8xf32, #CSR>, ...)
```

That attribute says the first level is stored densely and the second is *compressed*: the
matrix keeps only the entries that exist, plus the index structure needed to find them. It
is a statement about representation, attached to a type, and nothing in the operation
refers to it.

`--sparsification` reads it and writes a different program.

## What the encoding turns into

The three arrays a CSR matrix is made of come out of the type, not out of the IR:

```mlir
%0 = sparse_tensor.values %arg0      : tensor<8x8xf32, #sparse> to memref<?xf32>
%3 = sparse_tensor.positions  %arg0 {level = 1 : index} : ... to memref<?xindex>
%4 = sparse_tensor.coordinates %arg0 {level = 1 : index} : ... to memref<?xindex>
```

And the loop nest is built around them. The outer loop walks rows densely, because level 0
is `dense`. The inner loop is the interesting one:

```mlir
scf.for %arg3 = %c0 to %c8 step %c1 {
  %7  = memref.load %3[%arg3] : memref<?xindex>        // positions[row]
  %8  = arith.addi %arg3, %c1 : index
  %9  = memref.load %3[%8] : memref<?xindex>           // positions[row + 1]
  %10 = scf.for %arg4 = %7 to %9 step %c1 iter_args(%arg5 = %6) -> (f32) {
    %11 = memref.load %4[%arg4] : memref<?xindex>      // coordinates[k] -> the column
    %12 = memref.load %0[%arg4] : memref<?xf32>        // values[k]
    %13 = memref.load %1[%11] : memref<8xf32>          // x[column]  <-- the indirection
    ...
```

Its **bounds are data**: `positions[row]` to `positions[row + 1]`, read at run time. A
dense inner loop would have run `0` to `8` regardless of what is in the matrix. This one
visits only entries that are stored, and it pays for that with `%13` — an indexed load
through `coordinates`, which is the irregular access sparse formats are made of and the
reason a sparse kernel is not simply a faster dense one.

None of that structure was written by anyone. It was derived from four words in an
attribute.

## Why this is the payoff of the ladder

Getting here required the loop nest to still be *data* at the moment the type was known.
Had the program already been lowered to `scf.for` — as
[`01-the-abstraction-ladder.md`](01-the-abstraction-ladder.md) shows it can be, one pass
too early — the bounds would be constants, the access pattern would be baked in, and
recovering "this dimension is compressed, iterate its positions instead" would be a
loop-restructuring problem rather than a code-generation one.

This is also the idea BCIR's lane typing shares, from the other direction. In
[`../22-bcir-approach/`](../22-bcir-approach/) a lane type is a property of a value that
the planner reads to decide what is legal and what it costs — not an instruction the
program issues. Here an encoding is a property of a tensor that the compiler reads to
decide which loop to emit. In both cases the *declaration* carries information the
operation deliberately does not, and in both cases the win comes from asking the type
before committing to the code.

## Pitfall: the operation name survives in the provenance

Sparsification stamps the loops it produced with where they came from:

```mlir
} {"Emitted from" = "linalg.generic"}
```

So a grep for `linalg.generic` in the output finds two matches and none of them is an
operation. This corpus's gate for this fixture therefore forbids **`linalg.yield`**, which
appears only inside a real generic body, and not the operation's name. The general rule is
the one the `pdl` and `gpu` fixtures need too: *a substring is not a match*, and a forbid
list that names the obvious string usually names something else as well.

## Pitfalls checklist

- Do not read a sparse kernel as a dense one with fewer iterations. The indexed load
  through `coordinates` is a different memory behaviour, and it is where the time goes.
- Do not lower to loops before the sparsifier has run. The encoding is only actionable
  while the iteration is still declared rather than emitted.
- Do not assume `--sparsification` alone is the pipeline. `--sparse-reinterpret-map` runs
  first here to put the map in the form the sparsifier expects.
- Do not grep an operation's name to prove it is gone. Provenance attributes quote it.
- Do not treat the encoding as documentation. It is the input to code generation; changing
  `compressed` to `dense` changes the emitted program, not a comment.

## Checks

[`examples/sparse-matvec-to-loops.mlir`](examples/sparse-matvec-to-loops.mlir) is a Tier 3
fixture in [`../autograder/mlir-examples.json`](../autograder/mlir-examples.json). Its
source must carry `sparse_tensor.encoding`, `compressed` and `linalg.generic`; the output
must contain `sparse_tensor.positions`, `sparse_tensor.coordinates`, `sparse_tensor.values`,
`scf.for` and `memref.load`, and must contain no `linalg.yield`.

Requiring all three `sparse_tensor` accessors is the load-bearing part. A pass that merely
bufferized the generic would produce `scf.for` and `memref.load` too; only the positions
and coordinates arrays show that the *encoding* was read.
