# 01 — The abstraction ladder: `tosa` to `linalg` to loops

A machine-learning compiler is a stack of names. At the top an operation is called
`tosa.matmul` and everything about it — shapes, batching, accumulation order — lives in
that name. At the bottom there is a loop nest doing multiply-accumulate. Every level in
between exists because *some* transformation is easy at that level and impossible at the
others.

This chapter walks the ladder with a real toolchain, one rung per fixture, because the
interesting facts are all at the joints.

## Rung one: a framework operation becomes a linear-algebra operation

[`examples/tosa-matmul-to-linalg.mlir`](examples/tosa-matmul-to-linalg.mlir) is one
`tosa.matmul`. `--tosa-to-linalg-named` turns it into `linalg.batch_matmul` — and into two
operations the name had been hiding:

```mlir
%0 = tensor.empty() : tensor<1x2x4xf32>
%1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
%2 = linalg.batch_matmul ins(%arg0, %arg1 : ...) outs(%1 : ...)
```

`tosa.matmul` has no output operand: the result appears from nowhere, as it does in every
framework API. `linalg` works in destination-passing style —
[`../18-mlir-lowering-to-llvm/10-tensors-buffers-and-the-step-between.md`](../18-mlir-lowering-to-llvm/10-tensors-buffers-and-the-step-between.md)
is where that idea is introduced — so the destination has to become explicit, and so does
the fact that an accumulator starts at zero. A `tensor.empty` plus a `linalg.fill` is what
"returns a new tensor" costs once someone has to allocate it.

**The rule:** descending a level does not just rename operations. It forces previously
implicit obligations — who allocates, what the initial value is — to become operations
somebody can see, schedule and price.

## Rung two: a named operation becomes a `linalg.generic`

`--linalg-generalize-named-ops` replaces `linalg.matmul` with the thing every named
`linalg` op is sugar for:

```mlir
#map  = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

%0 = linalg.generic {
       indexing_maps = [#map, #map1, #map2],
       iterator_types = ["parallel", "parallel", "reduction"]}
     ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32>)
     outs(%arg2 : tensor<2x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
} -> tensor<2x4xf32>
```

This is the level worth understanding, because it is a loop nest **written as data**. Three
loop dimensions `d0, d1, d2`; three `affine_map`s saying how each operand is indexed by
them; a body saying what happens at one point; and `iterator_types` declaring that `d0` and
`d1` are parallel while `d2` is a reduction.

Nothing there is a loop yet, and that is the point. Tiling, fusion, interchange and
vectorization are rewrites on the maps and the iterator list — cheap, local, and checkable.
The same transformations applied to a `scf.for` nest are loop-dependence analysis, which is
the problem `linalg` exists to avoid having to solve.

## Rung three: the generic becomes an actual loop nest — after bufferization

`--convert-linalg-to-loops` produces exactly what the maps described:

```mlir
scf.for %arg3 = %c0 to %c2 step %c1 {
  scf.for %arg4 = %c0 to %c4 step %c1 {
    scf.for %arg5 = %c0 to %c3 step %c1 {
      %0 = memref.load %arg0[%arg3, %arg5] : memref<2x3xf32, strided<[?, ?], offset: ?>>
      %1 = memref.load %arg1[%arg5, %arg4] : memref<3x4xf32, strided<[?, ?], offset: ?>>
      %2 = memref.load %arg2[%arg3, %arg4] : memref<2x4xf32, strided<[?, ?], offset: ?>>
      %3 = arith.mulf %0, %1 : f32
      %4 = arith.addf %2, %3 : f32
      memref.store %4, %arg2[%arg3, %arg4] : memref<2x4xf32, strided<[?, ?], offset: ?>>
    }
  }
}
```

`d0` became the outer loop, `d1` the middle, `d2` the innermost; the three `affine_map`s
became the three subscript lists. And `iterator_types` became *nothing at all*. After this
pass, "these two loops are parallel and this one is a reduction" is a fact somebody would
have to re-derive by analysing the loads and stores.

That is the real cost of descending, and it is why the order of a pipeline matters more in
an ML stack than the choice of any individual pass: **information is destroyed downward,
and the transformations that need it must run before it is gone.**

## The joint where the pipeline silently does nothing

Note the types in that loop nest: `memref`, not `tensor`. `--convert-linalg-to-loops`
rewrites `linalg` on **buffers**, so the registered pipeline bufferizes first.

Run it against tensors instead and the pass does not fail. It matches nothing, exits zero
and leaves the `linalg.generic` exactly where it was.
[`examples/linalg-to-loops-needs-buffers.mlir`](examples/linalg-to-loops-needs-buffers.mlir)
is that run, registered so the leftover operation is its declared outcome.

A pass that reports success and changes nothing is the worst failure mode in a compiler
pipeline, because every downstream stage then fails somewhere else with a message about
something unrelated. It is the same shape as the `shape` dialect's missing normalizer in
[`../18-mlir-lowering-to-llvm/11-shapes-as-values.md`](../18-mlir-lowering-to-llvm/11-shapes-as-values.md),
and the question to ask is identical: *did the pass have nothing to do, or nothing it could
recognise?*

## Where BCIR sits on this ladder

BCIR does not own an instruction selector or a register allocator, and it does not own this
ladder either — it plans over it. The relevant correspondence is at rung two, not rung
three: `iterator_types` and the `affine_map`s are a *declared* structure, and a declared
structure is something a cost model can price without re-deriving it.
[`../22-bcir-approach/`](../22-bcir-approach/) is where K_BCIR and the twelve cost axes are
taught; the thing to carry over from here is that the level at which a plan is still legible
is the level at which it can be priced, and that level is above `scf.for`.

## Pitfalls checklist

- Do not assume a pass that exits zero did anything. Check for the operation you expected
  to disappear, not for the exit code.
- Do not lower `linalg` to loops before you have finished doing `linalg`-level work.
  Tiling and fusion after the loops exist is a different, harder problem.
- Do not read `linalg.generic` as "a fallback for operations without a name". It is the
  definition; the named operations are sugar over it.
- Do not bufferize later than you must, or earlier. `--convert-linalg-to-loops` needs
  buffers; the tensor-level rewrites need tensors.
- Do not treat `tensor.empty` + `linalg.fill` as pass noise. They are the allocation and
  the accumulator initialisation that the framework-level name concealed.

## Checks

All three fixtures are Tier 3 entries in
[`../autograder/mlir-examples.json`](../autograder/mlir-examples.json), graded by
[`../tools/verify-mlir-examples.sh`](../tools/verify-mlir-examples.sh).

- [`examples/tosa-matmul-to-linalg.mlir`](examples/tosa-matmul-to-linalg.mlir) must produce
  `linalg.batch_matmul`, `linalg.fill` and `tensor.empty`, and must not leave
  `tosa.matmul` behind. Requiring the fill and the empty is deliberate: they are the claim
  that descending made an obligation explicit, not merely that a name changed.
- [`examples/linalg-to-loops.mlir`](examples/linalg-to-loops.mlir) must produce `scf.for`,
  `memref.load`, `memref.store`, `arith.mulf` and `arith.addf`, with no `linalg.matmul`,
  `linalg.generic` or `linalg.yield` surviving.
- [`examples/linalg-to-loops-needs-buffers.mlir`](examples/linalg-to-loops-needs-buffers.mlir)
  is registered with an `illegal-ops-remaining` outcome naming `linalg.generic`, so the
  silent no-op above is a checked fact. If a future MLIR teaches that pass to handle
  tensors, the fixture goes red and this chapter is rewritten.
