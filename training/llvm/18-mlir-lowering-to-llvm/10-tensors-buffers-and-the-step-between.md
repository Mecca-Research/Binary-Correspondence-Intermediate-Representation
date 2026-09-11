# Tensors, buffers, and the step between

Every chapter in this directory that lowers something starts from `memref`. None of them
says where the `memref` came from, and for a reader coming from LLVM IR that is the
confusing part: LLVM has pointers and allocations and nothing that behaves like a tensor,
so the type that needs explaining is the one the corpus skipped.

## Two types, one difference

```mlir
tensor<4xf32>    // a VALUE:  no address, immutable, produced by an operation
memref<4xf32>    // a BUFFER: an address, mutable, written through
```

That is the whole distinction, and everything else follows from it.

A `tensor` behaves like an `i32` does in LLVM IR. You cannot take its address, you cannot
store into it, and an operation that "modifies" one actually produces a second one. Two
`tensor` values are interchangeable if they hold the same elements, so a compiler may
duplicate, sink, or rematerialise them freely — the same freedom that makes SSA values
cheap to reason about.

A `memref` behaves like a pointer. It aliases, it has a lifetime, writing through it is
observable, and moving an operation across another operation's write is a question rather
than a given.

**Bufferization is the pass that crosses the line**, and crossing it is a one-way trip: it
exchanges an easy aliasing story for the ability to allocate a fixed amount of memory and
reuse it.

## Destination-passing style, before any of that

Look at [`examples/tensor-to-buffer.mlir`](examples/tensor-to-buffer.mlir):

```mlir
%e = tensor.empty() : tensor<4xf32>
%r = linalg.generic ... ins(%t : tensor<4xf32>) outs(%e : tensor<4xf32>) { ... }
     -> tensor<4xf32>
```

`tensor.empty` is the operation people misread first. **It allocates nothing.** It is a
shape request — it names the shape of the result so `linalg.generic` has an `outs` operand
to describe its output through, and the value it produces has no defined contents. Reading
an element of it is not a bug you will be told about; it is simply meaningless.

The `ins`/`outs` split is *destination-passing style*, and it exists precisely so that
bufferization has somewhere to put the answer later. On tensors the `outs` operand is
almost a formality — the operation returns a fresh value regardless. After bufferization
it becomes the actual destination.

## What the pass does, both ways

Run it and the boundary behaviour is the interesting half.

**Default: the signature keeps its tensors.**

```console
$ mlir-opt tensor-to-buffer.mlir --pass-pipeline='builtin.module(one-shot-bufferize)'
```
```mlir
func.func @scale(%arg0: tensor<4xf32>, %arg1: f32) -> tensor<4xf32> {
  %0 = bufferization.to_buffer %arg0 : tensor<4xf32> to memref<4xf32, strided<[?], offset: ?>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
  linalg.generic ... ins(%0 : memref<...>) outs(%alloc : memref<4xf32>) { ... }
  %1 = bufferization.to_tensor %alloc : memref<4xf32> to tensor<4xf32>
  return %1 : tensor<4xf32>
}
```

The body is buffers; the signature is still values; `bufferization.to_buffer` and
`bufferization.to_tensor` sit at the seam.

**Those two operations are this pipeline's `unrealized_conversion_cast`.** They mean the
same thing it does in
[`../24-mlir-infrastructure/04-bytecode-and-partial-lowering.md`](../24-mlir-infrastructure/04-bytecode-and-partial-lowering.md):
*these two representations will agree once the rest of the conversion lands.* A surviving
one in output you expected to be fully bufferized is a diagnosis, not a nuisance — it says
something on one side of it never converted.

**With `bufferize-function-boundaries=true`: the signature converts too.**

```console
$ mlir-opt tensor-to-buffer.mlir \
    --pass-pipeline='builtin.module(one-shot-bufferize{bufferize-function-boundaries=true})'
```
```mlir
func.func @scale(%arg0: memref<4xf32, strided<[?], offset: ?>>, %arg1: f32) -> memref<4xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
  linalg.generic ... ins(%arg0 : memref<...>) outs(%alloc : memref<4xf32>) { ... }
  return %alloc : memref<4xf32>
}
```

Both boundary casts are gone, because there is no longer a boundary to bridge.

## Three things to notice in that output

**The result type disappeared from `linalg.generic`.** On tensors it was
`... -> tensor<4xf32>`; on buffers the operation returns nothing at all and the answer is
in `%alloc`. That is the value/buffer difference made concrete: a value-producing operation
became a memory-writing one, and destination-passing style is what made the rewrite
mechanical rather than clever.

**`tensor.empty` became `memref.alloc`.** The shape request turned into the allocation it
was always standing in for. This is also where a real allocation enters the program — and
where the question of who frees it begins, which is what the rest of the `bufferization`
dialect (`dealloc`, `dealloc_tensor`, `clone`, `materialize_in_destination`) exists to
answer.

**The argument type gained a layout.** `tensor<4xf32>` became
`memref<4xf32, strided<[?], offset: ?>>`, not plain `memref<4xf32>`. A caller may pass a
slice of a larger buffer, so the *most general* layout is the honest one at a boundary: an
unknown stride and an unknown offset. The local allocation keeps the plain contiguous type
because bufferization knows exactly how it was made. If you have ever wondered why MLIR
memref types acquire `strided<...>` noise at function edges and not inside, that is why.

## Pitfalls checklist

- Do not read `tensor.empty` as an allocation. It is a shape, its contents are undefined,
  and reading from one is meaningless rather than diagnosable.
- Do not expect `outs` on a tensor operation to be written into. It is the destination the
  *bufferized* form will use; before that it only carries the shape.
- Do not delete a surviving `bufferization.to_tensor` to tidy the output. It is telling
  you a conversion is incomplete, exactly as `unrealized_conversion_cast` does.
- Do not assume a bufferized function signature takes a contiguous `memref`. At a boundary
  the general strided layout is the correct one, and code that assumes otherwise breaks on
  the first caller that passes a slice.
- Do not bufferize and then expect tensor-level reasoning to still apply. Once there are
  addresses, motion across a write is a question about aliasing.

## Checks

[`examples/tensor-to-buffer.mlir`](examples/tensor-to-buffer.mlir) is registered in
[`../autograder/mlir-examples.json`](../autograder/mlir-examples.json) as a Tier 3
conversion and graded by
[`../tools/verify-mlir-lowering.py`](../tools/verify-mlir-lowering.py): it runs the
boundary-bufferizing pipeline above and requires `memref.alloc` and a plain
`memref<4xf32>` to appear, while forbidding `tensor.empty`, any `-> tensor<` result, and
**both** boundary casts. That last one is the check that matters — forbidding
`bufferization.to_tensor` and `bufferization.to_buffer` in the fully-bufferized output is
what would catch a conversion that stopped halfway and left the seam behind.
