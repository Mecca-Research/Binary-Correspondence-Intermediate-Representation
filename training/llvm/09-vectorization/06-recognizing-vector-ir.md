# Recognizing Vectorized IR

## How to recognize vectorized IR

Look for these patterns in the output of `opt -S` or `clang -S -emit-llvm`.

### `<N x T>` vector types

Vector IR uses types such as `<4 x i32>` or `<8 x float>`:

```llvm
%sum.vec = add <4 x i32> %a.vec, %b.vec
```

`N` is the lane count and `T` is the element type.

### Vector loads and stores

Vectorized contiguous memory often appears as vector memory operations:

```llvm
%wide.load = load <4 x i32>, ptr %p, align 4
store <4 x i32> %sum.vec, ptr %q, align 4
```

Depending on alignment, target, legality, and the chosen plan, LLVM may instead
use scalar loads inserted into vectors, masked loads/stores, or target-specific
intrinsics.

### `shufflevector`

`shufflevector` rearranges lanes, concatenates pieces, extracts groups, or
builds vectors from scalar values. It is especially common in SLP output and in
code that needs lane permutations.

### Reduction intrinsics or reduction patterns

A vectorized integer sum may end with a reduction intrinsic:

```llvm
%sum = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %vec)
```

Some LLVM versions or optimization stages may show an explicit tree of extracts
and adds instead. Both represent "combine the vector lanes into one scalar".

## Suggested workflow

1. Compile the C example with `clang -O3 -Rpass=loop-vectorize` and read the
   remark.
2. Compile it again with `-Rpass-missed=loop-vectorize` after changing the loop
   into a dependency-heavy form.
3. Run `opt -S -passes=loop-vectorize` on `sum-loop-before.ll` and search for
   `<N x T>` types. Compare with `sum-loop-after-loop-vectorize.ll`.
4. Run `opt -S -passes=slp-vectorizer` on `slp-scalars-before.ll` and look for
   vector arithmetic, vector stores, and `shufflevector`. Compare with
   `slp-scalars-after-slp.ll`.
5. Repeat with `-force-vector-width` and `-force-vector-interleave` to see how
   the generated IR changes.

## Pitfalls

- **Vector IR does not guarantee vector machine code.** Later passes and target
  lowering decide how vector IR maps to actual instructions.
- **No remark does not always mean no vectorization.** Check the pass name and
  remark regex, and inspect the IR or assembly.
- **`restrict` and `noalias` are promises.** Do not add them to make a demo
  vectorize unless the program really satisfies the aliasing contract.
- **Floating-point reductions need care.** Reassociation can change results;
  fast-math flags or ordered-reduction support determine what is legal and
  profitable.
- **Forced vectorization is a teaching tool.** It can create slower code or code
  that later passes simplify away.

## See also

- [`../06-metadata/03-profile-and-optimization-metadata.md`](../06-metadata/03-profile-and-optimization-metadata.md) — loop metadata and optimization hints.
- [`../02-types/02-composite-types.md`](../02-types/02-composite-types.md) — vector types as LLVM composite types.
- [`../04-memory/02-load-store.md`](../04-memory/02-load-store.md) — load/store syntax used by vector memory operations.
- [LLVM Auto-Vectorization documentation](https://llvm.org/docs/Vectorizers.html)
- [LLVM Optimization Remarks documentation](https://llvm.org/docs/Remarks.html)
