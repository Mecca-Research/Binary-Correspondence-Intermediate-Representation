# SLP Vectorizer

The SLP Vectorizer combines independent scalar operations inside a basic block or
short straight-line region. Unlike the Loop Vectorizer, it does not need a loop;
it searches for isomorphic scalar instruction trees that can become one vector
instruction tree.

## Best input shape

SLP likes repeated scalar lanes such as:

```llvm
%a0 = load i32, ptr %p0
%a1 = load i32, ptr %p1
%s0 = add i32 %a0, %b0
%s1 = add i32 %a1, %b1
```

If the operations have matching opcodes, compatible types, and profitable data
movement, SLP can pack them into vector loads/inserts, vector arithmetic, and
extracts or vector stores.

## Typical command loop

```bash
opt -S -passes='slp-vectorizer' llvm-training/09-vectorization/examples/slp-scalars.ll -o /tmp/slp.ll
opt -passes=verify /tmp/slp.ll -o /dev/null
```

Compare [`examples/slp-scalars-before.ll`](examples/slp-scalars-before.ll) with
[`examples/slp-scalars-after-slp.ll`](examples/slp-scalars-after-slp.ll).

## What to look for in IR

Successful SLP output often has:

- `insertelement` chains that pack scalar values;
- vector `add`, `mul`, `icmp`, or other lane-wise operations;
- `extractelement` if the surrounding ABI still needs scalar results;
- vector stores if the packed result is written contiguously;
- `shufflevector` when lane order has to be rearranged.

## Common misses

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Similar operations stay scalar | Types, flags, or opcodes do not match | Canonicalize with `instcombine` first. |
| Too many inserts/extracts | Packing cost exceeds vector compute benefit | Keep data contiguous or leave scalar. |
| Loads cannot be packed | Pointers are not consecutive or aliasing is unclear | Emit clearer GEPs and alignment facts. |
| One lane has a call | No vectorizable equivalent for that lane | Use an intrinsic or split the sequence. |

## BCIR lowering advice

When lowering batches of independent BCIR lane work, keep lane operations in a
regular order and avoid hiding lane identity behind opaque helper calls. SLP can
recover straight-line SIMD only when the scalar tree remains visible in IR.

## Lane-packing walkthrough

The pair [`examples/slp-scalars-before.ll`](examples/slp-scalars-before.ll) and
[`examples/slp-scalars-after-slp.ll`](examples/slp-scalars-after-slp.ll) shows
both the plain packed case and a lane-reordering case.

### 1. Run SLP and force a teaching-friendly threshold

```bash
opt -passes=verify llvm-training/09-vectorization/examples/slp-scalars-before.ll -o /dev/null
opt -S -passes='slp-vectorizer' -slp-threshold=-999 \
  llvm-training/09-vectorization/examples/slp-scalars-before.ll \
  -o /tmp/slp-scalars-after.ll
opt -passes=verify /tmp/slp-scalars-after.ll -o /dev/null
```

The negative threshold is useful for training because it makes SLP more willing
to show the transformation even when the default target cost model would keep the
small scalar sequence unchanged.

### 2. Pack matching scalar lanes

In `@slp_scalars`, the input has four isomorphic scalar trees:

```llvm
%s0 = add i32 %a0, %b0
%s1 = add i32 %a1, %b1
%s2 = add i32 %a2, %b2
%s3 = add i32 %a3, %b3
```

The teaching snapshot shows the packed form:

```llvm
%wide.a = load <4 x i32>, ptr %a, align 4
%wide.b = load <4 x i32>, ptr %b, align 4
%wide.sum = add <4 x i32> %wide.a, %wide.b
store <4 x i32> %wide.sum, ptr %c, align 4
```

Read lane `k` in `%wide.sum` as the old scalar `%sk`: lane 0 is `%a0 + %b0`,
lane 1 is `%a1 + %b1`, and so on. The vector `add` is not a horizontal sum; it
performs four independent additions side by side.

### 3. Explain `shufflevector` in the reordered store

`@slp_shuffle_candidate` computes the same four lane values but stores them in a
permuted order. The scalar source writes `%s0`, `%s2`, `%s1`, `%s3` to
contiguous memory. After packing the arithmetic in natural order, LLVM needs one
lane permutation before the vector store:

```llvm
%reordered = shufflevector <4 x i32> %wide.sum, <4 x i32> poison,
                          <4 x i32> <i32 0, i32 2, i32 1, i32 3>
store <4 x i32> %reordered, ptr %c, align 4
```

The mask `<0, 2, 1, 3>` means:

| Output lane | Input lane selected | Old scalar value |
| --- | --- | --- |
| 0 | 0 | `%s0` |
| 1 | 2 | `%s2` |
| 2 | 1 | `%s1` |
| 3 | 3 | `%s3` |

The second vector operand is `poison` because this shuffle only needs lanes from
the first vector. If a mask element selected lane 4 or higher in this two-vector
shuffle, it would read from the second operand instead.

### 4. Debug a disappointing SLP result

When a local LLVM build does not produce the compact snapshot, diff the result in
small stages:

```bash
opt -S -passes='instcombine,slp-vectorizer,instcombine' -slp-threshold=-999 \
  llvm-training/09-vectorization/examples/slp-scalars-before.ll \
  -o /tmp/slp-scalars-canonicalized.ll
diff -u llvm-training/09-vectorization/examples/slp-scalars-before.ll /tmp/slp-scalars-canonicalized.ll
```

If only inserts/extracts appear, the vectorizer found a pack but the surrounding
memory layout or ABI still forced scalar boundaries. If no vector operations
appear, inspect lane regularity first: mismatched opcodes, mixed integer widths,
unknown calls, or non-consecutive pointers commonly prevent packing.
