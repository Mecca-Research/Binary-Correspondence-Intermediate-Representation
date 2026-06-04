# Loop Vectorizer

The Loop Vectorizer turns repeated scalar loop iterations into vector operations.
It looks for a loop whose iterations are independent or can be made independent
with runtime checks, then executes multiple iterations per vector step.

## Best input shape

A friendly loop has:

- a single canonical induction variable;
- a predictable trip count or a guardable runtime trip count;
- unit-stride or simple-stride loads and stores;
- no unsafe loop-carried dependence;
- calls that are known intrinsics, math functions with vector forms, or otherwise
  legal to vectorize;
- alias information good enough to prove loads and stores do not conflict, or a
  loop shape where runtime alias checks are legal.

## Typical command loop

```bash
opt -S -passes='loop-vectorize' llvm-training/09-vectorization/examples/sum-loop.ll -o /tmp/sum-loop-vectorized.ll
opt -passes=verify /tmp/sum-loop-vectorized.ll -o /dev/null
```

For source-driven diagnostics:

```bash
clang -O3 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -c sum-loop.c -o /tmp/sum-loop.o
```

## What to look for in IR

Successful loop vectorization often introduces:

- a vector loop body using `<N x T>` operations;
- vector induction updates by the vector width;
- scalar prologue or epilogue loops for unaligned or remainder iterations;
- runtime checks for aliasing or trip-count conditions;
- reduction intrinsics or patterns for horizontal sums/min/max operations.

Compare [`examples/sum-loop-before.ll`](examples/sum-loop-before.ll) with
[`examples/sum-loop-after-loop-vectorize.ll`](examples/sum-loop-after-loop-vectorize.ll)
for the compact before/after shape.

## Common misses

| Symptom | Likely cause | Next file |
| --- | --- | --- |
| Remark says unsafe dependent memory operations | Loop-carried store/load dependence | [`../08-pitfalls/12-vectorization-blocked-by-aliasing.md`](../08-pitfalls/12-vectorization-blocked-by-aliasing.md) |
| Remark says call cannot be vectorized | Callee has no vector form or attributes | [`../13-advanced-ir/04-attributes.md`](../13-advanced-ir/04-attributes.md) |
| Only scalar cleanup remains | Trip count too small or profitability says no | [`README.md`](README.md) forcing experiments |
| IR verifies but is not faster | Target cost model rejected useful width or memory layout is poor | [`03-vector-predication.md`](03-vector-predication.md) for masks/tails |

## BCIR lowering advice

Emit simple induction variables and explicit alignment/noalias facts only when
proven. Vectorizers are good at canonical patterns; unusual pointer arithmetic or
missing alias facts can make a mathematically vectorizable BCIR loop look unsafe.

## Full scalar-to-vector walkthrough

This chapter keeps both a scalar input and a cleaned-up vectorized snapshot in
`examples/` so you can separate three questions: whether the input verifies,
whether the vectorizer reports success or failure, and what changed in the IR.

### 1. Verify the scalar input

```bash
opt -passes=verify llvm-training/09-vectorization/examples/sum-loop-before.ll -o /dev/null
```

The `@add_arrays` loop in that file is intentionally friendly: the three pointer
arguments are `noalias`, the loads and store are unit stride, and the induction
variable has a simple `%i.next = add ... %i, 1` recurrence.

### 2. Run the loop vectorizer with remarks

```bash
opt -S -passes='loop-vectorize' \
  -pass-remarks=loop-vectorize \
  -pass-remarks-missed=loop-vectorize \
  llvm-training/09-vectorization/examples/sum-loop-before.ll \
  -o /tmp/sum-loop-vectorized.ll \
  2> /tmp/sum-loop-vectorizer.remarks
cat /tmp/sum-loop-vectorizer.remarks
opt -passes=verify /tmp/sum-loop-vectorized.ll -o /dev/null
```

On many targets the reduction in `@sum_loop` is more sensitive to legality and
profitability choices than the map-style `@add_arrays` loop. For deterministic
training output, force a teaching width:

```bash
opt -S -passes='loop-vectorize' -force-vector-width=4 -force-vector-interleave=1 \
  llvm-training/09-vectorization/examples/sum-loop-before.ll \
  -o /tmp/sum-loop-vectorized-forced.ll
```

Then compare the generated file with the checked-in snapshot:

```bash
diff -u \
  llvm-training/09-vectorization/examples/sum-loop-after-loop-vectorize.ll \
  /tmp/sum-loop-vectorized-forced.ll
```

LLVM version, target features, and cost-model details can change block names or
cleanup-loop shape, so treat the checked-in `*-after-loop-vectorize.ll` file as a
stable teaching snapshot rather than a byte-for-byte contract.

### 3. Read the vectorized shape

For `@add_arrays`, the important before/after correspondence is:

| Scalar input | Vectorized output |
| --- | --- |
| `%i` advances by `1` | `%index` advances by vector width `4` |
| `load i32` from `%a[i]` and `%b[i]` | `load <4 x i32>` from the same base pointers |
| `%sum = add i32 %va, %vb` | `%wide.sum = add <4 x i32> %wide.load.a, %wide.load.b` |
| `store i32 %sum` | `store <4 x i32> %wide.sum` |
| one scalar loop | vector loop plus scalar remainder loop |

The `vector.check` and `middle.block` blocks are not noise: they guard small trip
counts and decide whether the scalar cleanup loop is needed for `n % 4 != 0`.
The scalar loop remains because the vectorizer must preserve behavior for array
lengths that are not a multiple of the chosen vector width.

### 4. Capture missed-vectorization remarks

Use deliberately hostile examples to learn what a miss looks like:

```bash
opt -S -passes='loop-vectorize' \
  -pass-remarks-missed=loop-vectorize \
  llvm-training/09-vectorization/examples/not-vectorizable-dependency.ll \
  -o /tmp/not-vectorizable-dependency.ll \
  2> /tmp/not-vectorizable-dependency.remarks
cat /tmp/not-vectorizable-dependency.remarks
```

The dependency example stores `a[i]` from `a[i - 1]`, so each iteration consumes
a value produced by the previous iteration. That is a true loop-carried
dependence, not merely missing syntax. The unknown-call example shows a different
miss:

```bash
opt -S -passes='loop-vectorize' \
  -pass-remarks-missed=loop-vectorize \
  llvm-training/09-vectorization/examples/not-vectorizable-call.ll \
  -o /tmp/not-vectorizable-call.ll \
  2> /tmp/not-vectorizable-call.remarks
cat /tmp/not-vectorizable-call.remarks
```

Here the scalar call to `@opaque` blocks vectorization because the optimizer does
not have a vector equivalent or enough side-effect information. Fixes are
semantic, not cosmetic: expose a vector intrinsic, add correct attributes, or
split that operation out of the vectorized loop.
