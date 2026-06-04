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
