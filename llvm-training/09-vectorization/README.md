# Auto-Vectorization: Loop Vectorizer and SLP Vectorizer

## TL;DR

LLVM has two main auto-vectorizers:

- **Loop Vectorizer**: widens a loop so one vector iteration performs the work of several original scalar iterations.
- **SLP Vectorizer**: combines nearby independent scalar operations into vector operations, usually inside one basic block or a small straight-line region.

Both are legality- and profitability-driven. Even legal vectorization may be skipped when the target cost model predicts scalar code is better.

Official references:

- [LLVM Auto-Vectorization documentation](https://llvm.org/docs/Vectorizers.html)
- [LLVM Optimization Remarks documentation](https://llvm.org/docs/Remarks.html)

## Chapter dispatcher

| Need | Read |
|---|---|
| Loop-carried legality, trip counts, reductions, and runtime checks | [`01-loop-vectorizer.md`](01-loop-vectorizer.md) |
| Straight-line scalar packing and lane-tree profitability | [`02-slp-vectorizer.md`](02-slp-vectorizer.md) |
| Masks, tails, predicated operations, and scalable-vector cautions | [`03-vector-predication.md`](03-vector-predication.md) |
| What makes vectorization legal or blocked | [`04-vectorization-legality.md`](04-vectorization-legality.md) |
| Commands, expected observations, and forced-width experiments | [`05-example-walkthroughs.md`](05-example-walkthroughs.md) |
| IR clues: `<N x T>`, vector loads/stores, `shufflevector`, reductions | [`06-recognizing-vector-ir.md`](06-recognizing-vector-ir.md) |
| Advanced masked and interleaved memory access patterns | [`07-masked-and-interleaved-access.md`](07-masked-and-interleaved-access.md) |

## Example files

- [`examples/sum-loop.c`](examples/sum-loop.c) — small C loop for `clang` vectorization diagnostics.
- [`examples/sum-loop.ll`](examples/sum-loop.ll) — original scalar LLVM IR loop kept as a compact `opt` input.
- [`examples/sum-loop-before.ll`](examples/sum-loop-before.ll) — scalar loop input before Loop Vectorization.
- [`examples/sum-loop-after-loop-vectorize.ll`](examples/sum-loop-after-loop-vectorize.ll) — cleaned-up Loop Vectorizer output with vector loop bodies and a reduction.
- [`examples/slp-scalars.ll`](examples/slp-scalars.ll) — original straight-line scalar operations kept as a compact SLP input.
- [`examples/slp-scalars-before.ll`](examples/slp-scalars-before.ll) — scalar straight-line operations before SLP Vectorization.
- [`examples/slp-scalars-after-slp.ll`](examples/slp-scalars-after-slp.ll) — cleaned-up SLP output with vector operations and a lane shuffle.
- [`examples/not-vectorizable-dependency.ll`](examples/not-vectorizable-dependency.ll) — loop-carried memory dependency example that should remain scalar.
- [`examples/not-vectorizable-call.ll`](examples/not-vectorizable-call.ll) — unknown-call example that should produce a missed-vectorization explanation.
- [`examples/masked-load-store-before.ll`](examples/masked-load-store-before.ll) — scalar conditional-store loop for masked vectorization experiments.
- [`examples/masked-load-store-after-vectorize.ll`](examples/masked-load-store-after-vectorize.ll) — cleaned-up masked-store vector IR snapshot.
- [`examples/interleaved-access-before.ll`](examples/interleaved-access-before.ll) — scalar AoS-to-SoA deinterleave loop with stride-2 input fields.
- [`examples/interleaved-access-after-vectorize.ll`](examples/interleaved-access-after-vectorize.ll) — cleaned-up interleaved-access vector IR snapshot with `shufflevector` deinterleaving.

## Loop Vectorizer vs SLP Vectorizer

| Vectorizer | Looks for | Typical shape | Result |
|---|---|---|---|
| Loop Vectorizer | Parallel work across loop iterations | `for (i = 0; i < n; ++i)` touching `A[i]`, `B[i]`, `C[i]` | A vector loop, plus possible scalar prologue/epilogue and runtime checks |
| SLP Vectorizer | Parallel work among independent scalar statements | several similar adds, multiplies, loads, or stores in straight-line code | Vector instructions in the same basic block or region |

The Loop Vectorizer asks: “Can iteration `i`, `i+1`, `i+2`, ... execute as one wider iteration?” The SLP Vectorizer asks: “Can these nearby scalar operations be bundled as lanes of one vector instruction?”

## See also

- [`../06-metadata/03-profile-and-optimization-metadata.md`](../06-metadata/03-profile-and-optimization-metadata.md) — loop metadata and optimization hints.
- [`../02-types/02-composite-types.md`](../02-types/02-composite-types.md) — vector types as LLVM composite types.
- [`../04-memory/02-load-store.md`](../04-memory/02-load-store.md) — load/store syntax used by vector memory operations.
