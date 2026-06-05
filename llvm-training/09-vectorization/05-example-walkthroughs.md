# Vectorization Example Walkthroughs

## Useful commands

From the repository root, start with the checked-in wrapper when you want one
short demo that prints commands, requests `clang` vectorization remarks, and
prints forced loop-vectorizer IR output:

```sh
llvm-training/tools/demo-vectorize.sh
```

Then vary the underlying commands directly:

```sh
# Show successful loop-vectorization remarks from clang.
clang -O3 -Rpass=loop-vectorize \
  llvm-training/09-vectorization/examples/sum-loop.c -c -o /tmp/sum-loop.o

# Show missed loop-vectorization remarks from clang.
clang -O3 -Rpass-missed=loop-vectorize \
  llvm-training/09-vectorization/examples/sum-loop.c -c -o /tmp/sum-loop.o

# Run only the Loop Vectorizer over scalar IR.
opt -S -passes=loop-vectorize \
  llvm-training/09-vectorization/examples/sum-loop-before.ll -o -

# Compare against the cleaned-up expected Loop Vectorizer shape.
cat llvm-training/09-vectorization/examples/sum-loop-after-loop-vectorize.ll

# Run only the SLP Vectorizer over scalar IR.
opt -S -passes=slp-vectorizer \
  llvm-training/09-vectorization/examples/slp-scalars-before.ll -o -

# Compare against the cleaned-up expected SLP shape.
cat llvm-training/09-vectorization/examples/slp-scalars-after-slp.ll

# Ask for loop-vectorization missed remarks while running opt.
opt -S -passes=loop-vectorize \
  -pass-remarks-missed=loop-vectorize \
  llvm-training/09-vectorization/examples/not-vectorizable-dependency.ll -o -

opt -S -passes=loop-vectorize \
  -pass-remarks-missed=loop-vectorize \
  llvm-training/09-vectorization/examples/not-vectorizable-call.ll -o -

# Inspect advanced predicated and interleaved Loop Vectorizer cases.
opt -S -passes=loop-vectorize \
  -pass-remarks=loop-vectorize \
  -pass-remarks-analysis=loop-vectorize \
  -pass-remarks-missed=loop-vectorize \
  llvm-training/09-vectorization/examples/masked-load-store-before.ll -o -

opt -S -passes=loop-vectorize \
  -pass-remarks=loop-vectorize \
  -pass-remarks-analysis=loop-vectorize \
  -pass-remarks-missed=loop-vectorize \
  llvm-training/09-vectorization/examples/interleaved-access-before.ll -o -

# Run the normal O3 optimization pipeline and print IR.
opt -S -passes='default<O3>' \
  llvm-training/09-vectorization/examples/sum-loop-before.ll -o -
```

Notes:

- Newer LLVM tools use the `-passes=...` pass-manager syntax shown above.
- `clang -Rpass...` diagnostics are emitted while compiling from source. Use
  `-g` or `-fsave-optimization-record` when you need richer source locations or
  serialized optimization records.
- Optimization remarks come in passed, missed, and analysis forms. The exact pass
  names in remarks are part of your experiment: try `loop-vectorize`, `slp-vectorizer`,
  or a broader regex such as `.`.
- The checked-in `*-after-*` files are intentionally cleaned-up teaching
  snapshots. LLVM's exact output depends on version, target, target features,
  cost model, vector width, and interleave count.

## Expected observations by example

| Example | Command | What to look for |
|---|---|---|
| `sum-loop-before.ll` | `opt -S -passes=loop-vectorize llvm-training/09-vectorization/examples/sum-loop-before.ll -o -` | A vector body may appear for `add_arrays`, and `sum_loop` may show a reduction pattern. Search for `<4 x i32>` or another `<N x i32>` lane count. |
| `sum-loop-after-loop-vectorize.ll` | `cat llvm-training/09-vectorization/examples/sum-loop-after-loop-vectorize.ll` | Teaching snapshot with vector loads, vector stores, and `@llvm.vector.reduce.add.v4i32` combining lanes into a scalar result. |
| `slp-scalars-before.ll` | `opt -S -passes=slp-vectorizer llvm-training/09-vectorization/examples/slp-scalars-before.ll -o -` | Packed straight-line operations may become `<4 x i32>` loads, adds, and stores. |
| `slp-scalars-after-slp.ll` | `cat llvm-training/09-vectorization/examples/slp-scalars-after-slp.ll` | Teaching snapshot with vector arithmetic and a `shufflevector` that changes lane order before storing. |
| `not-vectorizable-dependency.ll` | `opt -S -passes=loop-vectorize -pass-remarks-missed=loop-vectorize llvm-training/09-vectorization/examples/not-vectorizable-dependency.ll -o -` | Missed-vectorization diagnostics should point at a loop-carried dependency, and the loop should remain scalar. |
| `not-vectorizable-call.ll` | `opt -S -passes=loop-vectorize -pass-remarks-missed=loop-vectorize llvm-training/09-vectorization/examples/not-vectorizable-call.ll -o -` | Missed-vectorization diagnostics should explain that the unknown call cannot be vectorized safely or profitably. |
| `masked-load-store-before.ll` | `opt -S -passes=loop-vectorize -pass-remarks=loop-vectorize -pass-remarks-analysis=loop-vectorize -pass-remarks-missed=loop-vectorize llvm-training/09-vectorization/examples/masked-load-store-before.ll -o -` | Look for a lane mask derived from the compare and whether the target chooses a masked store, scalarization, or a missed/profitability remark. |
| `interleaved-access-before.ll` | `opt -S -passes=loop-vectorize -pass-remarks=loop-vectorize -pass-remarks-analysis=loop-vectorize -pass-remarks-missed=loop-vectorize llvm-training/09-vectorization/examples/interleaved-access-before.ll -o -` | Look for stride-2 recognition, wide loads, `shufflevector` deinterleaving, or analysis remarks about shuffle/interleave cost. |


## Forcing experiments

The vectorizers normally choose vector width and interleave count with a target
cost model. For learning, force them to make the decision visible:

```sh
clang -O3 -Rpass=loop-vectorize \
  -mllvm -force-vector-width=4 \
  -mllvm -force-vector-interleave=2 \
  llvm-training/09-vectorization/examples/sum-loop.c -c -o /tmp/sum-loop.o

opt -S -passes=loop-vectorize \
  -force-vector-width=4 \
  -force-vector-interleave=2 \
  llvm-training/09-vectorization/examples/sum-loop-before.ll -o -
```

Use forced settings as experiments, not as proof that a setting is fastest. A
forced vector width can make LLVM generate vector IR even when the cost model
would normally reject it. For the advanced fixtures, compare forced output with
[`07-masked-and-interleaved-access.md`](07-masked-and-interleaved-access.md) and
the checked-in `*-after-vectorize.ll` teaching snapshots.

## See also

- [`README.md`](README.md) — vectorization dispatcher and example list.
- [`04-vectorization-legality.md`](04-vectorization-legality.md) — legality facts that explain the missed examples.
- [`06-recognizing-vector-ir.md`](06-recognizing-vector-ir.md) — IR clues to inspect in command output.
