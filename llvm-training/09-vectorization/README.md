# Auto-Vectorization: Loop Vectorizer and SLP Vectorizer

## TL;DR

LLVM has two main auto-vectorizers:

- **Loop Vectorizer**: widens a loop so one vector iteration performs the
  work of several original scalar iterations.
- **SLP Vectorizer**: combines independent scalar operations that already
  appear near each other into vector operations, usually inside one basic
  block or across a small straight-line region.

Official references:

- [LLVM Auto-Vectorization documentation](https://llvm.org/docs/Vectorizers.html)
- [LLVM Optimization Remarks documentation](https://llvm.org/docs/Remarks.html)

Start with the examples:

- [`examples/sum-loop.c`](examples/sum-loop.c) — small C loop for `clang`
  vectorization diagnostics.
- [`examples/sum-loop.ll`](examples/sum-loop.ll) — scalar LLVM IR loop that
  `opt` can feed to the Loop Vectorizer.
- [`examples/slp-scalars.ll`](examples/slp-scalars.ll) — straight-line scalar
  operations that the SLP Vectorizer can pack together.

## Loop Vectorizer vs SLP Vectorizer

| Vectorizer | Looks for | Typical shape | Result |
|---|---|---|---|
| Loop Vectorizer | Parallel work across loop iterations | `for (i = 0; i < n; ++i)` touching `A[i]`, `B[i]`, `C[i]` | A vector loop, plus possible scalar prologue/epilogue and runtime checks |
| SLP Vectorizer | Parallel work among independent scalar statements | several similar adds, multiplies, loads, or stores in straight-line code | Vector instructions in the same basic block or region |

The Loop Vectorizer asks: "Can iteration `i`, `i+1`, `i+2`, ... execute as one
wider iteration?" The SLP Vectorizer asks: "Can these nearby scalar operations
be bundled as lanes of one vector instruction?"

Both are profitability-driven. Even legal vectorization may be skipped when the
target cost model predicts scalar code is better.

## What makes loops vectorizable

A loop is easiest to vectorize when it has these properties.

### Simple counted loops

A canonical counted loop has a clear induction variable, a clear increment, and
a clear exit test:

```c
for (int i = 0; i < n; ++i)
  c[i] = a[i] + b[i];
```

LLVM can often canonicalize nearby forms, but simple loops make the analysis and
diagnostics much easier to understand.

### Predictable memory access

Unit-stride array access is the best case:

```c
c[i] = a[i] + b[i];
```

This maps naturally to vector loads, vector arithmetic, and vector stores.
Strides, gathers, scatters, and indirect indexing can still vectorize on some
targets, but they are more expensive and more target-dependent.

### No unsafe dependencies

Iterations must be independent, or the dependency must be a recognized pattern
such as a reduction. This is usually safe:

```c
for (int i = 0; i < n; ++i)
  c[i] = a[i] + b[i];
```

This is a reduction pattern that LLVM can often vectorize:

```c
int sum = 0;
for (int i = 0; i < n; ++i)
  sum += a[i];
```

The final scalar `sum` is produced by combining vector lanes after the vector
loop.

### Known trip counts or runtime checks

A compile-time constant trip count is simple:

```c
for (int i = 0; i < 1024; ++i)
  c[i] = a[i] + b[i];
```

Unknown trip counts can still vectorize. LLVM may emit a vector loop guarded by
a runtime trip-count check and a scalar remainder loop for leftover iterations.
For pointer arguments, it may also emit runtime alias checks before entering the
vector loop.

## What prevents vectorization

### Loop-carried dependencies

A dependency from one iteration to a later iteration usually blocks normal loop
vectorization:

```c
for (int i = 1; i < n; ++i)
  a[i] = a[i - 1] + 1;
```

Iteration `i` needs the value written by iteration `i-1`, so LLVM cannot freely
run several iterations together as vector lanes.

### Unknown aliasing

If LLVM cannot prove that pointers refer to separate memory, it must preserve the
possibility that one store changes a later load:

```c
void add(int *a, int *b, int *c, int n) {
  for (int i = 0; i < n; ++i)
    c[i] = a[i] + b[i];
}
```

Depending on target and cost, LLVM may add runtime pointer checks. Source-level
`restrict`, noalias IR attributes, TBAA metadata, and alias-scope metadata can
make alias facts explicit, but they must be true.

### Unsupported control flow

Small, predictable branches can sometimes be if-converted into selects or masks.
Complex exits, irreducible control flow, side exits, or control flow requiring
unsupported masking can prevent vectorization.

### Calls without vector forms

A call in a loop is vectorizable only when LLVM can safely represent or replace
it for vector lanes. Intrinsics and some math-library calls have vector forms on
some targets and with suitable options. Unknown calls, calls with side effects,
or calls that may access external state often block vectorization.

## Useful commands

From the repository root:

```sh
# Show successful loop-vectorization remarks from clang.
clang -O3 -Rpass=loop-vectorize llvm-training/09-vectorization/examples/sum-loop.c -c -o /tmp/sum-loop.o

# Show missed loop-vectorization remarks from clang.
clang -O3 -Rpass-missed=loop-vectorize llvm-training/09-vectorization/examples/sum-loop.c -c -o /tmp/sum-loop.o

# Run only the Loop Vectorizer over scalar IR.
opt -S -passes=loop-vectorize llvm-training/09-vectorization/examples/sum-loop.ll -o -

# Run only the SLP Vectorizer over scalar IR.
opt -S -passes=slp-vectorizer llvm-training/09-vectorization/examples/slp-scalars.ll -o -

# Run the normal O3 optimization pipeline and print IR.
opt -S -passes='default<O3>' llvm-training/09-vectorization/examples/sum-loop.ll -o -
```

Notes:

- Newer LLVM tools use the `-passes=...` pass-manager syntax shown above.
- `clang -Rpass...` diagnostics are emitted while compiling from source. Use
  `-g` or `-fsave-optimization-record` when you need richer source locations or
  serialized optimization records.
- Optimization remarks come in passed, missed, and analysis forms. The exact pass
  names in remarks are part of your experiment: try `loop-vectorize`, `slp-vectorizer`,
  or a broader regex such as `.`.

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
  llvm-training/09-vectorization/examples/sum-loop.ll -o -
```

Use forced settings as experiments, not as proof that a setting is fastest. A
forced vector width can make LLVM generate vector IR even when the cost model
would normally reject it.

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
3. Run `opt -S -passes=loop-vectorize` on `sum-loop.ll` and search for
   `<N x T>` types.
4. Run `opt -S -passes=slp-vectorizer` on `slp-scalars.ll` and look for vector
   arithmetic, vector stores, and `shufflevector`.
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
