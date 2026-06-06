# Vectorization Legality and Blockers

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

## BCIR pipeline checks

Before asking whether vectorization is profitable, place legality facts in the pipeline where LLVM can still use them: truthful alias metadata, MemorySSA-invalidating custom passes, `freeze` before poison-capable control decisions, and `loop-rotate` after BCIR lowering but before `loop-vectorize`. The cross-chapter pipeline checklist is in [`../07-optimization/08-deep-optimization-lessons.md#putting-advanced-passes-into-a-bcir-pipeline`](../07-optimization/08-deep-optimization-lessons.md#putting-advanced-passes-into-a-bcir-pipeline), with New Pass Manager mechanics in [`../17-new-pass-manager/02-custom-passes-and-analyses.md`](../17-new-pass-manager/02-custom-passes-and-analyses.md).

## See also

- [`README.md`](README.md) — vectorization dispatcher.
- [`01-loop-vectorizer.md`](01-loop-vectorizer.md) — deeper Loop Vectorizer legality and runtime checks.
- [`03-vector-predication.md`](03-vector-predication.md) — masked and predicated vectorization.
- [`../08-pitfalls/12-vectorization-blocked-by-aliasing.md`](../08-pitfalls/12-vectorization-blocked-by-aliasing.md) — aliasing as a missed-vectorization cause.
- [`../07-optimization/08-deep-optimization-lessons.md#putting-advanced-passes-into-a-bcir-pipeline`](../07-optimization/08-deep-optimization-lessons.md#putting-advanced-passes-into-a-bcir-pipeline) — advanced passes as BCIR pipeline components.
