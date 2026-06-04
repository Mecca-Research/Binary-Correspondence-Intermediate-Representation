# Pitfall 12 — Vectorization Blocked by Aliasing

## BCIR instance

| Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|
| Training-only exemplar; no affected BCIR `.ll` file recorded | Unknown | `opt -passes=loop-vectorize -pass-remarks-missed=loop-vectorize <bcir-loop>.ll -o /dev/null` | Preserve alias, alignment, and loop-shape facts so the vectorizer can prove memory reordering is safe. | [`09-vectorization/README.md`](../09-vectorization/README.md); [`07-optimization/02-common-analysis-passes.md`](../07-optimization/02-common-analysis-passes.md); [`04-memory/02-load-store.md`](../04-memory/02-load-store.md) |

## The diagnostic

From `clang -Rpass-missed=loop-vectorize` or `opt` missed remarks, the symptom is
usually one of these forms:

```text
loop not vectorized: cannot prove it is safe to reorder memory operations
```

```text
loop not vectorized: unsafe dependent memory operations in loop
```

LLVM may still generate correct scalar code; the failure is missed performance.

## Minimal reproducer

```c
void add(float *a, float *b, int n) {
  for (int i = 0; i < n; ++i)
    a[i] = a[i] + b[i];
}
```

The loop looks vector-friendly, but if `a` and `b` may overlap, vectorizing could
change behavior for some inputs. Equivalent IR often lacks attributes or metadata
that would let alias analysis prove independence.

A telltale IR shape is pointer parameters with no aliasing contract:

```llvm
define void @add(ptr %a, ptr %b, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %pa = getelementptr float, ptr %a, i64 %i
  %pb = getelementptr float, ptr %b, i64 %i
  %av = load float, ptr %pa, align 4
  %bv = load float, ptr %pb, align 4
  %sum = fadd float %av, %bv
  store float %sum, ptr %pa, align 4
  %next = add nuw i64 %i, 1
  %done = icmp eq i64 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
```

## Why it happens

The loop vectorizer must preserve memory semantics. If a store through `%a` might
affect a later load through `%b`, widening or reordering memory operations can be
incorrect. LLVM uses alias analysis, parameter attributes, access metadata,
runtime checks, target cost models, and loop structure to decide whether
vectorization is legal and profitable.

If a frontend drops `restrict`-like facts, noalias scopes, alignment, or
range/stride information, the vectorizer may refuse a loop that humans know is
safe.

## Fix pattern

State the aliasing contract in the IR when it is true:

```llvm
define void @add(ptr noalias %a, ptr noalias readonly %b, i64 %n) {
  ; same loop body
}
```

Other useful patterns:

- preserve source `restrict` as `noalias` where the language rules allow it;
- add accurate `readonly`, `writeonly`, `nocapture`, and alignment attributes;
- use loop access metadata or runtime alias checks for more complex cases;
- keep affine induction variables and simple `getelementptr` patterns visible;
- inspect `-Rpass-analysis=loop-vectorize` and `-Rpass-missed=loop-vectorize`
  output before guessing.

Never add `noalias` just to force vectorization. A false aliasing promise creates
miscompiles.

## BCIR-relevant note

Recovered binary IR often starts with conservative memory modeling: many accesses
appear to touch one large memory object. That is safe but hostile to
vectorization. If BCIR can prove stack slots, globals, heap regions, or recovered
arrays are disjoint, encode that fact deliberately; otherwise expect scalar code
or runtime checks.

## See also

- [`../09-vectorization/README.md`](../09-vectorization/README.md) — vectorizer diagnostics and examples
- [`../07-optimization/02-common-analysis-passes.md`](../07-optimization/02-common-analysis-passes.md) — alias analysis and other analyses
- [`../04-memory/02-load-store.md`](../04-memory/02-load-store.md) — memory operations that vectorizers reason about
