# Pitfall 13 — Pass Pipeline Ordering Surprise

## BCIR instance

| Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|
| BCIR validation pipeline scripts: `runtime/llvm/validate_phase3.sh`; `runtime/llvm/validate_phase4.sh` | Unknown | `opt -passes=<bcir-analysis-pipeline> <bcir-module>.ll -o /dev/null` | Run BCIR analyses before destructive cleanup passes, and preserve required analyses between dependent passes. | [`07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md); [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md); [`07-optimization/04-optimization-levels.md`](../07-optimization/04-optimization-levels.md) |

## The symptom

```text
my pass works when run alone, but fails or does nothing inside default<O2>
```

Common command-line versions:

```text
unknown function attribute remains after my cleanup pass
```

```text
loop-vectorize did not run because canonical loop form was not available
```

This is usually not a parser diagnostic. It is a pipeline-design bug.

## Minimal reproducer

Suppose a custom pass expects loops to have simplified form and LCSSA. Running it
alone over arbitrary IR is fragile:

```sh
opt -S -passes='my-loop-pass' input.ll -o -        # ❌ assumptions may be false
```

The pass may only work when canonicalization runs first:

```sh
opt -S -passes='loop-simplify,lcssa,my-loop-pass' input.ll -o -  # ✓
```

The opposite problem also happens: a pass adds information too early, then later
canonicalization erases, folds, or invalidates the exact shape the pass expected
to preserve.

## Why it happens

LLVM pipelines are ordered for legality, profitability, and compile-time budget.
Many passes require a particular IR shape or analysis state:

- scalar transforms often expect `mem2reg`, `instcombine`, or `simplifycfg` to
  have exposed simple SSA patterns;
- loop passes often expect `loop-simplify` and `lcssa`;
- vectorization needs canonical loops, alias information, target information, and
  cost-model context;
- cleanup passes after a transform can erase evidence you expected to inspect
  later.

Under the new pass manager, analysis invalidation is also explicit. A transform
that mutates IR but claims to preserve too much can leave later passes reading
stale analysis results.

## Fix pattern

Make pass preconditions explicit:

```text
required IR shape: promoted scalars, simplified CFG, LoopSimplify, LCSSA
required analyses: DominatorTree, LoopInfo, ScalarEvolution, AliasAnalysis
invalidated by this pass: CFG, MemorySSA, ScalarEvolution, ...
```

Then put the pass at a pipeline point where those conditions are true, or add the
canonicalization passes before it in a custom pipeline:

```sh
opt -S -passes='default<O2>,function(my-cleanup),verify' input.ll -o -
```

For debugging, bisect the pipeline:

```sh
opt -S -passes='print<module>' input.ll -disable-output
opt -S -passes='default<O2>' -debug-pass-manager input.ll -o /dev/null
```

Always end experimental pipelines with `verify` while developing.

## BCIR-relevant note

BCIR validation often depends on exact IR structure: helper calls still present,
metadata still attached, or memory accesses still split in a recoverable way. If
you run a broad optimization pipeline before correspondence extraction, it may
legally erase those anchors. Put BCIR-specific analysis before destructive
cleanup, or mark/preserve the anchors intentionally.

## See also

- [`../07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md) — pass manager concepts
- [`../07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) — transforms and canonicalization
- [`../07-optimization/04-optimization-levels.md`](../07-optimization/04-optimization-levels.md) — what default optimization levels imply
