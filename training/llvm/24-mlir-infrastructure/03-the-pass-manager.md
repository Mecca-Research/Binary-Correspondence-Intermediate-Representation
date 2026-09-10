# The pass manager: what a pipeline string actually means

The corpus teaches how to *write* an MLIR pass in
[`../18-mlir-lowering-to-llvm/`](../18-mlir-lowering-to-llvm). This chapter is about
running one, which turns out to have a rule that catches everybody once.

## A pass is anchored on an operation

MLIR's pass manager is a tree, not a list. Every pass declares the operation it runs on,
and the pipeline says where in the nesting each pass sits:

```
builtin.module(func.func(affine-scalrep))
     ^ outer op        ^ inner op   ^ the pass
```

Read that as: *for the module, for each `func.func` inside it, run `affine-scalrep`.*

Name a function-scoped pass at module level and it is a **configuration error**, refused
before any IR is touched:

```console
$ mlir-opt m.mlir --pass-pipeline='builtin.module(affine-scalrep)'
failed to add `affine-scalrep` with options ``
$ echo $?
1
```

Nest it correctly and it runs:

```console
$ mlir-opt m.mlir --pass-pipeline='builtin.module(func.func(affine-scalrep))'
$ echo $?
0
```

This is the difference from LLVM's pass manager that costs the most time. In LLVM you say
`-passes=instcombine` and the pass manager works out that it is a function pass. In MLIR
the nesting is *yours to write*, because "the operation this runs on" is an open set — a
pass can be anchored on any op, including one from a dialect that did not exist when the
pass manager was written.

## The trap: a misspelled anchor is silent

```console
$ mlir-opt m.mlir --pass-pipeline='builtin.module(no.such_op(canonicalize))'
$ echo $?
0
```

**Exit zero. Nothing ran.** There is no operation named `no.such_op` in the module, so the
nested pipeline matches nothing and the pass manager has nothing to complain about.

The asymmetry is worth stating explicitly, because it is not intuitive:

| Mistake | Result |
| --- | --- |
| Right anchor name, wrong nesting level | **Loud failure** — `failed to add`, exit 1 |
| Anchor name that does not exist | **Silence** — exit 0, zero passes run |

So a typo in the *pass* name fails; a typo in the *op* name passes. If a pipeline appears
to do nothing, check the anchor spelling before you check the pass.

This corpus's own gate pins that behaviour rather than assuming it: if a future MLIR starts
rejecting a non-existent anchor, the check fails and this chapter gets rewritten instead
of quietly becoming wrong.

## Analyses are requested, not registered

LLVM has an analysis manager, registration, and `PreservedAnalyses` returned from every
pass. MLIR has neither registration nor a return value:

```cpp
void MyPass::runOnOperation() {
  auto &domInfo = getAnalysis<DominanceInfo>();   // computed on demand, cached
  ...
}
```

An analysis is any class constructible from the operation. Ask for it and you get it,
cached per operation.

**The default is the inverse of LLVM's, and this is the part to remember.** An LLVM pass
declares what it preserved and everything else is invalidated. An MLIR pass **invalidates
everything** unless it says otherwise:

```cpp
markAllAnalysesPreserved();               // I changed nothing
markAnalysesPreserved<DominanceInfo>();   // I changed nothing dominance cares about
```

A pass that examines IR without modifying it and forgets to say so is not incorrect — it
is slow, and invisibly so, because every analysis downstream is recomputed.

## Passes run in parallel by default

MLIR runs op-nested passes concurrently across the operations they are anchored on. Two
`func.func`s in a module get their passes run at the same time.

That is safe only because of the `IsolatedFromAbove` trait from
[`02-interfaces.md`](02-interfaces.md): an isolated op's regions cannot reference values
defined outside it, so two of them cannot interfere. This is the real reason for the rule
that **a pass must not mutate anything outside the operation it was given** — not style,
but the precondition that makes the concurrency sound.

`--mlir-disable-threading` turns it off, which is the first thing to try when a pass
behaves differently under load, and the correct way to get deterministic diagnostic
ordering when debugging.

## Pitfalls checklist

- Do not write a flat pass list and expect MLIR to infer nesting. It will refuse, or
  worse, silently match nothing.
- Do not debug a pipeline that "does nothing" by studying the pass. Check the anchor
  operation name first.
- Do not omit `markAllAnalysesPreserved()` from an analysis-only pass.
- Do not touch IR outside your own operation, however convenient. The pass manager has
  already promised somebody it is safe to run you in parallel.
- Do not chase a nondeterministic diagnostic order before trying
  `--mlir-disable-threading`.

## Checks

[`../tools/verify-mlir-infrastructure.py`](../tools/verify-mlir-infrastructure.py) runs all
three pipelines above and requires each outcome: the module-level anchoring of a
function-scoped pass must fail with `failed to add`, the correctly nested form must
succeed, and the non-existent anchor must still be accepted silently. The last of those is
a pin on current behaviour, phrased so that an improvement in MLIR reads as "update the
chapter" rather than as a mysterious failure.
