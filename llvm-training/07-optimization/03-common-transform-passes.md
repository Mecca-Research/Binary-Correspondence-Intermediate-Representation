# Common Transform Passes

Transform passes rewrite IR. Some are canonicalization passes that make IR
simpler for other passes; others directly target dead code, redundancy, or loop
structure.

Official references:

- [LLVM's Analysis and Transform Passes](https://llvm.org/docs/Passes.html)
- [Using the New Pass Manager](https://llvm.org/docs/NewPassManager.html)
- [`opt` command guide](https://llvm.org/docs/CommandGuide/opt.html)

## Quick command pattern

Use `-S` to inspect changed textual IR:

```bash
opt -S -passes=mem2reg examples/mem2reg-before.ll -o examples/mem2reg-after.ll
opt -S -passes=instcombine examples/instcombine-before.ll -o examples/instcombine-after.ll
opt -S -passes=simplifycfg examples/simplifycfg-before.ll -o examples/simplifycfg-after.ll
opt -S -passes=adce examples/dead-code-before.ll -o examples/dead-code-after-adce.ll
opt -S -passes=loop-rotate examples/loop-rotate-before.ll -o examples/loop-rotate-after.ll
opt -S -passes=loop-unroll examples/loop-unroll-before.ll -o examples/loop-unroll-after.ll
opt -S -passes=gvn examples/gvn-before.ll -o examples/gvn-after.ll
```

For a pure validity check:

```bash
opt -passes=verify examples/dead-code-before.ll -disable-output
```

## `mem2reg`

`mem2reg` promotes eligible stack allocations to SSA registers. It is one of
the most useful teaching passes because frontends often emit simple mutable
local variables as `alloca` + `load` + `store`, while optimized LLVM IR prefers
SSA values and PHI nodes.

Try:

```bash
opt -S -passes=mem2reg examples/mem2reg-before.ll -o -
```

Regenerate the checked-in paired output with:

```bash
opt -S -passes=mem2reg examples/mem2reg-before.ll -o examples/mem2reg-after.ll
```

Look for removed `alloca`, `store`, and `load` instructions.

## `instcombine`

`instcombine` performs local algebraic and canonicalizing rewrites, such as
folding neutral operations, simplifying comparisons, and preferring canonical
instruction forms.

Try:

```bash
opt -S -passes=instcombine examples/instcombine-before.ll -o -
```

Regenerate the checked-in paired output with:

```bash
opt -S -passes=instcombine examples/instcombine-before.ll -o examples/instcombine-after.ll
```

It is not just a peephole optimizer; its main value is often making later
passes see the same pattern consistently.

## `simplifycfg`

`simplifycfg` simplifies control-flow structure. It may merge blocks, fold
branches with known conditions, remove unreachable edges, or convert tiny
control-flow patterns into simpler expressions when legal.

Try:

```bash
opt -S -passes=simplifycfg examples/simplifycfg-before.ll -o -
```

Regenerate the checked-in paired output with:

```bash
opt -S -passes=simplifycfg examples/simplifycfg-before.ll -o examples/simplifycfg-after.ll
```

## `adce`

`adce` is aggressive dead code elimination. It removes instructions that do not
contribute to observable behavior. It can be more powerful after other passes
have simplified values and control flow.

Try:

```bash
opt -S -passes=adce examples/dead-code-before.ll -o -
```

Regenerate the checked-in paired output with:

```bash
opt -S -passes=adce examples/dead-code-before.ll -o examples/dead-code-after-adce.ll
```

## `gvn`

`gvn` stands for global value numbering. It removes redundant computations and
some redundant memory operations when analysis proves they compute or read the
same value.

Try a focused redundant-computation example:

```bash
opt -S -passes=gvn examples/gvn-before.ll -o -
```

Regenerate the checked-in paired output with:

```bash
opt -S -passes=gvn examples/gvn-before.ll -o examples/gvn-after.ll
```

Try combining it with canonicalization:

```bash
opt -S -passes='mem2reg,instcombine,gvn' examples/mem2reg-before.ll -o -
```

## `loop-rotate`

`loop-rotate` turns many while-shaped loops into a rotated form with the loop
condition on the latch. This can make the loop body the main repeated path and
can expose a cleaner shape to later loop optimizations.

Try:

```bash
opt -S -passes=loop-rotate examples/loop-rotate-before.ll -o -
```

Regenerate the checked-in paired output with:

```bash
opt -S -passes=loop-rotate examples/loop-rotate-before.ll -o examples/loop-rotate-after.ll
```

## `loop-unroll`

`loop-unroll` duplicates loop bodies to reduce branch overhead and expose more
straight-line code to later passes. It can improve throughput, but it may also
increase code size.

Try:

```bash
opt -S -passes=loop-unroll examples/loop-unroll-before.ll -o -
```

Regenerate the checked-in paired output with:

```bash
opt -S -passes=loop-unroll examples/loop-unroll-before.ll -o examples/loop-unroll-after.ll
```

If nothing obvious changes, the pass may have decided not to unroll without
more information, target details, or loop metadata.

## Passes compose

A single pass often looks underwhelming. Pipelines matter:

```bash
opt -S -passes='mem2reg,instcombine,simplifycfg,adce,gvn' examples/dead-code-before.ll -o -
```

Do not infer that a pass is useless because it does not change one small input.
It may be preserving legality, waiting for a canonical form, or intentionally
avoiding a non-profitable rewrite.

## See also

- [`01-pass-model.md`](01-pass-model.md)
- [`02-common-analysis-passes.md`](02-common-analysis-passes.md)
- [`04-optimization-levels.md`](04-optimization-levels.md)
