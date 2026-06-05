# Optimization Pass Model

LLVM optimizations are organized as **passes**: reusable units that inspect,
check, or rewrite IR. The `opt` tool is the quickest way to experiment with
these passes on `.ll` files.

Official references:

- [LLVM's Analysis and Transform Passes](https://llvm.org/docs/Passes.html)
- [Using the New Pass Manager](https://llvm.org/docs/NewPassManager.html)
- [`opt` command guide](https://llvm.org/docs/CommandGuide/opt.html)

## Three pass roles

| Role | What it does | Examples | Usually changes IR? |
|---|---|---|---|
| Analysis pass | Computes facts other passes can use | alias analysis, loop analysis, scalar evolution | No |
| Transform pass | Rewrites IR to improve or normalize it | `mem2reg`, `instcombine`, `simplifycfg`, `adce`, `gvn`, `loop-unroll` | Yes |
| Utility pass | Checks, prints, views, or serializes IR | `verify`, CFG printers/viewers | Usually no |

The categories are practical, not mystical. A transform pass may request many
analyses, and a utility pass may expose analysis results for humans.

## The new pass manager syntax

Modern `opt` examples should generally use the new pass manager's
`-passes=...` pipeline spelling:

```bash
opt -passes=verify input.ll -disable-output
opt -passes=mem2reg input.ll -S -o promoted.ll
opt -passes=instcombine input.ll -S -o combined.ll
opt -passes='default<O2>' input.ll -S -o o2.ll
```

Important details:

- Use `-S` when you want textual LLVM IR output instead of bitcode.
- Use `-disable-output` for pure checking or printing workflows.
- Quote pipelines that contain shell metacharacters, such as
  `default<O2>`.
- Some passes are module passes, some are function passes, and some are loop
  passes. The new pass manager can infer common nesting in simple pipelines,
  but complex pipelines may need explicit adaptors or nested syntax.

## PassBuilder and plugin pipelines

The new pass manager's textual pipelines are parsed by `PassBuilder`. That
matters when you move from built-in teaching passes to BCIR-specific plugin
passes: the command line is not just a list of legacy flags, it is a nested
pipeline grammar. A simple function-pass sequence may be inferred, while mixed
module/function/loop pipelines may need explicit nesting.

Common shapes:

```bash
opt -S -passes='function(mem2reg,instcombine),module(function(sccp))' input.ll -o out.ll
opt -load-pass-plugin=./libBCIROptPasses.so -passes='bcir-preserve-map,verify' input.ll -S -o out.ll
opt --print-passes | rg 'sccp|memoryssa|loop-rotate'
```

A custom plugin should document its textual pass name, the IR unit it runs on,
and which analyses it preserves. For BCIR, pair plugin pipelines with mapping
and metadata checks so a pass cannot silently erase provenance, collapse a
1:1 mapping into a many-to-one fold, or rotate loop headers before a lowering
phase that still expects the source-shaped CFG.

For a deeper BCIR-focused pass-manager lesson, see
[`08-deep-optimization-lessons.md`](08-deep-optimization-lessons.md).

## Inspecting changed IR

A small workflow:

```bash
opt -S -passes=verify examples/mem2reg-before.ll -o /dev/null
opt -S -passes=mem2reg examples/mem2reg-before.ll -o /tmp/mem2reg-after.ll
diff -u examples/mem2reg-before.ll /tmp/mem2reg-after.ll
```

`mem2reg` promotes eligible stack slots to SSA values. In this example, the
`alloca`, `store`, and `load` instructions should disappear, leaving direct SSA
arithmetic.

## Common pitfalls

### Relying on accidental pass order

Passes are not independent text filters. A pass may expose opportunities for a
later pass, invalidate analysis results, or depend on canonical forms created by
earlier passes. If a custom sequence only works because pass A happens to run
before pass B in one LLVM release, treat that as fragile.

Prefer named predefined pipelines (`default<O2>`, `default<Oz>`) for broad
optimization, and keep hand-written pipelines small and intentional when
teaching or debugging.

### Assuming `-O3` is always faster

`-O3` means LLVM is willing to use more aggressive transformations than `-O2`.
That can improve throughput, but it can also increase code size, compile time,
cache pressure, or branch misprediction costs. Benchmark on the target workload.
For size-sensitive code, `-Os` or `-Oz` may be the better starting point.

### Mixing legacy and new-pass-manager names

Legacy examples often use spellings like:

```bash
opt -instcombine input.ll -S -o out.ll
```

With the new pass manager, write:

```bash
opt -passes=instcombine input.ll -S -o out.ll
```

If an online example fails, first check whether it is using legacy pass-manager
syntax, an old pass name, or a pass that moved/changed in your installed LLVM
version.

## See also

- [`02-common-analysis-passes.md`](02-common-analysis-passes.md)
- [`03-common-transform-passes.md`](03-common-transform-passes.md)
- [`04-optimization-levels.md`](04-optimization-levels.md)
- [`08-deep-optimization-lessons.md`](08-deep-optimization-lessons.md)
