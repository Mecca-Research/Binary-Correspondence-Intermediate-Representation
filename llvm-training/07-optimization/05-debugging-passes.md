# Debugging Pass Pipelines

When optimized IR surprises you, debug the pass pipeline as data: list which
passes ran, print IR around suspicious passes, reduce the pipeline, and compare
before/after modules.

## See the pass pipeline

```bash
opt -disable-output -debug-pass-manager -passes='default<O2>' input.ll
```

`-debug-pass-manager` prints pass-manager scheduling: module passes, CGSCC
passes, function passes, loop passes, analyses, invalidations, and adaptors. Use
it when a pass did not run or ran at a different nesting level than expected.

For a single pass:

```bash
opt -S -passes='mem2reg' input.ll -o output.ll
```

Always include `-S` when you want textual IR; otherwise `opt` may write bitcode.

## Print IR in the middle of a pipeline

The new pass manager exposes print passes that can be inserted into a pipeline:

```bash
opt -S -passes='mem2reg,print<module>,instcombine' input.ll -o output.ll
```

Useful print forms include `print<module>`, `print<function>`, and analysis
printers such as `print<loops>` or `print<scalar-evolution>` when available in
your LLVM build. Pair them with `-disable-output` if you only want diagnostics.

## Bisect a pipeline

For large default pipelines, reduce the question:

1. Reproduce with the full pipeline, for example `default<O2>`.
2. Try nearby levels: `default<O1>`, `default<O2>`, `default<O3>`, `default<Os>`.
3. Use `-debug-pass-manager` to capture the pass list.
4. Rebuild a smaller explicit pipeline around the suspicious passes.
5. Insert `print<module>` before and after the suspected pass.

Some LLVM builds also support pass bisection controls for stopping after a pass
count. If available, use them to find the first pass that changes or breaks the
IR, then switch back to an explicit minimal pipeline for the checked-in repro.

## IR diff workflow

A repeatable diff loop is often enough:

```bash
opt -S -passes='verify' input.ll -o /tmp/before.ll
opt -S -passes='mem2reg,instcombine,simplifycfg' input.ll -o /tmp/after.ll
diff -u /tmp/before.ll /tmp/after.ll
opt -passes=verify /tmp/after.ll -o /dev/null
```

For noisy diffs, run one pass at a time and keep filenames named after the pass:
`foo-before.ll`, `foo-after-instcombine.ll`, `foo-after-simplifycfg.ll`.

## Optimization remarks

Frontend-driven investigations often need remarks rather than raw IR dumps:

```bash
clang -O2 -Rpass=loop-vectorize -Rpass-missed=loop-vectorize input.c -c -o /tmp/input.o
```

Use remarks to learn why a pass skipped a source loop; use `opt` print/diff
workflows to learn exactly how IR changed after you already have IR.

## BCIR checklist

- Start every debugging session with `opt -passes=verify` to separate invalid IR
  from optimizer behavior.
- Capture the smallest pass pipeline that reproduces the issue and put that
  command in the nearby example README or exercise prompt.
- Keep before/after IR examples in `examples/` only if both files are valid
  standalone modules.
- If a pass-ordering surprise becomes recurring guidance, link it from
  [`../08-pitfalls/13-pass-pipeline-ordering-surprise.md`](../08-pitfalls/13-pass-pipeline-ordering-surprise.md).

## Pass-specific walkthroughs

The examples below use checked-in modules under
[`examples/`](examples/) so a reader can rerun each command and compare the
printed IR with the corresponding `*-after.ll` snapshot.

### `mem2reg`: watch stack slots become SSA values

Start with a promotable diamond that stores two possible values into one stack
slot:

```bash
opt -passes=verify llvm-training/07-optimization/examples/mem2reg-diamond-before.ll -o /dev/null
opt -S -passes='print<function>,mem2reg,print<function>' \
  llvm-training/07-optimization/examples/mem2reg-diamond-before.ll \
  -disable-output
opt -S -passes=mem2reg \
  llvm-training/07-optimization/examples/mem2reg-diamond-before.ll \
  -o /tmp/mem2reg-diamond-after.ll
diff -u llvm-training/07-optimization/examples/mem2reg-diamond-before.ll /tmp/mem2reg-diamond-after.ll
```

What to verify:

- the entry-block `alloca`, branch stores, and merge-block `load` disappear;
- a `phi` in `merge` selects the value from `%then` or `%else`;
- no memory operations remain for the promoted local.

### `instcombine`: isolate algebraic canonicalization

`instcombine` is intentionally local. Debug it by printing functions before and
after the pass, then diffing against a stable snapshot:

```bash
opt -S -passes='print<function>,instcombine,print<function>' \
  llvm-training/07-optimization/examples/instcombine-canonical-before.ll \
  -disable-output
opt -S -passes=instcombine \
  llvm-training/07-optimization/examples/instcombine-canonical-before.ll \
  -o /tmp/instcombine-canonical-after.ll
diff -u llvm-training/07-optimization/examples/instcombine-canonical-after.ll /tmp/instcombine-canonical-after.ll
```

Look for folds that do not need control-flow reasoning: `icmp` with identical
operands becomes `true`, identity operations disappear, and equivalent shifts or
multiplies are canonicalized.

### `simplifycfg`: prove a control-flow-only rewrite

For branch folding or branch-to-`select` changes, use module prints because block
successors and predecessors matter:

```bash
opt -S -passes='print<module>,simplifycfg,print<module>' \
  llvm-training/07-optimization/examples/simplifycfg-select-before.ll \
  -disable-output
opt -S -passes=simplifycfg \
  llvm-training/07-optimization/examples/simplifycfg-select-before.ll \
  -o /tmp/simplifycfg-select-after.ll
diff -u llvm-training/07-optimization/examples/simplifycfg-select-before.ll /tmp/simplifycfg-select-after.ll
```

The expected change is structural: `%then`, `%else`, and `%merge` collapse into
one block, while the value choice survives as `select i1 %cond, ...`.

### `gvn`: confirm one redundant value class

Use a tiny example where no intervening write can change the loaded memory:

```bash
opt -S -passes='print<function>,gvn,print<function>' \
  llvm-training/07-optimization/examples/gvn-load-before.ll \
  -disable-output
opt -S -passes=gvn \
  llvm-training/07-optimization/examples/gvn-load-before.ll \
  -o /tmp/gvn-load-after.ll
diff -u llvm-training/07-optimization/examples/gvn-load-before.ll /tmp/gvn-load-after.ll
```

The second load should be replaced with the first loaded SSA value. If a local
reproducer does not fold, inspect attributes first: a missing `readonly`,
`readnone`, `noalias`, or intervening store can make the loads genuinely
different.

### Loop passes: print just enough nesting context

Loop passes run through function-to-loop adaptors, so `-debug-pass-manager` is
especially useful:

```bash
opt -disable-output -debug-pass-manager -passes='loop-rotate' \
  llvm-training/07-optimization/examples/loop-rotate-while-before.ll
opt -S -passes='print<loops>,loop-rotate,print<loops>' \
  llvm-training/07-optimization/examples/loop-rotate-while-before.ll \
  -disable-output
opt -S -passes='loop-unroll' \
  llvm-training/07-optimization/examples/loop-unroll-count2-before.ll \
  -o /tmp/loop-unroll-count2-after.ll
```

Use `print<loops>` to confirm the loop header, latch, and exit before comparing
IR. Rotation should turn a while-shaped header test into a guarded do-while
shape. Partial unrolling should leave a loop, but duplicate the body inside each
remaining iteration.

### Inspect the `-O2` pipeline before reducing it

A default pipeline is too large to debug by staring at final IR first. Capture
scheduling, then bisect or reconstruct a smaller explicit pipeline:

```bash
opt -disable-output -debug-pass-manager -passes='default<O2>' \
  llvm-training/07-optimization/examples/o2-pipeline-inspection.ll \
  2> /tmp/o2-pipeline.log
opt -S -passes='default<O2>' \
  llvm-training/07-optimization/examples/o2-pipeline-inspection.ll \
  -o /tmp/o2-pipeline-after.ll
diff -u llvm-training/07-optimization/examples/o2-pipeline-inspection.ll /tmp/o2-pipeline-after.ll
```

When supported by your LLVM build, `-opt-bisect-limit` stops the optimizer after
a pass count. Increase the limit until the first interesting change appears:

```bash
opt -S -passes='default<O2>' -opt-bisect-limit=1 \
  llvm-training/07-optimization/examples/o2-pipeline-inspection.ll \
  -o /tmp/o2-bisect-001.ll
opt -S -passes='default<O2>' -opt-bisect-limit=25 \
  llvm-training/07-optimization/examples/o2-pipeline-inspection.ll \
  -o /tmp/o2-bisect-025.ll
diff -u /tmp/o2-bisect-001.ll /tmp/o2-bisect-025.ll
```

After bisection identifies the first pass that matters, switch back to an
explicit pass list around that area, for example:

```bash
opt -S -passes='mem2reg,instcombine,simplifycfg,gvn' \
  llvm-training/07-optimization/examples/o2-pipeline-inspection.ll \
  -o /tmp/o2-reduced.ll
```

That reduced pipeline is the version to preserve in a bug report or a training
example because it avoids depending on every target- and LLVM-version-specific
choice in `default<O2>`.
