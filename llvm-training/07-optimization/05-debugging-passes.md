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
