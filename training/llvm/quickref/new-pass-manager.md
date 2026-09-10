# Quickref: New Pass Manager

## Pipeline scopes

- **Module** passes see the whole module.
- **CGSCC** passes operate on call-graph strongly connected components.
- **Function** passes transform or analyze one function at a time.
- **Loop** passes operate on loops inside function pipelines.

## `opt` patterns

```bash
opt -passes=verify input.ll -o /dev/null
opt -passes='default<O2>' input.ll -S -o output.ll
opt -passes='function(mem2reg,instcombine,simplifycfg)' input.ll -S -o output.ll
opt -passes='print<scalar-evolution>' input.ll -disable-output
```

## Debugging workflow

1. Minimize the IR while preserving the pass behavior or verifier failure.
2. Print the pipeline or run a small explicit pipeline before trying full `default<O2>`.
3. Capture before/after IR at the pass boundary where semantics or shape changes.
4. Remember that analyses are invalidated by transforms; pipeline order changes available facts.
5. Treat pass names and default pipelines as LLVM-version-sensitive details.

## Common surprises

| Symptom | Check |
| --- | --- |
| Pass works alone but not in a larger pipeline | Earlier canonicalization, invalidated analysis, or changed IR shape. |
| Expected loop/vector transform does not run | Missing loop form, alias facts, target info, or optimization level. |
| Metadata disappears | Transform does not preserve it, or the metadata no longer applies. |
| Different output across LLVM versions | Default pipeline changed; compare explicit pass lists. |

## Deep links

- [`../07-optimization/README.md`](../07-optimization/README.md)
- [`../07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md)
- [`../07-optimization/05-debugging-passes.md`](../07-optimization/05-debugging-passes.md)
- [`../08-pitfalls/13-pass-pipeline-ordering-surprise.md`](../08-pitfalls/13-pass-pipeline-ordering-surprise.md)
- [`../indexes/optimizer-passes.md`](../indexes/optimizer-passes.md)
