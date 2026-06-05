# Optimization: Passes, Analyses, Pipelines, and Diagnostics

## Key takeaways

- LLVM's new pass manager composes analysis and transform passes at module, CGSCC, function, and loop scopes.
- Transforms are pipeline-sensitive: a pass that works alone may depend on earlier canonicalization or invalidate later analyses.
- Use `opt` debugging flags and before/after IR snapshots to distinguish semantic changes from version-dependent formatting.
- PGO, LTO, and BOLT add profile- and link-time feedback layers; keep their assumptions separate from plain IR legality.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Pass model, pass scopes, and pipeline syntax | [`01-pass-model.md`](01-pass-model.md) |
| Common analysis passes and what facts they compute | [`02-common-analysis-passes.md`](02-common-analysis-passes.md) |
| Common transform passes and canonical before/after shapes | [`03-common-transform-passes.md`](03-common-transform-passes.md) |
| Optimization levels and default pipeline behavior | [`04-optimization-levels.md`](04-optimization-levels.md) |
| Debugging pass pipelines and reduced examples | [`05-debugging-passes.md`](05-debugging-passes.md) |
| PGO, LTO, and BOLT overview | [`06-pgo-lto-bolt.md`](06-pgo-lto-bolt.md) |
| BOLT layout walkthrough and evidence review | [`07-bolt-layout-walkthrough.md`](07-bolt-layout-walkthrough.md) |

## Examples

Open the `examples/` directory in this chapter for standalone artifacts and small fixtures that accompany the lessons. Files ending in `.ll` are intended to assemble unless the lesson or filename says they are intentionally invalid.
