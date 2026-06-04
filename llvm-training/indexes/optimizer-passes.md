# Index: By optimizer pass / `opt` flag

| Name or flag | Kind | Read |
|---|---|---|
| `opt -passes=verify` | Utility/checking pipeline | [`07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md), [`07-optimization/02-common-analysis-passes.md`](../07-optimization/02-common-analysis-passes.md) |
| `-debug-pass-manager` | Pass-manager scheduling trace | [`07-optimization/05-debugging-passes.md`](../07-optimization/05-debugging-passes.md) |
| `print<module>` | IR printing inside a pipeline | [`07-optimization/05-debugging-passes.md`](../07-optimization/05-debugging-passes.md) |
| `opt -S` | Textual IR output flag | [`07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md), [`07-optimization/04-optimization-levels.md`](../07-optimization/04-optimization-levels.md) |
| `-disable-output` | Suppress output for check/print workflows | [`07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md), [`07-optimization/02-common-analysis-passes.md`](../07-optimization/02-common-analysis-passes.md) |
| `mem2reg` | Transform | [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) |
| `instcombine` | Transform | [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) |
| `simplifycfg` | Transform | [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) |
| `adce` | Transform | [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) |
| `gvn` | Transform | [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) |
| `loop-unroll` | Transform | [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) |
| alias analysis | Analysis family | [`07-optimization/02-common-analysis-passes.md`](../07-optimization/02-common-analysis-passes.md) |
| CFG printing/viewing | Utility/analysis inspection | [`07-optimization/02-common-analysis-passes.md`](../07-optimization/02-common-analysis-passes.md) |
| loop analysis | Analysis | [`07-optimization/02-common-analysis-passes.md`](../07-optimization/02-common-analysis-passes.md) |
| scalar evolution / SCEV | Analysis | [`07-optimization/02-common-analysis-passes.md`](../07-optimization/02-common-analysis-passes.md) |
| `default<O1>`, `default<O2>`, `default<O3>`, `default<Os>`, `default<Oz>` | Predefined new-PM pipelines | [`07-optimization/04-optimization-levels.md`](../07-optimization/04-optimization-levels.md) |

| `-fprofile-generate`, `-fprofile-use` | PGO profile collection/use | [`07-optimization/06-pgo-lto-bolt.md`](../07-optimization/06-pgo-lto-bolt.md) |
| `llvm-profdata merge` | Profile data preparation | [`07-optimization/06-pgo-lto-bolt.md`](../07-optimization/06-pgo-lto-bolt.md) |
| `-flto=thin`, Full LTO | Cross-module optimization configuration | [`07-optimization/06-pgo-lto-bolt.md`](../07-optimization/06-pgo-lto-bolt.md) |
| `llvm-bolt` | Post-link binary layout optimization | [`07-optimization/06-pgo-lto-bolt.md`](../07-optimization/06-pgo-lto-bolt.md) |
