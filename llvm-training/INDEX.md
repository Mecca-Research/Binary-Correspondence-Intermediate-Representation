# INDEX — Dispatcher

Agent: choose one axis, then grep only that focused file. This top-level file is
kept intentionally small so agent readers do not have to scan every table for
every lookup.

| Need | Index |
| --- | --- |
| Concept or task area | [`indexes/topics.md`](indexes/topics.md) |
| Common LLVM IR instruction syntax | [`indexes/instructions.md`](indexes/instructions.md) |
| One-page quick references | [`quickref/`](quickref/) |
| `opt` flags and pass names | [`indexes/optimizer-passes.md`](indexes/optimizer-passes.md) |
| Auto-vectorization lessons and advanced masked/interleaved examples | [`09-vectorization/README.md`](09-vectorization/README.md) |
| Symbols such as metadata names, intrinsics, and files | [`indexes/symbols.md`](indexes/symbols.md) |
| Intrinsics and special types | [`indexes/intrinsics-special-types.md`](indexes/intrinsics-special-types.md) |
| Common intrinsic family quick reference | [`reference/intrinsics-quickref.md`](reference/intrinsics-quickref.md) |
| LLVM types and opaque-pointer migration | [`02-types/README.md`](02-types/README.md) |
| BCIR lowering guide | [`bcir-mapping/README.md`](bcir-mapping/README.md) |
| BCIR lowering patterns | [`indexes/bcir-patterns.md`](indexes/bcir-patterns.md) |
| MLIR bridge and integration | [`14-mlir-bridge/01-what-is-mlir.md`](14-mlir-bridge/01-what-is-mlir.md) |
| Backend/JIT diagnostics | [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) |
| Binary-analysis evidence schemas | [`15-binary-analysis/README.md`](15-binary-analysis/README.md) |
| Repair and prediction exercises | [`exercises/README.md`](exercises/README.md) |
| Expanded learning paths | [`CURRICULUM.md`](CURRICULUM.md) |
| Keywords where they are introduced | [`indexes/keywords.md`](indexes/keywords.md) |
| BCIR runtime/source cross-references | [`indexes/bcir-crossrefs.md`](indexes/bcir-crossrefs.md) |
| First-use protocol | [`START_HERE.md`](START_HERE.md) |
| Example and exercise conventions | [`EXAMPLES.md`](EXAMPLES.md) |
| LLVM version policy | [`SEMVER.md`](SEMVER.md) |
| Tool scripts and verifier behavior | [`tools/README.md`](tools/README.md) |
| Exercises and solution verification | [`exercises/README.md`](exercises/README.md) |
| Task-oriented lookup | [`RECIPES.md`](RECIPES.md) |
| Binary-analysis side-channel/profile/BCSA topics | [`15-binary-analysis/README.md`](15-binary-analysis/README.md) |
| Corpus self-test | [`EVAL.md`](EVAL.md) |
| Known gaps and future roadmap | [`ROADMAP.md`](ROADMAP.md) |

If you do not know where to start, open [`RECIPES.md`](RECIPES.md), pick the row
closest to your task, then jump to the linked chapter and examples. Detailed
topic and symbol lookup remains in [`llvm-training/indexes/`](indexes/).
