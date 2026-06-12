# INDEX — Dispatcher

Agent: choose one axis, then grep only that focused file. This top-level file is
kept intentionally small so agent readers do not have to scan every table for
every lookup.

| Need | Index |
| --- | --- |
| Concept or task area | [`indexes/topics.md`](indexes/topics.md) |
| Common LLVM IR instruction syntax | [`indexes/instructions.md`](indexes/instructions.md) |
| One-page quick references | [`quickref/`](quickref/) |
| Advanced IR quick reference | [`quickref/advanced-ir.md`](quickref/advanced-ir.md) |
| MLIR bridge quick reference | [`quickref/mlir-bridge.md`](quickref/mlir-bridge.md) |
| `opt` flags and pass names | [`indexes/optimizer-passes.md`](indexes/optimizer-passes.md) |
| Auto-vectorization lessons and advanced masked/interleaved examples | [`09-vectorization/README.md`](09-vectorization/README.md) |
| Symbols such as metadata names, intrinsics, and files | [`indexes/symbols.md`](indexes/symbols.md) |
| Intrinsics and special types | [`indexes/intrinsics-special-types.md`](indexes/intrinsics-special-types.md) |
| Advanced IR lessons | [`13-advanced-ir/README.md`](13-advanced-ir/README.md) |
| Operand bundles, GC statepoints, coroutines, matrix/convergence intrinsics, and token IR | [`13-advanced-ir/03-special-types-and-tokens.md`](13-advanced-ir/03-special-types-and-tokens.md), [`13-advanced-ir/07-operand-bundles.md`](13-advanced-ir/07-operand-bundles.md) |
| Common intrinsic family quick reference | [`reference/intrinsics-quickref.md`](reference/intrinsics-quickref.md) |
| LLVM types and opaque-pointer migration | [`02-types/README.md`](02-types/README.md) |
| BCIR lowering guide | [`bcir-mapping/README.md`](bcir-mapping/README.md) |
| BCIR lowering patterns | [`indexes/bcir-patterns.md`](indexes/bcir-patterns.md) |
| BCIR normal forms and verifier design | [`bcir-mapping/11-normal-forms-and-verification.md`](bcir-mapping/11-normal-forms-and-verification.md) |
| MLIR bridge and integration | [`14-mlir-bridge/README.md`](14-mlir-bridge/README.md) |
| Dedicated MLIR lowering to LLVM for BCIR | [`18-mlir-lowering-to-llvm/README.md`](18-mlir-lowering-to-llvm/README.md) |
| Hardware-aware lowering, GAADMSF pulses, Dragon Egg flows, RISC-V, and MIR | [`19-hardware-aware/README.md`](19-hardware-aware/README.md) |
| Backend/JIT diagnostics | [`12-backend-jit/01-codegen-pipeline.md`](12-backend-jit/01-codegen-pipeline.md) |
| Advanced ORC runtime integration, hot kernel re-JIT, and heterogeneous deployment | [`12-backend-jit/07-advanced-orc-runtime-integration.md`](12-backend-jit/07-advanced-orc-runtime-integration.md) |
| Binary-analysis evidence schemas | [`15-binary-analysis/README.md`](15-binary-analysis/README.md) |
| Exception-handling IR (`invoke`, landing pads, WinEH funclets) | [`16-exception-handling/README.md`](16-exception-handling/README.md) |
| Modern pass-manager infrastructure, PassBuilder plugins, and adaptive BCIR pipelines | [`17-new-pass-manager/README.md`](17-new-pass-manager/README.md) |
| Type metadata and CFI-style checked dispatch | [`06-metadata/04-type-metadata-cfi.md`](06-metadata/04-type-metadata-cfi.md) |
| Repair, prediction, BCIR, MLIR, backend/JIT, and advanced-debugging exercises | [`exercises/README.md`](exercises/README.md) |
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
| EH cleanup/resume and funclet operand-bundle topics | [`16-exception-handling/04-cleanups-and-resume.md`](16-exception-handling/04-cleanups-and-resume.md) |
| Corpus self-test | [`EVAL.md`](EVAL.md) |
| Known gaps and future roadmap | [`ROADMAP.md`](ROADMAP.md) |

If you do not know where to start, open [`RECIPES.md`](RECIPES.md), pick the row
closest to your task, then jump to the linked chapter and examples. Detailed
topic and symbol lookup remains in [`llvm-training/indexes/`](indexes/).

## Advanced chapter integration map

| Need | Primary chapter | Companion index or gate |
|---|---|---|
| New PM registration, analysis invalidation, plugins, adaptive pipelines, PGO/MLGO | [`17-new-pass-manager/README.md`](17-new-pass-manager/README.md) | [`indexes/optimizer-passes.md`](indexes/optimizer-passes.md) |
| MLIR legality, type conversion, materialization, Transform dialect, metadata translation | [`18-mlir-lowering-to-llvm/README.md`](18-mlir-lowering-to-llvm/README.md) | [`indexes/topics.md`](indexes/topics.md), [`tools/verify-mlir-examples.sh`](tools/verify-mlir-examples.sh) |
| ORC materialization units, transform layers, resource trackers, remote JITLink | [`12-backend-jit/07-advanced-orc-runtime-integration.md`](12-backend-jit/07-advanced-orc-runtime-integration.md) | [`reference/glossary.md`](reference/glossary.md) |
| GAADMSF/Dragon Egg dispatch, pulses, calibration, target extensions, MIR | [`19-hardware-aware/README.md`](19-hardware-aware/README.md) | [`reference/intrinsics-quickref.md`](reference/intrinsics-quickref.md), [`tools/smoke-llc-skip.txt`](tools/smoke-llc-skip.txt) |
| BCIR normal forms and mapping drift | [`bcir-mapping/11-normal-forms-and-verification.md`](bcir-mapping/11-normal-forms-and-verification.md) | [`indexes/bcir-patterns.md`](indexes/bcir-patterns.md), [`tools/verify-bcir-mapping.sh`](tools/verify-bcir-mapping.sh) |
| Artifact type and portability classification | [`examples/README.md`](examples/README.md) | [`tools/verify-manifest.sh`](tools/verify-manifest.sh) |
