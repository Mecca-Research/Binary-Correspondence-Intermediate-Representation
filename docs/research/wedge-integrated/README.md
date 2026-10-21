# BCIR full-global integrated recipe — 2026-10-18

Read `BCIR_Global_Integrated_Recipe_Report_2026-10-18.md` for findings, comparison limits and the next architecture hypothesis. The chart is `BCIR_Global_Integrated_Recipe_Frontier_2026-10-18.png`.

This bundle contains an executable research recipe: full rank-22 global head, compact/recomputed native plans, a reusable workspace, variable-length curricula and budget-aware execution. It does not modify the BCIR repository or implement a production IR/MLIR integration.

## Environment

Measured on Linux x86-64, AMD EPYC 9V74, GCC 13.3, Python 3.12.14 and four native pool threads. Model/tensor C sources are pinned to BCIR `aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d`; the original code license is `vendor/LICENSE`. The exact ctypes ABI declarations are vendored in `native_abi.py`, so a sibling BCIR checkout is not required for these scripts.

The generated `.so` files are measured-host artifacts. Rebuild them on another host. The backend already includes runtime-dispatched AVX2/AVX-512 kernels. Compiler/libm/CPU changes can alter floating-point results; original checkpoint hashes identify this measured environment, not a universal cross-platform bit pattern.

Training dependencies are NumPy 2.5.4 and PyArrow 26.0.0. Independent gradient checks additionally use CPU PyTorch 2.14.1; install its CPU wheel from the official CPU index. Plotting uses Matplotlib 3.11.2 and Pillow 13.0.0. Full direct versions are recorded in `requirements-measured.txt`.

## Run the combined recipe

From this directory, in a suitable environment:

```bash
python build_integrated.py
python validate_integrated.py
python run_recipe.py --describe --budget-mib 32
python run_recipe.py --case ordered_bucket_global --seed 20261021 --budget-mib 32 --output runs/global-demo
```

`--describe` shows admitted plans without training. The default case trains the full global head for 900 ordered-recall updates, using balanced homogeneous lengths 64/128/256, batch 4, causal trimming and budget-dependent reconstruction. Outputs include the selected checkpoint, optimizer state, events, validation history, data hash and execution-policy IDs.

`BCIR_GLOBAL_RECIPE.json` retains the measured 32 MiB configuration. The CLI also admits a 20 MiB global curriculum budget by using deeper reconstruction. An impossible budget is refused before the model arrays are allocated. The declared bound covers controlled native arrays, pool-control/staging storage and specified records; it excludes thread stacks, Python/data/checkpoint overhead and allocator-retained pages. It is not an RSS cap.

The supplied cost lookup belongs to the measured CPU. On a different CPU model the CLI uses the documented fallback heuristic until local recalibration; `--no-calibration` explicitly selects that fallback anywhere. Even on the same CPU model, new quotas/frequencies or workloads may change rankings. The report shows that the short calibration is not reliably optimal.

Other approved cases are listed by `python run_recipe.py --help`. The language hybrid is a separate case/checkpoint family. The single-run CLI writes to a new output directory and skips an already existing per-run result. It does not silently replace the archived research evidence.

## Inspect the retained evidence

* `protocol.json`: frozen main 30-run design, sources, seeds, data, optimizer and selection rule.
* `matched-budget/protocol.json`: separate three-run attention32 follow-up and fresh paired tests.
* `continuation-protocol.json`, `continuation-extension-protocol.json`: budget transitions, archival replay, shorter-context control and longer evaluation.
* `results/summary.json`: audited means/ranges, timing, memory, recall, language and continuation aggregates.
* `results/final-audit.json`: 42 checkpoint hashes/restores and frozen-source/input checks.
* `results/validation.json`, `results/accumulation-check.json`: native-reference state/resume checks, independent gradients, accumulation normalization and refusals.
* `results/calibration.json`, `results/budget-probe.json`: every timing repetition, selected plan, storage bound and observed RSS.
* `results/evaluation.json`, `results/counterfactuals.json`: main held-out scores and value-swapping tests preserving token multisets.
* `results/precision-audit.json`: audit-only guard/absent-plane counts, separate from timed builds.
* `results/registry-reference.json`: grammar-known C lookup control, explicitly separate from neural training.
* `checkpoints/`, `matched-budget/checkpoints/`: all selected from-scratch and final continuation model/moment arrays, with no pickle loading required.

Datasets and their hashes/provenance are retained in `data/`, `DATA_PROVENANCE.md` and the main protocol. Native code/license and generated sources are included. `SOURCE_MANIFEST.json` hashes the deliverable files and excludes itself and runtime caches.

## Reproduce the study

Preserve the original archive. Work in a copy for new measurements; scripts intentionally skip existing training-result files. For a fresh study, move the archived result/checkpoint directories aside before rebuilding, and retain the original frozen protocol for comparison. Do not tune candidates using final-test outcomes.

Run heavy stages sequentially:

```bash
python build_integrated.py
python validate_integrated.py
python calibrate.py
python train_integrated.py --freeze-only
python train_integrated.py
python budget_probe.py
python evaluate_integrated.py
python registry_reference.py
python continuation.py
python matched_budget.py
python archive_continuation.py
python accumulation_check.py
python summarize_integrated.py
python plot_integrated.py
```

`train_integrated.py` freezes source/calibration provenance and verifies retained source hashes; `summarize_integrated.py` also verifies the frozen calibration hash. A newly measured calibration differs from the archived one: move the original protocol aside with its results, and record a new protocol in a separate study directory rather than pretending it is the original run. The main held-out evaluator expects 30 main training files. The three additional attention runs live under `matched-budget/`, so the two stages remain distinguishable.

`archive_continuation.py` replays the existing long-context continuation and verifies its two final digests before writing final checkpoints. It also adds the update-matched T128 control and new T2048/T4096 evaluations. Its replay is verification work, not an additional independent discovery run.

## Important execution contracts

The model and numerical family must remain the same during budget resizing. C pool calls finish before a plan is rebound. Public inference copies outputs; internal activation views must not outlive a resize. Only arrays controlled by the declared budget are resized, and allocator RSS may remain high after their release.

Full effective microbatches keep the main training convention fixed. If accumulation is used, all microbatches must have equal supervised counts. BCIR stores the sum of microbatch-mean gradients and applies `1/accumulation` inside AdamW. Unequal counts are refused; a future variable-density packing path needs explicit global-denominator or weighted accumulation semantics.

Removing a causally unused suffix preserves the mathematical objective, but changes floating reduction partitions. The archived tests show identical logits/loss and tiny initial gradient differences, not bitwise-identical long training. Likewise, correct accumulation can lead to different extrapolation after many updates. Rerun the capability gates when changing execution/numerical conventions.

The registry reference knows the synthetic grammar and uses exact parsing/lookup. Its perfect score is a label/control check, not a learned-model result. The interior-product side channel in the report is a future hypothesis, not implemented or benchmarked here.

Latest LLVM/Clang/MLIR, Python 3.15, full BCIR CI, Windows, native ARM and accelerator qualification remain unperformed. The inspected repository worktrees remain unchanged; no commit, push or PR was created by this research session.
