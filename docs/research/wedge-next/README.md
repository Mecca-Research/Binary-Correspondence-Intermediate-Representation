# BCIR global-wedge advancement experiment — 2026-10-17

This is a research bundle, not a replacement for the BCIR repository or a production runtime PR.

Read `BCIR_Global_Wedge_Advancement_Report_2026-10-17.md` first. The figure is `BCIR_Wedge_Memory_Quality_Frontier_2026-10-17.png`; machine-readable aggregates are `results/summary.json`. Every timing repetition, learning checkpoint and main validation output is retained.

## Environment and repository pins

Tested on Linux x86-64, AMD EPYC 9V74, GCC 13.3, Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1, PyArrow 26.0.0, Matplotlib 3.11.2 and CPU PyTorch 2.14.1. There is no GPU requirement. Compiled `.so` files are host artifacts; rebuild for another machine. The CPU backend already dispatches AVX2/AVX-512 kernels where supported.

The scripts expect these sibling directories:

```
<parent>/wedge-next/
<parent>/BCIR-wedge-next/
<parent>/BCIR-gp-head/
```

Prepare the public repositories from `<parent>`:

```bash
git clone https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation.git BCIR-wedge-next
git -C BCIR-wedge-next checkout be6e377a2eb417cb4bfe3de5243ca09961ba1833
git -C BCIR-wedge-next fetch origin pull/818/head
git -C BCIR-wedge-next worktree add --detach ../BCIR-gp-head aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d
```

The old checkout supplies Python ctypes ABI declarations; `vendor/` retains the actual C sources compiled here. The newer checkout supplies the HAM/static-memory Python planners. No repository files are modified by the experiment. Keep heavy jobs sequential; pool thread count is an experimental condition. Latest-toolchain/full CI/hardware gates are not qualified by this bundle.

Use a Python environment with the versions above. Install CPU PyTorch using its official CPU wheel index. `requirements-measured.txt` records direct packages; hardware, compiler, hashes and optimization contracts are also recorded in `protocol.json` and build metadata.

## Inspect and verify the retained result

From `wedge-next/`, with the environment active:

```bash
python build_next.py
python validate_next.py
python precision_next.py
python limits_next.py
python summarize_next.py
python plot_next.py
```

`build_next.py` does not rebuild `hybrid.so`; its emitted source is retained as `build/hybrid.c`. To rebuild it without retraining, use:

```bash
python - <<'PY'
from build_next import compile_source
from pathlib import Path
compile_source(Path('build/hybrid.c').read_text(), 'hybrid')
PY
```

Rebuild all shared objects needed by `summarize_next.py`, including the hybrid, before auditing on a different host. `precision_next.py` and `limits_next.py` build their own variants. The archive also retains SVD variants; re-running `svd_next.py` rebuilds those and evaluates them again without training.

Validation re-runs may overwrite validation/build evidence. Preserve the original archive or work in a copy. Compilers and other CPUs may produce different bit patterns even under compatible mathematical contracts; the original hashes identify the measured environment.

## Repeat measurements and training

`train_next.py` intentionally skips existing per-run JSON files. For a genuinely fresh repetition, work in a separate extracted copy, preserve the archived `results/` and checkpoint directories elsewhere, then create empty output directories. Do not discard the frozen protocol or alter candidates after inspecting tests. The measured output remains available in the original archive.

Core sequence:

```bash
python build_next.py
python validate_next.py
python train_next.py
taskset -c 0 python benchmark_next.py --mode execution --threads 1
python benchmark_next.py --mode execution --threads 4
taskset -c 0 python benchmark_next.py --mode architecture --threads 1
python benchmark_next.py --mode architecture --threads 4
taskset -c 0 python profile_next.py --threads 1
python profile_next.py --threads 4
python precision_next.py
python budget_next.py
python ham_next.py
python assess_next.py
python svd_next.py
python hybrid_next.py
python limits_next.py
python followup_next.py
taskset -c 0 python benchmark_limits.py --threads 1
python benchmark_limits.py --threads 4
python summarize_next.py
python plot_next.py
```

If CPU 0 is unavailable, choose a permitted CPU and record it. Avoid running training and timing blocks concurrently. Timed regions use three warmups and ten measured updates per repetition in the main and additional execution comparisons. Budget experiments use two warmups and six measured effective updates. Original result files retain all repetition counts and values.

`hybrid_next.py` is a separate exploratory stage after the main run. Its nested global result files reference main-root checkpoints, while hybrid result files reference `hybrid/checkpoints/`; `summarize_next.py` handles this distinction explicitly. SVD files are inference-only, with no conversion of old AdamW state.

`run_post.py` records an earlier orchestration attempt that stopped on an overstrict precision assertion. The assertion was corrected to distinguish guarded and unguarded arithmetic, and the remaining jobs completed sequentially. `protocol-design-draft.json` predates the multi-subspace absent-plane fix; `protocol.json` is the final pre-training protocol. `debug-multi.npz` is the random-input duplicate-token gradient witness. These diagnostic artifacts are retained for provenance, not counted as successful training runs.

## Reading the evidence

* `results/validation.json`, `results/limits-validation.json`: exact execution, gradients, state/resume, UBSan.
* `results/benchmark-*.json`: raw alternating timing repetitions and final-state digests.
* `results/{byte_lm,recall_*}-*.json`: per-run validation, native events, selected checkpoint and test metrics.
* `results/length-stress.json`, `results/followup.json`: paired length/ordered-shift and hybrid/guarded tests.
* `results/profile-*`, `results/budget.json`: measured kernel shares and constrained native-buffer experiment.
* `results/ham.json`, `ham-artifacts/`: verified planner-only residency scenarios and exact scratch fixture. No hardware DMA was executed.
* `results/precision.json`, `results/svd.json`: separate numerical-contract and approximate-compression experiments.
* `results/final-audit.json`: source hashes and checkpoint/restoration evidence.
* `SOURCE_MANIFEST.json`: file hashes for the delivered bundle, excluding the manifest itself and runtime caches.

Memory figures describe native arrays, not RSS. Language metrics are on small WikiText-2 byte models. First-token recall is a synthetic task with explicit vocabulary conditions; it is not a general retrieval benchmark. This bundle does not reproduce the original paper’s reported full-scale tasks, benchmark PyTorch training speed or prove a global hardware optimum.
