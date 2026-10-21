# BCIR PR818 geometric-head rerun

Start with [the assessment](BCIR_PR818_Geometric_Head_Report_2026-10-18.md) and [the result figure](BCIR_PR818_Results_2026-10-18.png).

This is a standalone native-C research experiment. It contains a fixed-architecture FP32 regression (51 runs, 41,400 updates), a separately declared BF16 follow-up (15 runs, 12,600 updates), paired runtime measurements and actual C-plan buffer inventories. The models have approximately 0.53 million parameters. The custom full global/query head runs through the native training rail; lowering it through the general IR/GEM+ planner remains future integration work.

Source pins:

- Before: `aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d`, already inside PR818.
- After: merged main `237651126684569e5b756dcc8d1c89ed2e86d0e3`.
- The four files in each `vendor-old` / `vendor-new` snapshot match those Git commits byte for byte. Original licenses are retained beside them; `source-provenance.json` records the verification.

The archive retains selected checkpoints and raw records, including the previous study's baseline outputs. It excludes the full external Git repository and transient caches. Paths and commands recorded in build logs describe the measured environment; executable Python scripts locate dependencies relative to their current directory.

The report contains dependency and command instructions. Run heavy commands sequentially. Training/evaluation scripts preserve existing results; for a new from-scratch replication, work in a separate extracted copy and remove its generated `results`, `checkpoints`, `bf16/results` and `bf16/checkpoints` contents first. Keep the supplied frozen evidence intact in the original copy. Rebuild shared libraries on a different machine and record its compiler/CPU; prebuilt objects are host artifacts.

Key records:

- `protocol.json`, `bf16/protocol.json`: predeclared architecture, inputs, optimization and selection.
- `results/before-after-audit.json`: exact selected/final states, loss/norm streams and fixed-sample regression comparison.
- `results/paired-timing.json`: all paired observations, ratio intervals, workspace probes and observed page coverage.
- `results/buffer-inventory.json`: 100 native plan/packing queries and explicitly unimplemented mixed-layer planning estimates.
- `results/update-assessment.json`: structured comparison for later work.
- `results/validation.json`, `bf16/validation.json`, `upstream-gate/report.json`: focused implementation checks.
- `baseline/`: previous results, protocol and report, without duplicating all previous checkpoint files.
- `DATA_PROVENANCE.md`: retained language-data sources and notices. Recall tasks are generated from the saved seeds.
- `MANIFEST_SHA256.json`: per-file byte counts and hashes, excluding the manifest itself.

This CPU is AMD EPYC with AVX-512F and no Intel AMX. The upstream 9.84-million-parameter AMX/PyTorch comparison is a separate reported result; these measurements do not reproduce that hardware comparison. Controlled buffer totals are not process RSS.
