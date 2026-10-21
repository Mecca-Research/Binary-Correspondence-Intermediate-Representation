# BCIR native global Grassmann/Plücker experiment

Independent research extension of arXiv:2512.19428v1. The archive contains real
native C training, explicit backward kernels, raw measurements and checkpoints.
It is not the author's implementation or a merged/production BCIR feature.

## Reproduce

Use Python 3.12 with the versions in `requirements.txt` and GCC 13.3 on x86 Linux.
Clone `Mecca-Research/Binary-Correspondence-Intermediate-Representation` as the
sibling directory `../BCIR-gp-head`, checking out
`aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d`. The ctypes ABI comes from that checkout;
the exact native sources and license are also retained in `vendor/`.

From this directory:

```sh
python build_native.py --repo ../BCIR-gp-head
python validate_extended.py
python run_experiments.py
python run_measurements.py
python recall_length_probe.py
python position_ablation.py
taskset -c 0 python benchmark_position.py --threads 1
python benchmark_position.py --threads 4
python common_mode_probe.py
python verify_results.py
python summarize_extended.py
python write_report_extended.py
```

`run_experiments.py` resumes by skipping complete result JSONs. To genuinely
retrain the three new recipes, back up and remove their 18 task/kind/seed JSONs
first. Never overwrite the supplied evidence without making a copy.
The protocol file is frozen and reused, not silently regenerated. The 18
baseline checkpoints are retained, so evaluating them does not require the
earlier pilot directory. Their original metadata is in `pilot-results/`.
Baseline training itself is not repeated by this archive.

`run_measurements.py` serializes baseline evaluation, six-recipe full-update
timings, one-core primitive timings, streaming/subspace/common-factor probes,
summary and report generation. It uses `taskset -c 0` for one-core measurements;
select an actually permitted CPU on another host and record that change.
Do not run heavy jobs concurrently. The optional recall-length test performs
no fitting or checkpoint selection. Its results are folded into the final
summary/report by the last two commands.

The three new models use a strictly earlier prefix scan and current rotary
rank-22 projection. They compare normalized wedge, matched off-diagonal
symmetric bilinear features, and a regularized rank-two projector. The latter
has 0.073% more parameters. All train through native C; PyTorch is only the
independent gradient/forward reference. No GPU or PyTorch training-speed claim
is made. The global APIs reserve mutable cached-prefix space in the projected
feature arena and its adjoint in scratch; see `global_core.h`.

The training implementation hashes and protocol were frozen before training.
The measurement harness was later expanded, and the baseline ingestion helper
was made self-contained. `verify_results.py` checks the model/protocol hashes
and explicitly records those nontraining changes. The final manifest covers
every retained file except itself. Source-generated `.so` files are supplied
as evidence for the measured host; rebuild them on other systems.

## Read the evidence

- `REPORT.md`: results, mathematical derivations, limitations and next changes.
- `global_core.h`: C pair forward/VJP, prefix/reverse scans, direct control.
- `streaming_probe.c`: feature-only incremental API (not a full decoder).
- `results/validation.json`: independent oracle, degeneracies and UBSan.
- `results/common-factor-erratum.json`: correction to the pilot's generic
  averaging argument. Shared-first-vector local wedges remain decomposable.
- `results/summary.json`: aggregated learning/performance evidence.
- `results/recall-length.json`: post-training length stress test.
- `results/checkpoint-verification.json`: all 42 checkpoint hashes/state digests.
- `checkpoints/` and `baseline-checkpoints/`: weights, AdamW moments and step.
- `protocol.json`: fixed choices, seeds, environment and raw data hashes.
- `position-ablation/`: separate exploratory protocol, six trained checkpoints,
  independent validation, length tests and alternating runtime measurements.
- `results/common-mode.json`: cancellation/conditioning stress diagnostic.

## Attribution

Preserve `vendor/LICENSE`, BCIR's Non-Commercial License 1.0 (July 2026).
The retained WikiText-2 raw parquet files are from the Salesforce Hugging Face
dataset, originally by Stephen Merity, Caiming Xiong, James Bradbury and Richard
Socher, with original Wikipedia contributors retaining their rights. The
dataset card lists CC BY-SA 3.0 and GFDL. Preserve attribution and applicable
share-alike terms. Exact source URLs and hashes are in `protocol.json`.
The uploaded research PDF is not redistributed.

Related primary sources:
https://arxiv.org/abs/2512.19428 ;
https://math.mit.edu/~edelman/publications/geometry_of_algorithms.pdf ;
https://proceedings.mlr.press/v119/katharopoulos20a.html .
