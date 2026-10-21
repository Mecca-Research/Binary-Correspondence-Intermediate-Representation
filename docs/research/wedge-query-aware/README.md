# Full global wedge head with query-aware contraction

This is a self-contained BCIR native-C research implementation and controlled
learning study. The full rank22 head retains all231 packed wedge coordinates
and its dense231→128 projection. The new side channel is `c=q^T P`, with
`q=u/max(norm(u),1e-6)`; the mixer receives `Wp*p + Wc*c`.

The report and raw results determine whether this particular inductive bias
helps. Correct gradients and geometric identities alone do not establish
better learning or arbitrary key/value memory.

## Implementation

- `query_core.h`: allocation-free C contraction, VJP, fused frame adjoints and
  native pool dispatch. Packed antisymmetry gives the positive `q^T P` sign.
- `build_query.py`: exact-asserting C generation from pinned full-head sources.
  It produces cached compact, frame/FF recomputation and combined layer/frame/FF
  recomputation variants. The original full projection is preserved.
- `precision_core.h`: existing FP64 ill-conditioning guard; absent planes have
  zero feature/VJP. The guard is common to all global-family study arms.
- `experiment.py`, `native_abi.py`: parameter/checkpoint layout and exact C ABI.
- `policy.py`: reusable native workspace, budget admission and optimizer state.
  This study's global portfolio uses three guarded layouts. No timing lookup is
  borrowed from another model or machine.
- `oracle.py`: independent PyTorch equations for correctness only.
- `validate_query.py`, `conditioning_check.py`, `probe_query.py`: AD, directional
  finite differences, causal, numerical, state/replay, red-zone and UBSan checks.
- `study.py`: frozen three-seed,51-run training protocol.
- `evaluate_query.py`: fresh tests and association counterfactuals after
  checkpoint selection; per-case results are resumable.
- `results/`: raw training events, validation histories, gradients/checks and
  paired predictions/labels. `checkpoints/` holds selected model and AdamW state.

No sibling checkout, provider API, GPU, downloaded model or hidden service is
required. All model training/backward/AdamW arithmetic runs in C. Python hosts
the experiment, creates data, binds plans and selects validation checkpoints.
PyTorch is not the training-throughput engine in this study.

## Reproduce

The measured environment is Python3.12, GCC13.3, NumPy2.3.5, PyArrow26.0.0 and
CPU PyTorch2.14.1. The host uses4 native training threads; heavy jobs are
serialized. Run from this directory:

```sh
python3 -m venv .venv
.venv/bin/pip install numpy==2.3.5 pyarrow==26.0.0 matplotlib==3.10.8 pillow==12.3.0
.venv/bin/pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python build_query.py
.venv/bin/python validate_query.py
.venv/bin/python conditioning_check.py
.venv/bin/python study.py --freeze-only
.venv/bin/python study.py
.venv/bin/python probe_query.py
.venv/bin/python evaluate_query.py
.venv/bin/python summarize_query.py
.venv/bin/python plot_query.py
.venv/bin/python audit_query.py
.venv/bin/python build_report.py
```

The archive contains completed results. Existing case files are skipped. To
perform a genuinely fresh run, copy the source/data/foundation/vendor files into
a new directory and create empty `build`, `results` and `checkpoints` directories.
Use the archived protocol as an immutable reference; a new run's timestamps and
host-dependent timings will differ. Do not overwrite the only completed copy.

Individual training example:

```sh
.venv/bin/python study.py --task kv_curriculum --model query_matched --seed 20261018
```

Recipes are explicit: `global`=525696 parameters, `query_full`=534144 with FF257,
`query_matched`=526080 with FF250, `attention`=525696 with FF256 and RoPE.
The separate `hybrid` arm is a language control, not a shared task checkpoint.
All global arms use no RoPE. Both the baseline and contraction arms use the
same inherited precision safeguard to isolate the added readout.

## Bounds and qualifications

The32MiB training envelope bounds four FP32 model/optimizer arrays, the unified
activation/scratch arena, input/target staging, pool-control storage, alignment
slack and named C records. It excludes code, thread stacks, Python metadata,
datasets, checkpoint copies, inference copies and allocator-retained pages.
It is not RSS or an operating-system-enforced process memory limit.

The shared512MiB evaluation envelope is separate; no update occurs during test.
Attention may accumulate microbatches during training under32MiB. The native
gradient array stores a SUM of microbatch means and AdamW scales it by the
accumulation count. The controller refuses unequal active counts per microbatch.

This artifact is generated research C, not a registered production IR/MLIR op
or repository runtime merge. No repository source is modified. Native sources
derive from `aaa2e6c0ee0ee15b83fe4d0373b48bd893e80d6d`; inspected upstream main is
`be6e377a2eb417cb4bfe3de5243ca09961ba1833`. Qualification on the full BCIR CI
matrix, LLVM/Clang/MLIR23, Python3.15, native ARM/Windows and accelerators remains
outside this CPU study. See the report for measured outcomes and limits.
