# Compiled Triton-CPU / BCIR pilot, 2026-10-19

This directory supports the updated
[comparative analysis](../../../BCIR_TRITON_COMPARATIVE_ANALYSIS.md).
It contains two sessions of **protocol v2**, with no changes between sessions:

- [session-a.json](session-a.json) and [session-b.json](session-b.json): raw paired rounds,
  candidate-tuning samples, numerical checks, configuration/IR identities and environment.
- [provenance.json](provenance.json): source pins, verified wheel identity, CPU features,
  resource limits and SHA-256 hashes of the harness, native sources and result files.
- [measured-harness.py.txt](measured-harness.py.txt): exact source bytes used for sessions A/B,
  matching their recorded `harness_sha256`. The current reproducer differs only by explicit
  UTF-8/LF settings on its two JSON writes, added after CI's portability check. The kernel and
  timing code are unchanged. This archive is source evidence; restore it to its original
  `tools/perf/triton_cpu_compare.py` path to execute that exact historical source.
- [reproducer](../../../../../tools/perf/triton_cpu_compare.py): optional experiment tooling;
  NumPy/Triton are not added to BCIR's dependency-free core.

Both sessions finished `PASS`: ten shapes at one and two workers, 21 paired rounds per case,
48 checked Triton configurations per session. No GPU, model training, gradient test, quality
test, PyTorch comparator or PR #820 implementation was executed. Wall times on this VM are
observations, not CI thresholds or physical-hardware optimality certificates.

## Reproduce

Use this report's commit: its native C sources are unchanged from the BCIR pin recorded in
provenance.json. Verify their hashes before reproducing the historical comparison; the
`measured_harness` entry identifies the historical harness separately from the current tool.
Use Linux
x86-64 with at least two available CPUs, GCC/G++, Python 3.12, and the official
**experimental CPU snapshot**, not the standard Triton PyPI GPU wheel.
Install dependencies in an isolated environment; keep the core package unchanged.

```bash
python3.12 -m venv /tmp/bcir-triton-env
/tmp/bcir-triton-env/bin/python -m pip install numpy==2.3.5
```

Download the asset named
`triton-3.8.0+cpu-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
from the [official v3.8.0 CPU release](https://github.com/triton-lang/triton-cpu/releases/tag/v3.8.0).
Verify its 268,075,073 bytes and SHA-256 before installing it:

```python
import hashlib
from pathlib import Path

wheel = Path("/tmp/triton-3.8.0+cpu-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
assert wheel.stat().st_size == 268075073
with wheel.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == (
        "693f657501e6d791e889d5fd880d3b84a9bf4d8f3087dbb685534bb8533361fa"
    )
```

```bash
/tmp/bcir-triton-env/bin/python -m pip install --no-deps /tmp/triton-3.8.0+cpu-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
TRITON_DEFAULT_BACKEND=cpu OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 /tmp/bcir-triton-env/bin/python tools/perf/triton_cpu_compare.py --out /tmp/bcir-triton-session-a --rounds 21 --target-ms 8
TRITON_DEFAULT_BACKEND=cpu OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 /tmp/bcir-triton-env/bin/python tools/perf/triton_cpu_compare.py --out /tmp/bcir-triton-session-b --rounds 21 --target-ms 8
```

Run those commands sequentially from the repo root. Output directories must be fresh. The
script selects the first two allowed logical CPUs, builds the existing tensor C unit and
uses CPU-targeted compiled Triton kernels over typed NumPy buffers. It refuses another backend
or package version. Generated `.so` files, JIT caches and selected-stage IR stay in the output
directories; their identities are recorded in JSON. They are not committed here. For a shorter
functional check use `--smoke --rounds 5 --target-ms 2` in another fresh output directory.

## Protocol and limits

Triton vector blocks: 256, 1024, 4096. GEMM tiles `(BM, BN, BK)`:
`(16,32,32)`, `(32,32,32)`, `(32,64,32)`. RMS uses one program per row and the
next-power-of-two block width. All candidate outputs must pass before tuning. Five warm
tuning batches select a configuration; the 21 subsequent rounds use randomized implementation
order and separately calibrated repeat counts. Timing is per synchronous call, including the
host entry/launcher, over repeated buffers. It excludes reset, the 2 ms parking interval,
allocation, compilation and tuning. Both entries receive cached integer addresses; Triton's
compiled launcher is prebound. The Python collector is paused symmetrically.

The idle OpenMP team parks (`PASSIVE`, spin count 0), and BCIR's bounded spin/yield tail gets a
2 ms interval outside each timed batch. This prevents an inactive backend consuming the two
CPUs while its competitor is measured. It also means the two-worker rows represent coexistence
settings, not optimal standalone OpenMP tuning. The small residual/SwiGLU case fits one BCIR
work chunk, so two configured workers need not mean two workers actually compute that call.

Inputs are finite seeded random FP32 values; seeds are identical across thread counts and
sessions. Float64 NumPy references are outside timing, with OpenBLAS constrained to one thread.
Admission is `abs(error) <= tolerance * (1 + abs(reference))` per element: add `1e-7`,
SwiGLU `3e-6`, RMS/GEMM `1e-5`. No universal error bound or all-input semantic proof is claimed.
Reductions/exponentials may differ. GEMM enables FMA; other kernels disable contraction.
BCIR's fused and kblock paths are kept separately; SwiGLU includes libm and vector-exp v2.
Triton has no optional XSMM/oneDNN microkernel provider in this experiment.

The raw 95% intervals resample paired measurement rounds 2,000 times and take the ratio of
medians. They describe this sample, not all VM variability or all hardware. Known input/output
bytes are not peak memory: process caches, stacks, spills, launch allocations and compiler
resident memory were not profiled. The comparison does not time BCIR graph planning or ASN.1.

Exploratory smoke runs before protocol v2 identified idle-pool interference and per-call pointer
conversion overhead. Those runs were used to fix the measurement boundary and were not mixed
into either published session. The source/runtime pins and final protocol are identical in A/B.
