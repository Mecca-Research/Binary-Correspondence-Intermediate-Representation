# Backend optimality: the OT experiments and the BG gates, executed

> Dated 2026-10-21, updated 2026-10-22 (the PyTorch and Triton routes), non-normative. The
> executable side of the proposal in draft PR #821
> (`docs/research/BCIR_BACKEND_OPTIMALITY_AND_KERNEL_GATES_2026-10-20.md` on its branch, pinned at
> `c01a53c8`): sixteen optimality experiments, OT-01 to OT-16, and eight entry gates, BG0 to BG7.
> The proposal is #821's and stays there; this page records what of it runs in this tree, on
> which host, with what result, and what each gate still needs. Counts live in the generated
> [`STATUS.md`](../STATUS.md).

## 0. The answer

- **BG0, the measurement contract, is code.**
  [`tools/perf/ot_contract.py`](../../tools/perf/ot_contract.py) holds the proposal's vocabularies
  -- the comparator routes (its section 3), the kinds of gap a result reports (section 2), the host
  capabilities an experiment needs -- the record every experiment writes, held by `validate_record`
  to one shape, the sixteen experiments as a registry (section 5) and the eight gates as
  requirements over evidence (section 8), stated by `evaluate_gates`. BG0 passes when every
  experiment has a record the contract admits: measured, unavailable with the capabilities this host
  lacks, or blocked with the prerequisite the repository lacks. The proposal's BG0 also binds its
  interface tests (the SI matrix of #821's universal-interface roadmap) to the same contract; those
  tests are not in this tree, so this BG0 is the OT half.
- **Nine of the sixteen run here**: OT-01, OT-02, OT-03, OT-04, OT-05, OT-06, OT-10, OT-11 and OT-14
  -- on BCIR's own rails and, since 2026-10-22, against PyTorch (eager, FX, AOTAutograd and
  Inductor) and Triton-CPU. The rest are written as unavailable (a GPU, a multi-device host, QEMU or
  a device rig this host lacks: never simulated) or blocked (OT-15: no IPC adapter to decompose).
- **Against PyTorch and Triton**: BCIR's decoder trains faster than both PyTorch routes at four
  threads -- OT-10: 272 ms a step against eager's 533 and Inductor's 390, in under half their
  memory, its held-out losses within 0.021 nats of theirs -- and its plan costs milliseconds where
  Inductor's first call costs 44-80 s (OT-02, OT-11). Kernel by kernel it is ahead of Triton-CPU
  (OT-05's T-DSL rows: its GEMM 2.2-3.4 times faster, its RMSNorm 2.6-10 times, the SwiGLU product
  under the BCIR exponential at parity at a million elements). Against PyTorch's own SwiGLU block
  (OT-05), run as the trainer runs it -- kblock+exp, the k-blocked GEMMs with the BCIR exponential's
  passes -- it is faster at four threads in sixteen of eighteen comparisons (1.32-5.45 times) and at
  parity in two (512 x 768 x 1024); at one thread faster in ten (1.07-1.72 times), at parity in
  five, and slower in three: 512 x 768 x 1024 (1.37 and 1.38 times eager's and Inductor's time) and
  512 x 257 x 1025 against Inductor (1.10 times its time). The day's first run read 1.17-1.44 times
  PyTorch's time at one thread on the 128- and 512-row blocks and put it down to MKL's GEMMs; it had
  held PyTorch against v1's libm SiLU passes, which the trainer does not run (that arm reads
  1.15-1.47 times again here), and they take 3.4-7.3 times the BCIR exponential's time. Taken apart
  at one thread, the block's GEMMs keep pace with MKL's but on the largest block, whose A (1.5 MiB)
  is larger than the core's 1 MiB L2 cache -- its three GEMM forms, each run twice in the block,
  29.3 ms against MKL's 22.6, the nn and tn forms 1.35 and 1.45 times its time -- and in the weight
  adjoint's transposed form, behind at seven of the nine shapes (1.04-1.53 times). Those two name
  the next training update.
- **The gates on this host**: BG0 PASS, BG1 PASS, BG2 UNAVAILABLE, BG3 OPEN, BG4 UNAVAILABLE, BG5
  UNAVAILABLE, BG6 UNAVAILABLE, BG7 UNAVAILABLE.

## 1. What runs, and how a gate reads it

| Piece | What it does |
|---|---|
| `tools/perf/ot_contract.py` | the contract (BG0): routes, gaps, capabilities, the record and `validate_record`, the sixteen, the eight and `evaluate_gates`; `--list`, `--gates RECORDS.json` |
| `tools/perf/ot_run.py` | runs the experiments the host can run, writes all sixteen records under the contract, reads the GEM+ rows a gate names (`--rows`, `gemplus_baseline.py --json`), a Triton-CPU session (`--triton`) and a training frontier (`--frontier`) -- every input read before any experiment runs -- and states the gates |
| `tools/perf/ot_torch.py` | the PyTorch routes: P-EAGER's numerics for OT-01, the capture cost of P-FX, P-AOT and P-IND (OT-02), the block portfolio's P-EAGER and P-IND arms beside B-DIRECT's under each GEMM contract and under the trainer's own, kblock+exp (OT-05), and Inductor's cache lifecycle (OT-11), each compiled step in its own process with a fresh cache |
| PR #821's Triton-CPU pilot | T-DSL's rows (OT-05): its own interpreter and wheel (`triton-cpu`), its BCIR side built from this tree's `bcir_tensor.c`; a session is admitted only over this tree's tensor unit, passed and correct on both sides |
| `docs/research/measurements/2026-10-22/block-parts/block_parts.py` | OT-05's block taken apart at one thread: each GEMM form under BCIR's three contracts against torch.mm (MKL), each elementwise pass under v1's libm SiLU, the BCIR exponential's and PyTorch's |
| `docs/research/measurements/2026-10-11/train_comparison.py` | OT-10's runs: one prepared recipe, each route's learning curve, step time, tokens a second and memory |
| `bcir/tests/test_ot_contract.py` | the registry against the proposal, every record rule's refusal, the gates' semantics (nothing PASS without its evidence; UNAVAILABLE never a pass; no gate ahead of the gate it requires; every gate passable with complete evidence), OT-01's checks firing past their bounds and its verdict every route's, every experiment without a runner recorded on every host, the PyTorch routes' records keeping every route that ran and every failure, OT-05's arms each run under its own contracts and paired round by round, a Triton session admitted only as evidence about this tree, the runner's inputs refused before it runs, a record naming the PyTorch and threads it ran, a smoke run |
| `tools/testing/faults/ot-contract.json` | the faults each witness exists to catch |

A gate is **PASS** only when every requirement is met by evidence: an experiment's record
measured, a GEM+ row at its bound (0), a host capability, a driver-ladder milestone attested by a
report the repository holds, another gate's PASS. It is **UNAVAILABLE** when a requirement is a
capability this host lacks or an experiment unavailable here -- never a pass -- and **OPEN**
otherwise, naming what is missing.

## 2. The sixteen on this host

| OT | Experiment | Routes compared | Needs | Here |
|---|---|---|---|---|
| OT-01 | common operation/gradient contract | B-DIRECT, P-EAGER | cpu, numpy | **measured** (B-DIRECT, P-EAGER) |
| OT-02 | IR/capture cost and coverage | P-EAGER, P-FX, P-AOT, P-IND, B-PLAN | cpu, torch | **measured** (P-FX, P-AOT, P-IND, B-PLAN) |
| OT-03 | backward and optimizer identity | B-DIRECT, B-PLAN, P-EAGER | cpu | **measured** (B-DIRECT, B-PLAN) |
| OT-04 | plan intervention | B-DIRECT, B-PLAN | cpu | **measured** (B-DIRECT, B-PLAN) |
| OT-05 | kernel/block portfolio | B-DIRECT, B-MLIR, P-EAGER, P-IND, T-DSL, T-GLUON | cpu, torch | **measured** (P-EAGER, P-IND, B-DIRECT, T-DSL) |
| OT-06 | layout/packing/arena frontier | B-DIRECT, B-PLAN | cpu | **measured** (B-DIRECT, B-PLAN) |
| OT-07 | mapping and conversion | T-DSL, T-GLUON, B-DEVICE | gpu, triton | unavailable: no gpu, triton |
| OT-08 | pipeline and synchronization | T-DSL, T-GLUON, B-DEVICE | gpu, triton | unavailable: no gpu, triton |
| OT-09 | same image, different dispatcher | B-DEVICE | gpu | unavailable: no gpu |
| OT-10 | complete training frontier | B-DIRECT, B-PLAN, P-EAGER, P-IND, P-TUNED | cpu, torch | **measured** (P-EAGER, P-IND, B-DIRECT) |
| OT-11 | specialization/cache lifecycle | P-IND, P-TUNED, T-DSL, B-PLAN | cpu, torch | **measured** (P-IND, B-PLAN) |
| OT-12 | data/residency path | B-DEVICE, S-DEVICE | gpu | unavailable: no gpu |
| OT-13 | collectives/checkpoint | P-EAGER, B-PLAN | multi-device | unavailable: no multi-device |
| OT-14 | telemetry/TMOG observer and feedback | B-DIRECT, B-PLAN | cpu | **measured** (B-DIRECT) |
| OT-15 | IPC decomposition | OS-DIRECT, OS-LINUX, OS-NATIVE | cpu | blocked: RuntimeChannel has its direct adapter alone: no shared-memory queue, Linux process or native adapter exists to decompose against it (the driver ladder's M2, M3 and M8) |
| OT-16 | device/OS vertical slice | OS-DIRECT, OS-LINUX, OS-NATIVE | qemu, physical-device | unavailable: no qemu, physical-device |

## 3. The eight gates

| Gate | Title | Here | Unmet |
|---|---|---|---|
| BG0 | measurement contract | **PASS** | -- |
| BG1 | trace-to-execution | **PASS** | -- |
| BG2 | hosted accelerator slice | **UNAVAILABLE** | capability gpu: one real target with a resident provider; record OT-07: OT-07 on that target; record OT-08: OT-08 on that target; record OT-09: OT-09 on that target; record OT-12: OT-12 on that target |
| BG3 | direct driver slice | **OPEN** | milestone UART-D2: a D0-D2 UART package beside the RuntimeChannel loopback, with telemetry and bounded lifecycle tests |
| BG4 | Linux queue/transport proof | **UNAVAILABLE** | gate BG3: the direct driver slice; milestone M2: UART through Linux-hosted D3, direct/Linux parity; milestone M4: virtio console and block, direct/Linux parity; record OT-15: OT-15 measured; record OT-16: OT-16 measured |
| BG5 | native service/boot proof | **UNAVAILABLE** | gate BG4: the Linux queue/transport proof; capability qemu: a UEFI/reset boot proof; milestone M8: the native service: direct/Linux/native replay; record OT-16: direct/Linux/native replay for the supported devices |
| BG6 | targeted kernel policy/fast path | **UNAVAILABLE** | gate BG5: the native service/boot proof; capability physical-device: a profile on the declared workload's hardware; milestone M7: an adapter or kernel change that improves the declared workload |
| BG7 | integrated promotion | **UNAVAILABLE** | gate BG2: BG2 passed; gate BG4: BG4 passed; gate BG5: BG5 passed; gate BG6: BG6 passed |

## 4. The measured experiments

The records are in [`measurements/2026-10-22/ot/`](measurements/2026-10-22/ot/README.json)
(the block taken apart in [`measurements/2026-10-22/block-parts/`](measurements/2026-10-22/block-parts/README.json); the first run's, BCIR's routes alone, in [`measurements/2026-10-21/ot/`](measurements/2026-10-21/ot/README.json)):
the run on Intel(R) Xeon(R) Processor @ 2.80GHz (4 vCPUs; BCIR's timed experiments at one thread, the PyTorch routes at the host's), a provisional commit of this change (84e4db0) with nothing uncommitted below bcir, runtime/c or tools/perf, Ubuntu clang version 18.1.3 (1ubuntu1), NumPy 2.2.6, Python 3.11.15, PyTorch 2.13.0+cu130 (AVX512, BLAS mkl); the capabilities it found: cpu, numpy, torch.

- **OT-01** (B-DIRECT against NumPy float64): the first common block -- U = X Wg, V = X Wu, Y = silu(U) * V and its backward (dU, dV, dX, dWg, dWu) -- on 18 shapes (16x256x1024, 16x768x1024, 128x256x1024, 128x768x1024, 512x256x1024, 512x768x1024, 16x257x1025, 128x257x1025, 512x257x1025, 16x256x1024, 16x768x1024, 128x256x1024, 128x768x1024, 512x256x1024, 512x768x1024, 16x257x1025, 128x257x1025, 512x257x1025), every element within its binary32 bound: the worst error is 0.366 of its bound (a GEMM element within gamma_k |A| |B| of the exact product in any order, an elementwise one within 8 units of roundoff of its terms). The coverage ledger: dense row-major GEMMs and their transposes, the SiLU gate and its backward, the zero-size product a no-op; strided operands unsupported (the entries take dense rows).
- **OT-03** (B-DIRECT against B-PLAN): 27 of 27 state cases of the trace corpus train one state and one event sequence, bit for bit, from the direct C loop, from the step program and from its StreamPack and ExecutionPlanV1 lowered by the C.
- **OT-04** (plan intervention): every layer's q/k/v and gate/up projections a call each -- 145 launches become 163, the state the same, bit for bit -- times x1.0043 (95% interval 0.974-1.035) of the default's step over 24 rotated rounds, beside the A/A control's x0.9762 (0.943-1.010), not resolvable beside its control (d_model 384, 6 layers, 8x128 tokens, kblock+exp+reduce+attn, 1 thread).
- **OT-06** (arena frontier): the bench step's transient values take 999,424 bytes in the C plan's fixed spans, x1.326 of G26's bound (753,664); placed by their lifetimes (PX-4) 754,176, x1.000, a proved optimum.
- **OT-14** (observer cost): the step on the library with the launch trace probe times x1.1209 (95% interval 1.092-1.150) of the plain library's over 24 rotated rounds, beside the A/A control's x0.9811 (0.951-1.012); the same state on all three trainers, 5,550 launch records written and 0 dropped.

### 4.1 Against PyTorch and Triton

- **OT-01, P-EAGER beside B-DIRECT**: the same block and backward through PyTorch's autograd on the same draws, held to the same float64 reference and bounds: P-EAGER's worst element is 0.351 of its bound, B-DIRECT's 0.366; the verdict is both routes' (within bound).

- **OT-02, capture cost** (the 384-wide, 6-layer decoder's step at 8x128 tokens, 4 threads; each route in its own process with an empty cache; `module` captures `model(x, y)`, `loss` the whole loss as one graph):

| Route | Capture and first call (s) | Warm call (s) | Frames | Graph breaks | Why |
|---|---:|---:|---:|---:|---|
| P-FX/module | 5.55 | 0.5463 | 6 | 2 | Data-dependent branching |
| P-FX/loss | 3.45 | 0.5315 | 1 | 0 | -- |
| P-AOT/module | 9.17 | 0.7510 | 6 | 2 | Data-dependent branching |
| P-AOT/loss | 6.77 | 0.6574 | 1 | 0 | -- |
| P-IND/module | 76.24 | 0.4343 | 6 | 2 | Data-dependent branching |
| P-IND/loss | 44.25 | 0.4765 | 1 | 0 | -- |

  B-PLAN prices the same step without a compiler in the loop: the step program 4.02 ms (145 launches), its 10,752 words 0.52 ms, the training pack 9.69 ms, StreamPack and ExecutionPlanV1 lowered 67 ms (44,637 and 31,717 bytes); the C library it runs on builds once in 11.5 s for every shape.

- **OT-05, the block portfolio** (OT-01's SwiGLU block, forward and backward, binary32): B-DIRECT under the trainer's contract -- kblock+exp, the k-blocked GEMMs with the BCIR exponential's SiLU passes, as OT-10's B-DIRECT trains -- and under its three GEMM contracts with v1's libm SiLU passes, beside an A/A control, against P-EAGER and P-IND; each ratio is the first arm's time over the second's, the geometric mean of the rounds' paired ratios with its 95% interval (below 1 the first is faster):

| Shape (m x k x n) | Threads | kblock+exp ms | P-EAGER ms | P-IND ms | kblock+exp / P-EAGER | kblock+exp / P-IND | kblock+exp / kblock | kblock / fused | kblock / v1 | A/A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 x 256 x 1024 | 1 | 1.679 | 2.587 | 2.877 | 0.65 [0.62, 0.69] | 0.58 [0.54, 0.63] | 0.80 [0.73, 0.88] | 0.71 [0.68, 0.75] | 0.56 [0.52, 0.62] | 1.00 [0.93, 1.08] |
| 16 x 768 x 1024 | 1 | 4.970 | 5.666 | 6.114 | 0.93 [0.87, 1.00] | 0.84 [0.78, 0.91] | 0.84 [0.78, 0.91] | 0.57 [0.55, 0.60] | 0.50 [0.47, 0.54] | 1.06 [1.01, 1.10] |
| 128 x 256 x 1024 | 1 | 5.908 | 7.252 | 6.828 | 0.82 [0.79, 0.85] | 0.86 [0.82, 0.90] | 0.72 [0.67, 0.76] | 0.68 [0.66, 0.70] | 0.56 [0.53, 0.60] | 1.00 [0.97, 1.04] |
| 128 x 768 x 1024 | 1 | 16.166 | 19.506 | 19.081 | 0.86 [0.82, 0.89] | 0.87 [0.82, 0.92] | 0.91 [0.86, 0.97] | 0.59 [0.57, 0.61] | 0.45 [0.43, 0.47] | 0.95 [0.90, 1.00] |
| 512 x 256 x 1024 | 1 | 20.628 | 20.376 | 19.403 | 1.01 [0.99, 1.03] | 1.05 [1.01, 1.08] | 0.73 [0.71, 0.74] | 0.68 [0.66, 0.69] | 0.54 [0.52, 0.56] | 1.01 [0.99, 1.03] |
| 512 x 768 x 1024 | 1 | 67.485 | 50.454 | 50.507 | 1.37 [1.31, 1.43] | 1.38 [1.30, 1.47] | 0.94 [0.88, 1.00] | 0.67 [0.64, 0.71] | 0.51 [0.48, 0.54] | 0.98 [0.95, 1.01] |
| 16 x 257 x 1025 | 1 | 1.816 | 2.832 | 3.012 | 0.65 [0.58, 0.72] | 0.61 [0.54, 0.68] | 0.82 [0.72, 0.93] | 0.70 [0.64, 0.76] | 0.50 [0.43, 0.58] | 1.06 [0.99, 1.14] |
| 128 x 257 x 1025 | 1 | 5.824 | 6.744 | 6.512 | 0.88 [0.84, 0.93] | 0.92 [0.86, 0.98] | 0.72 [0.67, 0.77] | 0.65 [0.64, 0.67] | 0.52 [0.51, 0.54] | 1.01 [0.96, 1.06] |
| 512 x 257 x 1025 | 1 | 21.030 | 21.202 | 20.312 | 1.01 [0.96, 1.06] | 1.10 [1.04, 1.17] | 0.75 [0.69, 0.81] | 0.65 [0.63, 0.68] | 0.50 [0.47, 0.52] | 0.96 [0.90, 1.02] |
| 16 x 256 x 1024 | 4 | 0.591 | 2.530 | 3.213 | 0.24 [0.21, 0.28] | 0.18 [0.16, 0.21] | 0.57 [0.45, 0.72] | 0.81 [0.66, 1.00] | 0.69 [0.57, 0.82] | 1.19 [0.93, 1.52] |
| 16 x 768 x 1024 | 4 | 1.465 | 4.267 | 4.999 | 0.39 [0.31, 0.49] | 0.33 [0.27, 0.41] | 0.90 [0.73, 1.10] | 0.61 [0.58, 0.64] | 0.45 [0.42, 0.49] | 1.01 [0.91, 1.12] |
| 128 x 256 x 1024 | 4 | 2.013 | 4.499 | 4.538 | 0.46 [0.43, 0.48] | 0.46 [0.43, 0.49] | 0.77 [0.73, 0.81] | 0.62 [0.60, 0.63] | 0.55 [0.53, 0.57] | 0.96 [0.92, 1.00] |
| 128 x 768 x 1024 | 4 | 4.561 | 8.052 | 7.999 | 0.56 [0.53, 0.60] | 0.55 [0.50, 0.61] | 0.80 [0.74, 0.85] | 0.58 [0.54, 0.61] | 0.47 [0.44, 0.49] | 1.07 [1.01, 1.13] |
| 512 x 256 x 1024 | 4 | 6.973 | 9.724 | 9.857 | 0.68 [0.63, 0.73] | 0.69 [0.63, 0.74] | 0.73 [0.67, 0.80] | 0.64 [0.61, 0.67] | 0.52 [0.48, 0.56] | 1.01 [0.93, 1.09] |
| 512 x 768 x 1024 | 4 | 18.952 | 20.539 | 21.124 | 0.91 [0.87, 0.95] | 0.89 [0.82, 0.96] | 0.90 [0.86, 0.94] | 0.62 [0.60, 0.65] | 0.49 [0.47, 0.51] | 0.99 [0.94, 1.05] |
| 16 x 257 x 1025 | 4 | 0.726 | 2.789 | 3.582 | 0.29 [0.25, 0.35] | 0.23 [0.18, 0.30] | 0.76 [0.57, 1.02] | 0.72 [0.63, 0.83] | 0.58 [0.51, 0.67] | 1.12 [0.98, 1.28] |
| 128 x 257 x 1025 | 4 | 2.119 | 4.865 | 4.674 | 0.45 [0.41, 0.49] | 0.47 [0.43, 0.51] | 0.74 [0.69, 0.80] | 0.57 [0.53, 0.62] | 0.48 [0.45, 0.52] | 0.95 [0.86, 1.05] |
| 512 x 257 x 1025 | 4 | 7.886 | 10.677 | 10.603 | 0.74 [0.69, 0.80] | 0.76 [0.69, 0.84] | 0.74 [0.69, 0.80] | 0.59 [0.56, 0.61] | 0.48 [0.44, 0.52] | 1.04 [0.98, 1.10] |

  `kblock+exp / kblock` is the elementwise contract's share: the same k-blocked GEMMs, the BCIR exponential's passes in place of v1's libm SiLU.

  The block taken apart at one thread ([block-parts](measurements/2026-10-22/block-parts/README.json): each part's median over rotated rounds, BCIR's serial path and PyTorch at one thread; the GEMM columns BCIR's k-blocked time over torch.mm's, MKL's, the elementwise ones milliseconds):

| Shape (m x k x n) | nn: kblock / MKL | nt: kblock / MKL | tn: kblock / MKL | SiLU product v1 / v2 / PyTorch (ms) | SwiGLU VJP v1 / v2 / PyTorch (ms) |
|---|---:|---:|---:|---:|---:|
| 16 x 256 x 1024 | 0.92 | 0.74 | 0.98 | 0.19 / 0.03 / 0.06 | 0.20 / 0.04 / 0.14 |
| 16 x 768 x 1024 | 0.80 | 0.65 | 1.53 | 0.32 / 0.04 / 0.06 | 0.31 / 0.05 / 0.19 |
| 128 x 256 x 1024 | 0.70 | 0.95 | 1.19 | 1.19 / 0.23 / 0.30 | 1.33 / 0.34 / 1.01 |
| 128 x 768 x 1024 | 0.58 | 0.95 | 1.42 | 1.16 / 0.23 / 0.32 | 1.30 / 0.28 / 0.94 |
| 512 x 256 x 1024 | 0.93 | 1.12 | 1.04 | 4.71 / 0.87 / 1.80 | 5.49 / 1.22 / 12.02 |
| 512 x 768 x 1024 | 1.35 | 1.09 | 1.45 | 4.49 / 0.95 / 1.50 | 5.08 / 1.47 / 11.93 |
| 16 x 257 x 1025 | 1.13 | 0.79 | 0.90 | 0.20 / 0.03 / 0.05 | 0.21 / 0.03 / 0.12 |
| 128 x 257 x 1025 | 0.95 | 0.93 | 1.27 | 1.15 / 0.21 / 0.23 | 1.27 / 0.27 / 0.88 |
| 512 x 257 x 1025 | 0.84 | 0.99 | 1.17 | 4.60 / 0.94 / 1.54 | 5.14 / 1.25 / 11.57 |

  T-DSL, the Triton-CPU pilot's session over this tree's tensor unit (its own interpreter and wheel; BCIR's side over Triton's, below 1 BCIR is faster):

| Kernel | Shape | Threads | BCIR implementation | BCIR / Triton |
|---|---|---:|---|---:|
| add | 4096 | 1 | `bcir` | 0.34 [0.34, 0.36] |
| add | 1048579 | 1 | `bcir` | 0.83 [0.79, 0.87] |
| swiglu | 4096 | 1 | `bcir_libm` | 4.21 [4.14, 4.28] |
| swiglu | 4096 | 1 | `bcir_v2` | 0.84 [0.82, 0.87] |
| swiglu | 1048579 | 1 | `bcir_libm` | 6.01 [5.70, 6.37] |
| swiglu | 1048579 | 1 | `bcir_v2` | 1.01 [0.95, 1.09] |
| rms | 64 x 128 | 1 | `bcir` | 0.39 [0.38, 0.39] |
| rms | 256 x 257 | 1 | `bcir` | 0.12 [0.12, 0.12] |
| rms | 128 x 1024 | 1 | `bcir` | 0.18 [0.17, 0.19] |
| gemm | 128 x 128 x 128 | 1 | `bcir_fused` | 0.76 [0.76, 0.78] |
| gemm | 128 x 128 x 128 | 1 | `bcir_kblock` | 0.46 [0.46, 0.47] |
| gemm | 256 x 256 x 256 | 1 | `bcir_fused` | 0.72 [0.70, 0.76] |
| gemm | 256 x 256 x 256 | 1 | `bcir_kblock` | 0.45 [0.43, 0.47] |
| gemm | 129 x 257 x 255 | 1 | `bcir_fused` | 0.71 [0.69, 0.72] |
| gemm | 129 x 257 x 255 | 1 | `bcir_kblock` | 0.30 [0.29, 0.30] |
| add | 4096 | 2 | `bcir` | 0.18 [0.15, 0.19] |
| add | 1048579 | 2 | `bcir` | 0.72 [0.67, 0.86] |
| swiglu | 4096 | 2 | `bcir_libm` | 2.97 [2.82, 3.04] |
| swiglu | 4096 | 2 | `bcir_v2` | 0.52 [0.50, 0.55] |
| swiglu | 1048579 | 2 | `bcir_libm` | 6.13 [5.62, 6.29] |
| swiglu | 1048579 | 2 | `bcir_v2` | 1.06 [0.96, 1.09] |
| rms | 64 x 128 | 2 | `bcir` | 0.10 [0.10, 0.10] |
| rms | 256 x 257 | 2 | `bcir` | 0.13 [0.12, 0.14] |
| rms | 128 x 1024 | 2 | `bcir` | 0.20 [0.19, 0.22] |
| gemm | 128 x 128 x 128 | 2 | `bcir_fused` | 0.54 [0.52, 0.57] |
| gemm | 128 x 128 x 128 | 2 | `bcir_kblock` | 0.30 [0.28, 0.31] |
| gemm | 256 x 256 x 256 | 2 | `bcir_fused` | 0.87 [0.83, 0.89] |
| gemm | 256 x 256 x 256 | 2 | `bcir_kblock` | 0.41 [0.40, 0.42] |
| gemm | 129 x 257 x 255 | 2 | `bcir_fused` | 0.88 [0.62, 1.03] |
| gemm | 129 x 257 x 255 | 2 | `bcir_kblock` | 0.29 [0.25, 0.33] |

- **OT-10, the training frontier** (train_comparison.py, one prepared recipe: the 9,836,928-parameter decoder, 8x128 tokens, 300 updates of `lm` from the same weights):

| Route | Threads | Median step (ms) | Tokens a second | Training (s) | Validation CE | Test CE | Peak RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| P-EAGER (torch-eager) | 4 | 533.4 | 1,920 | 161.9 | 2.3643 | 2.3114 | 1,118 |
| P-IND (torch-compiled) | 4 | 390.0 | 2,626 | 125.7 | 2.3730 | 2.3366 | 1,059 |
| B-DIRECT (kblock+exp+reduce+attn) | 4 | 272.0 | 3,764 | 82.4 | 2.3525 | 2.3162 | 489 |

- **OT-11, the cache lifecycle** (the same step, 8x128 tokens, then 8x64; each case in its own process):

| Case | First call (s) | Warm call (s) | Second shape's first call (s) | Frames after it |
|---|---:|---:|---:|---:|
| P-IND cold, static shapes | 43.64 | 0.5144 | 29.24 | 2 |
| P-IND, its cache kept from the cold run | 7.94 | 0.4268 | 1.47 | 2 |
| P-IND cold, dynamic shapes | 80.11 | 0.4208 | 0.22 | 1 |

  B-PLAN: one C library for every shape: a new shape costs its plan alone; the two shapes' plans 5.27, 4.71 ms.

## 5. What moves each gate next

- **BG1 (trace-to-execution)** reads the four `train.plan.*` rows (PX-1, PX-2a, PX-2b, PX-3) at 0
  and OT-03 and OT-04 measured.
- **BG2 (hosted accelerator slice)** needs a GPU with a resident provider and OT-07, OT-08, OT-09
  and OT-12 on it: PX-6, pending, continues in a local session on the hardware.
- **BG3 (direct driver slice)** needs the first half of the driver ladder's M2 -- a D0-D2 UART
  package beside the RuntimeChannel loopback, with telemetry and bounded lifecycle tests --
  attested by its report.
- **BG4-BG6** follow the driver ladder (all of M2 and M4 for BG4, M8 for BG5, M7 for BG6), QEMU
  for the boot proof, a physical device rig for the targeted kernel work, and OT-14, OT-15 and
  OT-16 measured.
- **BG7 (integrated promotion)** needs BG1, BG2, BG4, BG5 and BG6, and OT-10's complete-workload
  benefit.
- **The routes still uncompared** are labelled so in their records: P-TUNED (a budgeted
  autotuning lifecycle), T-GLUON (Gluon's layouts target a GPU), B-MLIR on the block (BCIR's MLIR
  route lowers one matmul buffer nest), and T-DSL inside OT-11 (no Triton with a usable backend
  in the interpreter that runs PyTorch: the Triton-CPU pilot keeps its own).

## 6. The rules this follows (the proposal's section 10)

Correctness, refusal, trace identity and bounded accounting are deterministic gates; a timing is
this host's evidence with an interval, never a universal threshold, and is read beside an A/A
control -- a second trainer running the reference arm in the same rotation -- so an effect counts
only where its interval holds neither 1 nor any of the control's. That rule was earned here:
twelve-round runs without a control read the probe's cost at about 10% twice and at nothing once;
with the control, both timed effects sit inside this host's own noise. An absent capability is
UNAVAILABLE, never a simulated pass, and a GPU, a multi-device host or a device rig is declared
by its operator (`BCIR_OT_CAPABILITIES`), never inferred.
