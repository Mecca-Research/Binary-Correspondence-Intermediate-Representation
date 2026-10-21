# Backend optimality: the OT experiments and the BG gates, executed

> Dated 2026-10-21, non-normative. The executable side of the proposal in draft PR #821
> (`docs/research/BCIR_BACKEND_OPTIMALITY_AND_KERNEL_GATES_2026-10-20.md` on its branch, pinned at
> `c01a53c8`): sixteen optimality experiments, OT-01 to OT-16, and eight entry gates, BG0 to BG7.
> The proposal is #821's and stays there; this page records what of it runs in this tree, on
> which host, with what result, and what each gate still needs. Counts live in the generated
> [`STATUS.md`](../STATUS.md).

## 0. The answer

- **BG0, the measurement contract, is code.** [`tools/perf/ot_contract.py`](../../tools/perf/ot_contract.py)
  holds the proposal's vocabularies -- the comparator routes (its section 3), the kinds of gap a
  result reports (section 2), the host capabilities an experiment needs -- the record every
  experiment writes, held by `validate_record` to one shape, the sixteen experiments as a
  registry (section 5) and the eight gates as requirements over evidence (section 8), stated by
  `evaluate_gates`. BG0 passes when every experiment has a record the contract admits:
  measured, unavailable with the capabilities this host lacks, or blocked with the prerequisite
  the repository lacks. The proposal's BG0 also binds its interface tests (the SI matrix of #821's
  universal-interface roadmap) to the same contract; those tests are not in this tree, so this BG0
  is the OT half.
- **Five of the sixteen run here**, on BCIR's own rails:
  OT-01, OT-03, OT-04, OT-06 and OT-14. The rest are written as unavailable (a comparator, a device or a rig this
  host lacks: never simulated) or blocked (OT-15: no IPC adapter to decompose).
- **The gates on this host**: BG0 PASS, BG1 PASS, BG2 UNAVAILABLE, BG3 OPEN, BG4 UNAVAILABLE, BG5 UNAVAILABLE, BG6 UNAVAILABLE, BG7 UNAVAILABLE.

## 1. What runs, and how a gate reads it

| Piece | What it does |
|---|---|
| `tools/perf/ot_contract.py` | the contract (BG0): routes, gaps, capabilities, the record and `validate_record`, the sixteen, the eight and `evaluate_gates`; `--list`, `--gates RECORDS.json` |
| `tools/perf/ot_run.py` | runs the experiments the host can run, writes all sixteen records under the contract, reads the GEM+ rows a gate names (`--rows`, `gemplus_baseline.py --json`), states the gates |
| `bcir/tests/test_ot_contract.py` | the registry against the proposal, every record rule's refusal, the gates' semantics (nothing PASS without its evidence; UNAVAILABLE never a pass; no gate ahead of the gate it requires; every gate passable with complete evidence), OT-01's checks firing past their bounds, a smoke run |
| `tools/testing/faults/ot-contract.json` | the faults each witness exists to catch |

A gate is **PASS** only when every requirement is met by evidence: an experiment's record
measured, a GEM+ row at its bound (0), a host capability, a driver-ladder milestone attested by a
report the repository holds, another gate's PASS. It is **UNAVAILABLE** when a requirement is a
capability this host lacks or an experiment unavailable here -- never a pass -- and **OPEN**
otherwise, naming what is missing.

## 2. The sixteen on this host

| OT | Experiment | Routes compared | Needs | Here |
|---|---|---|---|---|
| OT-01 | common operation/gradient contract | B-DIRECT, P-EAGER | cpu, numpy | **measured** (B-DIRECT) |
| OT-02 | IR/capture cost and coverage | P-EAGER, P-FX, P-AOT, P-IND, B-PLAN | cpu, torch | unavailable: no torch |
| OT-03 | backward and optimizer identity | B-DIRECT, B-PLAN, P-EAGER | cpu | **measured** (B-DIRECT, B-PLAN) |
| OT-04 | plan intervention | B-DIRECT, B-PLAN | cpu | **measured** (B-DIRECT, B-PLAN) |
| OT-05 | kernel/block portfolio | B-DIRECT, B-MLIR, P-EAGER, P-IND, T-DSL, T-GLUON | cpu, torch | unavailable: no torch |
| OT-06 | layout/packing/arena frontier | B-DIRECT, B-PLAN | cpu | **measured** (B-DIRECT, B-PLAN) |
| OT-07 | mapping and conversion | T-DSL, T-GLUON, B-DEVICE | gpu, triton | unavailable: no gpu, triton |
| OT-08 | pipeline and synchronization | T-DSL, T-GLUON, B-DEVICE | gpu, triton | unavailable: no gpu, triton |
| OT-09 | same image, different dispatcher | B-DEVICE | gpu | unavailable: no gpu |
| OT-10 | complete training frontier | B-PLAN, P-EAGER, P-IND, P-TUNED | cpu, torch | unavailable: no torch |
| OT-11 | specialization/cache lifecycle | P-IND, P-TUNED, T-DSL, B-PLAN | cpu, torch, triton | unavailable: no torch, triton |
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
| BG7 | integrated promotion | **UNAVAILABLE** | gate BG2: BG2 passed; gate BG4: BG4 passed; gate BG5: BG5 passed; gate BG6: BG6 passed; record OT-10: a reproducible complete-workload benefit |

## 4. The measured experiments

The records are in [`measurements/2026-10-21/ot/`](measurements/2026-10-21/ot/README.json):
the run on Intel(R) Xeon(R) Processor @ 2.10GHz (4 vCPUs, one thread), a provisional commit of this change (33d07ce) with nothing uncommitted below bcir, runtime/c or tools/perf, Ubuntu clang version 18.1.3 (1ubuntu1), NumPy 2.2.6, Python 3.11.15; the capabilities it found: cpu, numpy.

- **OT-01** (B-DIRECT against NumPy float64): the first common block -- U = X Wg, V = X Wu, Y = silu(U) * V and its backward (dU, dV, dX, dWg, dWu) -- on 9 shapes (16x256x1024, 16x768x1024, 128x256x1024, 128x768x1024, 512x256x1024, 512x768x1024, 16x257x1025, 128x257x1025, 512x257x1025), every element within its binary32 bound: the worst error is 0.366 of its bound (a GEMM element within gamma_k |A| |B| of the exact product in any order, an elementwise one within 8 units of roundoff of its terms). The coverage ledger: dense row-major GEMMs and their transposes, the SiLU gate and its backward, the zero-size product a no-op; strided operands unsupported (the entries take dense rows).
- **OT-03** (B-DIRECT against B-PLAN): 27 of 27 state cases of the trace corpus train one state and one event sequence, bit for bit, from the direct C loop, from the step program and from its StreamPack and ExecutionPlanV1 lowered by the C.
- **OT-04** (plan intervention): every layer's q/k/v and gate/up projections a call each -- 145 launches become 163, the state the same, bit for bit -- times x1.0222 (95% interval 0.970-1.077) of the default's step over 12 paired rounds, not resolvable: the interval holds 1 (d_model 384, 6 layers, 8x128 tokens, kblock+exp+reduce+attn, 1 thread).
- **OT-06** (arena frontier): the bench step's transient values take 999,424 bytes in the C plan's fixed spans, x1.326 of G26's bound (753,664); placed by their lifetimes (PX-4) 754,176, x1.000, a proved optimum.
- **OT-14** (observer cost): the step on the library with the launch trace probe times x1.0953 (95% interval 1.021-1.175) of the plain library's over 12 paired rounds, the same state, 2,886 launch records written and 0 dropped.

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
- **The P-\* and T-\* routes** need PyTorch and Triton, which this host does not have: OT-02,
  OT-05, OT-10 and OT-11 run where they are installed, under the same contract.

## 6. The rules this follows (the proposal's section 10)

Correctness, refusal, trace identity and bounded accounting are deterministic gates; a timing is
this host's evidence with an interval, never a universal threshold. An absent capability is
UNAVAILABLE, never a simulated pass, and a GPU, a multi-device host or a device rig is declared
by its operator (`BCIR_OT_CAPABILITIES`), never inferred.
