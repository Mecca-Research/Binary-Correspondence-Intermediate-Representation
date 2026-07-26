# BCIR onboarding deep dive

> Orientation for the package-version `0.2.0` tree. This guide is descriptive, not
> normative. [`BCIR_LANGREF.md`](BCIR_LANGREF.md) owns language/wire law,
> [`STATUS.md`](STATUS.md) owns generated static counts, and
> [`REPO_CURRENT_STATE_AUDIT.md`](REPO_CURRENT_STATE_AUDIT.md) owns the dated capability
> snapshot.

## 1. Read this first

BCIR is a registry-first, phase-ordered, lane-typed, cost-governed correspondence IR.
It preserves the relationship between semantic intent, legal realization, physical
schedule, executable artifact, and telemetry evidence.

The central planning problem is:

```text
K_BCIR(G | H, Θ) = min_{π ∈ Legal(G,H)} M(π, Θ)
                   subject to R(π, Θ) ⪯ B(H, Θ)
```

- `G` is the semantic claim graph.
- `H` is the target/device profile.
- `Θ` is live but generation-bound machine state.
- `π` is a legal realization path.
- `M` is schedule-aware makespan; serial work adds and parallel waves use `max`.
- `R` is the additive 12-axis integer/Q8 resource ledger.
- `B` contains hard budgets such as thermal or power limits.

BCIR owns planning, verification, device-command schemas, transport-neutral artifacts,
telemetry meaning, and certified adaptation. LLVM/Clang/GCC and vendor toolchains retain
general instruction selection, register allocation, object production, and vendor
kernel internals unless a measured decision gate says otherwise.

## 2. The three implementation rails

| Rail | Role | Key property |
|---|---|---|
| `bcir/` | Executable Python conformance oracle plus opt-in hosted adapters | Default/core behavior is dependency-free; hosted model execution is import-quarantined |
| `mlir/` | ODS/TableGen/C++ law rail | R1–R24, native optimizer/GEM passes, IRDL projection, and partial lowering |
| `runtime/c/` | Production C compiler/runtime rail | Freestanding execution plus hosted compiler/model tools and direct RuntimeChannel hooks |

`runtime/cpp/` is a narrow orchestration seam above the C ABI. It does not redefine
legality, ownership, or planning.

`bcir/hosted/models/` is a fourth execution *adapter*, not a new semantic rail. It uses a
pinned PyTorch/Safetensors environment for scalable tensor execution, but every export must
re-enter through the dependency-free strict-ingest oracle before BCIRQ8 or C deployment.

Agreement is scoped to the rails that implement a concept:

- oracle ↔ MLIR for semantic types, laws, costs, plans, and GEM contracts;
- oracle ↔ C for codecs, compiler claims/emission, model artifacts, and runtime behavior;
- direct ↔ future Linux/native adapters for driver lifecycle behavior.

Parity is demonstrated with pinned anchors, independent recomputation, generated
adversarial modules, malformed-artifact corpora, and compiler behavior differentials.
It is not inferred from similarly named code.

## 3. From source to execution

```text
source / ROP / MAP / binary record / model
                     │
                     ▼
BCIR-0..2: semantic claims + shaped resources + placement candidates
                     │  verify legality (R1–R24 where applicable)
                     ▼
BCIR-3: K_BCIR legal realization + budgeted plan
                     │
                     ▼
BCIR-4: GEM schedule + StreamPack + event/DMA/channel contracts
                     │
                     ▼
BCIR-5: portable C / partial LLVM / WASM / JVM / CIL / SYCL / device commands
                     │
                     ▼
resident toolchain or RuntimeChannel adapter
                     │
                     ▼
versioned telemetry → offline calibration/replay → certified immutable artifact
```

Learning can propose search order, select among certified policies, or produce frozen
Q8 calibration artifacts. It never decides legality and does not run in the execution
hot path. Promotion occurs only after replay/certificate checks at a quiescent generation
boundary with rollback available.

## 4. Core semantic and optimizer packages

### `bcir/model/` and `bcir/verify/`

- A `Resource` has a registry ID, domain, shape/layout, access mode, and generation.
- A `Claim` names an operation, read/write resources, lane/stride/hazard/bounds
  contracts, and optional timing/lifetime metadata.
- A `Phase` is a dependency node; event-triggered phases make asynchronous entry
  explicit.
- R1–R24 cover registry, resolution, domains, DAGs, hazards, lanes, bounds, costs,
  plans, stream provenance/generation, lowering/provenance, smart lowering, accuracy,
  call graphs, timing/CDC/lifetime, and GEM shape/dtype seams.

The MLIR law rail carries the full current numbered set. The C front is explicitly
scoped to its implemented subset; unsupported C routes through the documented fallback
contract.

### `bcir/kbcir/`

The optimizer owns:

- fixed-order 12-dimensional `CostVector` arithmetic;
- exact tropical shortest-path selection;
- resource-constrained shortest path and Pareto fronts;
- `(max,+)` overlap/schedule pricing;
- deterministic calibration, provenance, replay, and regret ledgers;
- certified learned helpers whose outputs freeze to integer artifacts;
- reference ML/training, quantization, autodiff, and model components.

Legality is decided before cost. A learned ranker may reduce work but cannot change the
optimum or turn an illegal path into a candidate.

### `bcir/gem/`

GEM hydrates a selected plan into executable scheduling records: lane segments,
prefetch/block/trace data, phases/waves, event phases, DMA descriptors, channel/device
bindings, and StreamPack. Duration-aware scheduling, affinity, bandwidth knees, and
generation checks remain deterministic.

StreamPack v1 is frozen. v2/v3 add records append-only while preserving old walkers.
The exact bytes are owned by
[`BCIR_STREAMPACK_ABI.md`](kernel/BCIR_STREAMPACK_ABI.md), Python codec code, and the C
header/decoder together.

## 5. Frontends, lowering, and machine boundary

### C frontend

The Python oracle under `bcir/frontends/cfront/` and the C twin under `runtime/c/`
implement a broad driver-oriented C23 subset. The gate compares claim structure,
storage extents, bounds guards, compiler behavior, diagnostics, target ABI behavior,
project outcomes, and emitted artifacts.

It is not complete ISO C23. Unsupported constructs are rejected or routed to the
resident compiler under `--fallback`; `_Decimal*` remains blocked by the chosen
differential reference. Start with
[`CFRONT_GUIDE.md`](languages/CFRONT_GUIDE.md).

### Lowering and objects

- Portable C23 is the broad resident-toolchain path.
- Python LLVM AOT/JIT supports exactly one 2-read/1-write add/sub/mul elementwise
  claim and rejects additional executable claims.
- MLIR `bcir-aot` is partial preparation and may leave BCIR/GEM operations in mixed IR.
- JVM/CIL/WASM and SYCL/SPIR-V paths have explicitly bounded validation surfaces.
- Real ELF objects for documented scalar slices are produced through resident
  Clang/LLVM and checked for expected machine type.
- BCAB v1 packages preserved standard artifacts with target/feature/provenance metadata;
  Python and allocation-free C validate and select the same variant, while MLIR records
  the directory, selection, and additive ASN.1 projection. Read
  [`BCIR_ARTIFACT_BUNDLE_ABI.md`](kernel/BCIR_ARTIFACT_BUNDLE_ABI.md) before changing it.

BCIR does not implement a general CPU instruction selector, register allocator, or
linker. The decision can change only through
[`BCIR_NATIVE_OBJECT_GATE.md`](BCIR_NATIVE_OBJECT_GATE.md).

### Assembly and driver edge

Typed MMIO, port-I/O, fences, control-register/MSR access, an x86-64 long-mode C handoff,
descriptor/segment operations, and an ordinary interrupt trampoline are represented and
object/disassembly tested. They are not a reset vector or complete exception subsystem;
NMI/IST/paranoid entry, feature policy, and direct hardware execution remain explicit
work.

## 6. Runtime memory and ownership

[`C_MEMORY_DISCIPLINE.md`](languages/C_MEMORY_DISCIPLINE.md) divides C code into:

1. a heap-free freestanding core;
2. hosted compiler/model tools with allocator injection and checked growth;
3. driver adapters using opaque handles and byte offsets across boundaries.

Outputs initialize to a safe state, destruction is idempotent where promised, growth is
two-phase, and fault-injection tests fail each hosted allocation point. IPC is not a
memory-safety shortcut and must not enter the freestanding core.

## 7. Models, training, and BCIRQ8

The model/reference stack includes manifest and safetensors ingestion, SentencePiece,
Llama-family decoding, GQA and KV cache, training specifications/optimizers, continuous
batching references, and quantized evaluation. The optional hosted lab adds an independent
PyTorch Llama reference, AdamW training, safe exact-resume generations, and strict HF-style
export without changing default `bcir` imports.

BCIRQ8 v1 is a deterministic groupwise signed-int8 decoder-artifact format with
power-of-two exponents, canonical tensor order, CRCs, SHA-256 provenance, strict bounds,
and a portable C loader. The normative contract is
[`BCIR_LANGREF.md`](BCIR_LANGREF.md#16-bcirq8-v1-decoder-artifact-contract) §16.

The pinned real-model gate performs tokenizer/checkpoint verification, Q8 export, Python
inference, standalone C inference, generated-ID parity, and deterministic report
generation. Checkpoints, tokenizers, generated weights, binaries, and logits are local
cache/build products, never repository assets. A second micro gate trains a 90,688-element
model from random weights for 64 CPU steps and proves Safetensors→BCIRQ8→standalone-C parity;
it is a composition test, not a useful language model.

Stable repeated AI work has a cold opt-in C data plane: Q8/Q4 conversion, Q8 decoder/head
projections, exact hard-filtered Q15 retrieval, group-32 Q4×Q8 accumulation, and bounded native
measurement. The Python implementation remains the independent semantic oracle and owns laws,
schemas, planning, provenance, and hosted orchestration. Read
[`BCIR_PYTHON_NATIVE_BOUNDARY_AUDIT.md`](machine-learning/BCIR_PYTHON_NATIVE_BOUNDARY_AUDIT.md)
before proposing another port; file size or loop count alone is not a placement argument.

A third optional micro gate trains a 32-wide GNN/Transformer over availability-aware telemetry,
memory-bank topology, and the finite payload-free placement portfolio. Exact K_BCIR outcomes
create reward/DPO/PPO targets; bounded PUCT proposes a winner; deterministic claims, StreamPack,
bank moves, and aligned static addresses dispose. Its six episodes are simulated, so the live
promotion gate refuses them. Real counter corpora, rematerialization/spill execution, and
two-target qualification remain open. Production-scale training, serving, and broader
model/low-bit support remain open; see
[`BCIR_ML_AI_INTEGRATION_ROADMAP.md`](machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md).

The metadata-only HAM rail generalizes model movement without loading weights. `HAMResource` and
`HAMAccess` describe a semantic dependency/access trace; `bcir-ham-plan` routes it only over
declared directed hardware links, replays residency/capacity/generations independently, and lowers
the result through existing claims and StreamPack. Context-shard manifests reference existing
BCIRQ8/Q8-table/Safetensors/StreamPack payloads and activate only at quiescent generation
boundaries with rollback. The exact Q15 optimization-memory reference applies hard facts before
similarity results are visible. Read
[`BCIR_HAM_MEMORY_FABRIC.md`](kernel/BCIR_HAM_MEMORY_FABRIC.md) before changing this boundary.

## 8. Drivers, kernel, telemetry, and IPC

The present foundation includes device manifests, bank/move constraints, event phases,
DMA descriptors, StreamPack, a signal registry, telemetry codec/metrics/serializers,
direct RuntimeChannel hooks, and loopback behavior.

It does **not** yet include a resident UART/virtio/device driver, Linux module, stable
UAPI, physical GDS/P2PDMA/CXL/NVMe memory-fabric adapter, BCIR-Linux fork, native kernel,
or native IPC. UART/GPIO sources currently prove
compiler shapes; they are not deployed drivers.

The dependency order is:

1. finish telemetry identity and the live bounded SPSC contract;
2. prove UART in-process against a simulator and direct RuntimeChannel;
3. add Linux-hosted adapter parity, then virtio-console/virtio-blk;
4. freeze UAPI only after MMIO/event and queue/DMA classes agree;
5. implement native IPC only after direct/Linux/native traces justify its shape.

Read [`BCIR_DRIVER_KERNEL_ROADMAP.md`](kernel/BCIR_DRIVER_KERNEL_ROADMAP.md),
[`TELEMETRY_PIPELINE_RESEARCH.md`](kernel/TELEMETRY_PIPELINE_RESEARCH.md), and
[`SIGNAL_REGISTRY.md`](kernel/SIGNAL_REGISTRY.md) together.

## 9. Current evidence boundary

The source-backed snapshot is [`REPO_CURRENT_STATE_AUDIT.md`](REPO_CURRENT_STATE_AUDIT.md).
Important limits for new contributors:

- local x86-64 execution is not ARM, board, GPU-driver, or kernel evidence;
- modeled FPGA/NVMe/PIM channels are not resident backends;
- serialization shapes are not live OTLP/Prometheus/Redfish transports;
- `bcir-aot` is not arbitrary-graph AOT;
- compiler fixtures are not drivers;
- a clean tool/hardware skip is not a passing execution measurement.

Generated counts in [`STATUS.md`](STATUS.md) are static inventories, not proof that the
listed tests were run in the current checkout.

## 10. Validation workflow

Use bounded concurrency on local hardware:

```bash
python -m bcir.tests.run_all --tier quick -j 2
python -m bcir.tests.run_all --tier thorough -j 2
python tools/perf/run_tmsao_audit.py --repeats 3
bash tools/c/check_runtime.sh
bash tools/cpp/check_handoff.sh
bash tools/wsl/check_passes.sh       # when a coherent LLVM/MLIR toolset exists
bash tools/irdl/check_corpus.sh
python tools/docs/gen_status.py --check
python tools/docs/check_links.py
git diff --check
```

The TMSAO report is strict on deterministic results and structural invariants but records
wall-clock latency as host-relative evidence. Read
[`PERFORMANCE_AUDIT.md`](PERFORMANCE_AUDIT.md) before interpreting those numbers as a
target performance claim.

Quick mode intentionally hides compiler/toolchain capabilities and expects explicit
skip results. Thorough mode restores the real host toolset. CI owns Windows, Ubuntu,
native ARM, longer fuzz/analyzer campaigns, and future kernel matrices. Do not start an
unbounded emulation, fuzzing, or nested multiprocessing loop on the local workstation.

Before publication, map the change to every required GitHub check and wait for all of
them to pass. The test registry prevents adding an uncollected `test_*.py` file.

## 11. Reading and change-placement map

| Need | Canonical entry |
|---|---|
| Language, laws, BCIRQ8 | [`BCIR_LANGREF.md`](BCIR_LANGREF.md) |
| Multi-backend artifact and binary selection | [`kernel/BCIR_ARTIFACT_BUNDLE_ABI.md`](kernel/BCIR_ARTIFACT_BUNDLE_ABI.md) |
| Current implementation truth | [`REPO_CURRENT_STATE_AUDIT.md`](REPO_CURRENT_STATE_AUDIT.md) |
| Static inventory | [`STATUS.md`](STATUS.md) |
| Oracle/MLIR/C correspondence | [`PARITY.md`](PARITY.md) |
| Portfolio execution order | [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) |
| Driver/kernel sequence | [`kernel/BCIR_DRIVER_KERNEL_ROADMAP.md`](kernel/BCIR_DRIVER_KERNEL_ROADMAP.md) |
| ML/model sequence | [`machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md`](machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md) |
| Python/native AI placement | [`machine-learning/BCIR_PYTHON_NATIVE_BOUNDARY_AUDIT.md`](machine-learning/BCIR_PYTHON_NATIVE_BOUNDARY_AUDIT.md) |
| C usage and ownership | [`languages/CFRONT_GUIDE.md`](languages/CFRONT_GUIDE.md), [`languages/C_MEMORY_DISCIPLINE.md`](languages/C_MEMORY_DISCIPLINE.md) |
| Machine/backend gaps | [`BCIR_MACHINE_CODE_HAL_ISA_AUDIT.md`](BCIR_MACHINE_CODE_HAL_ISA_AUDIT.md) |
| Merged chronology | [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md) |
| Repository ownership | [`BCIR_Repo_Structure.md`](BCIR_Repo_Structure.md) |
| Performance/TMSAO evidence | [`PERFORMANCE_AUDIT.md`](PERFORMANCE_AUDIT.md) |

Place semantic changes in the oracle first, then the applicable law/production twin and
a differential regression. Place stable byte changes in the ABI document and both
implementations together. Place landing notes in development history—not in the master
roadmap.
