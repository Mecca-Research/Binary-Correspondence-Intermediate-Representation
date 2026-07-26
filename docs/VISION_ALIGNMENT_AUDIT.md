# BCIR vision-alignment audit — 2026-07-17

> A conservative, source-backed comparison between BCIR’s “C as macro assembly,
> IR-owned physical planning, certified learning, and AI-native driver/kernel” thesis
> and the package-version `0.2.0` implementation. Detailed current state is
> [`REPO_CURRENT_STATE_AUDIT.md`](REPO_CURRENT_STATE_AUDIT.md); future execution order is
> [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md). Static counts belong only in
> generated [`STATUS.md`](STATUS.md).

## 1. Thesis under audit

BCIR aims to make:

- C a typed registry/MMIO definition language and transparent macro target;
- the IR owner of semantic claims, layouts, schedules, budgets, provenance, and
  device-command contracts;
- K_BCIR/GEM the deterministic optimizer and execution-artifact path;
- learned models a quarantined source of frozen, certified planning artifacts—not
  legality or in-flight control;
- driver packages reusable across a Linux evidence rail and a future native BCIR kernel;
- inference and selected training kernels deployable through portable C while C++ is
  reserved for orchestration that genuinely needs it.

The audit distinguishes four evidence levels: **landed** (code plus deterministic gate),
**partial** (real bounded slice), **modeled** (executable model without device binding),
and **missing** (no implementation). Compiler fixtures, serializers, and clean skips are
not promoted to driver, transport, or hardware evidence.

## 2. Scorecard

| Pillar | Verdict | Evidence boundary |
|---|---|---|
| C registry/macro-assembly surface | **Landed core; partial language breadth** | MMIO/volatile/atomic/bitfield/ABI/project/link/fallback paths are dual-railed; not full ISO C23 and not a resident driver |
| IR ownership of planning and execution shape | **Landed** | R1–R24, K_BCIR, GEM, StreamPack v1–v3, event/DMA/device contracts; arbitrary-graph LLVM AOT remains absent |
| Certified optimization and learning | **Landed reference; hardware evidence partial** | Exact search and frozen-Q8/replay/provenance controls exist; most target calibration is not yet driver/hardware qualified |
| Math, AD, precision, and library substrate | **Advanced but bounded** | BCIRQ4T/AVX2/SmoothQuant, measured schedule artifacts, expanded closed-set AD, and workload-scoped numerical evidence exist; whole-model and multi-target qualification remains |
| Model inference and training | **Real reference + hosted micro/C gates; not production/bare-metal complete** | Planned/streamed semantics, owned safe-resume pretraining, offline SFT/RM/DPO/PPO/reasoning/embedding stages, and TinyLlama/hosted→BCIRQ8→standalone-C parity exist; 32M/GPU/serving remain open |
| Driver, kernel, ABI, and IPC | **Foundation only** | Direct RuntimeChannel, manifests, event/DMA and ordinary x86 edges exist; no resident driver, Linux module/fork, stable UAPI, native kernel, or native IPC |
| Telemetry/control plane | **Codec/meaning landed; live plane missing** | Registry, BTLM, metrics, deterministic exposition and ring baseline exist; identity envelope, live SPSC and transports/providers are open |

## 3. C as registry definition and macro target

### Landed

- The C-front twins lower register-shaped structs, MMIO, volatile access, atomics,
  fences, inline assembly, port I/O, control/MSR edges, bitfields, pointer extents,
  target ABI contracts, and driver-shaped project headers into claims.
- Clang/GCC behavior differentials, storage-extent comparison, emitted-C compilation,
  project outcomes, fallback, and two-translation-unit linking pin the bounded surface.
- Portable C lowering receives a selected plan; scheduling and legality do not live in
  the emitter.
- The C runtime has explicit freestanding, hosted-tool, and adapter memory classes,
  allocator injection, checked growth, and allocation-failure regressions.

### Still open

- Complete ISO C23 and later C++/Python/Java frontend programs are separate language
  efforts. `_Decimal*` remains blocked by the selected reference-compiler method.
- UART/GPIO sources are compiler fixtures. They do not prove an open/map/submit/event/
  cancel/close driver lifecycle or device ownership.
- The full-model BCIRQ8 loader/inference CLI is hosted (`fopen`/`malloc`); it is not a
  no-libc bare-metal whole-model runtime.
- A formal generalized arbitrary-claim-graph C artifact contract and broader production
  source/debug behavior remain work even though several kernel families lower today.

## 4. IR ownership, machine edges, and backend boundary

### Landed

- Registry IDs, phases/events, lanes, bounds/hazards, timing/lifetime, generation, cost,
  selected plan, StreamPack and lowering contracts are represented and checked.
- K_BCIR provides exact min-plus, RCSP/Pareto, and schedule `(max,+)` rails. GEM owns
  hydration, waves, affinity, event phases, DMA descriptors, and device/channel metadata.
- StreamPack v1 is frozen; v2/v3 evolve append-only and have Python/C malformed-input
  gates.
- BCAB v1 preserves standard backend images inside one bounded content-addressed directory,
  with Python/C/C++ deterministic selection and MLIR metadata projection. This defines binary
  selection compatibility without claiming a new object format or cross-ISA execution.
- Typed x86-64 long-mode handoff, descriptor/segment operations, and an ordinary
  interrupt trampoline lower through LLVM and reach real object/disassembly checks.

### Deliberate or open boundary

- Tropical algebra optimizes *realization cost*; it does not rewrite user arithmetic
  into min-plus and thereby change program meaning.
- General CPU instruction selection/register allocation/object linking remains with
  resident LLVM/Clang/GCC. No seeded target passes the native-object GO gate.
- Artifact signatures, relocation/W^X loader policy, and revocation remain MC14 work; BCAB's
  CRC/SHA integrity and compatibility checks are necessary but do not authenticate code.
- Python LLVM lowering is one supported elementwise claim; MLIR `bcir-aot` is partial
  mixed-dialect preparation. Arbitrary-graph AOT is a separate backend program.
- The x86 edge assumes long mode. Reset transition, NMI/IST/paranoid entry, CR3/PTI,
  SMAP/CET/IBT, speculation and extended-state policy, unwind/CFI, and direct QEMU/
  hardware execution remain open.

## 5. Certified optimization and AI substrate

### Landed

- Cache/bank contention signals, layout pivoting, fusion/deforestation/CSE, tile/loop
  selection, frozen priors, and cost-aware scheduling are deterministic inputs to legal
  exact search.
- Learned ranking/routing/calibration artifacts are quantized, generation-tagged,
  content-addressed, replay-gated, and reversible. They may change search effort or choose
  certified alternatives, never an R-law verdict.
- A1 includes BCIRQ4T signed-Q4 storage, SmoothQuant/outlier calibration, portable C and
  AVX2 packed compute. B1 has content-addressed analytic/measured schedule evidence and
  selected-schedule MLIR. B3 includes six transcendental VJPs, rematerialization, bounded
  loop/call handling, and explicit quarantine. B5 has workload-scoped measured provider
  selection over the existing library families.

### Still open

- whole-decoder Q4, model-level compactness/drift/NLL, ARM/other packed compute, and
  independently specified additional formats;
- bounded exhaustive schedule evidence on at least two real CPU/GPU/device targets and
  reviewed artifact promotion into a target transform;
- representative-model qualification of AD ordering/rematerialization; aliased mutation,
  recursion, unbounded control, and dynamic higher-order calls remain quarantined;
- target/device calibration and Q8-prior promotion from resident-driver telemetry.

The correct mathematical description of the AD/rewrite structure is
monoidal/string-diagram/PROP rewriting. “Gradients as operad 2-cells” is not an accepted
implementation claim.

## 6. Model inference, training, and C++ boundary

### Landed

- Activations/losses/optimizers, planned and streamed execution, optimizer-state claims,
  partial batches, deterministic dataset utilities, transformer/recurrent/classical ML
  references, and finite-difference gates exist.
- Manifest/safetensors ingestion, SentencePiece, Llama-family full/KV decode, GQA,
  batching references, tied/untied heads, checkpoint RMSNorm epsilon, quantized drift and
  NLL are implemented.
- BCIRQ8 v1 has deterministic Python read/write and a strict portable C loader. The
  pinned TinyLlama gate verifies source hashes, tokenizer IDs, compact export, Python/C
  generated IDs, finite logits, and final-logit tolerance without committing assets.
- The import-quarantined hosted lab independently implements Llama/GQA in PyTorch, accepts
  only explicit AdamW/device/precision modes, checkpoints without pickle, and proves exact
  same-host resume plus random-weights→Safetensors→BCIRQ8→standalone-C composition.
- The sibling hosted-training package adds deterministic corpus/BPE preparation, typed
  SFT/preference/PPO/reasoning records, RM/DPO/PPO/reasoning/embedding objectives, three
  bounded non-LLM model families, and an append-only pipeline ledger. Recorded teacher and
  offline remote-compute adapters prove the provider-neutral boundary without a live API.
- The C++ handoff has a small compiled single-node seam and explicit ownership rules.

### Still open

- The pinned 16K tokenizer and canonical BCIR-TinyStories-32M run, live provider adapters,
  distributed training,
  production tokenizer/runtime integration, sampling, safety policy, broad architectures,
  long-context/device kernels, robust serving/evaluation, and physical accelerator qualification.
- A freestanding whole-decoder profile with caller-owned memory if bare-metal deployment
  is required; the present standalone C decoder is hosted.
- Real dynamic-graph and distributed MPI/NCCL orchestration. Current C++ backends beyond
  the bounded seam are honest stubs and need suitable multi-node/device evidence.

## 7. Driver, kernel, telemetry, and IPC alignment

### Landed foundation

- Device manifests and profiles, bank typing, distance-priced moves, event phases, DMA
  descriptors, StreamPack v3 metadata, direct RuntimeChannel v1 hooks, and bounded
  loopback behavior.
- Signal IDs/units/provenance in Python, strict Python/C BTLM frame parity, continuity
  evidence, derived metrics/sensitivity, deterministic Prometheus text/OTLP JSON/Redfish
  shapes, and a quiescent shared-ring baseline.
- A proof-carrying driver-package maturity model and separate BCIR-Linux/native-kernel
  dependency tracks are documented.

### Missing implementation

- Generated fixed-width C signal table and ID-range policy.
- Source/session/generation/clock/loss telemetry envelope; live publish/backpressure/
  peer-death SPSC ring; UART, HTTP, OTLP, Prometheus host, Redfish/BMC, GPU, and other
  live providers/transports.
- Resident UART, virtio, storage, network, USB, accelerator, or physical-device driver.
- Linux bridge/module and actual `Mecca-Research/BCIR-Linux` patch queues.
- Stable userspace ABI, out-of-process Linux adapter, native kernel, and capability IPC.

The dependency is evidence-driven: telemetry identity → direct UART lifecycle → Linux
adapter parity → virtio queue/DMA proof → UAPI v1 → native IPC after direct/Linux/native
behavior agrees. Linux compatibility is additive; BCIR does not promise a stable Linux
kernel-internal ABI or replace POSIX/Linux syscalls.

## 8. Highest-leverage remaining work

1. Complete the pre-driver telemetry identity and live bounded SPSC contracts.
2. Build UART as the first resident in-process proof-carrying driver with simulator,
   faults, cancellation, teardown, telemetry, and replay evidence.
3. Add Linux-hosted parity, then use virtio-blk to prove queue/DMA/reset/saturation.
4. Gather bounded real-hardware calibration and performance evidence; promote immutable
   priors only through certificates and quiescent generation swaps.
5. Advance the bounded AI-substrate gaps—low-bit formats, portable schedule export, and
   AD breadth—without delaying the driver dependency chain.
6. Keep arbitrary-graph AOT, native isel, BCIR-Linux, native kernel, and native IPC behind
   their explicit evidence/dependency gates.

## 9. Bottom line

The central compiler thesis is credible and code-backed: C is a useful registry/macro
surface, the IR owns legality/planning/schedule, K_BCIR/GEM are executable, learned
artifacts are quarantined, and a real small model crosses a deterministic Python-to-C Q8
gate.

The system is not yet an AI-native driver/kernel platform. Its most important missing
evidence is a resident device lifecycle and the telemetry identity/backpressure semantics
needed to train and certify device-specific optimizers. Calling compiler fixtures,
modeled channels, serialization outputs, or cross-compiled objects “drivers,” “live
telemetry,” or “hardware support” would overstate the repository. The roadmap now makes
those boundaries and promotion gates explicit.
