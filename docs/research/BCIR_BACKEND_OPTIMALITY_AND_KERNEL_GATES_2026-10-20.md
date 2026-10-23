# BCIR backend optimality tests and driver/kernel entry gates

**2026-10-20 UTC. Proposed test specification and integration blueprint; no new benchmark results.**

Develop the executed IR/native-training path and a bounded hardware/driver path **in parallel**.
Do not wait to beat every framework before beginning device bring-up. Do not make a custom GPU
driver or independent OS a prerequisite for qualifying GPU kernels through existing runtimes.
Promote each path on evidence appropriate to its scope.

The architectural target is a shared semantic/resource contract linking graph decisions,
kernel implementations, device work submission, memory ownership and measured feedback. The
independent bootable OS is a separate deployment of that contract, with its own correctness,
device-support and performance gates. Its success criterion is useful complete-workload
behavior on named hardware, rather than boot independence alone.

This specification extends the [three-way architecture/IR analysis](BCIR_TRITON_PYTORCH_ARCHITECTURE_2026-10-19.md)
and [Triton CPU pilot](BCIR_TRITON_COMPARATIVE_ANALYSIS.md). It supplies experiments for the
existing [training acceleration](BCIR_TRAINING_ACCELERATION_ROADMAP.md) and
[canonical driver/kernel](../kernel/BCIR_DRIVER_KERNEL_ROADMAP.md) ladders; it does not replace
those ladders or create new runtime functionality.

The [universal serialization/interface roadmap](../BCIR_UNIVERSAL_INTERFACE_ROADMAP.md)
adds the UDSI program and twelve proposed SI experiments for typed service descriptions,
schema mappings, foreign formats, ABNF/CSN.1, buffer views and cryptographic-provider syntax.
Its compatibility and ownership contracts extend OT-12/15/16 and BG0/BG3–BG6 without
changing the sixteen OT definitions or freezing a new IPC/UAPI.

The [driver/IPC data and AI qualification report](BCIR_DRIVER_IPC_AI_QUALIFICATION_2026-10-20.md)
adds applicable hardware-specific data, inference and training profiles to these gates.
Workload evidence guides IPC and precedes physical HAM/native HAM/SYCL integration. It
retains the existing HAM planner and hosted SYCL oracle as earlier software foundations.

## 1. Pinned baseline and the pending PX ladder

| Surface | Evidence used here | Status |
|---|---|---|
| BCIR main | `12aa6dc5ba422418742f49e4bfc00aafdc17e07f` | Baseline implementation; the earlier CPU pilot directly called its C kernels |
| PyTorch | 2.14.1, `5c4886908584029761b579af026dcfb627c84070` | Source-inspected FX/ATen, AOTAutograd, Inductor and native/compiler interfaces |
| Triton | 3.8.0, `c01b6774b1865984607d89d89d3a10833de92037` | Source-inspected GPU compiler; no NVIDIA/AMD GPU execution on this host |
| Triton-CPU | Experimental snapshot `5a4885bf9b0b349a525b490a1df25f449f12d4e9` | Earlier limited CPU overlap experiment, not a production GPU comparison |
| Pending BCIR #820 | `b6c50f11d5ed2d18bd7b7cae026547b93db235fe`; open, draft, unmerged at refresh | GEO-3 memory/recompute work, PX-1 launch witness, PX-2a checked launch-program execution and PX-2b StreamPack/ExecutionPlanV1 execution; source-inspected separately, not merged or benchmarked here |

Full comparative model experiments remain blocked on **#820 completing and merging**. Freeze
the actual merged SHA before collecting new training results. PyTorch's own Triton dependency
pin must be respected: the independently inspected upstream Triton release is not automatically
a compatible replacement for its bundled/pinned compiler.

### The trace gap has progressed

Pending #820 progressed from `native_launch.run_launches` and its launch witness (PX-1),
through whole-program checking/execution (PX-2a), to lowering a StreamPack and ExecutionPlanV1
through GEM into that executable program (PX-2b). The author reports 221 traced configurations
at PX-1 and 356 matching pack/program/state runs at PX-2b [B2]. These are the PR author's
results, not measurements reproduced here. Source inspection confirms the pack/lifetime and
C lowering contracts. PX-3–PX-5 remain open at the inspected head; the remaining ladder is:

| Existing follow-up | Purpose | Comparison evidence to add |
|---|---|---|
| PX-1 | Describe and witness the current native launches | Preserve exact call coverage and counted probe overflow; distinguish a witnessed direct loop from a plan-driven executor |
| PX-2a / PX-2b | Implemented on pending #820: checked launch-program execution, then StreamPack/ExecutionPlanV1 lowering and execution, one claim per kernel call | Review after merge; preserve kernel entries, operand spans, state and refusal behavior; compare lowering/dispatch cost with the direct loop |
| PX-3 | Carry per-call choices in the plan and measure a legal intervention | Change the admitted plan, observe the changed calls/resources, and measure the predicted effect |
| PX-4 | Place transient storage by lifetime | Compare actual overlapping lifetimes and peak owned bytes under the same step and recomputation contract |
| PX-5 | Add GF(2) layouts with packer parity | Check bit-layout/conversion semantics and executable pack reuse, rather than descriptor expressivity alone |
| PX-6 | GPU lowering and synchronization | Execute on a real supported target; generated PTX or a host fallback is not the exit gate |

Use these identifiers for implementation work; the OT identifiers below name experiments.
Do not reset this ladder to "no tracing exists", rebuild the pending PX-2 bridge, or declare
PX-3 complete because a pack executes. Measured per-call choice and lifetime/layout optimization
have their own gates. Optional preflight launches and trace capacity remain part of the witness.

## 2. Distinguish the kinds of optimality gap

The comparison must produce several results, rather than one "TMSAO completion" percentage.

| Gap | Definition / evidence | Interpretation |
|---|---|---|
| Capability | Required operation, gradient, shape, layout, effect or device path is unsupported | Report missing/refused explicitly; there is no meaningful speed ratio until an implementation exists |
| Realization | Better legal kernel, partition, packing, layout or schedule is known but absent from BCIR's portfolio | Add the implementation or an optional provider and test it under the declared contract |
| Selection | A better feasible candidate exists in the evaluated portfolio but the planner chooses another | Compare chosen performance against independently validated candidates under an equal search budget |
| Integration | Selected graph/plan differs from executed calls, storage or synchronization | Trace and intervene; output parity alone can miss unnecessary work |
| Implementation versus competitor | `T_BCIR / T_reference - 1` on matched hardware/work/semantics | A measured competitive gap; the reference is not thereby the physical optimum |
| Bound slack | `(T_BCIR - L) / L` for a justified positive lower bound `L` | Possible headroom relative to that bound; tightness/attainability must be stated |
| Compilation/tuning | Cold compile, specialization, cache and search costs | Include amortization and deployment lifetime, not just warm execution |
| System | Submission, mapping, copies, queueing, isolation and event/service overhead | Measure with the same computation/artifact where possible; separate kernel math from runtime/OS effects |

For a declared finite candidate set, model cost `C` and exact model optimum `C*`, report
`C(chosen)-C*` in the model's units; normalize only with a stated valid denominator. This
certificate does not establish the fastest hardware implementation. For measured selection
regret, select candidates on tuning data and validate them on independent rounds/held-out
shapes; choosing the lowest noisy sample overstates the achievable frontier.

Use compute, bandwidth, dependency-span and communication lower bounds where justified for
the declared algorithm and hardware. Report the assumptions. An observed best implementation,
peak datasheet FLOP rate, or an analytical model prediction is not by itself a tight lower
bound. Track the time/memory/quality/compile-cost Pareto frontier; an optimization may be useful
because it fits a required memory budget even when its step takes longer.

## 3. Comparator routes and what each isolates

Three meanings of "kernel" must remain separate: a CPU math microkernel, a GPU compute kernel,
and an OS microkernel. Tests at one level cannot establish superiority at another.

| Route | Execution path | Question |
|---|---|---|
| P-EAGER | Current PyTorch eager tensor/module/autograd path | Cost and coverage of ordinary eager execution, including native/library dispatch |
| P-FX | Capture through Dynamo with a non-optimizing/eager backend | Capture, guards and graph-break overhead; not Inductor performance |
| P-AOT | AOTAutograd forward/backward capture with an eager execution backend | Functionalization/decomposition/partitioning effects and cost, separate from kernel code generation |
| P-IND | `torch.compile` with Inductor | Current compiled model/block baseline; record backend, graph breaks, fallbacks and actual generated/library calls |
| P-TUNED | An explicitly budgeted Inductor tuning configuration | Best admitted compiled result under the same tuning resources; not an unlimited-search comparator |
| B-DIRECT | BCIR's existing native C tensor/training entry points | Native library quality and host entry overhead; no attribution to an unexecuted optimizer |
| B-PLAN | Pending PX-2a/PX-2b program/pack execution; PX-3 measured per-call selection remains open | Plan/dispatch and graph-level optimization effects on the same kernels, after merge and qualification |
| B-MLIR | BCIR-generated MLIR lowered to an executable, on its supported subset | Actual compiled law/lowering quality; compare separately from hand-written C kernels |
| T-DSL / T-GLUON | Triton tiled DSL / explicit-layout Gluon, same target and math contract | Compiler-generated kernel quality, layout/control trade-offs and specialization cost |
| B-DEVICE | BCIR dispatch of the **same device image** used by a direct resident provider | Orchestration, ownership, residency and synchronization costs without changing machine code |
| S-DEVICE | A qualified SYCL device path, when available | Another backend implementation under the same operation and device contract; mark portable host fallback separately |
| OS-DIRECT / OS-LINUX / OS-NATIVE | Equivalent RuntimeChannel operations through direct, Linux-hosted and native adapters | Transport/service/protection cost and complete-workload behavior |

These routes are an experimental design, not a list of implementations already available.
B-PLAN requires the pending branch to merge and qualify; B-DEVICE and native hardware routes
require their implementation gates first.
Triton needs a documented host/autodiff/optimizer harness for a full training experiment.
P-FX/P-AOT are diagnostic compiler routes; they do not replace P-EAGER/P-IND as the principal
user-facing comparison [P1, P2].

For a common supported functional ATen subset, supply identical forward/backward graphs and
the same legal kernel portfolio to BCIR and a comparison planner. Separately compare each
stack's own preferred portfolio/code generation. The first experiment isolates decisions;
the second tests total delivered capability. Capture/import must start from equivalent work
when measuring compiler overhead. Time graph construction, verification, planning, lowering,
code generation, loading and execution separately.

## 4. Gap register beyond integer-affine addressing

Rows identify opportunities in the inspected implementations. The list is a bounded engineering
inventory, not a claim that all hardware/compiler gaps have been discovered. "Pending" always
means #820 is not in main. OT tests are specified in §5.

| Area | BCIR boundary / opportunity | Comparator mechanism to study or adopt | Tests |
|---|---|---|---|
| Graph capture and semantic coverage | Claim/resource frontends exist; no complete functional ATen importer or broad PyTorch program capture | FX/operator schemas, supported decompositions and explicit custom-backend import | OT-01, OT-02 |
| Autodiff and training graph | Native decoder has forward/backward; that does not establish arbitrary graph differentiation | AOTAutograd joint graphs, effect-aware partitioning, generated backward coverage | OT-01, OT-03 |
| Executed plan and task grouping | Main's direct C loop is separate; pending PX-1 witnesses it and PX-2a/PX-2b execute program/pack choices | Review pending executor; PX-3 measured per-call selection; retain direct C reference | OT-02, OT-04 |
| CPU register/cache kernels | Native kernels are substantial; MLIR matmul-buffer route is a scalar tiled nest | Inductor C++ templates, admitted BLAS/library implementations, vector/register blocking and pack reuse | OT-05, OT-06 |
| Layout algebra and conversions | Integer-affine and resource layout models do not provide GPU lane/warp/register swizzles or conversion costs | Triton GF(2) layouts, explicit Gluon layouts, preimages for replicated layouts | OT-06, OT-07 |
| GPU instruction/tensor lowering | PTX text checks and SYCL seams do not qualify tensor-core training | TritonGPU/target dialects, LLVM/vendor assembly, tensor operation selection and real device images | OT-07, OT-08 |
| Asynchronous GPU pipelines | Event/DMA models exist; no qualified resident GPU training pipeline here | Shared-memory staging, pipelining, warp specialization, CUDA/HIP streams and SYCL dependencies | OT-08, OT-09 |
| Fusion and materialization | Cost/fusion concepts must be realized as fewer/more efficient executed operations | Inductor fusion/extern decisions; Triton fused kernels with resource feedback | OT-04, OT-05, OT-08 |
| Memory/recompute/packing | Main arena scheme; pending GEO-3 adds exact workspace tiers and per-layer remat; not global liveness optimum | Joint-graph rematerialization, pooled buffers, pack lifetime and budgeted selection | OT-04, OT-06, OT-10 |
| Attention algorithm | Native attention storage remains quadratic; BF16 contracts do not remove that asymptotic storage | Qualified blocked/streaming attention with supported backward and precise masking/numerical policy | OT-05, OT-10 |
| Dynamic shapes and compilation | Static upper-bound costing differs from symbolic executable shapes/specialization | Guarded symbolic shapes, bounded specialization/cache policies, AOT deployment for supported inference | OT-02, OT-11 |
| Precision and numerical policy | Explicit contracts and references exist; backend/shape coverage and fast paths remain target-specific | Storage/accumulation rules, hardware low-precision kernels, quality-constrained provider admission | OT-03, OT-05, OT-10 |
| Optimizer and persistent state | Native update path exists; broader optimizer graphs and state/residency choices need measurement | Fused updates, master-weight/state layouts, gradient accumulation and checkpoint interfaces | OT-03, OT-10, OT-13 |
| Device residency and movement | RuntimeChannel value ABI and HAM planning are foundations; no physical GPU binding implied | Resident allocations/images, explicit events, supported vendor loaders, transfer/packing overlap | OT-09, OT-12 |
| Telemetry and ranking | Signal registry/envelopes/calibration exist; absent device counters remain unavailable | Work-correlated timing/counters, uncertainty-aware cost calibration, bounded autotune pruning | OT-12, OT-14 |
| Task manager / TMOG integration | Relevant telemetry and control pieces exist; a unified device/task monitor is a proposed integration | Correlated task/resource view over the same execution trace and generation, not a second planner | OT-04, OT-14 |
| Distributed training and data pipeline | Shard/HAM/control contracts are not device collectives or distributed training qualification | Optional established collectives, topology-aware overlap, checkpoint/restart and measured data staging | OT-12, OT-13 |
| Universal interfaces and serialization | Existing ASN.1/JER plans and direct value ABI; broader service-IDL/format/grammar adapters remain proposed | Directional schema mappings, independent codecs, native bindings, Arrow C Data/IPC and explicitly negotiated profiles | OT-12, OT-15/16; SI-01–SI-12 |
| Native IPC / device service | Direct loopback exists; resident hardware bindings, native IPC and independent OS remain unbuilt at the inspected pin | Canonical D0–D7/M0–M10 sequence; seL4-style IPC/IRQ/mapping measurement dimensions | OT-15, OT-16 |

Do not copy whole frameworks to obtain these mechanisms. Start with an optional provider,
importer, compiler/backend interface or a small licensed algorithm implementation, keeping
provenance and its semantic/parity tests. Promote a native reimplementation when profiling
shows the external interface blocks a useful transformation or required deployment property.
Retain the external implementation as a differential reference where practical.

## 5. Test catalogue

All tests below are **planned**, except the separately cited historical CPU pilot and pending
PX-1/PX-2 evidence. Implement runners as subsequent slices; this document introduces no executable
test command or fabricated result.

Build on the existing [CPU overlap harness](../../tools/perf/triton_cpu_compare.py),
[GEM+ baseline](../../tools/perf/gemplus_baseline.py),
[before/after audit](../../tools/perf/ab_audit.py),
[native decoder C gate](../../tools/c/sections/decoder_train.sh), and pending PX-1/PX-2 witnesses.
Their roles differ: regression/parity rows and the CPU pilot are not substitutes for the
new GPU, graph-import or full-model comparison runners. Reuse semantic fixtures and evidence
formats rather than creating a second unverified definition of the math.

### 5.1 Semantic graphs, compilation and native kernels

| ID | Experiment / controls | Outputs and exit criterion |
|---|---|---|
| OT-01 — common operation/gradient contract | Start with residual add, RMSNorm, SwiGLU, GEMM and their required backward pieces. Define ATen/BCIR operation mapping, effects, aliases, layouts, precision, accumulation and RNG. Include non-contiguous inputs, ragged shapes and zero/invalid sizes where the APIs admit/refuse them | Operation/gradient coverage ledger; numerical checks against an independent reference and each contract; unsupported operations refused or explicitly labeled fallback |
| OT-02 — IR/capture cost and coverage | P-EAGER/P-FX/P-AOT/P-IND versus supported B routes. Generate chains, fan-out/shared-input DAGs and reduction graphs, varying size/depth independently. Run supported static and variable-shape sequences; use a graph-break-refusing configuration where appropriate and separately record realistic partial capture | Capture/import/verify/plan/lower/codegen/load time, compiler peak memory, node/artifact sizes, guards/recompiles, graph breaks, fallbacks and final calls; equal input boundary and no hidden eager segment counted as generated BCIR code |
| OT-03 — backward and optimizer identity | One block and a tiny decoder; fixed initial parameters/batches/update semantics. Compare forward, loss, gradients, clipping, optimizer state and checkpoint/resume. Include tied weights, accumulation, provider and recompute cases | Exact state where the contract promises exactness; otherwise predeclared tolerances and learning/quality checks. Equal effective batch alone does not prove identical numerical updates |
| OT-04 — plan intervention | Extend PX-1 through PX-2/PX-3. Direct C versus plan-executed C, same entries/buffers first. Then vary only grouping, a pack reuse, a recompute choice or an admitted implementation | Declared choices match the complete executed trace, including optional preflight and drop accounting. Plan change causes expected calls/storage/work change; measure the complete-step benefit/cost separately from the instrumented witness |
| OT-05 — kernel/block portfolio | Direct C, generated BCIR MLIR, P-EAGER/P-IND library/generated calls, T-DSL/T-GLUON where target overlaps. Sweep small-call, cache-resident, streaming and dense-math regimes; forward/backward/update separately | Cold and warm latency/throughput, useful work/bytes, launch/barrier counts, pack time, tail handling and resources. Same dtype is insufficient without an equal accumulation/approximation contract |
| OT-06 — layout/packing/arena frontier | Shared-input SwiGLU projections, transposed gradients and reused weights. Compare no pack, per-call pack, persistent pack and producer-written panels when legal. Sweep workspace and activation budgets at fixed full micro-batch | Actual pack count/bytes/time, liveness overlap, peak owned storage and state identity. Do not double count shared packs or count released-but-in-flight storage as reusable. PX-4/PX-5 are implementation dependencies |

First common block: `U = X @ Wg; V = X @ Wu; Y = silu(U) * V`, followed by its backward.
Add residual/RMSNorm as a second block once the common subset passes. A proposed initial
CPU shape grid is `M = 16,128,512`, `K = 256,768`, `N = 1024`, with selected
`K=257/N=1025` tails and transposed/strided cases. Prune combinations explicitly by memory
budget; these shapes are a starting protocol, not a validated tuning optimum. GPU grids must
be selected for the actual target, not inherited as the only tensor-core workloads.

### 5.2 GPU code generation and resident execution

| ID | Experiment / controls | Outputs and exit criterion |
|---|---|---|
| OT-07 — mapping and conversion | T-DSL versus Gluon and any BCIR-generated candidate on the **same GPU**. Vary register tiles, lane/warp layouts, swizzles, replication, tails and required layout conversions under the same math | Semantic conversion tests, final image identity/disassembly, register/spill/shared-memory use, measured memory behavior and latency. Tile/thread names or PTX text alone do not pass |
| OT-08 — pipeline and synchronization | Sweep tile/stage/warp configurations, synchronous versus qualified asynchronous staging, fusion partitions and tensor instructions. Preserve producer/consumer and barrier semantics | Device results and ordering stress tests, resource/occupancy feedback, achieved useful work, stalls/spills where observable, complete block time. An extra stage/warp is not presumed faster |
| OT-09 — same image, different dispatcher | Load one compiled device image. Execute through the direct resident provider, then BCIR with identical resident input/output buffers; compare uncaptured and captured/batched submission separately | Host submission, device event time, queue wait, explicit copies/allocations, end-to-end critical path and jitter. Changed compilation/grid/buffers invalidate the pure orchestration attribution |
| OT-10 — complete training frontier | After #820 merges: fixed 10M-class recipe plus supported smaller/larger and variable-length cases; P-EAGER and P-IND separately. Same initialization, batches, optimizer, numerical policy and memory cap; then a separate best-admitted-recipe study | Learning versus wall time/tokens, held-out quality, true peak memory by scope, throughput/latency and compilation/tuning. A Triton comparator identifies its host/autodiff/optimizer harness; architecture changes are labeled |
| OT-11 — specialization/cache lifecycle | Static, repeated and changing lengths/shapes; cold process, warm in-process and persistent-cache cases. Change target/compiler/options/semantic contract to test invalidation | Guard/cache correctness, cold-to-warm break-even, compile/tune memory/time and code volume. Cache reuse cannot cross incompatible capability/precision/shape assumptions |

The first GPU integration should reuse a supported CUDA/HIP/SYCL runtime for allocations,
loading and events, with optional C++ glue. Inspect generated LLVM/PTX/device images as needed;
runtime/driver replacement is not required to compare instruction quality. PTX is an intermediate
ISA, while the loaded device image and executed instructions are the relevant final artifact
on NVIDIA [N3]. Compare AMD and NVIDIA results on separate target tables, rather than using
different chips as proof of compiler superiority.

### 5.3 Movement, telemetry, distribution and OS services

| ID | Experiment / controls | Outputs and exit criterion |
|---|---|---|
| OT-12 — data/residency path | Kernel-only resident data versus complete staging/pack/compute/readback. Vary payload size/reuse, transfer overlap and the supported pinned/staged paths; ASN.1/StreamPack parsing timed as control/data ingestion where actually used | Per-stage bytes/time/allocations, useful throughput and same payload/results. Host codec speed does not imply faster tensor arithmetic. HAM routes require physical bindings before a direct-DMA performance label |
| OT-13 — collectives/checkpoint | Only on a qualified multi-device host: compare optional established collective providers, placement/overlap policies and restore semantics at fixed work | Communication/compute critical path, scaling efficiency, topology, checkpoint pause/I/O and exact resume contract. Shard serialization or simulated networks do not establish distributed training throughput |
| OT-14 — telemetry/TMOG observer and feedback | Run the same artifact with tracing/monitoring off, sampled and detailed. Correlate task/claim, artifact, device, generation, timestamps and availability; replay missing/dropped/stale samples; calibrate offline and change only a quiescent next generation | Measured monitoring overhead, coverage/loss, timestamp uncertainty, ranking regret and held-out frontier. Unknown is not zero pressure; observed correlation is not a causal proof; no telemetry changes legality or live instructions |
| OT-15 — IPC decomposition | Direct call, shared-memory queue plus notification, Linux process adapter and eventual native adapter. Sweep message size, batch depth, cross-core/NUMA placement, request/reply and saturation with equivalent isolation/work | One-way/round-trip latency, p50/p95/p99, throughput, copies/bytes, context switches, wakeups, cache/TLB behavior where exposed, CPU/energy and backpressure. Separate CPU instruction-per-cycle "IPC" from inter-process communication |
| OT-16 — device/OS vertical slice | Canonical UART then virtio-console/blk: direct/Linux/native behavioral replay, queue/mapping/interrupt/cancel/reset lifecycles; UEFI boot proof separately. Later run the same admitted useful workload/device program on supported physical hardware | Correct ownership/order/completion, bounded storage, direct/Linux/native parity and recovery; QEMU proves functional behavior, not physical latency or superiority over GPU drivers |

Extend OT-12/15/16 with the [SI conformance/performance matrix](../BCIR_UNIVERSAL_INTERFACE_ROADMAP.md#10-conformance-and-performance-qualification):
hold method effects, request values, schema/profile, validation and ownership constant.
Measure codec, copy/layout conversion, transport and complete request separately. Preserve
semantic trace parity when adapters use different formats; require byte parity only for a
shared wire profile. Arrow C Data callbacks and native vtables remain local bindings.

For IPC/OS comparisons, preserve the isolation and scheduling contract. An in-process call
is a useful lower-overhead reference but does not provide a separate service's isolation.
Polling, notification and batching have different idle-power/latency/load trade-offs. seL4's
benchmark suite usefully separates IPC, IRQ, page mapping, scheduler and synchronization
measurements; use that decomposition without treating its published figures as BCIR targets
measured on another machine [K1].

Linux DMA separates CPU virtual, CPU physical and device/bus addresses; coherence also does
not remove ordering requirements. Model mapping lifetimes, ownership, completion and barriers
at the ABI boundary. Reusing a register table alone does not supply these driver services [K2].

## 6. Admission of new IR operations and pre-LLVM transformations

Do not hard-code one benchmark's winner into the semantic law. Keep semantic contracts,
target capability legality and calibrated optimization policy separate.

| Change type | Required justification before promotion |
|---|---|
| New operation/representation | Clear semantics, verifier/refusals, supported lowering and examples showing the missing expressive/analysis need. Not every semantic operation must independently make a benchmark faster |
| Required legalization/lowering | Preserve the declared semantics and reach a supported target; compare alternatives when available. Required target support is a legitimate reason even without a faster baseline |
| Optional optimization pass | A/B pass enabled/disabled on the same pipeline, target, inputs and numerical contract; verify effects on final code/resources and complete workload; predeclare acceptable regressions/compile budget |
| Memory/quality trade-off | Demonstrate an admitted time/memory/quality frontier or fit a required budget with the preserved contract; disclose extra computation or approximation |
| Target-specific instruction/layout | Capability predicate, executable conversion/lowering, numerical/ordering checks, resource limits and target measurements; refuse unsupported hardware |
| Learned/telemetry-guided ranking | Legality first; calibrated selection and held-out validation with bounded search cost, safe incumbent and reproducible future-generation activation |

For a proposed pass retain: pass/contract/target IDs, before/after IR and image hashes, admitted
shapes and numerical policy, legality witnesses, raw paired samples, useful bytes/work, peak
resources, compile/tune costs, mechanism and counterexamples. Interact passes intentionally:
test important pairs such as fusion × register pressure, packing × lifetime, recompute ×
workspace, and scheduling × peak memory. An individual win may become a combined regression.

For repeated execution, an extra compile/search cost `DeltaC` and a measured per-run saving
`DeltaT > 0` have a simple break-even `N > DeltaC/DeltaT`, before other lifecycle costs.
Select optional passes under the workload's lifetime and budgets. This is performance-driven
IR engineering without making transient timing evidence part of semantic correctness.

## 7. Hardware intimacy: make it a measurable contract

MAP currently parses macro-assembly-like resource/operation declarations into BCIR claims;
the inspected parser puts them in phase zero. It is not a full vendor instruction assembler
or a physical register allocator. BCIR already has typed assembly/ISA edges through its
LLVM lowering. Ordinary user-mode CPU instructions and GPU kernel instructions do not require
inventing a new OS; privileged device control and ownership have different requirements [B3, N1].

Use three linked descriptions instead of waiting for every detail of all hardware:

| Description | Required content | Owner / change boundary |
|---|---|---|
| Semantic and capability manifest | Operation/precision/ordering, ISA feature/ABI, address spaces, legal layouts, resource limits, ownership and required device/firmware identity | Compiler/verifier and provider contract; documented/probed facts and unknowns are explicit |
| Compiled resource/artifact record | Actual kernel/image, grid/block/warp choices, register/spill/shared-memory/scratch requirements, conversions and event dependencies | Backend output bound to target/compiler/options/shape contract; validate against capability limits |
| Measured execution profile | Workload-correlated timing/traffic/counters, clocks/thermal state, availability/loss, contention and confidence/measurement window | Telemetry/calibration; update costs or choose a later artifact, never rewrite legality from noisy readings |

Mapping the relevant device slice is sufficient to begin optimizing it. Grow coverage through
probes, specifications, compiler feedback and experiments. Some implementation details remain
opaque or vary at runtime; use conservative bounds, capability refusal and measured selection.
Complete knowledge of all physical registers or all future hardware is not a prerequisite.

A TMOG-style view should read these same records to display tasks, dependencies, device work,
memory lifetimes, queues and pressure. It should not independently reconstruct a conflicting
schedule. Sampling/trace collection itself consumes resources; OT-14 is a promotion gate for
the observer as well as the optimizer. Current signal/envelope/ring code provides foundations,
not a delivered universal hardware task manager [B4].

### Driver as compiler: the precise boundary

The compiler package can turn verified claims into device commands, queue descriptors,
transfer programs and dispatch dependencies. A resident adapter still establishes mappings,
publishes work, handles completions/faults and maintains lifecycle/ownership. The same package
can target direct, Linux-hosted and native adapters. Converting a Linux driver into a BCIR
package therefore means extracting and re-expressing supported protocols/contracts, reusing
appropriate runtime/driver code and qualifying behavior. It does not mean that encoding its
register tables into MLIR compiles away the OS or device lifecycle.

Use CUDA/HIP/SYCL queue/event semantics as interoperability references [N2, A1, S1]. Hardware-
specific IPC must preserve those semantics where required, including DMA visibility and
completion before storage reuse. A future fast path can specialize a service/copy/notification
sequence after its contract is proven; generic, direct and hosted paths remain references.

## 8. When to start drivers, IPC and the bootable OS

There is no defensible universal percentage of PyTorch performance that must be reached first.
Use dependency and evidence gates instead.

| Gate | Minimum exit / start condition | Relationship to existing roadmap |
|---|---|---|
| BG0 — measurement contract | OT route/semantic/target IDs, independent references, raw records, memory scopes and unsupported/fallback labels agreed; interface tests also bind schema/mapping/profile and SI contract | Enables current research and D0 contracts; no full-model benchmark until #820 merges |
| BG1 — trace-to-execution | PX-1 reviewed after merge; PX-2 identity and PX-3 one causal plan intervention pass on a common block | Primary IR/training integration milestone; need not block a bounded UART/boot lab |
| BG2 — hosted accelerator slice | One real target: a resident provider, exact ownership/events and OT-07–OT-09/OT-12 evidence; device image identity retained | Parallel GPU integration using supported runtimes; addresses PX-6 incrementally |
| BG3 — direct driver slice | Existing RuntimeChannel loopback plus a D0–D2 UART package, telemetry and bounded lifecycle tests | Start bounded driver work here; no requirement to win a complete training leaderboard |
| BG4 — Linux queue/transport proof | UART plus virtio console/block direct/Linux parity, measured OT-14/OT-15/OT-16 and generation/ownership behavior | Existing D3/D4 and M2–M5; freeze UAPI only after both device classes qualify |
| BG5 — native service/boot proof | UEFI/reset, memory/VM, clocks/interrupts and applicable DMA/PCIe prerequisites; direct/Linux/native replay for supported devices | Existing M8/native dependency track; boot correctness and device performance are separate records |
| BG6 — targeted kernel policy/fast path | Profile identifies a meaningful service/mapping/scheduling overhead or a required feature unavailable through stock interfaces; an adapter/kernel change improves the declared workload without violating its contract | Existing M7, then device D7/M9 qualification; retain stock-hosted baseline and fallback |
| BG7 — integrated promotion | Reproducible complete-workload time/memory/quality and tail-latency benefit or a stated independent-OS capability; required CI and physical target gates pass | Promote only the supported workload/device/OS scope, not all architectures or all training tasks |

Native boot research can begin as a small QEMU correctness track while hosted ML optimization
continues. UART is the first driver proof because its lifecycle is tractable; it is not a GPU
performance surrogate. A full native GPU replacement is a later device/firmware/memory/queue
engineering program, justified by a concrete requirement or observed blocker. It does not
need to precede every LLVM/PTX/ISA optimization experiment.

The allocation of development effort follows the measured dominant gap: improve a weak kernel
when math dominates; improve packing/liveness when memory dominates; improve submission and
batching when host gaps dominate; investigate native IPC/driver paths when a measured remaining
service boundary or deployment requirement warrants it. Use complete critical-path and resource
evidence rather than summing overlapping kernel/event durations.

## 9. Gateable implementation slices and integration order

| Slice | Deliverable | Gate / focused tests |
|---|---|---|
| Next after #820 | Review PX-1/PX-2a/PX-2b and their merged identity/refusal evidence; qualify the existing program/pack executor | BG0/BG1; OT-01/03/04; direct C remains reference; PX-3 intervention is still required |
| Plan-selected native execution | PX-3 per-call choices and one measured intervention; reuse memory/recompute contract | BG1; OT-04/06; costs, selected plan, trace and state refer to the same configuration |
| Compact buffers and layouts | PX-4 lifetime placement and PX-5 pack/layout semantics | OT-06/07; measured time/memory frontier, not descriptor count |
| Optional graph bridge | Functional ATen subset and AOTAutograd forward/backward import in an integration package | OT-01/02/03; compare BCIR and Inductor decisions on common kernels before native rewrites |
| Resident GPU provider | Optional supported-runtime adapter and image/event/resource records, one real target first | BG2; OT-07/08/09/12; precise device/fallback reporting |
| Broader kernel optimization | Add profitable fusion, tensor/attention/optimizer kernels under common contracts; keep provider/compiler alternatives | OT-05/08/10/11; compile/resource regression budgets and held-out validation |
| Telemetry/task view | Work-correlated signal intake, trace/UI consumers and offline calibration; availability-aware ranking | OT-14 and canonical telemetry ladder; observer overhead measured, generation switch quiescent |
| Universal interface slice | UDSI-0/1 service description and local binding, then one independently conforming foreign provider; bulk/profile/grammar extensions separately | SI-01–SI-12 as applicable, OT-12/15/16; direct RuntimeChannel and canonical ABI authority retained |
| Driver/IPC vertical proof | DDK UART then virtio, stock Linux adapter, bounded boot/native adapter track | BG3–BG6 and canonical D/M gates; OT-15/16; physical performance gate later |
| Data/AI workload qualification | Per-driver data/inference/training matrix and correlated dataset-to-completion/checkpoint evidence; measurements select IPC candidates | Applicable OT/SI/D/M gates; qualified relevant drivers/IPC precede physical HAM and native HAM/SYCL |

Keep arena/dispatch execution in C. Use C++ for compiler/framework/accelerator APIs when useful;
Python capture/tuning tools remain optional. Adopting an upstream compiler or library at a
provider boundary is a way to improve BCIR while preserving the independent runtime strategy.
Avoid a wholesale Triton/PyTorch/Linux fork to obtain an isolated optimization.

## 10. Reproducible result and pass/fail policy

Each experiment records: semantic/operation/graph and experiment IDs; all source/compiler/
runtime/driver/firmware versions; device and topology; admitted transformations/precision;
input/seed/shape/stride distributions; actual provider/fallback and image identities; selected
plan and executed trace; resource ownership/liveness; compile/cache/tune state; collection
availability and clock domain; raw timings, errors, quality and failure/refusal records.

Timing protocol: pin threads/affinity where applicable, disclose power/frequency settings,
warm deliberately, keep cold-start results separate, randomize paired rounds and repeat on
independent sessions. Choose enough work per timed region to resolve timer/launch overhead,
without hiding the per-call workload being studied. Define relevant regression margins and
confidence procedures before collecting the comparison. Time trace-disabled production paths;
measure the trace observer separately. Include held-out shapes and multiple seeds for learning
quality; report ties and regressions.

Memory protocol: distinguish logical tensor bytes, simultaneously live owned allocations,
workspace/persistent packs, runtime/allocator reserved device memory, process RSS and compile
memory. Include overlap, delayed completion, alignment and framework/library caches. Compare
the same scope on both systems. Inspect dtype/accumulation and real task grouping before
interpreting an apparent memory or launch saving.

Correctness/refusal/trace identity and bounded accounting are deterministic gates. Physical
timing is target-specific evidence with uncertainty; shared CI VM timing is not a universal
speed threshold. This host lacks a qualified GPU/PMU/energy/native-driver test rig, so those
results remain **unavailable**, not simulated performance passes. A simple kernel, QEMU boot
or a trace-only witness cannot promote the full training/OS stack.

## Source ledger

Repository implementation outranks roadmaps/research for status. Primary upstream interfaces
were checked on 2026-10-20 UTC. These sources establish mechanisms, not benchmark outcomes.

- **[B1]** [BCIR parity](../PARITY.md), [native training](../machine-learning/BCIR_NATIVE_DECODER_TRAINING.md),
  [MLIR matmul lowering](../../mlir/lib/passes/BCIRLowerGemMatmulBufferPass.cpp),
  [C tensor contracts](../../runtime/c/bcir_tensor.h), [ExecutionPlan](../../bcir/gem/execution_plan.py).
- **[B2]** Pending [#820](https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/pull/820),
  [pinned native launch model](https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/blob/6c01fcad1c2ace951a077956d8d0058b6490b28f/bcir/hosted/models/native_launch.py),
  [probe](https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/blob/6c01fcad1c2ace951a077956d8d0058b6490b28f/runtime/c/bcir_decoder_train.c),
  [witness](https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/blob/6c01fcad1c2ace951a077956d8d0058b6490b28f/bcir/tests/test_native_launch.py),
  [PX ladder](https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/blob/6c01fcad1c2ace951a077956d8d0058b6490b28f/docs/research/BCIR_TRAINING_ACCELERATION_ROADMAP.md).
  Refreshed at `b6c50f11d5ed2d18bd7b7cae026547b93db235fe`: the
  [PX-2b pack model](https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/blob/b6c50f11d5ed2d18bd7b7cae026547b93db235fe/bcir/hosted/models/native_pack.py)
  and [C lowering contract](https://github.com/Mecca-Research/Binary-Correspondence-Intermediate-Representation/blob/b6c50f11d5ed2d18bd7b7cae026547b93db235fe/runtime/c/bcir_decoder_pack.h).
- **[B3]** [MAP parser](../../bcir/frontends/map.py), [claim model](../../bcir/model/graph.py),
  [LangRef assembly contract](../BCIR_LANGREF.md), [RuntimeChannel v1](../../runtime/c/bcir_runtime_channel.h),
  [data-plane handoff](../kernel/BCIR_DATA_PLANE_HANDOFF.md),
  [driver/kernel roadmap](../kernel/BCIR_DRIVER_KERNEL_ROADMAP.md),
  [AMD runtime-first strategy](../kernel/BCIR_AMD_AI_DRIVER_ROADMAP.md).
- **[B4]** [Signal registry](../kernel/SIGNAL_REGISTRY.md),
  [envelope ABI](../kernel/TELEMETRY_ENVELOPE_ABI.md), [live ring](../kernel/BCIR_LIVE_RING_ABI.md),
  [telemetry intake](../../bcir/telemetry_intake.py), [hardware limits](../kernel/HARDWARE_VALIDATION.md).
- **[T1]** [Pinned Triton NVIDIA pipeline](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/third_party/nvidia/backend/compiler.py),
  [layout algebra](https://github.com/triton-lang/triton/blob/c01b6774b1865984607d89d89d3a10833de92037/include/triton/Tools/LinearLayout.h),
  [Gluon layouts](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html),
  [autotuning](https://triton-lang.org/main/python-api/generated/triton.autotune.html).
- **[P1]** [PyTorch custom backends / AOTAutograd](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_custom_backends.html),
  [compiler FAQ](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_faq.html),
  [dynamic shapes](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html).
- **[P2]** [Pinned Inductor IR](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/ir.py),
  [scheduler](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/scheduler.py),
  [memory planning](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_inductor/codegen/memory_planning.py),
  [joint-graph partitioner](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_functorch/partitioners.py).
- **[N1]** [CUDA hardware/kernel programming and inline PTX](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html).
- **[N2]** [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).
- **[N3]** [PTX virtual ISA and translation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
- **[A1]** [ROCr runtime / user-mode queues](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/what-is-rocr-runtime.html).
- **[S1]** [SYCL 2020, revision 12](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html).
- **[K1]** [seL4bench: IPC, IRQ, mapping, scheduler and synchronization](https://docs.sel4.systems/projects/sel4bench/).
- **[K2]** [Linux DMA mapping and ordering](https://docs.kernel.org/core-api/dma-api-howto.html).
