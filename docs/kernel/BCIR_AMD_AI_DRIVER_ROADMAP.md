# BCIR AMD AI Driver / Kernel Development Roadmap (2026-07-05)

**Goal**: bootstrap AMD's current AI stack and **maximize AI inference on AMD chips**, with a
build order — *inherit-and-enhance*, never replace. BCIR becomes the planning / verification /
cost / provenance brain **around** AMD's shipping stack (ROCm on the GPU rail, XDNA/Ryzen-AI on
the NPU rail), riding the resident compilers (AMDGPU-LLVM for the GPU, Peano/MLIR-AIE for the
NPU) and calling real kernels through Triton + AITER/CK — and owns only what BCIR uniquely
provides (K_BCIR cost algebra, Q8-frozen certificate-gated priors, `-bcir-verify` legality,
StreamPack provenance, event-phase/DMA-ring IR).

**Method**: a 14-agent research pass (AMD silicon/ISA, the ROCm + Linux `amdgpu`/KFD/`amdxdna`
driver machinery, Triton's AMD backend + the BCIR-Triton compiler design, the inference/
training/quant ecosystem, the framework supplements, and a scoping-only pass on the deferred
Linux-inheritance) with six load-bearing strategy claims put through adversarial verification
(4 confirmed, 2 partial, 0 refuted). Every BCIR-side statement is `file:line`-anchored; AMD-side
claims anchor to AMD/ROCm docs and project READMEs. This doc extends the **Part IX driver
catalog** and consumes the **[BCIR–Triton comparative-analysis](../research/BCIR_TRITON_COMPARATIVE_ANALYSIS.md)** verdict; it is a roadmap,
not a build — each device class gets its own `BCIR_<DEV>_DRIVER_BLUEPRINT.md` authored in its
own research session before its slices (the Part IX per-driver contract).

> **Four corrections the verification forced — they are messaging discipline.** (1) **BCIR's GPU
> compute codegen is not real today** (matmul lowers to a scalar `scf.for`, no Matrix Cores; the
> only PTX is elementwise and never executed; there is *no* `amd`/`rocm` channel and *no*
> `amdgcn` target). The AMD channel is a **thin routing seam** that hands isel/regalloc to the
> resident AMDGPU-LLVM backend and routes real kernels through Triton/AITER — native `amdgcn`
> Matrix-Core codegen is a **later, separately-gated** deliverable, not Phase 1. (2) **Triton is
> the portable *baseline*, not the perf default.** On Instinct the recommended path is
> `VLLM_ROCM_USE_AITER=1` → AITER (`ROCM_AITER_FA`/`ROCM_AITER_MLA`) and Composable Kernel;
> Triton is the always-available single-source fallback. Say "ROCm with Triton as the portable
> baseline **and** AITER/CK as the native perf backends." (3) **GPU and NPU are distinct
> accelerators** — CDNA (MFMA), RDNA-iGPU (WMMA), and XDNA-NPU (spatial dataflow) are **three
> device-class blueprints**, and Instinct has *no* NPU, so the NPU+iGPU hybrid is a client-APU
> pairing needing a cross-device router. (4) **The ML-framework supplement boundary splits by
> workload**: training + general autograd cross a *live FFI/process* edge into the eager API
> (PyTorch/JAX); ONNX/StableHLO carry only *lowered inference* graphs (ONNX cannot train, and the
> ONNX-Runtime ROCm EP was removed at ORT 1.23 — use the **MIGraphX EP**).

---

## 0. Executive strategy

**Inherit-and-enhance via interop-not-fork.** BCIR does not write an AMD kernel-mode driver, an
`amdgcn` instruction selector, or a ROCm reimplementation. It **inherits** AMD's driver tables
and communication machinery from Linux (deferred Phase 0), **rides** the resident AMDGPU-LLVM /
Peano compilers and ROCm/XRT runtimes through the resident-compiler gate, **calls** real kernels
on-demand (Triton on-call, AITER/CK/hipBLASLt as supplements), and **layers around** them the
one thing nobody else in the AMD stack has: a cost-governed, certificate-gated, legality-verified,
provenance-carrying plan. The forking of `amdgpu`/`amdkfd`/`amdxdna` (GPL-2.0, firmware-coupled),
the AMDGPU LLVM backend, `mlir-aie`, Triton, vLLM/SGLang, or ROCm is **prohibited** — it would
drag BCIR into owning firmware bring-up and instruction selection it must never own.

**The single hardest sequencing fact**: BCIR's differentiators (K_BCIR cost, learned priors,
`guided == exhaustive` certificates) **presuppose a working target + runtime + real reference
compiler to measure and certify against**. They cannot precede the channel/manifest/runtime/
Triton phases. A prior trained against a seeded (`cal_gen = 0`) uncalibrated AMD profile is
meaningless — **microbench calibration via the calibloop is a hard prerequisite** for any
defensible AMD prior.

---

## 1. The honest starting point

**What already exists to build on** (the AMD path is a data-and-binding problem, not an
optimizer rewrite):

- `HardwareChannel` + `register_channel()` (`channels.py:222`) is the single extension point —
  an `amd_cdna`/`amd_rdna`/`amd_xdna` channel is added without editing the optimizer or executor
  (interop-not-fork applied to the tower). `nvidia_ptx` (`channels.py:184`) + `TargetProfile.
  nvidia_ptx()` (`cost.py:242`) are the direct copy-template; the `warp` field already models a
  wavefront-like wide-lane machine.
- `TargetProfile` is an **open container** with a factory + `TARGETS` registry
  (`cost.py:181,292`); `MemoryHierarchy` already prices an **HBM tier at ~4× bandwidth**
  (`cost.py:166`) — Instinct HBM placement is priced today.
- The full **D-R driver substrate** is generic (`device_manifest.py`): banks-as-types (D-R2),
  Q8 distance-priced moves (D-R3), the **StridedView native-tile law** (D-R4), veto-not-steer
  (D-R1). The **A2 DMA descriptor rings** (`dma.py`) and **A1/B1 event phases** (`events.py`,
  EV1–EV3) give host↔VRAM scatter-gather + doorbell/AQL-completion/interrupt modeling ready to
  bind to an AMD SDMA engine.
- The **D3 learned-prior recipe** is shipped and certificate-gated: `tile_prior`/`channel_prior`
  are Q8-frozen, `guided == exhaustive` (mismatches must be 0), cal_gen-staleness-refused
  (`tile_prior.py:147,168`, `channel_prior.py:137`) — exactly the §7 ML placement card the
  blueprint contract mandates, and they train against *whatever* TargetProfile is passed in.
- `orchestrate()` + `HeterogeneousPlan` with a real FABRIC/SYNC transfer model
  (`channels.py:391,365`) will place claims onto an AMD channel and price host↔VRAM offload the
  instant it registers; StreamPack's per-segment channel tag (`streampack.py:56`, ABI v3) carries
  an `amd_rocm` segment with **zero artifact/ABI change**.
- The **resident-compiler gate** is already the codegen posture (`codegen.py:36`, `llvm.py:46`) —
  the discipline that lets BCIR ride AMDGPU-LLVM instead of hand-rolling AMDGCN isel. The Triton
  analysis already **decided** the AMD codegen posture (interop-not-fork). Part IX defines the
  process, the ML-seam mandate, and the D15 RDNA3 backend row.
- **Two-truth is structurally enforced**: `device_manifest`, `dma`, `events`, `tile_prior`,
  `channel_prior` import **no verifier** — an AMD prior/cost can never move the legality verdict.

**What is missing for AMD** (the deliverables of this roadmap): an `amd_*` channel; CDNA/RDNA/
XDNA `TargetProfile` factories; an `amdgcn` `CodegenTarget` (as a *delegating stub*, not isel);
AMD `DeviceManifest` instances (HBM/LDS/Infinity-Cache banks, Infinity-Fabric Q8 distance,
native_tile = MFMA/WMMA fragment); a ROCm/HSA RuntimeChannel execution + telemetry binding
(`amd-smi`/HIP driver API); the BCIR-Triton on-call module; trained AMD priors with a *calibrated*
cal_gen; a real MFMA/WMMA matmul codegen path (the honest-state gap, a later gate); and the
`BCIR_AMD_GPU_DRIVER_BLUEPRINT.md` / `BCIR_XDNA_DRIVER_BLUEPRINT.md` themselves.

---

## 2. The vertically-integrated stack

| Layer | Source (AMD/external) | BCIR's role |
|---|---|---|
| **L0 Silicon** | CDNA3 gfx942 (MI300X/MI325X), CDNA4 gfx950 (MI350/MI355, MXFP4/6), RDNA3 gfx11 / RDNA4 gfx12 (WMMA, fp8 + 4:2 sparsity), XDNA2 NPU (AIE-ML spatial array) → converging into **UDNA/gfx13** ~2026 | `TargetProfile` factories + `DeviceManifest` per device class, designed as **one `gfxN` family converging on UDNA**; Matrix-Core fragment = StridedView `native_tile`; NPU modeled as a **separate** spatial-dataflow class |
| **L1 Inherited KMD** *(Linux, deferred Phase 0)* | `amdgpu` DRM (GEM/GPUVM/CS/USERQ) + `amdkfd`/KFD (`/dev/kfd`) for GPUs; `amdxdna` (DRM accel) for the NPU — GPL-2.0, firmware-coupled | **Inherit-not-fork**: the source of DeviceManifest data + ABI binding targets; the resident KMD owns the kernel side |
| **L2 Topology tables + thunk** | KFD sysfs topology tree (nodes: banks, caches, io_link distances, wavefront); `libhsakmt`/ROCT | The concrete **"AMD driver tables"** to inherit — parse topology into a DeviceManifest digest (banks + Q8 distance from io_link weights); D-R1 veto-not-steer keeps it a compile-time attested manifest |
| **L3 ROCm/HSA/HIP runtime** *(interop)* | ROCr HSA runtime (AQL queues, doorbells, code-object loader) + HIP driver API (`amdhip64`) | RuntimeChannel execution + telemetry binding — **call** the runtime for launch/queues/signals (HIP as the pragmatic first launch target); BCIR keeps only cost/verification/provenance above; two-truth quarantines telemetry |
| **L4 Resident compiler** *(the gate)* | AMDGPU LLVM backend (`amdgcn-amd-amdhsa`, all gfx) for GPUs; Peano + MLIR-AIE/IRON for the NPU | Emit LLVM IR / HSA code objects (GPU) or ObjectFifo/tile IR (NPU) and hand isel/regalloc to the resident backend — **never** hand-roll MFMA/WMMA/AIE selection |
| **L5 BCIR-Triton on-call compiler** | Triton-on-ROCm / AOTriton (in-tree `third_party/amd` backend) wrapped as an **out-of-process** `TritonInvoker` plugin adapter (`TRITON_PLUGIN_DIRS`) | On-call GPU kernel provider **isolated from bcir core** (own venv/subprocess, StreamPack seam); the portable baseline to real AMDGCN *today*; migrate the measure-then-freeze idea, not the code; `triton-shared` is reference-only (unmaintained/CPU) |
| **L6 ROCm kernel libraries** *(supplement)* | **AITER** (recommended vLLM/SGLang default on Instinct), hipBLASLt (default GEMM), rocBLAS, Composable Kernel, rocWMMA, MIOpen, RCCL | Supplemental-call providers for tuned GEMM/attention/MoE/conv/collectives BCIR cannot yet emit; correctness oracles + K_BCIR calibration baselines; two-truth (results/telemetry only, never the verdict) |
| **L7 Serving / inference interop** | vLLM-ROCm, SGLang-ROCm (RadixAttention); Ryzen AI client: **Lemonade SDK / OnnxRuntime-GenAI / GAIA** (NPU+iGPU hybrid) | Interop serving front-ends that **host** BCIR-emitted kernels (no fork); Lemonade/OGA as supplemental-call for Ryzen AI client inference BCIR has no codegen for yet; migrate the backend-selection / prefix-reuse / capability-negotiation *ideas* |
| **L8 ML-framework supplements** | PyTorch-ROCm (first-class), JAX-ROCm/OpenXLA (first-class, StableHLO peer), TensorFlow-ROCm (legacy); AMD **Quark** + **torchao** (inference quant) | Supplemental delegation for training / full autograd / missing ops via a live FFI-or-process edge; StableHLO for lowered graphs; ONNX interchange via **MIGraphX EP**; BCIR owns planning/verification/cost |
| **L9 BCIR enhancement layer** *(what BCIR owns)* | — | K_BCIR 12-d cost algebra, Q8-frozen certificate-gated tile/channel priors, `-bcir-verify` legality, StreamPack provenance, A1/B1 + A2 IR, per-driver blueprints — layered **around (never inside)** Triton/ROCm; two-truth structurally enforced |

---

## 3. The phased build order

**Phase 0 — DEFERRED (scope only): Linux AMD driver-table + machinery inheritance.** *Deps: none
(deferred).* Enumerate, do not build, the AMD userspace↔kernel surface BCIR will inherit so later
phases have a named landing zone (§7). No code, no Linux-internals deep-dive; bootstrap on Linux,
never fork the GPL KMD.

**Phase 1 — AMD device-class manifests + TargetProfiles + channel registration (thin routing
seam).** *Deps: Phase-0 scope for manifest provenance; uses existing open containers only.* Give
BCIR a **name and a cost model** for each AMD device class with **no native codegen**:
`register_channel()` entries `amd_cdna`(gfx942)/`amd_cdna4`(gfx950)/`amd_rdna4`(gfx12)/`amd_xdna2`
(GPU channels `kind=gpu`; NPU a distinct class; `llvm_triple=amdgcn-amd-amdhsa`; `modeled=False`);
`TargetProfile` factories (`warp`→wavefront=64, wave32 option for RDNA; HBM/Infinity-Cache
`mem_channels`; `isa_features {mfma|wmma, fp8, mxfp4/6, sparsity4to2}`) as **one `gfxN` family
converging on UDNA**; `DeviceManifest` instances (banks {VGPR/SGPR, LDS 64 KB, L1, L2,
Infinity-Cache/MALL, HBM/VRAM}, `native_tile` = MFMA/WMMA fragment wired into the StridedView law
so a mis-tiled matmul refuses at plan time; Q8 distance LDS→L2→InfinityCache→HBM, cross-XCD +
Infinity-Fabric far hops; Strix Halo unified-memory = near-zero inter-bank distance); an `amdgcn`
`CodegenTarget` added **only as a delegating routing stub**. *Pure new data instances — no
optimizer edits.*

**Phase 2 — ROCm/HSA/HIP RuntimeChannel binding (execution + telemetry).** *Deps: Phase 1.* Make
the AMD channel executable and observable by **calling the resident runtime** — the far side of
the gate: a RuntimeChannel execution binding via the **HIP driver API**
(`hipModuleLoadData`/`GetFunction`/`LaunchKernel` over `amdhip64`) as the pragmatic first launch
target (abstracting over raw HSA/ROCr AQL for later fine-grained control); telemetry via
`amd-smi`/`rocm-smi` filling the declared-but-unbound GPU power/thermal/throttle/RAS signals; A2
DMA rings bound to an AMD SDMA copy engine; A1/B1 event phases bound to AQL/doorbell completion;
`dma-buf` fd exchange wired to IPC-R2 generation-guarded handoff. **Two-truth: all telemetry
feeds cost only, never the legality verdict.**

**Phase 3 — BCIR-Triton on-call interop compiler (isolated from core).** *Deps: Phase 2.* Deliver
competitive AMD inference **today** without waiting for native Matrix-Core codegen: a
`TritonInvoker` **out-of-process** channel-plugin adapter (own venv/subprocess, declared via
`channel_plugin.py` `channel.json`) that runs *whole unmodified Triton* (its LLVM pin, PyTorch
coupling, autotuner, MFMA/WMMA passes) **entirely outside `bcir` core's import graph**. BCIR core
hands it a K_BCIR-selected plan (tile size / `num_stages` / `waves_per_eu`) and receives a
StreamPack GPU-artifact segment (hsaco + HIP launch metadata) — **no ABI bump** (the v3 channel
tag is free-form). Lead mechanism: the out-of-tree Triton **plugin backend** (`TRITON_PLUGIN_DIRS`)
targeting AMD's existing in-tree `third_party/amd` backend + AMDGPU LLVM backend; **do NOT
register BCIR under Triton's frontend** (inverts control, demands isel BCIR lacks).
`microsoft/triton-shared` (`triton-to-linalg`) is an architectural **reference only** (unmaintained,
CPU-focused), not a live dependency. Budget for Triton version-pin churn (3.3/3.6/3.7).

**Phase 4 — BCIR enhancement layer over the AMD path (what BCIR uniquely owns).** *Deps: Phases
2+3 (a working kernel path to measure/certify against).* Layer BCIR's differentiators **around
(never inside)** Triton/ROCm: a Q8-frozen, certificate-gated (`guided == exhaustive`,
cal_gen-stale-refused) AMD `tile_prior` (MFMA tile size / `num_stages` / `waves_per_eu`) +
`channel_prior` (launch-config/occupancy/memory-tier) — the principled replacement for Triton's
brittle 24h/GPU `do_bench` autotune; **K_BCIR cost calibration** via a TurnkeyML-style
cross-hardware harness + Triton `do_bench` as a *calibration source* (telemetry only, never
selection/legality); `-bcir-verify` R-law legality over the AMD plan; StreamPack provenance
across the on-call GPU seam; migrated ideas landed (OCP MXFP4/6/8 + blockwise/NF4 into
`quantize.py` under R17; CK fusion-as-composition into `gem.*` seam laws; torchtitan device-mesh
sharding-as-a-plan into channel orchestration). Author `BCIR_AMD_GPU_DRIVER_BLUEPRINT.md` per the
Part IX contract with its §7 learned placement card.

**Phase 5 — XDNA NPU device class (separate manifest, blueprint, hybrid router).** *Deps: Phase 1
machinery; Phase 4 pattern.* Treat the Ryzen AI NPU as a **distinct accelerator** (spatial
dataflow, its own driver `amdxdna`, ISA, compiler Peano/MLIR-AIE, runtime XRT/ERT): a
`BCIR_XDNA_DRIVER_BLUEPRINT.md` with banks {AIE-tile 64 KB L1 ×32, memtile 512 KB L2 ×8,
shim→DDR} mapped onto ObjectFifo/buffer-descriptor dataflow via the existing A2/A1/B1 vocabulary;
a Q8-frozen NPU `tile_prior`/`channel_prior`; **Block FP16** as an NPU accuracy tier; the
resident-compiler gate for the NPU resolves to **Peano + MLIR-AIE/IRON** (call, never hand-roll
AIE placement/DMA); and a **cross-device hybrid router** seam modeling the shipping
`prefill → NPU / decode → iGPU` mode with a two-device residency/handoff cost (Strix Halo unified
memory = near-zero copy). *Instinct has no NPU — this is a client-APU-only pairing, a third
device-class blueprint.*

**Phase 6 — Serving/inference + ML-framework interop and supplement wiring.** *Deps: Phase 4;
Phase 5 for the Lemonade hybrid.* Bootstrap on AMD's shipping serving/quant stack: route
higher-perf native kernels through **AITER** (`ROCM_AITER_FA`/`MLA` — the recommended vLLM/SGLang
default on Instinct) + hipBLASLt/CK/MIOpen/RCCL as supplemental-call providers behind the gate;
vLLM/SGLang-ROCm as serving front-ends that **host** BCIR-emitted kernels (no fork); execute the
interop ledger (§5); wire the ML-framework supplements (§6); supplement Ryzen AI client inference
BCIR cannot yet codegen via **Lemonade Server / OnnxRuntime-GenAI**; use **AMD Quark + torchao**
as the called quant path (**not** bitsandbytes, whose ROCm fork is beta/disabled).

---

## 4. The three device classes (never one)

| Class | Silicon | Matrix unit | Memory model | Resident compiler | Notes |
|---|---|---|---|---|---|
| **CDNA (datacenter GPU)** | gfx942 (MI300X/MI325X), gfx950 (MI350/MI355) | **MFMA** (4/CU, wave64, fixed 16×16×K / 32×32×K fragments; SMFMAC sparse) | Chiplet on-package NUMA (XCD), 192–288 GB HBM3/3e @5.3–8 TB/s, Infinity Cache, LDS 64 KB | AMDGPU-LLVM (`amdgcn`) | Inference is bandwidth-bound at batch-1, matrix-core-bound at batch; MXFP4/6 on gfx950 |
| **RDNA (consumer/edge iGPU)** | gfx11 (RDNA3), gfx12 (RDNA4, RX 9070) | **WMMA** (wave32/64, smaller unit; native FP8 + 4:2 sparsity on gfx12) | GDDR6 + large Infinity Cache/MALL (no HBM) | AMDGPU-LLVM (`amdgcn`) | Distinct TargetProfile (dtype set, warp options, memory system), **same** resident backend |
| **XDNA (NPU / AI Engine)** | XDNA2 (Ryzen AI, Strix Point/Halo) | Spatial dataflow, **not SIMT** — 32 AIE tiles, explicit L1 scratchpads + memtiles + shim DMA, ObjectFifo/BD movement | Software-managed tile memory, ~50 TOPS INT8 / Block FP16 | **Peano + MLIR-AIE/IRON** | No register allocator to hand off to — perf is tile placement + double-buffering + DMA dataflow; maps onto BCIR's A2/A1/B1 vocabulary |

The **hybrid**: Ryzen AI's default LLM flow splits a *single* model — compute-bound prefill on
the NPU, decode on the RDNA iGPU — so the roadmap needs a cross-device router with a two-device
residency/handoff cost (Phase 5); on Strix Halo the shared unified memory makes the handoff
near-zero-copy. CDNA/Instinct never pairs with an NPU.

---

## 5. The per-project interop ledger

The Triton discipline (comparison → interop → import only portable ideas) applied to every named
project:

| Project | AMD status | Verdict | Best idea to import |
|---|---|---|---|
| **Triton** (Triton-on-ROCm / AOTriton) | In-tree AMD backend; vLLM's portable always-available baseline (not the perf default) | **interop** (on-call out-of-process plugin); do NOT fork; do NOT put BCIR under Triton's frontend | Measure-then-freeze: replace 24h/GPU `do_bench` with a Q8-frozen certificate-gated prior; single-source cross-vendor portability |
| **vLLM** | Mature ROCm backend; recommends `VLLM_ROCM_USE_AITER=1` (AITER native default); Triton fallback | **interop** (serving front-end hosting BCIR kernels) | Attention-backend auto-selection heuristic (arch/head-size/dtype/quant) → BCIR's priced registry selection (cost-side) |
| **SGLang** | First-class ROCm (MI300X/MI355, day-one DeepSeek); AITER default | **interop** | RadixAttention prefix-KV reuse → correspondence-keyed prefix sharing scored by a learned prior |
| **Lemonade SDK** | AMD-native; Ryzen AI hybrid NPU(prefill)+iGPU(decode) via OGA | **supplemental-call** (client inference BCIR can't codegen) + **migrate-idea** | Capability-driven `modality→engine→backend→device` negotiation as the shape of the blueprint device manifest; a Lemonade-style router sits *above* the separate NPU/iGPU blueprints |
| **GAIA** | AMD-native desktop agentic-RAG app on Ryzen AI | **do-not-import** (application layer above the IR) | Validates the Lemonade-server-as-supplement path; nothing at IR/codegen altitude |
| **TurnkeyML** | AMD-native no-code ONNX build→run→benchmark (Lemonade's lineage) | **migrate-idea** (harness only) | Reproducible cross-hardware build→run→benchmark→report harness feeding two-truth cost/telemetry + certificate checks (never the verdict) |
| **TokenSpeed** (LightSeek) | Cross-vendor Gluon kernels 1.6–3.6× over portable Triton on MI355X | **migrate-idea** (LLM-operator API surface only) | The attention/MoE/GEMM operator *vocabulary* — the registry/numerics/selection *design* is already what BCIR is |
| **Digest AI** | AMD-affiliated model ingestion/analysis (HF ONNX) | **migrate-idea** (ingest/report flow) | Automated model-graph analysis-to-report → BCIR provenance + registry ingestion |
| **Unsloth** | Official ROCm 6.0+, Triton kernels HIPified; disables bitsandbytes on AMD | **interop** (on-call fine-tuning) + **migrate-idea** | Exact, memory-frugal reformulated backward passes (fused/chunked CE, RMSNorm/RoPE) → value-invariant cost-governed BCIR rewrites (the algebra only) |
| **torchtitan** | AMD-optimized fork, MI300X/MI325X-validated | **supplemental-call** (delegate distributed training) + **migrate-idea** | Declarative N-D parallelism as a device-mesh sharding *spec* → BCIR's placement-candidate graph + channel-orchestration cost (RCCL/PyTorch keep the comm codegen) |
| **bitsandbytes** | ROCm is a **beta multi-backend fork, not mainline** (disabled by Unsloth on AMD) | **migrate-idea (formats)** + **do-not-depend** (prefer torchao/Quark) | Blockwise per-block absmax + NF4 datatype → `quantize.py`/`precision.py` under R17 (hardware-agnostic format) |
| **LlamaIndex** | Runs on AMD only by pointing at a ROCm inference backend | **do-not-import** (application-layer orchestrator) | At most expose a ROCm-backed BCIR inference endpoint it calls; nothing at IR altitude |

---

## 6. The ML-framework supplement boundary

BCIR **calls** these for what it cannot yet do; it never reimplements or imports their graph/
scheduler/runtime (no two-planners-fighting):

- **PyTorch-ROCm** (primary): the delegate for **autograd on exotic ops, full training loops, and
  any op BCIR has no Matrix-Core kernel for**. Boundary: a **live FFI/ATen (libtorch) or
  subprocess edge** for training + general autograd (these *cannot* ride ONNX); StableHLO/ONNX
  only for *lowered inference* graphs. PyTorch reaches Matrix Cores via rocBLAS/hipBLASLt/CK.
  (`torch.compile`/TorchInductor itself emits Triton.) BCIR owns planning/K_BCIR/verification/
  provenance.
- **JAX-ROCm / OpenXLA** (cleanest IR-to-IR peer): boundary is an **IR-level StableHLO** emit/
  consume edge for lowered graphs (including a `jax.export`'d training-step function); live
  `jax.grad`/`jax.jit` over an FFI/process edge for dynamic autograd. StableHLO is an MLIR dialect
  that aligns with BCIR's own law dialect. Delegate XLA fusion/autotuning; BCIR owns
  correspondence/legality/cost.
- **TensorFlow-ROCm** (legacy, weak leg): SavedModel/ONNX exchange or subprocess only; delegate
  *pre-existing* TF/Keras model inference exclusively. Do not build TF-specific plumbing.
- **ROCm kernel libs — AITER / hipBLASLt / CK / MIOpen / RCCL**: supplemental-call **behind the
  gate**; BCIR plans and orchestrates over them and uses them as correctness oracles + K_BCIR
  calibration baselines. Two-truth: results/telemetry only, never the verdict. Never fork; never
  hand-roll their isel.
- **AMD Quark + torchao** (inference quant): AMD's official vLLM-wired quant path (FP8/MXFP4/6/OCP,
  weight+activation+KV) + torchao (PyTorch-native int4/float8/QAT). Supplemental-call the
  quantizer; enhance the *calling* side with BCIR cost/provenance/verification; migrate only the
  portable quant-*format* ideas (blockwise/NF4, MXFP) into `quantize.py` under R17. **Not
  bitsandbytes** (beta fork).
- **Interchange caveats** (from verification): ONNX is **inference-only** and its opset lags ATen;
  the ONNX-Runtime **ROCm EP was removed at ORT 1.23** → use the **MIGraphX EP** for AMD ONNX
  delegation. Any "ONNX for training" or "ONNX Runtime ROCm EP" assumption is wrong.

---

## 7. The deferred Phase-0 Linux inheritance (scope, not build)

The user's strategy inherits AMD's driver tables + communication machinery from Linux, but the
Linux-internals deep-dive is **explicitly deferred / out of scope** for this pass. Phase 0 names
four inheritance targets and one fallback so later phases have a landing zone:

1. **Driver tables** = the **KFD sysfs topology tree** (`/sys/class/kfd/.../topology/nodes`:
   memory heaps as typed banks, caches, `io_link` bandwidth/latency/weight distances, wavefront
   size, BDF order) parsed into a **re-verified, digest-sealed BCIR DeviceManifest** (banks + Q8
   distance) — a near-1:1 map onto D-R1/D-R2/D-R3, a **pure data ingest, not code**. *(Precision:
   the topology is originally sourced from firmware — ACPI CRAT — parsed by KFD and re-exposed.)*
2. **Communication machinery** = the enumerable **KFD ioctl ABI** (`kfd_ioctl.h` v1.23:
   CREATE/DESTROY_QUEUE, ALLOC/MAP/UNMAP_MEMORY_OF_GPU, SVM, CREATE/WAIT_EVENTS,
   GET/IMPORT/EXPORT_DMABUF, GET_PROCESS_APERTURES_NEW) + the **amdgpu DRM UAPI** (GEM typed
   domains, GPUVM, CS chunk model, USERQ) + the separate **`amdxdna`** NPU driver — all as future
   **RuntimeChannel direct hook-table** binding targets (Part VIII MC8), bound as an ABI, with the
   resident KMD owning the kernel side. *(Precision: AQL packets / doorbells are ROCr/HSA-runtime
   + hardware constructs, not KMD tables — inherit the ABI, bind the runtime.)*
3. **The HSA AQL doorbell ring + Linux-6.16 user-mode queues** (MES/MQD/wptr-doorbell) as the
   ring-shaped precedent that may inform a later BCIR-IPC adapter (IPC-R1..R4) and event-phase (A1/B1). Note
   the forward caveat: user-mode queues partially *supersede* KFD-ioctl submission, but the
   AQL/doorbell abstraction is future-proof (only the submission wrapper changes).
4. **DMA-buf fd handoff** (KFD + amdgpu PRIME, emerging P2P) as the **IPC-R2 generation-guarded
   zero-copy** capability primitive (the substrate RCCL/RDMA ride).
5. **The Strategy-3 "Linux Master Kernel" fallback** (wave 15): keep `amdgpu.ko`/KFD/ROCr resident
   as a peer over an initially simple `SOCK_SEQPACKET` + bounded shared-memory transport,
   migrating only measured-hot submission
   paths off it (telemetry-ring driven), the cold tail staying on Linux indefinitely — the bridge
   that lets the AMD-compute wave run **before** the native BCIR-IPC ring substrate (driver-catalog
   D7) lands.

**Not inheriting**: the GPL-2.0 firmware-coupled in-kernel `amdgpu`/`amdkfd`/`amdxdna` drivers
themselves (fork prohibited); System V IPC / legacy signals / CRIU (cold-tail bloat).

**The deferred trade to resolve later** (data-driven, not a-priori): full ROCm/Linux
backward-compat (instant MI300/MI350 enablement, but a second scheduler/telemetry BCIR cannot
govern — a two-truth tension) **vs** a slim BCIR-native inheritance (topology tables + AQL-ring
vocabulary + KFD event/dma-buf model + a ~2–3-generation ioctl shim, cost-governed and
two-truth-clean, but greenfield and still resident on AMDGPU-LLVM). **Measure which paths are hot
and native-ize only those.**

---

## 8. Risks / messaging discipline

- **Overclaim trap.** BCIR's GPU codegen is not real today; the `amd` channel is a **thin routing
  seam** that delegates isel to AMDGPU-LLVM and routes real kernels through Triton/AITER. Native
  `amdgcn` Matrix-Core codegen is a later, separately-gated deliverable — never imply BCIR emits
  competitive Matrix-Core kernels itself.
- **"Triton is the default" oversimplification.** Triton is the portable always-available baseline
  and V1-enablement vehicle; **AITER/CK are the native perf defaults** on Instinct (and the vLLM
  default is version-sensitive). Say "ROCm with Triton as the portable baseline **and** AITER/CK
  as native perf backends."
- **GPU/NPU conflation.** XDNA is a physically separate accelerator (driver `amdxdna`, ISA, memory
  model, compiler Peano/MLIR-AIE, runtime XRT/ERT). A shared GPU manifest cannot describe
  tile-to-memtile DMA. **Three device-class blueprints** (CDNA, RDNA-iGPU, XDNA-NPU) + a hybrid
  router; Instinct has no NPU.
- **Framework call-boundary error.** ONNX cannot carry training/general autograd (inference-only,
  opset lags ATen); training/autograd **must** cross a live FFI/process edge into the eager API.
  The ONNX-Runtime **ROCm EP was removed at ORT 1.23** — use the MIGraphX EP.
- **`triton-shared` staleness.** Unmaintained and CPU-focused — an architectural **reference** for
  TTIR→linalg only, not a live AMD dependency. AMD's Triton backend is *in-tree*, so BCIR's
  out-of-tree plugin builds **on top of** `third_party/amd`, not in place of it.
- **Two-truth leak.** AMD's dynamic features (XNACK/HMM page migration, library autotuning, SMI
  telemetry, `do_bench` timings) are cost/telemetry inputs that must **never** touch the
  `-bcir-verify` legality verdict — a mis-migrated page or mis-tuned kernel is *slower, never
  illegal*. The Strategy-3 Master Kernel's own scheduler/telemetry is a "second truth" BCIR cannot
  govern — keep it quarantined.
- **Resident-compiler-gate violation.** Hand-rolling MFMA/WMMA selection, LDS staging,
  buffer_load/store, block-pingpong, or AIE tile placement inside BCIR duplicates a fast-moving
  GPU/NPU middle-end and breaks the gate. Any `amdgcn` `CodegenTarget` is a **delegating stub**.
- **Moving-target / version drift.** The `amdgpu` KMD is now versioned separately from ROCm
  (driver 30.x vs ROCm 7.x, ~1-year compat window); Ryzen AI Software / OGA / EP interfaces and
  AITER/vLLM defaults shift per release. Interop-not-fork is mandatory — pin ROCm 7.x +
  gfx942/gfx950 and Ryzen AI Software ~1.7.x in blueprints and treat every dependency as a
  sync-burden line item.
- **Sequencing risk.** BCIR enhancements presuppose a working target + runtime + real reference
  compiler to certify against — they cannot precede Phases 1–3, and priors trained against a
  seeded (`cal_gen=0`) profile are meaningless. **Microbench calibration is a hard prerequisite.**

---

## 9. Recommended next steps (ranked)

1. **Author the blueprints first** (normative-before-code, Part IX contract):
   `BCIR_AMD_GPU_DRIVER_BLUEPRINT.md` (with its §7 learned-placement card) and a separate
   `BCIR_XDNA_DRIVER_BLUEPRINT.md` — **three device classes** (CDNA, RDNA-iGPU, XDNA-NPU), not one.
2. **Land Phase 1**: `register_channel()` `amd_cdna`/`amd_cdna4`/`amd_rdna4` + TargetProfile
   factories + DeviceManifest instances (banks + Q8 distance, `native_tile`=MFMA/WMMA fragment
   wired into the StridedView law) as one `gfxN` family — **pure new data, no optimizer edits**.
3. **Prototype the KFD-topology-sysfs → DeviceManifest ingest** (banks + `io_link` Q8 distances) as
   the concrete, in-scope slice of the deferred Phase 0 (data-only).
4. **Build the `TritonInvoker`** out-of-process plugin adapter (own venv, `channel.json`,
   StreamPack GPU-artifact segment) leading with the out-of-tree Triton plugin backend over AMD's
   in-tree `third_party/amd`; keep Triton entirely out of `bcir` core's import graph.
5. **Bind the RuntimeChannel** via the HIP driver API (`hipModule*` over `amdhip64`) as the first
   launch target + `amd-smi`/`rocm-smi` telemetry, so `orchestrate()` can route to AMD and price
   host↔VRAM offload the instant it lands.
6. **Stand up a TurnkeyML-style cross-hardware benchmark harness** + Triton `do_bench` as a K_BCIR
   *calibration* source; run the calibloop to give the AMD profile a real `cal_gen` **before**
   training any prior.
7. **Wire the supplemental-call seams** (AITER/hipBLASLt/CK/MIOpen/RCCL as oracles + higher-perf
   tier; PyTorch-ROCm / JAX-StableHLO delegation; AMD Quark/torchao quant) — all behind the channel
   seam, two-truth-clean.
8. **Execute the interop ledger's migrate-idea items**: OCP-MXFP + NF4 into `quantize.py` (R17),
   RadixAttention prefix-reuse into provenance/priors, torchtitan device-mesh sharding-as-a-plan
   into channel orchestration, Unsloth exact-backward reformulations as value-invariant rewrites,
   measure-then-freeze into the Q8-frozen certificate-gated prior.
9. **Draft the Phase-5 hybrid NPU+iGPU router seam** (prefill→NPU / decode→iGPU) with a two-device
   residency/handoff cost on the event-phase vocabulary, and a Peano/MLIR-AIE resident-compiler
   binding for the NPU (call, never hand-roll AIE placement).
10. **Validate every AMD device execution bit-for-bit** against the existing LLM decode-rail host
    oracle (reference decode + KV-cache twin) before any performance claim, and register the AMD
    channel into the C-twin routing table so both rails route the identical claim→channel map.
