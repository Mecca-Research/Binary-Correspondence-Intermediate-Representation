# BCIR vs Triton — comparative analysis + the fork / interop / migrate decision (2026-07-04)

**The question** (verbatim intent): BCIR and Triton share a philosophy — write better, more
efficient AI code — but BCIR's scope is much larger and Triton is missing several things BCIR
does. Should BCIR **fork** Triton, **incorporate** it some other way, or is Triton (and its
MLIR approach) **superseded** by what BCIR attempts? Is anything in Triton advanced and unique
enough to **migrate**, and if so, all of it or a subset?

**Method**: the 2019 MAPL paper (Tillet/Kung/Cox — the LLVM-IR origin) read in full, then a
14-agent research pass on the *modern* system (Triton 3.7.x, MLIR TritonGPU dialect, Linear
Layouts, TMA/WGMMA/warp-specialization, Gluon, Triton-CPU, the ecosystem) with six load-bearing
claims put through **adversarial verification** (each skeptic tried to *refute* it with
independent research). Every BCIR-side statement is `file:line`-anchored; Triton-side claims are
anchored to the paper, the public repo, or published follow-on papers (ASPLOS'26 Linear Layouts,
the Tawa warp-spec paper, IBM's "Anatomy of a Triton Attention Kernel"). This doc sits in the
comparison family with `CLANG_COMPARISON.md` and `OPENAI_BCIR_INTEGRATION_RESEARCH.md`.

> **Two corrections the verification forced — they shape everything below.** (1) **Triton does
> NOT hand-roll instruction selection.** It emits LLVM IR and delegates isel + register
> allocation to LLVM's NVPTX/AMDGPU backends and ptxas; its value-add is the tile DSL + the
> *TritonGPU middle-end* (layout, coalescing, shared-memory, pipelining). That is the **same
> "emit IR, never hand-roll isel" philosophy as BCIR's resident-compiler gate** — so Triton is
> *architecturally aligned* with the gate and slots in cleanly as one resident GPU backend, not
> a rival hiding behind it. (2) **"BCIR proves, Triton merely searches" overclaims.** BCIR's
> `guided == exhaustive` certificate proves the learned prior reproduces BCIR's *own* exhaustive
> optimum under BCIR's *own* analytical cost model — optimality-**relative-to-a-model**, not
> proof of hardware-optimality. Triton's `@triton.autotune` measures *real latency* (hardware
> ground truth). The two are **complementary**, not a strict-dominance win. Market BCIR as
> "analytical + certificate-gated + cost-quarantined-from-legality," not "we prove, they guess."

---

## 0. Executive verdict

**Do NOT fork Triton. Interop with it, and selectively migrate a short list of portable *ideas*
re-derived against BCIR's own rails.**

- **Fork? No.** A wholesale fork imports a large C++/MLIR codebase (≈38% MLIR, ≈27% C++, ≈34%
  Python) transitively pinned to a multi-million-LOC LLVM stack, on a fast ~2–4-month
  PyTorch-coupled cadence with periodic LLVM bumps — a perpetual-rebase liability whose *entire
  content* is GPU-microarchitecture-specific middle-end + numerics BCIR deliberately reaches via
  the resident compiler. MIT-licensed, so a fork is *legal*; the economics are prohibitive and
  the scope is wrong. The case against forking rests on **scale, cadence, LLVM-pin coupling,
  GPU-specificity, and scope mismatch** — *not* on an isel-philosophy collision (that premise is
  false, per correction 1).
- **Superseded? No — and this is the honest part.** Triton runs **SOTA multi-vendor GPU code
  today**: it is the default TorchInductor GPU backend and the default vLLM AMD attention
  backend, reaching ~106% of FlashAttention-3 from one source, ~64M PyPI downloads/month. BCIR's
  GPU compute path is **early/deferred**: `gem.matmul` lowers only to a scalar `scf.for` nest
  (`BCIRLowerGemMatmulBufferPass.cpp:137-174`) with plain `arith.mulf/addf` — **no tensor cores,
  no shared memory** — and the only PTX BCIR emits is an elementwise kernel via `llc` that is
  asm-marker-checked and **never executed on a device** (`codegen/targets.py:39`). Until BCIR's
  PTX actually runs and self-checks on hardware, any claim of GPU parity is unfounded.
- **The correct positioning is dimensional, not containment.** BCIR is **broader** (kernels +
  compilers + drivers + systems, plus verification, provenance, analytical + certified-learned
  cost, two-truth quarantine, a portable ABI); Triton is **deeper on GPU codegen that ships at
  scale**. They overlap at the AI-kernel / tile-planning slice and diverge in depth. Drop
  "BCIR strictly contains Triton" — it is refutable. Use **"broader in breadth, shallower on GPU
  depth."** Triton's *design goals* are narrower than BCIR's; Triton's *GPU realization* is far
  ahead of BCIR's.
- **Migrate? A short list of ideas, yes — code, no.** The standout is the **Linear Layout F₂
  algebra** (below). A handful of tile-level passes and the MXFP numerics are portable. The
  GPU-bound machinery (WGMMA/TMA/tcgen05, warp-shuffle swizzle, the autotuner) is not.

---

## 1. The comparison matrix

| Axis | Triton | BCIR | Edge |
|---|---|---|---|
| **Schedule selection / cost model** | No shipped analytical/learned cost model; default `@triton.autotune` is empirical min-of-measured-timings (`do_bench`, L2-flushed) over a hand-written `Config` list, re-paid per new shape key. Exposes a `prune_configs_by`/`perf_model` *hook* it never fills. | Analytical K_BCIR tropical 12-d costvec (`cost.py:43`) + Q8-frozen learned tile/channel priors certificate-gated so `guided == exhaustive`, 0 mismatches (`tile_prior.py:168`); cost/learning quarantined from legality (two-truth). | **BCIR on design** (a-priori, certified-relative-to-model, cost-quarantined) — but the certificate is proof-relative-to-BCIR's-model; Triton's search measures real hardware. **Complementary.** |
| **Formal verification / legality** | None. Correctness rests on MLIR op verifiers + fixed-shape `allclose` tests + (now) Linear-Layout algebraic closure. Documented silent-miscompile history (`tl.dot` wrong results, default-TF32 truncation, masked `other=None` undefined). External Z3 verifiers exist *because* Triton lacks one. | First-class `-bcir-verify` R1–R23 laws + op verifiers + cross-op seam laws (R22/R23), C twins, Python oracle, value-invariance proofs (`matmul_tiled` bit-identical to reference). | **BCIR (decisive).** |
| **Tensor layout algebra** | **Linear Layouts**: every layout as an F₂/GF(2) linear map over register/lane/warp/block bits; conversion = B⁻¹∘A; proven closed under shape ops; fixed ~12% of Triton bugs, up to 1.40×. Genuinely novel, published (ASPLOS'26). | Coarse SoA↔AoS pivot priced through stride/memory terms + `LayoutCertificate` (`layout.py:206`); affine per-resource addressing (`offset`/`stride_k`/`stride_class`). No register/shared/bank layout algebra. | **Triton (real gap for BCIR; the standout migrate-idea).** |
| **GPU compute codegen, today** | Production multi-vendor: TTIR→TTGIR→LLVM→PTX/AMDGCN, WGMMA/tcgen05/TMA/warp-spec; default TorchInductor + vLLM-AMD backends. | Early/deferred: matmul → scalar `scf.for`, no tensor cores, no shared memory; only elementwise PTX via `llc`, asm-marker-checked, never run on device. | **Triton (decisive, real-today vs deferred).** |
| **Codegen philosophy (isel handoff)** | Emits LLVM IR; delegates isel + regalloc to LLVM NVPTX/AMDGPU + ptxas; owns only the GPU middle-end. | Resident-compiler gate emits C23/LLVM IR, hands to clang/llc; never hand-rolls isel. | **Aligned (same principle)** — refutes the "Triton is the isel surface BCIR defers" premise; makes Triton a natural resident GPU backend. |
| **Scope / breadth** | GPU tensor kernels only (NVIDIA CC8.0+, AMD ROCm6.2+; CPU experimental). Single-kernel scope (cross-kernel fusion lives in TorchInductor *above* it). | Registry-first, phase-ordered across ML kernels + compilers + drivers + systems; whole-program planner + heterogeneous channel orchestrator (`channels.py:59`); eBPF/CBLAS/FFTW/LAPACK FFI, CXL tiers. | **BCIR on breadth** (not on maturity/adoption). |
| **Provenance / reproducibility** | Opaque JIT compile cache; documented nondeterminism (autotune re-selection, atomic-add order, warp scheduling). | Frozen, versioned, CRC'd StreamPack ABI (`streampack_abi.py:34`) + registry provenance (R11 stale-pack, R13 manifest, replay); determinism-by-construction C23 subset. | **BCIR.** |
| **Ecosystem / adoption** | De-facto GPU-kernel IR of the PyTorch era: TorchInductor/vLLM/SGLang/Liger/FlagGems, MIT, large kernel-gen research ecosystem (TritonBench). | Research IR with reference oracles and *modeled* (driverless) GPU/FPGA/NVMe channels. | **Triton (decisive).** |
| **Numerics / mixed precision** | fp16/bf16/fp8 tensor-core accumulation, `dot_scaled` with OCP MXFP4/6/8 microscaling mapped to hardware; but a dangerous silent-TF32 default. | Q8 integer + libm-quarantined f32 reference (`precision.py`/`quantize.py`); no device half/fp8 matmul today. | **Triton on device numerics; the MXFP *format* is a portable migrate-idea.** |

---

## 2. Where the two systems actually touch (BCIR surfaces, anchored)

The overlap is exactly the **AI-kernel / tile-planning slice** — and BCIR already has an
analytical, proof-carrying analog for most of Triton's *planning* concerns, while Triton has the
*execution* BCIR lacks:

| BCIR component | Anchor | Triton analog | Maturity |
|---|---|---|---|
| `plan_matmul` tile/loop-order search + `cost_of` roofline (dual-semiring min,+/max,+; value-invariant vs `matmul_reference`) | `kbcir/matmul.py:160,123,43` | `@triton.autotune` over BLOCK_M/N/K + the compiler's `tl.dot` block-tiling | Mature analytic oracle — *picks* from a CPU-shaped roofline; never benchmarks a real GPU kernel |
| K_BCIR 12-d cost algebra + `TargetProfile` (incl. an `nvidia_ptx` warp profile) + `MemoryHierarchy` (L1..CXL/SSD) | `kbcir/cost.py:43,181,153` | *None direct* — Triton has no analytic cost model | Analytic where Triton is empirical |
| Learned tile prior (Q8 logistic, admissible early-exit + `TilePriorCertificate`, guided==exhaustive); `channel_prior` one level up | `kbcir/tile_prior.py:147,168`; `channel_prior.py:137` | ML-autotuner cost models (a research direction; stock Triton ships none) | **Novel & differentiating: learned-AND-proven, ordering-only, stale-refusing** |
| `gem.matmul` law-rail op + R22 native-tile law + R22/R23 gem seams | `BCIRGEMOps.td:53`; `BCIRVerifyPass.cpp:1638,1491` | `tt.dot` + its shape/type verification | Mature law rail (a 15×15-vs-16-native tile is refused at compile time) — records the *plan*, not executable |
| `gem.matmul_buffer` → tiled `scf.for` (memref load/store + `arith.mulf/addf`) | `BCIRLowerGemMatmulBufferPass.cpp:137-174` | TritonGPU lowering of `tt.dot` to tensor-core MMA | **Early** — scalar accumulation, no GPU, structural-test-only |
| LLM decode/serving rail (rmsnorm/rope/gqa/kv-cache/paged-kv) + law ops + decode-chain R22/R23 seam | `frontends/models/{decode,serve,paged_kv}.py`; `BCIRGEMOps.td:480,498` | FlashAttention / paged-attention Triton kernels | Correctness **reference** + legality seam, *not* a fused high-perf kernel |
| Resident-compiler codegen gate (emit C23/LLVM IR → clang/llc; real ELF for x86-64/aarch64/riscv64/bpf; PTX text via `llc`) | `lower/llvm.py:46`, `c_kernel.py:56`; `codegen/codegen.py:36,155` | Triton's TritonGPU→LLVM→PTX→ptxas→cubin | End-to-end **only for elementwise** kernels; nvptx64 PTX is asm-marker-checked, never run; matmul never reaches PTX |
| StreamPack binary ABI (magic/version/gens/CRC32; per-segment channel dispatch) | `gem/streampack.py:34,56`; `abi/streampack_abi.py:34` | Compiled-cubin + launch-metadata cache | Production-frozen versioned ABI — carries lane segments + generation tags, not cubins |
| Duration-aware scheduler (EFT waves, token-DAG pipelining) + `gem.prefetch`/`gem.schedule` pipeline_depth | `gem/schedule.py:63,70`; `BCIRGEMOps.td:24,34` | `num_stages` K-loop software-pipelining + `cp.async` + mbarrier | Analytic placement/makespan model + double-buffer ABI — *not* async-copy GPU codegen |
| Heterogeneous channel orchestration (per-backend `HardwareChannel`; one plan → one StreamPack across a tower) | `channels.py:59,47,92` | *None* — Triton is one GPU kernel per invocation | Architecture in place, but GPU/FPGA/NVMe channels are **modeled** (no resident driver) |

**BCIR has that Triton lacks**: separable proof-carrying legality (R1–R23); an analytic cost
algebra; learned-*and-proven* priors; two-truth quarantine (cost/layout/learning provably never
touch the legality verdict); provenance + generation validity (R11/R13, replay); value-invariance
proofs per realization; systems/driver + FFI breadth; a deterministic C23 subset; a frozen
StreamPack ABI; a cross-device transfer cost model; cost governance (RCSP budgets); and the
registry-first scope spanning kernels + compilers + drivers + systems.

**Triton has that BCIR lacks**: real GPU codegen to PTX/cubin; tensor-core/WGMMA/MMA lowering;
TMA/`cp.async` async copy; shared-memory allocation + swizzle + bank-conflict staging; the full
register/shared **layout algebra**; a hardware-executed pipelined K-loop; the block/warp/lane
programming model as *executed* code; a benchmark-driven autotuner that runs candidates on
silicon; mixed-precision GPU numerics (fp16/bf16/fp8 tensor-core accumulation); ptxas/driver/
launch integration + occupancy modeling; executed fused attention; and production adoption.

---

## 3. The migration ledger

### 3a. Migrate the IDEA (re-derive against BCIR's rails — not a code port)

Ranked by value × fit. "Re-derive" is load-bearing: several Triton passes are **not cleanly
separable** from the GPU layout machinery (in Triton, *coalesce IS layout assignment*), so
migration means re-implementing the algorithm against K_BCIR + the R-laws, not lifting code.

1. **Linear Layout F₂/GF(2) algebra** *(medium-high)* — the single most novel, portable Triton
   contribution: all layouts as linear maps over bit-vectors, conversion as B⁻¹∘A, proven closed
   under shape ops. It fits BCIR's proof-carrying/two-truth model *better than it fits Triton's
   otherwise-heuristic stack*. BCIR's affine `Ax+b` addressing is already a **superset** of the
   linear map the paper itself wishes it had (it wants the `+b` flip/slice offset). Migrate the
   *algebra* as an R-law-verified, cost-governed addressing module extending `kbcir/layout.py`;
   leave the warp-shuffle/bank-conflict *instantiation* on the GPU. Inherit its power-of-two-shape
   limitation (note the "Hexcute" critique for non-power-of-two / mixed-precision).
2. **OCP microscaling numerics** *(medium)* — MXFP4/6/8 block-scaled formats + a `dot_scaled`-style
   scaled-matmul op. Pure hardware-agnostic data format + dequant algorithm; lands in
   `precision.py`/`quantize.py` and extends `gem.matmul` under an R22-style law with block-scale
   operands, keeping scaling in the IR/cost model.
3. **Tile-peephole / Combine rewrite algebra** *(low-medium)* — the cleanest port: pure IR pattern
   rewrites, no layout dependence (dot+add fusion, nested `addptr` merge, `select+load`→masked-load,
   broadcast-constant fold). Re-derive as BCIR rewrites gated by the cost algebra. **First migrate
   to attempt.**
4. **Modulo-scheduling software-pipeliner formalism** *(medium)* — initiation-interval +
   resource-reservation tables to make StreamPack `pipeline_depth` *analytical*, parameterized by
   K_BCIR instead of Triton's acknowledged-brittle latency heuristics.
5. **Layout-propagation anchor-and-rematerialize** *(medium)* — the forward/backward
   remove-layout-conversions + reduce-data-duplication algorithm, ideal for K_BCIR + two-truth
   (a cost-reducing rewrite system). Caveat: not cleanly separable from coalesce in Triton.
6. **Shared-memory liveness allocation + barrier insertion** *(medium)* — a general staging/liveness
   algorithm useful if BCIR ever grows a staged-buffer target.
7. **Explicit precision/accumulator/determinism as a checkable law** *(low-medium)* — Triton
   exposes `input_precision` but with a silent-TF32 default and no enforcement; encode as a BCIR
   law so precision/determinism is *verified*, not hoped.
8. **KPerfIR/Proton-style profiling-as-an-MLIR-pass** *(medium)* — attribute measured cost back onto
   IR regions/costvec dimensions, letting a BCIR law assert *analytical ≈ measured* agreement —
   turns Proton/`do_bench` into **calibration for K_BCIR**, not a reimplementation target.
9. **Backend-plugin discovery pattern** *(low-medium)* — Triton's `triton.backends` entry-point
   registry (one `BaseBackend`+`DriverBase` pair per backend) is congenial to BCIR's registry-first
   design for its own codegen backends.
10. Lower-priority, GPU-cost-profile-only: a **wave-quantization / tail-occupancy** cost term and
    **`num_stages`/`num_ctas` as first-class planned axes** in `plan_matmul` — add analytically to
    the search space *if* BCIR grows a real GPU cost profile, rather than benchmarking them.
11. **Architectural patterns** (idea, not code): the **Gluon two-tier** design (an automatic tier +
    an explicit expert escape-hatch sharing one IR/cost model) reinforces BCIR's registry/two-truth
    separation; and the **`@triton.jit` block/pointer/mask authoring ergonomics** are worth
    borrowing as a kernel-authoring *frontend surface* in `bcir/frontends/` (crushes CUDA) — borrow
    the surface, not the GPU-bound lowering.

### 3b. Interop — reach Triton's ecosystem WITHOUT owning it

- **Consume Triton via `microsoft/triton-shared`** (`triton-to-linalg`): lower a real Triton kernel
  into standard MLIR (linalg/tensor/memref) that BCIR's law dialect, K_BCIR cost algebra, and
  planner can reason over — no core fork. *Lowest-risk first prototype.*
- **Register BCIR as an out-of-tree Triton backend** (the intel-xpu plugin pattern), or **emit
  Triton**, so BCIR rides existing NVIDIA/AMD/Intel vendor backends without maintaining them.
- **Treat Triton as one GPU codegen backend *below* the resident-compiler gate** — natural
  precisely because both emit LLVM IR and defer isel to LLVM/ptxas.
- **Be a TorchInductor source/sink**: consume Inductor-generated Triton kernels to put BCIR on the
  de-facto PyTorch GPU codegen path (~64M downloads/mo of reach) without inheriting the compiler.
- **Use Proton / `do_bench` / TritonBench** (183 GitHub + 166 PyTorch-aligned kernels) as
  empirical ground-truth harnesses to **calibrate and validate** the analytical K_BCIR costvec and
  to check `guided == exhaustive` selections against *measured* optima.
- **Use Triton kernel libraries** (FlashAttention-in-Triton, Liger-Kernel, FlagGems, vLLM
  paged-attention) as correctness oracles and performance baselines for BCIR's LLM decode/serving
  rail.

### 3c. Do NOT migrate (GPU-bound or paradigm-mismatched)

- **A wholesale fork** — perpetual LLVM-pinned rebase liability, wrong scope (see §0).
- **The TritonGPU middle-end** (hand-written layout assignment, coalescing, shared-mem alloc,
  software pipelining) and **PTX/AMDGCN emission** — GPU-microarchitecture-specific; reach via
  interop. (Nuance: this is the *middle-end*, not isel — LLVM/ptxas does isel.)
- **WGMMA / tcgen05 / TMEM / TMA / mbarrier / warp-spec knobs** — Hopper/Blackwell/AMD-bound
  lowering; belongs behind the resident-compiler boundary. Only the *async-DMA-with-completion-token*
  and *producer/consumer-overlap* **concepts** are already in BCIR (`gem/async_tokens.py`,
  `overlap.py`, `kbcir/dma.py`).
- **`@triton.autotune` empirical config search** — the anti-pattern BCIR's analytical cost +
  certificate-gated priors are designed to *supersede*. (But do not overclaim the replacement —
  it proves optimality-relative-to-BCIR's-model, not hardware-optimality.)
- **The SPMD "one program per block" imperative model + grid-lambda launch** — paradigm mismatch;
  BCIR is a declarative correspondence/registry IR; the launch grid is a backend concern.
- **Gluon's concrete GPU primitives** (`tcgen05_mma`, `allocate_tensor_memory`,
  `shared_memory_descriptor`, `warp_specialize`) — bound to Blackwell TMEM/warp/mbarrier hardware.
  Migrate the two-tier *pattern*, not these primitives.
- **Provably-optimal swizzle / warp-shuffle generation** — algorithm portable in principle, but its
  objective (bank conflicts, warp-shuffle exchange) is SIMT-lane-hardware-specific with no BCIR
  analog.
- **Persistent kernels + static launch grid + CUDA/HIP-graph capture** — tied to the GPU SPMD
  launch model; no clean BCIR analog.
- **LLM/RL-guided kernel-gen agents** (TritonForge, AutoTriton, GEAK, KernelBand) — non-deterministic
  trial-and-error with no correctness or optimality guarantee — incompatible with BCIR's
  certificate-gated, proof-first philosophy. Their *benchmark suites* are interop targets; the
  *agents* are not.
- **The experimental Triton-CPU backend** — WIP/unreleased and redundant with BCIR's freestanding C
  twins + resident-compiler C23/LLVM path, which already covers CPU as a first-class target.

---

## 4. Direct answers to the three questions

1. **Fork Triton into BCIR?** **No.** The economics (LLVM-pinned, PyTorch-cadence rebase treadmill),
   the scope (its entire content is GPU middle-end + numerics BCIR reaches through the resident
   compiler), and the philosophy (Triton *already* delegates isel to LLVM — forking gains BCIR
   nothing on its actual codegen gate) all argue against it. A fork is legal (MIT) and pointless.
2. **Incorporate it another way?** **Yes — interop, not fork.** In priority order: a `triton-shared`
   → linalg ingestion prototype (consume one real kernel through BCIR's law/cost/planner rails);
   BCIR-as-an-out-of-tree-Triton-backend / Triton-as-a-resident-GPU-backend; TorchInductor
   source/sink; and TritonBench/Proton as **calibration harnesses** for K_BCIR.
3. **Superseded, or migrate?** **Not superseded** — Triton ships real multi-vendor GPU code at scale
   that BCIR cannot yet emit. Triton's MLIR *approach* is, if anything, a **model to interoperate
   with**, and its layout work is *ahead* of BCIR on GPU depth. **Migrate a subset, not all**: the
   Linear Layout F₂ algebra (re-derived as a proof-carrying addressing module — the one genuinely
   unique, portable idea), MXFP microscaling numerics, the tile-peephole rewrites, and the
   modulo-scheduling formalism. Everything hardware-bound stays on the far side of the
   resident-compiler gate.

---

## 5. Recommended next steps (ranked)

1. **Lowest-risk reach play first**: stand up a `triton-shared` ingestion prototype — lower one real
   Triton kernel to linalg and run it through BCIR's law/cost/planner rails for cross-validation;
   and/or register BCIR as an out-of-tree Triton backend.
2. **Prototype the Linear Layout F₂ algebra** as an R-law-verified addressing module extending
   `kbcir/layout.py`, unified with BCIR's affine `Ax+b` per-resource addressing (BCIR is already
   ahead on the offset term the paper wants); document the power-of-two / Hexcute limits.
3. **Add MXFP4/6/8 block-scaled numerics** to `precision.py`/`quantize.py` + a scaled `gem.matmul`
   variant under an extended R22 law, validated bit-exact against a C-twin reference.
4. **Stand up TritonBench + Proton/`do_bench` as an empirical calibration harness for K_BCIR** —
   quantify the analytical-vs-measured gap on real GPU kernels; use it to *sharpen, not oversell*
   the certificate story (target a KPerfIR-style law asserting analytical ≈ measured).
5. **Decide the GPU-rail posture explicitly**: commit to interop (emit Triton / ride vendor
   backends) **or** fund a real tensor-core lowering — and until PTX actually executes and
   self-checks on a device, stop asserting GPU parity in any external framing.
6. **Port the Combine tile-peephole rewrites** as the first, cleanest migrate (pure IR patterns, no
   layout dependence), gated by the cost algebra.

---

## 6. Messaging discipline (the corrections, restated so they don't drift back)

- **"BCIR is broader; Triton is deeper on GPU."** Not "BCIR contains Triton." Migrating Linear
  Layouts narrows but does not close the GPU-depth gap (BCIR still lacks real tensor-core lowering).
- **"Analytical + certificate-gated, cost-quarantined-from-legality."** Not "BCIR proves; Triton
  searches." The certificate is optimality-relative-to-BCIR's-model; Triton's `do_bench` is hardware
  ground truth. Complementary.
- **"Triton is architecturally aligned with the resident-compiler gate, not behind it."** It
  delegates isel to LLVM/ptxas — the same philosophy. Any fork-avoidance argument built on an "isel
  collision" is factually wrong; base it on scale / cadence / GPU-specificity / scope.
- **BCIR's GPU codegen is not real today.** Matmul is scalar `scf.for`; PTX is elementwise-only and
  never executed. No GPU-parity claim is defensible until it runs and self-checks on silicon.
