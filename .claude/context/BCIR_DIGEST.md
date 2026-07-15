# BCIR repo digest (compressed knowledge base — read this INSTEAD of exploring)

<!-- KNOWLEDGE:BEGIN -->
## Curated knowledge (current through PR #638 and package 0.2.0, 2026-07-15)

**What BCIR is.** A registry-first, phase-ordered, lane-typed, cost-governed
correspondence IR: `K_BCIR(G|H,Θ) = min_π M(π,Θ) s.t. R(π,Θ) ⪯
B(H,Θ)`. It selects a legal realization under a 12-dimensional integer/Q8 cost
vector, live pressure Θ, and RCSP budgets. BCIR is a planning, verification, artifact,
and runtime layer above resident LLVM/GCC/vendor toolchains; it is not a replacement
instruction selector by default.

**Implementation rails.**
- `bcir/` is the dependency-free Python conformance oracle: R1–R23 semantics,
  K_BCIR planning/certified learned organs, GEM scheduling and StreamPack v1–v3,
  frontends/ETL, telemetry, model/training references, BCIRQ8, and hosted lowering.
  Python LLVM AOT/JIT accepts exactly one supported elementwise claim; arbitrary graphs
  are rejected rather than truncated.
- `mlir/` is the ODS/TableGen/C++ law rail: verifier and optimizer passes, partial AOT
  preparation, IRDL projection, and typed x86 entry/descriptor/segment/ordinary-interrupt
  edges. `bcir-aot` may intentionally leave residual BCIR/GEM operations.
- `runtime/c/` contains three enforced classes: heap-free freestanding code; hosted
  compiler/model tools with allocator injection and fail-every-allocation tests; and
  driver adapters using handles/offsets rather than cross-boundary pointers. It includes
  the C-front/compiler rail, RuntimeChannel v1 loopback, StreamPack/telemetry codecs, and
  standalone BCIRQ8 Llama inference. `runtime/cpp/` is a narrow orchestration seam;
  single-node dispatch is real and dynamic/distributed backends remain explicit stubs.
- `llvm-training/` is a standalone LLVM/MLIR curriculum and evaluation corpus, never a
  runtime dependency. Its aggregate CMake gate is bounded to two lit workers.

**Contracts and ownership.** `docs/BCIR_LANGREF.md` is normative and owns BCIRQ8 v1.
`docs/BCIR_MASTER_ROADMAP.md` owns portfolio order, not history. `docs/kernel/` owns
driver/kernel/StreamPack/telemetry contracts; `docs/machine-learning/` owns model and ML
programs; `docs/languages/` owns language rails and C memory discipline;
`docs/research/` holds comparative/feasibility evidence. Counts live only in generated
`docs/STATUS.md`; chronology lives in `docs/DEVELOPMENT_HISTORY.md` and git.

**Landed versus open.** Typed x86 long-mode entry, descriptor/task/segment operations,
and an ordinary interrupt-frame trampoline are assemble-smoke-gated. RuntimeChannel v1,
event phases, DMA descriptors, frozen Q8 priors, telemetry registry/frame/metrics/export
serialization, and the pinned TinyLlama BCIRQ8→standalone-C parity gate are code-backed.
Resident UART/virtio/device drivers, Linux modules/UAPI, BCIR-Linux, native IPC/kernel,
live telemetry transports, and reset/paranoid-exception execution remain unbuilt or
hardware-gated. UART and virtio-blk evidence must precede UAPI v1; direct behavior must
stabilize before any IPC transport.

**ML-substrate closure.** Exact-width low-bit lanes/group-32 quantization (A1),
deterministic matmul schedule search (B1), closed-set content-addressed reverse AD (B3),
and the current Area-B library wrappers (B5) landed. Packed INT2–INT6 compute/wire
formats, activation quantization/outlier handling, portable schedule export plus measured
hardware evidence, and general higher-order/control-flow AD remain in
`docs/machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md`.

**Release and validation discipline.** Package `0.2.0` is current; `0.3b` is an
unreleased draft. Two-truth quarantine keeps learned policy off legality. Measured and
modeled evidence must stay labeled. Before publication run the complete local x86
inventory with at most two workers, C sanitizer/fuzz/memory gates, coherent LLVM/MLIR
gates, model parity, docs governance, and `git diff --check`; native Windows and ARM are
CI/hardware gates, not emulated locally. Do not count a capability skip as hardware proof.

**History and agent gotchas.** PRs #2–#638 cover the C++ seed, LLVM-first period, PR #153
oracle/law pivot, optimizer and C-front arcs, ML/telemetry/driver foundations, correctness
and portability work, C memory discipline, and driver/kernel roadmap reconciliation.
Quick tier intentionally hides toolchains; thorough must use one coherent LLVM major.
The default target is host-adaptive. Keep local concurrency at two and never launch
unbounded fuzzing, inference, emulation, or nested build loops.
<!-- KNOWLEDGE:END -->

## Generated inventory (do not edit — rebuild with build_digest.py)

Top-level: ./bcir ./channels ./docs ./llvm-training ./mlir ./runtime ./tools

### STATUS.md counts (generated source of truth)
| Metric | Value |
|---|---|
| Static Python `test_*` function inventory | **2071** across 178 files |
| Static MLIR ODS op-definition inventory (`mlir/include/BCIR/*.td`) | **116** |
| Static registered-pass inventory | **37** |
| Static MLIR fixture inventory (`mlir/test/`) | **107** files; 205 `expected-error` markers |
| Static runtime C source/header inventory (`runtime/c/`) | **256** files |
| Verifier-law negative-fixture tag inventory | **R1–R23** (23/23 present) |
| Registered hardware-channel inventory | **9** (cpu, fpga, gpu, memory, storage) |
| R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | R16 | R17 | R18 | R19 | R20 | R21 | R22 | R23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Channel | Kind |
|---|---|
| `arm64_neon` | cpu |
| `arm64_sve` | cpu |
| `fpga_systolic` | fpga |
| `hbm_pim` | memory |
| `nvidia_ptx` | gpu |
| `nvme_stream` | storage |
| `riscv_rvv` | cpu |
| `x86_avx2` | cpu |
| `x86_avx512` | cpu |

### docs/ heading map
- **docs/BCIR_LANGREF.md** (1105L): 0. Stance · 1. The multi-level IR · 2. Central equation · 3–9. Laws (summary) · 10. Verifier laws (R1–R23) · 11. Rewrite laws (the building-blocks engine) · 12. Lowering contracts · 13. Learning placement (normative policy) · 14. The two-truth separation (MOPC) · 15. The enriched-operad memory interface (the higher intelli · 16. BCIRQ8 v1 decoder-artifact contract · 17. Conformance profiles and external-contract boundary · 18. Thesis
- **docs/BCIR_MACHINE_CODE_HAL_ISA_AUDIT.md** (215L): 0. Executive verdicts · 1. The machine model as built (the "BCIR ISA") · 2. The toolchain as built · 3. What the research docs add to the frame · 4. HAL/ABI as built · 5. The ABI ledger (what is frozen today) · 6. The gap register — the MC-track (code-backed) · 7. Standing positions this audit confirms (no change needed)
- **docs/BCIR_MASTER_ROADMAP.md** (328L): 1. Mission and non-negotiable invariants · 2. Architecture and current baseline · 3. Dependency order · 4. Active workstreams · 5. Program milestones · 6. Release policy · 7. Validation and publication gate · 8. Decision boundaries · 9. Risk register · 10. Document ownership · 11. Immediate priority queue
- **docs/BCIR_NATIVE_OBJECT_GATE.md** (143L): 1. The decision · 2. The warranted slice (done): real native objects end-to-en · 3. GO criteria — what would warrant BCIR-native isel · 4. STOP criteria — if a native-isel experiment is taken · 5. Current verdict and evidence boundary
- **docs/BCIR_Repo_Structure.md** (171L): 1. Top-level ownership · 2. Oracle package (`bcir/`) · 3. Law rail (`mlir/`) · 4. C and C++ runtime classes · 5. Contract ownership · 6. Documentation taxonomy · 7. Build and validation entry points · Fast dependency-free oracle tier · Full local oracle/toolchain tier, with bounded concurrency · Production C and C++ boundaries · MLIR/IRDL rails when the coherent LLVM toolset is installed · Documentation governance · 8. Change-placement rules
- **docs/DEVELOPMENT_HISTORY.md** (355L): 1. The development method · 2. The PR arc (eras) · 3. Condensed dated changelog · 4. Capability closure ledger migrated from the former master · 5. Where the detailed notes live now
- **docs/ONBOARDING_DEEP_DIVE.md** (275L): 1. Read this first · 2. The three implementation rails · 3. From source to execution · 4. Core semantic and optimizer packages · 5. Frontends, lowering, and machine boundary · 6. Runtime memory and ownership · 7. Models, training, and BCIRQ8 · 8. Drivers, kernel, telemetry, and IPC · 9. Current evidence boundary · 10. Validation workflow · 11. Reading and change-placement map
- **docs/PARITY.md** (360L): Enum value parity (normative) · Concept parity · Python ↔ C artifact and runtime parity · Python ↔ C frontend twin (`runtime/c/`) · Worked-example parity · Generated, adversarial parity (the proof, not the hope) · How parity is enforced today
- **docs/RELEASE_NOTES_0.3b.md** (98L): Candidate baseline already landed · Release blockers · Explicit non-goals · Candidate validation
- **docs/REPO_CURRENT_STATE_AUDIT.md** (180L): Snapshot · Confirmed strengths · Confirmed limitations · Recommended next milestones · Changelog
- **docs/STATUS.md** (50L): Verifier-law negative-fixture inventory (R1–R23) · Hardware channel / target matrix · Runtime C components
- **docs/VISION_ALIGNMENT_AUDIT.md** (196L): 1. Thesis under audit · 2. Scorecard · 3. C as registry definition and macro target · 4. IR ownership, machine edges, and backend boundary · 5. Certified optimization and AI substrate · 6. Model inference, training, and C++ boundary · 7. Driver, kernel, telemetry, and IPC alignment · 8. Highest-leverage remaining work · 9. Bottom line
- **docs/kernel/BCIR_AMD_AI_DRIVER_ROADMAP.md** (384L): 0. Executive strategy · 1. The honest starting point · 2. The vertically-integrated stack · 3. The phased build order · 4. The three device classes (never one) · 5. The per-project interop ledger · 6. The ML-framework supplement boundary · 7. The deferred Phase-0 Linux inheritance (scope, not build) · 8. Risks / messaging discipline · 9. Recommended next steps (ranked)
- **docs/kernel/BCIR_DRIVER_KERNEL_ROADMAP.md** (678L): 1. Mission and product split · 2. Current baseline · 3. The BCIR driver package contract · 4. Execution, telemetry, and continual optimization · 5. Driver maturity and build order · 6. BCIR-Linux: compatibility oracle and experimental fork · 7. Universal ABI, POSIX compatibility, and IPC · 8. Validation and promotion policy · 9. Risks and stop conditions · 10. Historical rationale retained from roadmap v1
- **docs/kernel/BCIR_STREAMPACK_ABI.md** (135L): Conventions · Header (64 bytes, cache-line aligned) · Body (sequential, length-prefixed) · Trailer · v2 (append-only): pipelined phases + double-buffer prefetch · v3 (append-only): on-wire segment dispatch + channel · Semantic trust boundary (R10/R11 in C) · Versioning (the freeze) · Why a frozen ABI now
- **docs/kernel/BCIR_UART_DRIVER_BLUEPRINT.md** (1048L): 0. How to use this document (for the implementing model) · 1. The normative 16550 device model · 2. The variant matrix (what the registry must parameterize) · 3. Field-reality quirks (research; not in any datasheet) · 4. Architecture: how each piece maps onto existing BCIR mach · 5. The build slices · U8 adds: "tl16c750" (mode64_key="dlab", rx_triggers_alt=(1,1 · "tl16c750e" (fifo_depth=128, rx_triggers=(1,4,120,124), tx_t · flow_mech="efr_tcr", has_tlr/has_frac_divisor/has_sleep=True · reset_honest=False), · "h16750s_64" (a chosen synthesis point: fifo_depth=64, rx_tr · "lattice16550_lmmi" + "lattice16550_apb" (has_scr=False, fcr · reset_honest=False; the APB one is stride=4).
- **docs/kernel/HARDWARE_VALIDATION.md** (104L): What IS validated here (real, measured — `bcir/tests/test_si · What is BLOCKED in this sandbox (and why) · The rig required for FULL hardware validation · The runbook (push-button) · Honest status line
- **docs/kernel/HETEROGENEOUS_CHANNELS.md** (152L): The problem it solves · The abstraction (`bcir/channels.py`) · The unified core — every channel plans the same way · Heterogeneous orchestration — one binary graph across the to · Cross-device placement cost (fabric/sync) · Adding a backend (the extension path) · Status
- **docs/kernel/SIGNAL_REGISTRY.md** (120L): What it is · Core types · Providers · Builders + the channel↔provider mapping · Honest real/unavailable split (typical sandbox) · Status and pre-driver boundary
- **docs/kernel/SYCL_INTEROP.md** (121L): Resident dispatch (the channel executes) · The bright line: SYCL is a compiler MODE, not a `c.call.libm
- **docs/kernel/TELEMETRY_FRAME_ABI.md** (153L): Conventions · Frame · Resync semantics · Host decode reuses RT3 (two-truth) · Frozen-v1 scope and pre-driver extension · Egress over UART (documented adapter, not built here)
- **docs/kernel/TELEMETRY_PIPELINE_RESEARCH.md** (298L): 0. What BCIR already has (the substrate this builds on) · 1. Layered model of telemetry sources (where each tool sits) · 2. Metric taxonomy → BCIR cost dimensions · 3. Abstractions worth copying (the design DNA) · 4. Gaps to add to BCIR's existing surfaces · 5. Recommended architecture (vendor-neutral, two-truth-safe) · 6. Suggested build order (each a gated segment) · 7. Driver/kernel integration gate · Sources
- **docs/languages/CFRONT_GUIDE.md** (337L): Quickstart · compile a file (verified C + R1–R18 status to stdout) · syntax/semantic check only — Clang-style diagnostics, no out · machine-readable diagnostics (for editors / CI) · lay the types out for another target ABI · graceful degradation: report a fallback-to-LLVM signal inste · Command-line options · Diagnostics · The target ABI matrix · The LLVM-backend fallback contract · Pointer-bounds policy (LangRef §4) · Pointer-lifetime policy (R21, LangRef §10) · Inline assembly (ASM1)
- **docs/languages/CPP_HANDOFF_BOUNDARY.md** (243L): Honest depth (read this first) · Why a boundary at all · What STAYS on the C/IR rail (below the boundary) · What CROSSES to C++ (above the boundary) · The seam · Why C++ (and not C / IR) · The scaffold (what is built) · Risks / follow-ups (what a real distributed/dynamic implemen
- **docs/languages/C_MEMORY_DISCIPLINE.md** (87L): Runtime classes · Required ownership rules · Direct driver ABI first · IPC boundary · Validation cadence
- **docs/machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md** (857L): 0. Stance — why an IR becomes intelligence · 1. The intelligence already in BCIR (the substrate this buil · 2. The ordered build-out · 3. The continuous-development discipline (how every layer ke · 4. Capability-track placement (do we need CT6 / CT7?) · 5. Risk register / honest boundaries (out of the dreamy pote · 6. AI-substrate closure register · 7. Open-weight model ingestion (GLM / Gemma / Qwen) — the LL · 8. Feasibility audit — the deeper-integration program (2026-
- **docs/machine-learning/BCIR_WHOLE_MODEL_REFERENCE.md** (166L): 1. The overlap — BCIR already owns the reference numerics · 2. What landed and what remains · 3. What NOT to import · 4. Where each piece lands (anchored homes) · 5. Build-slice status (WMR-1 … WMR-4) · 6. The larger implication — closing the train → export → ser · 7. Coherence with the rest of the system
- **docs/machine-learning/ML_LANGUAGE_PLACEMENT_ANALYSIS.md** (394L): 1. Executive summary — the thesis · 2. The five placement criteria · 3. The four language tiers · 4. The classification table (the heart) · 5. The migration map · 6. Conclusion — the clean hierarchy
- **docs/machine-learning/OPENAI_BCIR_INTEGRATION_RESEARCH.md** (455L): 1. Repository capability map · 2. Current OpenAI developer capability research · 3. How deep ChatGPT can integrate into BCIR · 4. Proposed architecture · 5. Proposal versions · 6. Recommended next implementation steps · 7. Core conclusion
- **docs/machine-learning/THIRD_PARTY_MODELS.md** (25L): Maykeye/TinyLLama-v0
- **docs/research/BCIR_GAME_OPTIMIZATION_ROADMAP.md** (438L): 1. The exact-vs-approximate split — the load-bearing thesis · 2. The overlap — what BCIR already embodies (map, don't re-b · 3. Per-game principles — the full ledger · 4. Lessons applied to **GEM** (the StreamPack hot path) · 5. Lessons applied to **K_BCIR** (the tropical cost model, e · 6. Lessons applied to the **StreamPack ABI** (frozen artifac · 7. Ranked build slices · 8. Risks & myth-flags · 9. The bottom line
- **docs/research/BCIR_NATIVE_BACKEND_FEASIBILITY.md** (217L): 1. What "native backend" means here · 2. Current state — the codegen spectrum BCIR already populat · 3. What a *general* native backend requires (and why it is e · 4. The gate, restated and assessed (status: all GO criteria  · 5. The candidate bounded targets, priced and ranked · 6. Development roadmap (executed ONLY if the gate opens for  · 7. What to do *now* (and how it de-risks any future native w · 8. Bottom line
- **docs/research/BCIR_TRITON_COMPARATIVE_ANALYSIS.md** (271L): 0. Executive verdict · 1. The comparison matrix · 2. Where the two systems actually touch (BCIR surfaces, anch · 3. The migration ledger · 4. Direct answers to the three questions · 5. Recommended next steps (ranked) · 6. Messaging discipline (the corrections, restated so they d
- **docs/research/CLANG_COMPARISON.md** (98L): The fair frame · Results · Where we WIN · Where we MATCH · Where we LOSE (honest) · Bottom line
