# BCIR repo digest (compressed knowledge base — read this INSTEAD of exploring)

<!-- KNOWLEDGE:BEGIN -->
## Curated knowledge (distilled from a full-tree + full-PR-history deep dive, 2026-07-01)

**What BCIR is.** A registry-first, phase-ordered, lane-typed, cost-governed correspondence IR:
`K_BCIR(G|H,Θ) = min_π M(π,Θ) s.t. R(π,Θ) ⪯ B(H,Θ)` — tropical (min,+) selection of a legal
realization plan π under a 12-d integer/Q8 cost vector (compute, memory, fabric, sync, compile,
thermal, power, reliability, security, accuracy, contention, verification), live Θ pressures, and
RCSP budgets. Positioning: a planning+verification layer ABOVE LLVM (never a Clang replacement);
wins = gather avoidance ~6–16×, budget feasibility as correctness; matches Clang on dense.

**Three rails, one law.**
- `bcir/` — Python conformance oracle (dep-free, deterministic). model/=BCIR-0..2 (Lane U/UX/T/GGG/A/H,
  StrideClass, Domain, 18 opcodes, Resource/Claim/Phase/Module, RID registry, optional Timing/Lifetime).
  kbcir/=BCIR-3 (cost.py 12-d CostVector+couple(Q8), realize.py legal candidates+min-plus DAG,
  rcsp.py label-DP+Pareto, softdp T-annealed twin, accel exact B&B+frozen ranker, microbench/bayescal/
  calibrate→FrozenCalibrator, moegate GNN→FrozenGate, egraph/memory fixpoints, portfolio+replay gate,
  regret+MDL ΔL, provenance FNV manifest, proof explain/replay/reduce, twotruth quarantine, compose
  region trees, bundle, fusion, layout, cache_predict, precision R17, quantize; ML: autodiff closed set
  {const,var,neg,add,sub,mul,div,dot,select} machine-proven, losses, training, ols/pca/transformer/
  recurrent/classical/unsupervised). gem/=BCIR-4 (StreamPack hydrate v1..3, deterministic executor,
  concurrency waves+GGG tail, EFT/token scheduling, overlap M(π,Θ): makespan+gain==serial, cim, dvfs).
  lower/=BCIR-5 (llvm.py ONE elementwise claim shape→.ll; jit lli; wasm; c_kernel.py wide portable C23
  emitters + Area-B wraps BLAS/FFTW/LAPACK/GSL/SLEEF/libcerf; stackify JVM exec-validated, CIL skip).
  channels.py 9 channels (6 real+3 modeled: fpga_systolic/nvme_stream/hbm_pim), channel_plugin.json v1.
  Pinned: vector_add avx512 cool latency = 7808 vec16; budget 700 → 9472 vec8; hot → 13952.
- `mlir/` — the law: 102 ODS ops, 15 enums, 4 attrs (Precision/Timing/Lifetime/CostVector), 4 types
  (only !bcir.token live), 37 passes + 5 pipelines (audit/optimize/hydrate/lower-llvm/aot).
  -bcir-verify = R1–R21 (~1400 lines, BCIRVerifyPass.cpp); optimizer core bit-exact vs oracle;
  asm-edge ops bcir.asm/portio/volatile_load/store/creg_*/msr_* → llvm.inline_asm, assemble-smoke-gated.
  irdl/ = structural-only projection for stock mlir-opt (dots→underscores, no c_pred). LLVM 22 gating.
- `runtime/c/` — 233 components. Freestanding: bcir_runtime/exec/encode/hydrate/plan/binrec/
  telemetry_frame (+q8 tables #embed). Host: bcir_cpp→bcir_cfront (5.3k LOC twin)→bcir_verify→bcir-cc;
  quarantine BCIR_CHK (read clampable via recorded decide) / BCIR_CHK_W (write noreturn). 182 cfront_*.c
  fixtures; parity = 9-count summary + FNV-1a structural digest byte-identical to oracle; 7 CRC-fixed
  corruption classes → exact BCIR_ERR_*. runtime/cpp/ = G8 seam (single-node real, dyn/dist stubs).

**Non-negotiables.** Two-truth quarantine (graded informs, never verdicts; L0 no learned inference,
L1 frozen Q8, L2 replay-gated, L3 human-actuated ΔL); prototype-then-port with parity gates;
provenance digest = plan commit hash; new laws vacuous-by-default (non-disturbance); native isel
DEFERRED behind gate; measured-vs-modeled honesty (the one deferred result: measured bare-metal replan).

**llvm-training/** — separate 612-file agent context pack (NOT the IR): 20 modules, bcir-mapping/,
42 exercises + autograder (7 executable-registered), 42-record dataset, EVAL 30/30 answer key.
24 files carry retired-material banners (they reference the removed early LLVM-IR-schema runtime tree). Stale: its README "Seed
(~40 files)" table; tools/README semantic-only marker spelling.

**History (PRs #2–#607, 2026-04-23→07-01, ~606 PRs; see docs/DEVELOPMENT_HISTORY.md).**
#2–12 C++ skeleton (1 day) → #13–31 LLVM-first seed → #32–152 llvm-training → **#153 pivot**
(oracle+law+PARITY) → #163–211 R1–R12 + Phases 12–26 organs → #212–253 C++ optimizer port+LLVM22 →
#254–262 ARM/channels/governance → #263–510 cfront arc (~250 PRs, six-artifact gate, twin port #266,
fuzzer bug-hunt #428–449, R19–21 seeded) → #511–555 ML roadmap, R19–21 promoted (#514–5), RT1–7
red-team, vision audit #539 + G1–G8 → #556–604 T1–4 telemetry, M1–3+E1–7 ML, ASM/SEG/D driver arc →
#605–607 OpenAI research. Test ratchet 25→~1846.

**Gotchas for agents.** Counts live in generated docs/STATUS.md (gen_status.py --check is a CI gate;
also check_links.py + check_retired_paths.py). Quick test tier hides the toolchain (19 cfront
"failures" on default run = toolchain-gated, not real). Concurrency: 4 cores → agent cap 2.
Entry function = LAST function in a cfront unit; loop cost bound = 1024; default target host-adaptive.
<!-- KNOWLEDGE:END -->

## Generated inventory (do not edit — rebuild with build_digest.py)

Top-level: ./bcir ./channels ./docs ./llvm-training ./mlir ./runtime ./tools

### STATUS.md counts (generated source of truth)
| Metric | Value |
|---|---|
| Python conformance tests (`python -m bcir.tests.run_all`) | **1846** across 146 files |
| MLIR ODS ops (`mlir/include/BCIR/*.td`) | **102** |
| Registered `-bcir-*` passes | **37** |
| MLIR FileCheck tests (`mlir/test/`) | **94** (170 `expected-error` negatives) |
| Runtime C components (`runtime/c/`) | **233** |
| Verifier laws | **R1–R21** (21/21 covered) |
| Hardware channels | **9** (cpu, fpga, gpu, memory, storage) |
| R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | R16 | R17 | R18 | R19 | R20 | R21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
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
- **BCIR_DRIVER_KERNEL_ROADMAP.md** (303L): 0. The one-paragraph orientation · Part I — The pre-driver hardening gate (Phase 0: do this fir · Part II — The language-placement model (the "more important  · Part III — Kernel/firmware dependency ordering (what's befor · Part IV — The sequenced slice plan · 5. Decision summary
- **BCIR_LANGREF.md** (748L): 0. Stance · 1. The multi-level IR · 2. Central equation · 3–9. Laws (summary) · 10. Verifier laws (R1–R21) · 11. Rewrite laws (the building-blocks engine) · 12. Lowering contracts · 13. Learning placement (normative policy) · 14. The two-truth separation (MOPC) · 15. The enriched-operad memory interface (the higher intelli · 16. Milestone map · 17. Thesis
- **BCIR_MASTER_ROADMAP.md** (2158L): 1. What BCIR is (positioning) · 2. Current state at a glance (measured) · 3. The MLIR / C / C++ / Python placement map (the two-truth  · 4. What's done (the landed work) · 5. The forward roadmap (what's next) · 6. Next build steps (concrete, prioritized) · 7. Release ladder (reconciled) · 8. Risk register · Appendix A — capability tracks & the build history (what's b · Appendix B — what was consolidated / removed
- **BCIR_ML_AI_INTEGRATION_ROADMAP.md** (443L): 0. Stance — why an IR becomes intelligence · 1. The intelligence already in BCIR (the substrate this buil · 2. The ordered build-out · 3. The continuous-development discipline (how every layer ke · 4. Capability-track placement (do we need CT6 / CT7?) · 5. Risk register / honest boundaries (out of the dreamy pote · 6. Where to start (the first concrete, gateable slices) · 7. Open-weight model ingestion (GLM / Gemma / Qwen) — the LL
- **BCIR_NATIVE_BACKEND_FEASIBILITY.md** (210L): 1. What "native backend" means here · 2. Current state — the codegen spectrum BCIR already populat · 3. What a *general* native backend requires (and why it is e · 4. The gate, restated and assessed (status: all GO criteria  · 5. The candidate bounded targets, priced and ranked · 6. Development roadmap (executed ONLY if the gate opens for  · 7. What to do *now* (and how it de-risks any future native w · 8. Bottom line
- **BCIR_NATIVE_OBJECT_GATE.md** (135L): 1. The decision · 2. The warranted slice (done): real native objects end-to-en · 3. GO criteria — what would warrant BCIR-native isel · 4. STOP criteria — if a native-isel experiment is taken · 5. Current verdict
- **BCIR_Repo_Structure.md** (164L): Problem this structure solves · Top-level separation · The IR pipeline and section ownership · IRDL vs MLIR vs LLVM — what each builds, and why they are se · How separation is enforced · Migration notes (this reorg) · Build matrix (current — post-fold) · bcir/ -- the oracle (no third-party deps; CI jobs oracle / c · mlir/ -- the dialect law (needs libmlir-NN-dev + llvm-NN-dev · One tree (the `ir/` fold is complete) · Documentation inventory (`docs/`)
- **BCIR_STREAMPACK_ABI.md** (120L): Conventions · Header (64 bytes, cache-line aligned) · Body (sequential, length-prefixed) · Trailer · v2 (append-only): pipelined phases + double-buffer prefetch · v3 (append-only): on-wire segment dispatch + channel · Semantic trust boundary (R10/R11 in C) · Versioning (the freeze) · Why a frozen ABI now
- **CFRONT_GUIDE.md** (311L): Quickstart · compile a file (verified C + R1–R18 status to stdout) · syntax/semantic check only — Clang-style diagnostics, no out · machine-readable diagnostics (for editors / CI) · lay the types out for another target ABI · graceful degradation: report a fallback-to-LLVM signal inste · Command-line options · Diagnostics · The target ABI matrix · The LLVM-backend fallback contract · Pointer-lifetime policy (R21, §5.12) · Inline assembly (ASM1) · Port-mapped I/O (ASM2)
- **CLANG_COMPARISON.md** (98L): The fair frame · Results · Where we WIN · Where we MATCH · Where we LOSE (honest) · Bottom line
- **CPP_HANDOFF_BOUNDARY.md** (242L): Honest depth (read this first) · Why a boundary at all · What STAYS on the C/IR rail (below the boundary) · What CROSSES to C++ (above the boundary) · The seam · Why C++ (and not C / IR) · The scaffold (what is built) · Risks / follow-ups (what a real distributed/dynamic implemen
- **DEVELOPMENT_HISTORY.md** (267L): 1. The development method · 2. The PR arc (eras) · 3. Condensed dated changelog · 4. Where the detailed notes live now
- **HARDWARE_VALIDATION.md** (104L): What IS validated here (real, measured — `bcir/tests/test_si · What is BLOCKED in this sandbox (and why) · The rig required for FULL hardware validation · The runbook (push-button) · Honest status line
- **HETEROGENEOUS_CHANNELS.md** (152L): The problem it solves · The abstraction (`bcir/channels.py`) · The unified core — every channel plans the same way · Heterogeneous orchestration — one binary graph across the to · Cross-device placement cost (fabric/sync) · Adding a backend (the extension path) · Status
- **ML_LANGUAGE_PLACEMENT_ANALYSIS.md** (390L): 1. Executive summary — the thesis · 2. The five placement criteria · 3. The four language tiers · 4. The classification table (the heart) · 5. The migration map · 6. Conclusion — the clean hierarchy
- **ONBOARDING_DEEP_DIVE.md** (535L): 0. The one-paragraph thesis · 1. The repository is two separate things · 2. The semantic model (`bcir/model/`) · 3. The K_BCIR optimizer core (`bcir/kbcir/`) · 4. The intelligence layer — CT5 learned organs (`bcir/kbcir/ · 5. GEM, ETL, frontends, lowering (`bcir/gem`, `etl`, `fronte · 6. The verifier laws (`bcir/verify/`, `mlir/lib/passes/BCIRV · 7. The MLIR dialect — the law (`mlir/`) · 8. The C frontend / driver (`runtime/c/`, `bcir-cfront` / `b · 9. The C runtime — StreamPack, channels, native-object gate  · 10. The development history · 11. Current state & verification · 12. LLVM training — the corpus
- **OPENAI_BCIR_INTEGRATION_RESEARCH.md** (455L): 1. Repository capability map · 2. Current OpenAI developer capability research · 3. How deep ChatGPT can integrate into BCIR · 4. Proposed architecture · 5. Proposal versions · 6. Recommended next implementation steps · 7. Core conclusion
- **PARITY.md** (334L): Enum value parity (normative) · Concept parity · Python ↔ C frontend twin (`runtime/c/`) · Worked-example parity · Generated, adversarial parity (the proof, not the hope) · How parity is enforced today
- **REPO_CURRENT_STATE_AUDIT.md** (139L): Snapshot · Confirmed strengths · Confirmed limitations · Recommended next milestones (see [`BCIR_MASTER_ROADMAP.md`]( · Changelog
- **SIGNAL_REGISTRY.md** (106L): What it is · Core types · Providers · Builders + the channel↔provider mapping · Honest real/unavailable split (typical sandbox) · Next (T2–T4)
- **STATUS.md** (50L): Verifier law coverage (R1–R21) · Hardware channel / target matrix · Runtime C components
- **SYCL_INTEROP.md** (121L): Resident dispatch (the channel executes) · The bright line: SYCL is a compiler MODE, not a `c.call.libm
- **TELEMETRY_FRAME_ABI.md** (109L): Conventions · Frame · Resync semantics · Host decode reuses RT3 (two-truth) · Egress over UART (documented adapter, not built here)
- **TELEMETRY_PIPELINE_RESEARCH.md** (239L): 0. What BCIR already has (the substrate this builds on) · 1. Layered model of telemetry sources (where each tool sits) · 2. Metric taxonomy → BCIR cost dimensions · 3. Abstractions worth copying (the design DNA) · 4. Gaps to add to BCIR's existing surfaces · 5. Recommended architecture (vendor-neutral, two-truth-safe) · 6. Suggested build order (each a gated segment) · Sources
- **VISION_ALIGNMENT_AUDIT.md** (469L): The thesis being audited · Scorecard · Pillar-by-pillar evidence · Prioritized remaining-work backlog · Bottom line
