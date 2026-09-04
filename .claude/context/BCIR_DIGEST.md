# BCIR repo digest (compressed knowledge base — read this INSTEAD of exploring)

<!-- KNOWLEDGE:BEGIN -->
## Curated knowledge (current through PR #749 and package 0.2.0, 2026-09-03)

**What BCIR is.** A registry-first, phase-ordered, lane-typed, cost-governed
correspondence IR: `K_BCIR(G|H,Θ) = min_π M(π,Θ) s.t. R(π,Θ) ⪯
B(H,Θ)`. It selects a legal realization under a 12-dimensional integer/Q8 cost
vector, live pressure Θ, and RCSP budgets. BCIR is a planning, verification, artifact,
and runtime layer above resident LLVM/GCC/vendor toolchains; it is not a replacement
instruction selector by default.

**Implementation rails.**
- `bcir/` is the dependency-free Python conformance oracle: R1–R25 semantics,
  K_BCIR planning/certified learned organs, GEM scheduling and StreamPack v1–v3,
  frontends/ETL, telemetry, model/training references, BCIRQ8, hosted lowering, and the
  ASN.1 portfolio (`bcir/asn1/`): X.680–X.683 schema rail, DER-out/BER-in X.690, PER,
  OER, XER, JER (bounded J1 oracle + schema-plan compiler), complete X.692 ECN, and
  cost-governed encoding selection with calibrated/certified native tables.
  Python LLVM AOT/JIT accepts exactly one supported elementwise claim; arbitrary graphs
  are rejected rather than truncated.
- `mlir/` is the ODS/TableGen/C++ law rail: verifier and optimizer passes, partial AOT
  preparation, IRDL projection, typed x86 entry/descriptor/segment/ordinary-interrupt
  edges, and the `bcir.asn1.*` (R24) / `bcir.ecn.*` (R25) schema-legality dialects.
  `bcir-aot` may intentionally leave residual BCIR/GEM operations. Targets LLVM 23 (22 still in the CI matrix).
- `runtime/c/` contains three enforced classes: heap-free freestanding code; hosted
  compiler/model tools with allocator injection and fail-every-allocation tests; and
  driver adapters using handles/offsets rather than cross-boundary pointers. It includes
  the C-front/compiler rail, RuntimeChannel v1 loopback, StreamPack/telemetry codecs,
  standalone BCIRQ8 Llama inference, the allocation-free BCAB reader/selector, and
  freestanding ASN.1 twins (X.690, PER incl. the bit-oriented writer + plan-driven
  decoder, OER, XER, JER bounded reader) — every trust boundary fuzzed under
  ASan/UBSan. `runtime/cpp/` is a narrow orchestration seam (plus the J5 SIMD JER
  structural index); single-node dispatch is real and dynamic/distributed backends
  remain explicit stubs.
- `llvm-training/` is a standalone LLVM/MLIR curriculum and evaluation corpus, never a
  runtime dependency. Its aggregate CMake gate is bounded to two lit workers.

**Contracts and ownership.** `docs/BCIR_LANGREF.md` is normative and owns BCIRQ8 v1 and
the ASN.1/ECN law sections (§17; a 2026-08-11 direct commit also appended a large
"comprehensive system report" after §19). `docs/BCIR_ASN1_X690_ABI.md` owns the
DER/BER contract and the A1–A5 laws; `docs/BCIR_ASN1_BUILDOUT_ROADMAP.md` +
`docs/BCIR_ASN1_JSON_ROADMAP.md` own the ASN.1/JER phase ladders;
`docs/research/BCIR_GEMPLUS_ROADMAP.md` owns GEM+/TMSAO slices G0–G10 and the
exact/ratio/wall + TMSAO-1..4 measurement discipline; `docs/BCIR_TARGET_ACCESS.md`
records probed host-capability limits. `docs/BCIR_MASTER_ROADMAP.md` owns portfolio
order, not history. `docs/kernel/` owns driver/kernel/StreamPack/BCAB/telemetry
contracts; `docs/machine-learning/` owns model and ML programs; `docs/languages/` owns
language rails and C memory discipline; `docs/research/` holds
comparative/feasibility/security evidence. Counts live only in generated
`docs/STATUS.md`; chronology lives in `docs/DEVELOPMENT_HISTORY.md` and git. Docs carry
machine-checked claim markers (`test_docs_claims`).

**Landed versus open.** Typed x86 long-mode entry, descriptor/task/segment operations,
and an ordinary interrupt-frame trampoline are assemble-smoke-gated. RuntimeChannel v1,
event phases, DMA descriptors, frozen Q8 priors, telemetry registry/frame/metrics/export
serialization, and the pinned TinyLlama BCIRQ8→standalone-C parity gate are code-backed.
The ASN.1 portfolio is complete on its documented subsets with cost-governed encoding
selection (the build-out thesis executed: value + target + budget → certified legal wire
format); ECN's refusal list is empty; two calibration targets are admitted (Cortex-X4 +
Cortex-A520, same phone, deliberately per-core tables). GEM+ G0 (ExecutionScopeV1 scope
identity + TMSAO-1..4 ladder) landed; G9 half-landed (no more blanket `noalias` lies to
LLVM); everything BCIR emits is TMSAO-4 until G4's lower-bound stack. Open, deliberately:
GEM+ G1–G8/G10; the two-rail provenance-hash memory-hierarchy closure; R11 per-resource
generation vectors; resident UART/virtio drivers, Linux modules/UAPI, BCIR-Linux, native
IPC/kernel, live telemetry transports, reset/paranoid-exception execution. UART and
virtio-blk evidence must precede UAPI v1; direct behavior must stabilize before any IPC
transport. J6 hardware counters and J7 driver work are blocked on *access* (no PMU in
any available host — `BCIR_TARGET_ACCESS.md`), not code.

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
#639–#650 added consolidation/security passes, the hosted model lab, and three bounded
architecture labs (adaptive/byte-native/sequence-interface). #651–#739 is the ASN.1 era:
X.690 rail + R24 (#653–#654), the phase-A–H build-out (X.680 front-end, constraints, PER,
OER, XER, open types), the JER J1–J6 ladder (bounded oracle → schema plan → C twin →
family/profile rail → hosted SIMD J5 → certified K_BCIR selection J6 with real target
calibration), ECN in ~30 slices ending with R25 and an empty refusal list, the plan-driven
PER decoder, and PER/ECN canonicality closures. #740–#748: GEM+/TMSAO architecture +
frozen baseline harness + G0/G9, two 2026-08-12 security audits (Class A second-spelling /
Class B vacuous-check; the host-parser rule), tools-namespace and safetensors-mapping
fixes, and the setuptools-83-floor arc (#747 raised it; #748 opened as a revert, then
reversed course — the floor stands and a policy regression now rejects any lower floor).
#749 (merged): the maintained assurance rails — secret scanning, inventory-first
dependency audit, bounded fuzz/decoder campaigns, a malformed differential, subprocess
policy, and fail-closed independent review — hardened over 42 adversarial review rounds
and 240 graded findings. Its durable output is `docs/security/laws.md`: **21 gate-authoring
laws (L1–L21)**, each with witness tests and a C/C++ port note, plus the per-finding
harvest in `docs/security/pr749-harvest.csv` (20 NEW-LAW / 181 INSTANCE / 39 LOCAL). Read
that registry before writing or reviewing any gate. Side effects that are now repo facts:
the Python floor is **3.11** (the 3.10 TOML fallback is deleted, and a `python-floor` CI
job refuses to run on any other interpreter), the packaged wheel is a tested artifact
(`bcir.tests.run_all` runs from an installed wheel, and `_REPO_ONLY_MODULES` is a
per-test-derived registry, not a per-module guess), and `tools/security/` carries shared
predicates — `git_index`, `proc_bounds`, `report_hygiene` — that every rail must reuse
rather than re-implement.
Gotchas: quick tier intentionally hides toolchains; thorough must use one coherent LLVM
major (23, with 22 still passing — an LLVM-18 host cannot build `mlir/`, honest skip). The default target is
host-adaptive. Keep local concurrency at two and never launch unbounded fuzzing,
inference, emulation, or nested build loops. New C sources must land in the
check_runtime.sh gate block AND `native_bench._SOURCES` together (the #719 wiring trap).
New `test_*.py` files must be registered in `run_all.py`. Regenerate `STATUS.md` last;
regenerate this digest (`build_digest.py`) when its inventory drifts.
**2026-09-03 whole-system analysis** (`docs/research/BCIR_SYSTEM_ANALYSIS_2026-09-03.md`, PR #751):
every local gate green at PR #750 (quick + thorough tiers, C rail with clang/gcc sanitizers and
libFuzzer, docs governance, the six assurance rails, the MLIR rail on conda-forge LLVM 22.1.8);
Python 3.14.7 and 3.15.0rc2 pass the quick tier; clang 22/23 pass the C gates. Toolchain
lessons: build `bcir-opt` with the compiler that built the MLIR libraries (system g++ 13 objects
against conda's GCC-15-built archives crash at startup); set `LLVM_BIN` for the assemble-smoke
resolver. LLVM 23.1.0 is released and the rail needs one rename to compile on it
(`applyPatternsAndFoldGreedily` -> `applyPatternsGreedily` in `BCIRPromotePass.cpp`; LLVM 22
already provides the new name). The report's D1-D10 ledger lists the live documentation drift
(stale verifier-law ranges in ~20 docs, the LangRef's appended 3,190-line snapshot, the stale PER
hand-off note, ASN.1 roadmap self-contradictions) and §9 ranks the next slices.
**MLIR rail on LLVM 23 (PR #752, after the 2026-09-03 analysis).** `applyPatternsAndFoldGreedily`
was renamed to `applyPatternsGreedily` and all 51 `builder.create<OpTy>(...)` call sites became
`OpTy::create(builder, ...)` (both spellings exist in LLVM 22; the old ones are gone/deprecated in 23);
`mlir-rail-validate` now runs `llvm: ["22", "23"]`, the tool scripts try the `-23` names first, and
`tools/local/{setup_mlir,check_rail,env_mlir}.sh` take `MLIR_MAJOR` (default 23). Locally the rail
passed identically on conda-forge 22.1.8 and 23.1.0 (tblgen, IRDL, ODS, passes incl. asm-smoke,
malformed differential, bytecode, training tiers) built with each toolchain's own clang++.
The advisory half of the dependency audit is real from 2026-09-03: `security-assurance`
installs `pip-audit==2.10.1` and runs `audit_dependencies.py --require-advisory` — a resolved
run (pip's closure) and a floor run (each declaration at its lowest admitted version, which is
also what covers `setuptools`, dropped by pip-audit's scratch venv) reconciled against the
declared names; every other job asserts the inventory only. Dated findings and the currency
table: `docs/security/DEPENDENCY_AUDIT_2026-09-03.md`.

2026-09-04 full-surface dependency audit (`docs/security/DEPENDENCY_AUDIT_2026-09-04.md`,
evidence in `docs/security/audit-2026-09-04/`, tooling in `tools/security/audit/`): 0 known
vulnerabilities across 77 package-versions by two sources (pip-audit/PyPI API and an offline OSV
evaluation); every action/hook pin is a release commit; checkout/setup-python/cache/upload-artifact were
2–3 majors behind and moved to v7.0.1 / v7.0.0 / v6.1.0 / v7.0.1 in the follow-up PR; the runners' apt `nodejs` was
the EOL 18 line with 20 open Ubuntu advisories (dropped; the WASM tests run Node 24 through
SHA-pinned setup-node in the oracle jobs); the hosted train-to-C jobs audit their installed
closure with `audit_dependencies.py --installed` (exact public pins, engine in its own venv); kafka-python became the declared `telemetry-kafka`
extra (floor 2.3.2). A first-pass OSV false positive on torch (non-PEP-440 `last_affected`
values in PYSEC records) is root-caused in `osv_pypi.py`'s docstring.

<!-- KNOWLEDGE:END -->

## Generated inventory (do not edit — rebuild with build_digest.py)

Top-level: ./bcir ./channels ./docs ./llvm-training ./mlir ./runtime ./tools

### STATUS.md counts (generated source of truth)
| Metric | Value |
|---|---|
| Static Python `test_*` function inventory | **3538** across 254 files |
| Static MLIR ODS op-definition inventory (`mlir/include/BCIR/*.td`) | **133** |
| Static registered-pass inventory | **37** |
| Static MLIR fixture inventory (`mlir/test/`) | **117** files; 300 `expected-error` markers |
| Static runtime C source/header inventory (`runtime/c/`) | **301** files |
| Verifier-law negative-fixture tag inventory | **R1–R25** (25/25 present) |
| Registered hardware-channel inventory | **9** (cpu, fpga, gpu, memory, storage) |
| R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | R11 | R12 | R13 | R14 | R15 | R16 | R17 | R18 | R19 | R20 | R21 | R22 | R23 | R24 | R25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
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
- **docs/BCIR_ASN1_BUILDOUT_ROADMAP.md** (1508L): 0. The thesis: encoding rules are a realization choice · 1. Standards inventory (verified 2026-07-26) · 2. What is built (baseline taken at PR #670; §4 records the  · 3. Dependency order · 4. The phases · 5. Sequencing recommendation after PR #670 · 5.1 Access limits on the remaining phases · 6. Stop conditions and decision boundaries · 7. Risk register · 8. What this is not · 9. Encoding-rule coverage, and the rules deliberately not bu · 10. Information objects beyond X.681, and the ASN.1 bindings
- **docs/BCIR_ASN1_COMPILER_COMPARISON.md** (271L): 1. What each one is · 2. Transfer syntaxes · 3. Schema language coverage · 4. What BCIR has that none of the other three do · 5. What the other three do that BCIR does not · 6. What to import, ranked · 7. What not to import
- **docs/BCIR_ASN1_JSON_ROADMAP.md** (1237L): 1. Decision · 2. Source-backed baseline (taken at PR #670; §7 tracks what  · 3. Standards and terminology · 4. Trust and ownership contract · 5. Compiled schema and lowering contract · 6. K_BCIR selection and artifacts · 7. Delivery phases · 8. Validation and performance method · 9. Driver and kernel boundary · 10. Risk register · 11. References
- **docs/BCIR_ASN1_X690_ABI.md** (317L): 1. The stance: DER out, BER in · 2. Coverage · 3. The BCIR-StreamPack module · 3a. The DER → native fast path · 4. The BCIR-ArtifactBundle module · 5. The laws · 6. The law rail (R24) · 7. Trust boundary · 8. Transfer syntax identity · 9. Validation
- **docs/BCIR_JSON_PROGRAM_REPRESENTATION.md** (373L): 1. The proposal, restated precisely · 2. Prior art, because none of the core ideas are new — and t · 3. Three claims in the proposal that must be corrected befor · 4. The three deficiencies, and the mechanisms that answer th · 5. Control flow as a cost problem — the technically stronges · 6. Self-modification: the staged model · 7. Phase ladder · 8. What this changes in the existing roadmaps · 9. Risk register · 10. References
- **docs/BCIR_LANGREF.md** (1345L): 0. Stance · 1. The multi-level IR · 2. Central equation · 3–9. Laws (summary) · 10. Verifier laws (R1–R25) · 11. Rewrite laws (the building-blocks engine) · 12. Lowering contracts · 13. Learning placement (normative policy) · 14. The two-truth separation (MOPC) · 15. The enriched-operad memory interface (the higher intelli · 16. BCIRQ8 v1 decoder-artifact contract · 17. ASN.1 in BCIR · 18. Conformance profiles and external-contract boundary
- **docs/BCIR_MACHINE_CODE_HAL_ISA_AUDIT.md** (219L): 0. Executive verdicts · 1. The machine model as built (the "BCIR ISA") · 2. The toolchain as built · 3. What the research docs add to the frame · 4. HAL/ABI as built · 5. The ABI ledger (what is frozen today) · 6. The gap register — the MC-track (code-backed) · 7. Standing positions this audit confirms (no change needed)
- **docs/BCIR_MASTER_ROADMAP.md** (387L): 1. Mission and non-negotiable invariants · 2. Architecture and current baseline · 3. Dependency order · 4. Active workstreams · 5. Program milestones · 6. Release policy · 7. Validation and publication gate · 8. Decision boundaries · 9. Risk register · 10. Document ownership · 11. Immediate priority queue
- **docs/BCIR_NATIVE_OBJECT_GATE.md** (152L): 1. The decision · 2. The warranted slice (done): real native objects end-to-en · 3. GO criteria — what would warrant BCIR-native isel · 4. STOP criteria — if a native-isel experiment is taken · 5. Current verdict and evidence boundary
- **docs/BCIR_Repo_Structure.md** (212L): 1. Top-level ownership · 2. Oracle package (`bcir/`) · 3. Law rail (`mlir/`) · 4. C and C++ runtime classes · 5. Contract ownership · 6. Documentation taxonomy · 7. Build and validation entry points · Fast dependency-free oracle tier · Full local oracle/toolchain tier, with bounded concurrency · Production C and C++ boundaries · Optional pinned hosted-model CPU gate (one thread in CI) · MLIR/IRDL rails when the coherent LLVM toolset is installed · Documentation governance
- **docs/BCIR_TARGET_ACCESS.md** (130L): 1. Privilege is not capability · 2. What the two available hosts actually provide · 3. What each open phase needs · 4. The bare-metal targets that would unblock it · 5. How to record a new host
- **docs/DEVELOPMENT_HISTORY.md** (434L): 1. The development method · 2. The PR arc (eras) · 3. Condensed dated changelog · 4. Capability closure ledger migrated from the former master · 5. Where the detailed notes live now
- **docs/ONBOARDING_DEEP_DIVE.md** (320L): 1. Read this first · 2. The three implementation rails · 3. From source to execution · 4. Core semantic and optimizer packages · 5. Frontends, lowering, and machine boundary · 6. Runtime memory and ownership · 7. Models, training, and BCIRQ8 · 8. Drivers, kernel, telemetry, and IPC · 9. Current evidence boundary · 10. Validation workflow · 11. Reading and change-placement map
- **docs/PARITY.md** (378L): Enum value parity (normative) · Concept parity · Python ↔ C artifact and runtime parity · Python ↔ C frontend twin (`runtime/c/`) · Worked-example parity · Generated, adversarial parity (the proof, not the hope) · How parity is enforced today
- **docs/PERFORMANCE_AUDIT.md** (146L): 1. Gate and evidence contract · 2. Defects and bottlenecks found · 3. Local before/after evidence · 4. What remains hardware- and workload-gated
- **docs/RELEASE_NOTES_0.3b.md** (147L): Candidate baseline already landed · Release blockers · Explicit non-goals · Candidate validation
- **docs/REPO_CURRENT_STATE_AUDIT.md** (330L): Snapshot · Confirmed strengths · Confirmed limitations · Recommended next milestones · Changelog
- **docs/STATUS.md** (50L): Verifier-law negative-fixture inventory (R1–R25) · Hardware channel / target matrix · Runtime C components
- **docs/VISION_ALIGNMENT_AUDIT.md** (212L): 1. Thesis under audit · 2. Scorecard · 3. C as registry definition and macro target · 4. IR ownership, machine edges, and backend boundary · 5. Certified optimization and AI substrate · 6. Model inference, training, and C++ boundary · 7. Driver, kernel, telemetry, and IPC alignment · 8. Highest-leverage remaining work · 9. Bottom line
- **docs/kernel/BCIR_AMD_AI_DRIVER_ROADMAP.md** (383L): 0. Executive strategy · 1. The honest starting point · 2. The vertically-integrated stack · 3. The phased build order · 4. The three device classes (never one) · 5. The per-project interop ledger · 6. The ML-framework supplement boundary · 7. The deferred Phase-0 Linux inheritance (scope, not build) · 8. Risks / messaging discipline · 9. Recommended next steps (ranked)
- **docs/kernel/BCIR_ARTIFACT_BUNDLE_ABI.md** (269L): 1. Wire conventions and limits · 2. Header (128 bytes) · 3. Directory entry (448 bytes) · 4. Kinds and payload formats · 5. Compatibility and deterministic selection · 6. Interfaces and backend boundary · 7. Additive ASN.1 transfer syntax · 8. Binary compatibility strategy · 9. Conformance
- **docs/kernel/BCIR_DRIVER_KERNEL_ROADMAP.md** (775L): 1. Mission and product split · 2. Current baseline · 3. The BCIR driver package contract · 4. Execution, telemetry, and continual optimization · 5. Driver maturity and build order · 6. BCIR-Linux: compatibility oracle and experimental fork · 7. Universal ABI, POSIX compatibility, and IPC · 8. Validation and promotion policy · 9. Risks and stop conditions · 10. Historical rationale retained from roadmap v1
- **docs/kernel/BCIR_HAM_MEMORY_FABRIC.md** (269L): 1. Decision · 2. What the proposal got right—and what required correction · 3. Source-backed inventory · 4. Semantic metadata and planning contract · 5. Lowering and verification · 6. Context shards: transport crystallized artifacts, not “ne · 7. Dual-rail optimization memory · 8. Tiny-model placement · 9. Driver/kernel/firmware sequence · 10. Acceptance and promotion
- **docs/kernel/BCIR_STREAMPACK_ABI.md** (140L): Conventions · Header (64 bytes, cache-line aligned) · Body (sequential, length-prefixed) · Trailer · v2 (append-only): pipelined phases + double-buffer prefetch · v3 (append-only): on-wire segment dispatch + channel · Semantic trust boundary (R10/R11 in C) · Versioning (the freeze) · Why a frozen ABI now
- **docs/kernel/BCIR_UART_DRIVER_BLUEPRINT.md** (1050L): 0. How to use this document (for the implementing model) · 1. The normative 16550 device model · 2. The variant matrix (what the registry must parameterize) · 3. Field-reality quirks (research; not in any datasheet) · 4. Architecture: how each piece maps onto existing BCIR mach · 5. The build slices · U8 adds: "tl16c750" (mode64_key="dlab", rx_triggers_alt=(1,1 · "tl16c750e" (fifo_depth=128, rx_triggers=(1,4,120,124), tx_t · flow_mech="efr_tcr", has_tlr/has_frac_divisor/has_sleep=True · reset_honest=False), · "h16750s_64" (a chosen synthesis point: fifo_depth=64, rx_tr · "lattice16550_lmmi" + "lattice16550_apb" (has_scr=False, fcr · reset_honest=False; the APB one is stride=4).
- **docs/kernel/HARDWARE_VALIDATION.md** (109L): What IS validated here (real, measured — `bcir/tests/test_si · What is BLOCKED in this sandbox (and why) · The rig required for FULL hardware validation · The runbook (push-button) · Honest status line
- **docs/kernel/HETEROGENEOUS_CHANNELS.md** (152L): The problem it solves · The abstraction (`bcir/channels.py`) · The unified core — every channel plans the same way · Heterogeneous orchestration — one binary graph across the to · Cross-device placement cost (fabric/sync) · Adding a backend (the extension path) · Status
- **docs/kernel/SIGNAL_REGISTRY.md** (125L): What it is · Core types · Providers · Builders + the channel↔provider mapping · Honest real/unavailable split (typical sandbox) · Status and pre-driver boundary
- **docs/kernel/SYCL_INTEROP.md** (126L): Resident dispatch (the channel executes) · The bright line: SYCL is a compiler MODE, not a `c.call.libm
- **docs/kernel/TELEMETRY_FRAME_ABI.md** (153L): Conventions · Frame · Resync semantics · Host decode reuses RT3 (two-truth) · Frozen-v1 scope and pre-driver extension · Egress over UART (documented adapter, not built here)
- **docs/kernel/TELEMETRY_PIPELINE_RESEARCH.md** (310L): 0. What BCIR already has (the substrate this builds on) · 1. Layered model of telemetry sources (where each tool sits) · 2. Metric taxonomy → BCIR cost dimensions · 3. Abstractions worth copying (the design DNA) · 4. Gaps to add to BCIR's existing surfaces · 5. Recommended architecture (vendor-neutral, two-truth-safe) · 6. Suggested build order (each a gated segment) · 7. Driver/kernel integration gate · Sources
- **docs/languages/CFRONT_GUIDE.md** (337L): Quickstart · compile a file (verified C + R1–R18 status to stdout) · syntax/semantic check only — Clang-style diagnostics, no out · machine-readable diagnostics (for editors / CI) · lay the types out for another target ABI · graceful degradation: report a fallback-to-LLVM signal inste · Command-line options · Diagnostics · The target ABI matrix · The LLVM-backend fallback contract · Pointer-bounds policy (LangRef §4) · Pointer-lifetime policy (R21, LangRef §10) · Inline assembly (ASM1)
- **docs/languages/CPP_HANDOFF_BOUNDARY.md** (243L): Honest depth (read this first) · Why a boundary at all · What STAYS on the C/IR rail (below the boundary) · What CROSSES to C++ (above the boundary) · The seam · Why C++ (and not C / IR) · The scaffold (what is built) · Risks / follow-ups (what a real distributed/dynamic implemen
- **docs/languages/C_MEMORY_DISCIPLINE.md** (93L): Runtime classes · Required ownership rules · Direct driver ABI first · IPC boundary · Validation cadence
- **docs/machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md** (1363L): 0. Stance — why an IR becomes intelligence · 1. The intelligence already in BCIR (the substrate this buil · 2. The ordered build-out · 3. The continuous-development discipline (how every layer ke · 4. Capability-track placement (do we need CT6 / CT7?) · 5. Risk register / honest boundaries (out of the dreamy pote · 6. AI-substrate closure register · 7. Open-weight model ingestion (GLM / Gemma / Qwen) — the LL · 8. Feasibility audit — the deeper-integration program (2026-
- **docs/machine-learning/BCIR_PYTHON_NATIVE_BOUNDARY_AUDIT.md** (176L): 1. Verdict · 2. Scope and method · 3. Native surface implemented by this audit · 4. Safety, determinism, and portability contract · 5. Bounded local evidence · 6. Complete placement register · 7. Next native milestones
- **docs/machine-learning/BCIR_WHOLE_MODEL_REFERENCE.md** (301L): 1. The overlap — BCIR already owns the reference numerics · 2. What landed and what remains · 3. What NOT to import · 4. Where each piece lands (anchored homes) · 5. Build-slice status (WMR-1 … WMR-4) · 6. The larger implication — closing the train → export → ser · 7. Coherence with the rest of the system
- **docs/machine-learning/ML_LANGUAGE_PLACEMENT_ANALYSIS.md** (394L): 1. Executive summary — the thesis · 2. The five placement criteria · 3. The four language tiers · 4. The classification table (the heart) · 5. The migration map · 6. Conclusion — the clean hierarchy
- **docs/machine-learning/OPENAI_BCIR_INTEGRATION_RESEARCH.md** (466L): 1. Repository capability map · 2. Current OpenAI developer capability research · 3. How deep ChatGPT can integrate into BCIR · 4. Proposed architecture · 5. Proposal versions · 6. Recommended next implementation steps · 7. Core conclusion
- **docs/machine-learning/THIRD_PARTY_MODELS.md** (129L): Maykeye/TinyLLama-v0 · CUDA-LLM comparison boundary · TinyStories dataset planned for BCIR-TinyStories-32M · Adaptive-architecture research boundary · Byte-native architecture research boundary · Sequence-interface and progressive-growth research boundary
- **docs/research/BCIR_ADVANCED_TECHNIQUE_TRIAGE.md** (257L): The framing that decides most of the list · A. Already built · B. Already in the GEM+ roadmap · C. LLVM's job — BCIR's job is to supply the fact · D. Genuinely new — the roadmap additions · Summary: what changes in the roadmap
- **docs/research/BCIR_GAME_OPTIMIZATION_ROADMAP.md** (442L): 1. The exact-vs-approximate split — the load-bearing thesis · 2. The overlap — what BCIR already embodies (map, don't re-b · 3. Per-game principles — the full ledger · 4. Lessons applied to **GEM** (the StreamPack hot path) · 5. Lessons applied to **K_BCIR** (the tropical cost model, e · 6. Lessons applied to the **StreamPack ABI** (frozen artifac · 7. Ranked build slices · 8. Risks & myth-flags · 9. The bottom line
- **docs/research/BCIR_GEMPLUS_ROADMAP.md** (485L): 0. The measurement discipline, and why it comes first · 1. Scope identity: `S` · 2. Certificate classes · 3. The slices · 4. The sublinearity question, answered precisely · 5. The learned-optimization boundary · 6. Order of work · 7. What this roadmap will not claim
- **docs/research/BCIR_NATIVE_BACKEND_FEASIBILITY.md** (217L): 1. What "native backend" means here · 2. Current state — the codegen spectrum BCIR already populat · 3. What a *general* native backend requires (and why it is e · 4. The gate, restated and assessed (status: all GO criteria  · 5. The candidate bounded targets, priced and ranked · 6. Development roadmap (executed ONLY if the gate opens for  · 7. What to do *now* (and how it de-risks any future native w · 8. Bottom line
- **docs/research/BCIR_SECURITY_AUDIT_2026-08-12.md** (173L): 1. The two failure classes · 2. Class A — canonical-byte defects · 3. Class B — vacuous checks · 4. The one finding left half-closed · 5. Not reproduced · 6. Verification · 7. Recommended next
- **docs/research/BCIR_SECURITY_AUDIT_2026-08-12b.md** (170L): 1. What the previous pass handed this one · 2. Confirmed and fixed · 3. Investigated and cleared · 4. Swept clean · 5. Verification · 6. What remains open
- **docs/research/BCIR_SECURITY_RED_TEAM_AUDIT_2026-07-15.md** (193L): Executive verdict · Method and safety boundary · Observed environment and exploitability · Confirmed findings and fixes · Copy Fail, Dirty Frag, and local-escalation analogues · Investigated and closed as non-exploitable in the current en · Dependency and CI disposition · Validation record · Residual risk and required follow-up
- **docs/research/BCIR_SECURITY_THREAT_MODEL.md** (143L): Security objective · System and trust boundaries · Assets · Attacker capabilities · Security invariants · Principal abuse paths · Future driver/UAPI requirements · Accepted residual risk
- **docs/research/BCIR_SYSTEM_ANALYSIS_2026-09-03.md** (531L): 1. Executive summary · 2. What the system is, by the numbers · 3. Evidence: what was executed on this host · 4. Architecture assessment by subsystem · 5. History and process review · 6. Documentation audit: drift and inconsistencies found · 7. Code-quality observations · 8. Risks, ranked · 9. Recommended next development steps · 10. Reproducing the toolchains on a network-restricted host · Appendix A. Reading map used for this report
- **docs/research/BCIR_SYSTEM_REPORT_2026-08-10.md** (3202L): 1. Provenance and scope · 2. What BCIR is · 3. Authority hierarchy · 3.1 Normative language and law rail · 3.2 Python executable conformance oracle · 3.3 Freestanding and hosted C · 3.4 C++ adapter layer · 3.5 Educational rail · 3.6 Research rail · 4. Mechanical repository inventory · 5. End-to-end architecture · 5.1 Stable embeddable facade · 5.2 Concrete fresh example
- **docs/research/BCIR_TMSAO_ARCHITECTURE_AND_PERFORMANCE_REPORT.md** (942L): 1. Executive answer · 2. What “TMSAO” can and cannot mean · 3. Audit method and limitations · 4. Current architecture: what is genuinely strong · 5. Full operational and data-structure performance · 6. Exact differentials: where the current optimizer is not o · 7. Native measurement validity finding · 8. Is a DAG the right fallback? · 9. Is tropical min-plus the master algorithm? · 10. What current state-of-the-art systems add · 11. Proposed GEM+ / K_BCIR architecture · 12. Prioritized implementation program · 13. Decisions
- **docs/research/BCIR_TMSAO_ASN1_JSON_DRIVER_PROPOSAL.md** (974L): 1. Executive verdict · 2. Operational definition of TMSAO · 3. Source-backed state at PR #739 · 4. GEM+: the canonical architecture · 5. Solver portfolio and lower-bound stack · 6. Scaling, scheduling, and memory program · 7. ASN.1 through PR #739 and its GEM+ role · 8. Python-to-C++ migration roadmap · 9. Hardware profiles and the native measurement rig · 10. API, database, service, and IPC architecture · 11. Driver, kernel, FPGA, and SASOS implications · 12. Prioritized implementation program · 13. Risk register and decision rules
- **docs/research/BCIR_TRITON_COMPARATIVE_ANALYSIS.md** (271L): 0. Executive verdict · 1. The comparison matrix · 2. Where the two systems actually touch (BCIR surfaces, anch · 3. The migration ledger · 4. Direct answers to the three questions · 5. Recommended next steps (ranked) · 6. Messaging discipline (the corrections, restated so they d
- **docs/research/CLANG_COMPARISON.md** (98L): The fair frame · Results · Where we WIN · Where we MATCH · Where we LOSE (honest) · Bottom line
- **docs/security/DEPENDENCY_AUDIT_2026-09-03.md** (233L): 1. Verdict · 2. Inventory and currency · 3. The advisory scan — method and result · 4. What the rail enforces from this slice on · 5. What this audit does not cover · 6. Recommendations
- **docs/security/DEPENDENCY_AUDIT_2026-09-04.md** (288L): 1. Verdict · 2. Scope and method · 3. Inventory · 4. Advisory results · 5. Findings and dispositions · 6. Changes landed with this audit · 7. Reproduction · from the repository root; a scratch venv with pip-audit==2.1
- **docs/security/laws.md** (740L): The harvest protocol · The staleness rule (declared, not discretionary) · The laws · Campaign classification summary
