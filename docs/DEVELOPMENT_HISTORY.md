<!-- allow-retired-paths -->
<!-- allow-law-ranges -->
# BCIR Development History

> **Purpose.** The single summary of *how BCIR was built*: the development method, the
> era-by-era PR arc, and the condensed changelog. It consolidates the detailed
> development-step notes that used to be scattered across the docs — the dated changelog
> formerly in [`REPO_CURRENT_STATE_AUDIT.md`](REPO_CURRENT_STATE_AUDIT.md), the PR-arc
> section formerly in [`ONBOARDING_DEEP_DIVE.md`](ONBOARDING_DEEP_DIVE.md), and the
> slice-by-slice build notes formerly embedded in the roadmaps. Roadmap docs now describe
> *what exists and what is next*; this doc records *how it got here*. For current counts
> (tests, ops, passes, laws) see the generated [`STATUS.md`](STATUS.md) — nothing here is
> a live count. This revision is current through merged PR #757 (2026-09-04) and package version
> `0.2.0`. Sources are the GitHub PR record, first-parent history, implementation/tests,
> and pre-consolidation document revisions retained in git.

---

## 1. The development method

BCIR was developed almost entirely by AI coding agents under a single human operator,
with the recorded arc running from PR #2 through #638 between 2026-04-23 and 2026-07-15. Two agent fleets
are visible in the branch names: **OpenAI Codex** (`codex/*` branches — the initial
scaffolding, the `training/llvm/` corpus, and the closing integration-research docs) and
**Claude Code** (`claude/*` branches — the long engineering arcs, one long-lived session
branch per arc emitting dozens-to-hundreds of sequential PRs). The recurring process
patterns:

1. **Prototype-then-port (dual/triple-rail).** Every capability is prototyped in the
   Python oracle (`bcir/`), then ported to a production rail — the MLIR/C++ law
   (`mlir/`) for plan-time decisions, or C (`runtime/c/`) for the runtime and the
   plug-in C compiler — locked together by parity gates (bit-exact scores,
   byte-identical artifacts, an FNV-1a structural digest). The rule was made an explicit
   non-negotiable in PR #266 ("stop extending the prototype as if it were the product")
   and codified in the `PARITY.md` twin ledger (#357), which also records intentional
   non-ports (the two-truth line).
2. **PR-sized slices inside named ladders.** Nearly every PR is one gateable slice of a
   pre-declared program: L1–L8 (the cfront ladder), Phase 2/3/4 (C language /
   preprocessor / toolchain), RT1–RT7 (security red-team), G1–G8 (the vision-gap
   program), A1/B1–B5 (the AI substrate), M1–M3 + E1–E7 (ML tiers), T1–T4 (telemetry),
   ASM1–ASM3b (trusted asm edges), SEG1–SEG8 with D0/D1 and H1–H5 sub-slices (the
   driver arc), RUNG 0–7 (driver bring-up). Roadmap docs define the ladder before the
   slices land; PR titles carry the slice IDs.
3. **Verification-first PRs, a ratcheting test count.** Every substantive PR body ends
   with a Verification section quoting exact gate outputs. The conformance-test ratchet
   is visible across PR bodies: 25 (#153) → 91 (#163) → 468 (#212) → 631 (#260) →
   824 (#428) → 1009 (#534) → 1466 (#570) → ~1795 (#604).
4. **Generated, adversarial verification over curated pins.** Curated worked examples
   were systematically replaced by generated differentials: the Python↔MLIR plan
   differential, verifier fault-injection campaigns (every law must catch its own
   injected violation), a three-way C-frontend fuzzer (C twin vs oracle vs Clang),
   libFuzzer+ASan/UBSan on every trust-boundary decoder, and sanitizer sweeps over the
   whole fixture corpus.
5. **Non-disturbance for new laws.** New optional semantics (R17 accuracy, R19/R20
   timing, R21 lifetime) ship *vacuous-by-default*: the entire existing corpus must
   verify byte-identically with the new law wired in, proving the addition cannot
   perturb existing plans, scores, or digests.
6. **Generated status + docs governance (since #260).** After repeated count drift,
   `docs/STATUS.md` became a generated artifact with a CI drift gate
   (`tools/docs/gen_status.py --check`), joined by a broken-link checker, a
   retired-path checker, and a hot/cold import-quarantine gate. Prose links to
   STATUS.md instead of hard-coding numbers.
7. **Reconciliation waves.** When the plan fell behind the code, a docs PR reconciled
   it: #212/#225 (15 docs → one master roadmap), #260 (governance), #357–#358 (parity
   ledger + honest repositioning), #510 (the §5.14 arc), #513 (the onboarding
   deep-dive), #539/#555 (the vision-alignment audit + scorecard), #592/#594
   (feasibility + driver roadmaps) — and this consolidation.
8. **Red-team + honesty culture.** Dedicated adversarial slices (RT1–RT7, the Area-B
   numerical red-team, asm/port-I/O malformed-input red-team); fuzzer-found miscompiles
   itemized per PR (#428–#449); "execution honesty" work replacing shape-only tests
   with real execution (JVM `.class` assembly + run; "assemble-smoke" anti-masking
   gates on the asm-edge lowerings); expensive directions put behind written GO/STOP
   gates (the native-object gate #224, the #592 feasibility verdict *against* a general
   native backend); false wins retracted; modeled numbers labeled modeled; "measured"
   never claimed from synthetic telemetry.

---

## 2. The PR arc (eras)

### #2–#12 — the one-day C++ skeleton (2026-04-23)
The first incarnation, Codex-generated in a single day: a modular CMake build, a
textual BCIR tokenizer/parser/AST, registry type forms (the U/UX/T/GGG/A/H lane
enums), an aggregated ROP verifier, macro/MAP surface lowering, a GEM runtime with
deterministic controls and telemetry, golden fixtures + CI. Later retired.

### #13–#31 — the LLVM-first seed (2026-05-26/27)
BCIR re-expressed as hand-authored LLVM `.ll` master-reference modules: a 64-byte claim
schema, ops, registry lookup, a GEM executor and worklist, StreamPack/batch scheduling
seeds, and a `bcir-as` assembler, validated by `llvm-as`/`opt` scripts (`runtime/llvm/`,
also later retired — its teaching value survives in `training/llvm/` as
explicitly-bannered historical material).

### #32–#152 — the `training/llvm/` corpus (2026-05-28 → 06-06)
PR #32 (the first Claude Code PR) founded the agent-context LLVM/MLIR curriculum; a
~115-PR Codex wave built it out: 20 chaptered modules, the pitfalls catalog, ~42
exercises, CI example verifiers and tripwires, then the deterministic autograder,
declarative exercise manifests, the provider-neutral eval runner, the dataset exporter,
and the secure submission harness. This era explains the repo description
("LLVM AI agent training included"); the corpus is explicitly *not* part of the IR.

### #153–#161 — the BCIR Stack pivot (2026-06-07/08)
The founding of the current architecture. **PR #153**: the repo restructure — a
runnable Python **K_BCIR oracle** (`bcir/`) realizing
`K_BCIR(G|H,Θ) = min_π Σ Tᵢ⊗fᵢ(π)`, the **MLIR/IRDL dialect law** (`mlir/`), the
LangRef/Blueprint/PARITY doc set, the `mlir-rail-validate` CI job, and the legacy C++
`ir/` tree retired. Then in quick succession: `bcir-opt` becomes a real compiler
(#158), the frozen StreamPack ABI + WASM/stackify (#159), the freestanding C StreamPack
runtime + `!bcir.token` async + the memory model (#160), and data-driven per-target
codegen via `llc` for aarch64/riscv64/nvptx/bpf/x86-64 with a portable C fallback (#161).

### #163–#211 — verifier completion + the intelligence layer (2026-06-12 → 06-16)
PR #163 completed verifier laws R1–R12 on both rails ("the project's largest
correctness gap: optimized output was trusted, not proven"). RCSP/Pareto constrained
planning (#175) and the numbered Phases 12–26 followed: duration-aware scheduling +
StreamPack v2, physics-anchored calibration, R13 policy provenance + the regret ledger,
the soft-DP temperature dial, the MDL retune law, the Bayesian/conformal cost model,
the GNN MoE gate, the propose-verify search accelerator, provenance manifest +
deterministic replay, the e-graph building-blocks engine + the L1 cost throttle, the
memory-module fixpoint, the two-truth quarantine, and the enriched-operad memory
interface. Plus: the MLIR-native GEM pipeline (#196), the closed calibration loop
(#197), the portable C23 kernel backend (#198), the first measured gather-avoidance
wins (~6.5×/16×, #200–#201), the adaptive "smart layer" + R14/R15/R16 (#205–#208),
`kbcir.precision` (#209), and fusion/CSE/deforestation (#210–#211).

### #212–#253 — the C++ optimizer-core port; LLVM 22 (2026-06-16/17)
The measured BCIR-vs-Clang comparison (match on dense, 6–14× wins on intent) + the
master roadmap (#212); the optimizer core ported to C++ MLIR passes in five steps
(`-bcir-cost-model` → fusion/CSE → `-bcir-plan` → `-bcir-overlap` →
`-bcir-rcsp-plan`, #215–#221); the six-target capability matrix pinned on the MLIR
rail (#223); C23 `_BitInt`/`#embed` + the native-object decision gate (#224); the docs
consolidation into one master roadmap (#225); the C executor/encoder, R17 compensated
precision, bundle optimization, proof-carrying records (#226–#228); compositional
semantics + the Tier-2 passes (cim/dvfs/schedule-eft/alloc-pool/async/power-rail/
replay) + the R18 call-graph law (#229–#242); the move to LLVM/MLIR 22 with
conda-forge local validation (#243–#246); generative fault injection for all 18 laws
(#247–#248); shared PlanAnalysis, per-op verifiers, IRDL fidelity (#249–#253).

### #254–#262 — hardware-agnostic + governance day (2026-06-18)
One day of strategic PRs: the last law-rail gaps closed (#254); ARM/Raspberry-Pi-5
first-class with a native aarch64 CI job (#255); a 5.8× faster quick test chain
(#256–#257); the heterogeneous hardware channels (#258); the dependency-ordered
plug-in-compiler roadmap — C frontend → drivers → ML → frontends → ecosystem (#259);
docs governance — generated STATUS.md + link/retired-path CI (#260); test tiers, perf
budgets, import quarantine (#261); the channel-plugin boundary (#262).

### #263–#510 — the cfront arc (~250 PRs, 2026-06-18 → 06-25)
The largest arc: the dual-rail C compiler. The oracle-side MVP with the six-artifact
gate (#263) and ladder stages L5–L8 (#264–#265); the recorded course correction
porting the frontend to a production C twin (`runtime/c/bcir_cfront.c`, #266); the
C compile→execute loop closing with no Python, real register-map and UART drivers
end-to-end (#267–#275); the `bcir-cc` driver (#276–#277); the Phase-2 language waves
(#278–#303); the Phase-3 preprocessor (#304–#312); strings/floats/libm (#313–#324);
Phase-4 Clang-grade diagnostics, the target-ABI matrix, IPO/alias analysis, the LLVM
fallback contract, fuzzing, the driver CLI + user guide (#325–#339); the C-twin parity
backfill ending in the PARITY twin ledger (#340–#357); the honest repositioning as a
*freestanding driver-subset C23 compiler candidate* (#358); scalable IR with no fixed
caps (#359/#365); the type/expression breadth waves through `_Complex` (#360–#427);
the differential-fuzzer bug-hunt (#428–#449, ~21 real miscompiles found and gated);
layout completion — `alignas`, anonymous members, zero-width bitfields (#450–#465);
the emerging laws R19/R20 (timing) + R21 (lifetime) and the §5.12 bounds-quarantine
naked-pointer track (#468–#485); VLAs, lvalue-as-value, computed goto, the full
array-compound-literal surface + the fourth (storage-extent) parity axis (#486–#509);
and the §5.14 MLIR-catch-up + driver-release plan (#510).

### #511–#555 — ML/AI substrate, red-team, vision audit + gap program (2026-06-25 → 06-28)
The ML/AI integration roadmap (#511); the 512-PR onboarding deep-dive (#513);
**R19/R20/R21 promoted to first-class laws — the generated status reports R1–R21**
(#514–#515); CI parallelization (#516–#518); the AI-substrate slices — A1 per-group
quantization certified by R17, B1/B5 `gem.matmul` + the trusted BLAS edge, B3
reverse-mode autodiff as content-addressed graph rewrites, the C23 `_BitInt` frontend
(#519–#530); memory-stress + GCC↔Clang differential + emit→re-parse idempotence gates
(#531–#533); the **security red-team RT1–RT7** — 13 real use-after-frees fixed, parser
DoS depth guards, telemetry-stream integrity, the structural per-claim digest + R1.1,
StreamPack on-wire R10/R11 enforcement (#534–#538); the **vision-alignment audit**
(#539) whose gap backlog drove the **G1–G8 program** (#540–#554): `gem.activation`/
`conv`/`attention` claims, the SoA↔AoS layout pivot, the cache/bank-contention
predictor, the baked-weights inference emitter, the forward/backward training kernel +
SGD, the C↔C++ hand-off seam — then the G-series MLIR law ports (conformance
956 → 1235).

### #556–#604 — breadth: Area-B, SYCL, telemetry, ML tiers, the asm/driver rail (2026-06-28 → 06-30)
Area-B library wraps (LAPACK, GSL, SLEEF; later libcerf `erfcx` and 2-D FFTW); the
SYCL SPIR-V channel with resident dispatch + differential oracle; the **T1–T4
telemetry pipeline** (signal-provider registry, the CRC-sealed UART frame ABI
dual-rail, derived metrics + plan-cost sensitivity, OTLP/Prometheus/Redfish export);
**ML Tier-1 M1–M3** (losses, momentum/RMSprop/Adam, the training loop — BCIR trains
logistic regression and an MLP end-to-end, #570) and **E1–E7 breadth** (OLS, PCA, a
full Transformer block, RNN/LSTM/GRU, classical-ML predict wraps, unsupervised +
pipeline, the language-placement capstone, #571–#577); **ASM1–ASM3b** (inline asm as
an ISA-neutral trusted opaque edge, port-mapped I/O, per-ISA fences, barriers as
first-class ordering edges); the **SEG series** — gem cost/parity passes on MLIR, the
machine-proven closed-set autodiff DAG + `gem.autodiff` law op, order-parameterized
fences dual-rail, the native-backend feasibility verdict (#592: do *not* build a
general register-machine backend), `bcir.asm` → `llvm.inline_asm` (#593); the
**driver/kernel roadmap** (#594) and its slices — the pre-driver hardening gate
(sanitizer sweep into CI, Area-B numerical red-team) and the D1 driver ops
(`bcir.portio`, `bcir.volatile_load/store`, `bcir.creg_read/write`, `bcir.msr_read/
write`) with an "actually assembles" smoke gate (#595–#603); final hardening — the
`c.asm`/`c.portio` red-team, ML-convergence gates for the E-series demos, and
execution-validating the stackify JVM target (#602–#604).

### #605–#607 — integration research (2026-07-01)
Codex-authored research: [`OPENAI_BCIR_INTEGRATION_RESEARCH.md`](machine-learning/OPENAI_BCIR_INTEGRATION_RESEARCH.md) (OpenAI
Responses/Agents/Apps-MCP surfaces mapped onto BCIR's oracle/law/corpus structure,
staged V0–V5 proposals), merged serially and deepened by direct commits. Its
open-weight-model material now lives in
[`BCIR_ML_AI_INTEGRATION_ROADMAP.md`](machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md) §7, and this
docs consolidation followed.

### #608–#610 — documentation and feasibility reset (2026-07-02)
Development history was centralized; stale current-state claims were refreshed; the open-weight
track moved into the ML roadmap. Follow-on feasibility work separated deeper ML/model integration
from kernel/Linux planning and removed a duplicated driver-roadmap section. This was the first
explicit attempt to keep history, current state, and execution order in different documents.

### #611–#623 — thirteen implementation waves (2026-07-02 → 07-04)
The wave series closed several formerly speculative programs: R22/R23 shape/dtype laws; C extent
provenance, project mode, cross-TU linking and ABI contracts; planned/GEM/C streamed training; model
manifest, SentencePiece, reference decoder, real safetensors ingest, group-Q8 artifact, GQA/KV-cache,
serving/TokenDFA, paged KV, and continuous batching; tile/channel priors with exact certificates;
device manifests, bank typing, distance-priced moves, event phases, DMA descriptors, and driver-seam
rules. It also produced the UART U0–U9 blueprint and closed the former A1–A5 gap register where code
and tests existed. The important historical correction is that these were reference/compiler
substrates—not resident drivers, stable UAPI, or production model serving.

### #624–#630 — machine, driver, accelerator, and kernel design programs (2026-07-04 → 07-09)
The machine-code/HAL audit defined MC1–MC15 and native-backend boundaries; the driver catalog defined
proof-carrying package maturity; the Triton comparison chose interop/migration over a fork; the AMD
roadmap chose inherit-and-enhance over replacing ROCm/XDNA/AMDGPU; the whole-model and game-optimization
studies routed reusable mechanisms into BCIR; and the kernel roadmap established BCIR-Linux as a
separate evidence rail rather than making Linux internals normative.

### #631–#638 — governance, correctness, memory, and pre-driver foundation (2026-07-09 → 07-15)
Generated inventory was refreshed and the project moved to the BCIR Non-Commercial License v1.0 with
drafting corrections. Correctness/portability work fixed the C bitfield statement-expression case,
training-spec execution, untied-head quantization, quick-tier semantics, coherent LLVM discovery,
Windows spawn/link behavior, partial-AOT honesty, and the pinned TinyLlama→BCIRQ8→standalone-C gate.
Two C sweeps then hardened trust boundaries and established allocator injection, fail-every-allocation
tests, explicit freestanding/hosted/driver memory classes, and direct RuntimeChannel hooks. PR #638
completed the ordinary x86-64 assembly edge, MC1/MC2 operator tools, strict StreamPack/telemetry
semantics, and the source-backed driver/kernel roadmap v2; all required push and pull-request jobs
passed before merge.

### #639–#646 — consolidation, security, and bounded model/control planes (2026-07-15 → 07-19)
The docs tree was consolidated by subject and its current-state records were reconciled. A
repository-wide post-#538 security pass then hardened memory-sensitive parsing, artifact handling,
race windows, and publication checks. The hosted model lab established deterministic random-weight
training, safe exact resume/export, strict ingest, BCIRQ8, and standalone-C parity. Follow-on work
added bounded corpus/tokenizer and alignment pipelines, provider-neutral offline contracts,
payload-free model inventory/placement plans, a simulated hardware-policy GNN/Transformer with
measured-only promotion, a TMSAO performance/regression harness, and the HAM/context-shard/
dual-memory compiler-simulator baseline. These are compiler and control-plane foundations, not
claims of large-model training or physical-device speedup.

### Post-#646 — measured Python/native AI boundary (2026-07-19)

A source-wide placement audit kept laws, schemas, K_BCIR/HAM planning, hardware search, and hosted
training in the independent Python control/oracle rail while moving stable repeated work into a
portable no-heap C ABI. Q8/Q4 conversion, standalone-decoder Q8 projections, exact hard-filtered
Q15 retrieval, group-32 Q4×Q8 accumulation, and bounded native model measurement gained strict
Python parity, malformed-input, sanitizer, import-quarantine, and host-portability gates. No new
C++ layer was added because the kernels need a reusable C ABI, not another owner; C++ remains
gated on a measured asynchronous serving/device lifecycle.

### Post-#649 — sequence-interface adaptation and constructive growth (2026-07-22)

Seven tokenizer/representation papers and the pinned Apache-2.0 PGT/Embeddings repositories were
audited without importing source or assets. The independent implementation added exact-byte
student/teacher chunk alignment and probability-conserving credit projection, continued-BPE
decomposition plus copy/mean/freeze ownership, bounded Thunder-style unigram segmentation,
multi-objective tokenizer evidence, causal float32/FSQ time-series prefixes, fixed binary token
interfaces, and explicit active-parameter/optimizer-state growth schedules. A tiny one-thread
PyTorch gate proves copied rows and earlier blocks remain unchanged; dense growth lowers through
ordinary verified claims and StreamPack. Large tokenizers/models, glyph/PCA artifacts, LoRA
execution, GPU kernels, and external data remain promotion-gated.

### Post-#649 — multi-backend artifact compatibility (2026-07-22)

BCAB v1 established a bounded deterministic envelope around unmodified StreamPack and standard
backend payloads. The landing added Python encode/decode/selection and tooling, an allocation-free
C reader/selector, a borrowed C++ facade, MLIR directory/selection operations, a real bounded JVM
class assembler, resident compiler/linker adapters, and malformed-wire/differential gates. It did
not add a native linker, OS loader, signature policy, or claim that one ISA image runs on another.
After the ASN.1 rail landed, BCAB gained an additive DER/BER and COER/OER projection whose
round trip reconstructs byte-identical native artifacts without changing BCAB v1.

---

## 3. Condensed dated changelog

The full per-landing entries (one detailed paragraph each, 2026-06-07 → 2026-06-25,
~90 entries) lived in `REPO_CURRENT_STATE_AUDIT.md` and remain in its git history
(`git log -p -- docs/REPO_CURRENT_STATE_AUDIT.md`). The condensed arc:

- **2026-04-23:** the one-day C++ skeleton (#2–#12).
- **2026-05-26 → 06-06:** the LLVM-first seed (#13–#31); the `training/llvm/` corpus
  build-out (#32–#152).
- **2026-06-07 → 06-08:** the BCIR Stack pivot — oracle + law + PARITY; `bcir-opt`
  real passes; the frozen StreamPack ABI; the freestanding C runtime; per-target
  codegen (#153–#161).
- **2026-06-12 → 06-15:** verifier R1–R12 both rails; Phases 12–26 (calibration, R13
  provenance + regret, soft-DP, MDL, Bayesian/conformal, MoE gate, accelerator,
  replay, e-graph, memory fixpoint, two-truth, operad); the GEM MLIR pipeline; the
  closed calibration loop; the C23 kernel backend; the measured gather wins; the
  adaptive smart layer + R14–R16.
- **2026-06-16 → 06-17:** the Clang comparison + master roadmap; the optimizer-core
  C++ port (five steps, bit-exact); the six-target matrix; the C
  executor/encoder + R17 + bundle + proof records; compositional semantics + Tier-2
  passes + R18; LLVM/MLIR 22; fault-injection for all laws; the docs consolidation
  into one master roadmap.
- **2026-06-18:** ARM first-class; channels; the plug-in-compiler roadmap; docs
  governance (generated STATUS + gates); test tiers; the cfront arc opens — L1–L8 on
  the oracle, the production C twin, the no-Python compile→execute loop, register-map
  + UART drivers end-to-end, `bcir-cc`, and the first Phase-2 language waves.
- **2026-06-19 → 06-25:** the C-surface completion campaign (preprocessor,
  diagnostics, ABI matrix, IPO, fallback contract, fuzzing; floats/`_Complex`/
  variadics/`_Generic`/`typeof`/VLAs/computed-goto/compound literals; the fuzzer
  bug-hunt; R19–R21 seeded vacuous; the bounds-quarantine track; the storage-extent
  parity axis; `_Decimal*` honestly blocked), closed by the §5.14 plan (#510).
- **2026-06-25 → 06-28:** the ML/AI roadmap; the onboarding deep-dive; **R19–R21
  promoted to first-class (R1–R21)**; the A1/B1/B3/B5 AI-substrate slices; the RT1–RT7
  security red-team; the vision-alignment audit + the G1–G8 gap program (conformance
  956 → 1235).
- **2026-06-28 → 06-30:** telemetry T1–T4; SYCL resident dispatch; ML Tier-1 M1–M3 +
  breadth E1–E7; ASM1–ASM3b; SEG1–SEG8 (gem cost passes, the autodiff closure proof +
  law op, Area-B to six libraries, dual-rail ordered fences, the asm-edge law ops with
  assemble-smoke gates); the driver/kernel roadmap + pre-driver hardening; the
  close-out red-team and convergence-gate slices.
- **2026-07-01:** the OpenAI + BCIR integration research (#605–#607); the first docs
  consolidation (changelog extraction, the open-weight-track move, staleness fixes).
- **2026-07-02 → 07-04:** the documentation/feasibility reset (#608–#610) and the
  thirteen implementation waves (#611–#623): R22/R23, project-mode C, streamed
  training, whole-model ingest/inference, certified Q8 priors, device manifests,
  event/DMA contracts, and the UART blueprint.
- **2026-07-04 → 07-09:** the machine-code/HAL, driver, Triton, AMD, whole-model,
  game-optimization, and kernel programs (#624–#630) established the present
  resident-toolchain, inherit-and-enhance, proof-carrying-driver, and BCIR-Linux
  boundaries.
- **2026-07-09 → 07-15:** generated-status governance and licensing (#631–#633),
  correctness/Windows/real-model Q8 closure (#634), two C bug sweeps (#635–#636),
  hosted memory discipline and RuntimeChannel hooks (#637), and the pre-driver
  machine/telemetry/assembly foundation (#638).
- **2026-07-15 → 07-17:** the hosted model lab reached a deterministic random-weight
  train→safe-checkpoint→strict-ingest→BCIRQ8→standalone-C gate (#641). The follow-on
  offline model-development slice added corpus/BPE preparation, SFT/RM/DPO/PPO/reasoning/
  embedding stages, small MLP/GRU/encoder confirmation models, provider-neutral teacher
  and remote-compute contracts, BCIRQ4T/AVX2/SmoothQuant, measured schedule artifacts,
  expanded AD/rematerialization, and workload-scoped numerical-provider evidence.
- **2026-07-17:** payload-free model planning added exact tensor/format/KV/training/bank
  accounting, bounded prefill/decode evidence, resident/layer-stream/host-device candidates,
  and verified claim/StreamPack execution-plan artifacts. Synthetic large headers and the
  existing hosted micro checkpoint validate the rail without large local inference.
- **2026-07-17:** the first bounded hardware-RL slice added availability-aware telemetry tokens,
  bank/link graph and ordered placement encodings, K_BCIR metric rewards, a quarantined
  GNN/Transformer reward+DPO+PPO trainer, bounded root-PUCT, exact per-bank static tensor
  addresses, and measured-only quiescent promotion. Its tiny CI corpus is simulated and proves
  deterministic machinery, not a hardware speedup or live hot-swap.
- **2026-07-18:** the TMSAO sweep pinned canonical data-structure behavior and bounded performance
  evidence across GEM, StreamPack, K_BCIR, unsupervised ML, and the small AI organs; observed
  regressions remain gates rather than theoretical-maximum claims.
- **2026-07-19:** HAM/context-shard/dual-memory planning landed, followed by the measured
  Python/native boundary: portable Q8/Q4 conversion and projection, exact Q15 retrieval, native
  model measurement, and standalone-decoder integration with the Python oracle retained.
- **2026-07-22:** the bounded adaptive-architecture lab added independent LoopDeepNorm,
  fixed-residual variable-width, reference-sliding, H0/H1 exogenous-anchor, and coarse-to-fine
  multi-patch contracts. Tiny one-thread hosted probes, exact size/lower-bound reports, and
  verified claim/StreamPack lowering landed without importing upstream source or changing BCIRQ8.
- **2026-07-22:** the raw-byte laboratory added strict byte/UTF-8 reference semantics, incremental
  entropy and learned patches, local/global/local BLT, joint autoregressive/block-diffusion
  training, exact self-speculative and diffusion-draft verification, a readable MambaByte
  selective-SSM rail, failure-atomic global-weight transplantation, measured ingest selection,
  and R-law/StreamPack lowering. The deterministic CPU gate performs only tiny confirmation runs;
  no pretrained weights, upstream source, GPU kernel, or useful-scale training entered the tree.
- **2026-07-22:** the sequence-interface laboratory added exact-byte DPCA/credit projection,
  continued-BPE expansion and stage ownership, bounded substring/unigram selection, Pareto
  tokenizer evidence, prefix-stable FSQ series coding, frozen binary interfaces, and
  active-budget constructive growth. Tiny hosted training and verified StreamPack lowering landed
  without importing PGT/Embeddings code or claiming useful-scale quality.
- **2026-09-03 → 09-04:** the whole-repository analysis report (#751); the MLIR rail moved to
  LLVM/MLIR 23 with 22 kept in the matrix, plus a documentation currency sweep and the law-range
  gate (#752); the full-surface dependency audit with reproducible tooling and dated evidence
  (#753), then its dispositions: the four major action bumps (#754), the WASM tests off apt's
  EOL Node 18 onto SHA-pinned setup-node / Node 24 (#755), the installed-closure advisory audit
  in the hosted train-to-C jobs — which fired on its first run, on `setuptools` 78.1.0 from the
  PyTorch index, now pinned — with the two pre-commit hook revisions (#756); and the audit's last
  two rows, the vendored LLVM IR grammar refreshed to upstream HEAD and re-verified against LLVM
  23, and the clang-format policy decided by measurement (C++ rails only; `runtime/c` is dense
  hand-formatted C outside any configuration) (#757). The GEM+/TMSAO program was then re-staged
  against the 2026-07/08 assessment: its P−1 blockers dispositioned (four closed, three partly),
  the frozen rows re-measured (G1/G2 unchanged), eight missing contracts added as G11–G18, and
  six stages with PR-sized sections declared (#758). Its first section, S0-A: the EV1–EV3 event
  laws entered the canonical verifier; R9 became scope-aware on both rails — it re-derives the
  planner's actual offer (the old re-derivation rejected every fused consumer), every step cost
  through the one predicate the planner prices with, and budget feasibility, while the C planner
  stopped writing element counts into lane widths and the C R9 re-derives costs; the planner's
  hot paths lost their dataclass copies (−21% plan time on the 4,096-claim fixture, score
  unchanged); two harness rows freeze the finding and its price (`verify.plan.r9.vacuous`
  1.0 → 0.0, `verify.plan.scope.overhead` ≈ 1.07× the planner, G17's row to move); the audit
  command says what it is not. The Python tree was then reformatted once under `ruff format`
  (537 of 555 tracked files; every change proved AST-identical, docstring whitespace aside), ruff
  pinned exactly in the `dev` extra, the last lint findings dispositioned (closures bound at
  definition, two exception chains, `UP042` and lit's injected `config` ignored with their
  reasons), and a CI "Python style" job now runs the hooks' two commands under the same pin, so
  a hook can never rewrite a file CI accepts -- the `ruff (format)` hook had rewritten every
  file the S0-A slice touched, wholesale, because the tree had never been formatted. S0-B took
  the MLIR rail's three correctness-closure items: `bcir-optimize` and `bcir-hydrate` are
  verifier-checkpointed (the two pipelines that advertised checkpoints and ran none), the two
  inert fixtures execute and a quick-tier gate reconciles every pass fixture against the
  runners, the verify/select/GEM passes are anchored at `bcir.module` through one scope
  predicate (a claim can no longer resolve another module's resource; namesake paths price per
  module), and the IRDL projection's subset is a manifest reconciled against ODS both ways
  (38 of 133 operations declared unprojected, the underscore naming rule stated once). S0-C
  (2026-09-05) landed the shared structural-law corpus: `bcir/verify/structural_corpus.py`
  holds 93 rail-neutral cases (widths, alignments, shapes, strides, phase identity and the one
  canonical phase order, the isolated-domain rule, the target descriptor, the address width
  under a declared target, the R13 manifest/portfolio/calibration records, the M5 descriptors,
  the convolution wire domain, the MAP/ROP derived domain) that the quick tier runs on the oracle
  and a generated, drift-gated `structural_corpus.mlir` runs under `check_passes.sh`, every
  mismatch a finding. Seven assessment rows closed with the parent build's defects measured
  first: the oracle folded a zero stride to 1 and refused an HBM-only MAP program; the law rail
  admitted a duplicate phase id, a dangling dependency, an i32 device-register address under
  x86_64 and an unsorted manifest artifact record, ignored a calibration certificate's
  constants, validated no M5 descriptor, sorted phases by numeric id (an exec order that broke
  the phase DAG), and lowered a verifier-legal one-tile convolution to a `count = 0` block.
  S0-D (2026-09-05) widened the two cross-rail content hashes on both rails in one commit:
  `hash_target` folds the memory hierarchy (`target.capability` gains the tier arrays; the law
  rail's walk pins the default hierarchy for IR without it) and `hash_module` folds the
  claims in declared order -- the audit's two collisions, a thirty-fold score move and a
  reordered pair of claims under one digest, now move the hash -- with the emitter writing
  every hashed field and `test_hash_parity.py` holding the rails together. The Codex review
  of #761/#762 (13 findings, 2026-09-05) was triaged under the harvest protocol -- 1 NEW-LAW
  (L22, every rule admits a witness), 10 INSTANCE, 2 LOCAL -- and fixed on both rails in one
  PR: R9 over the emitted plan on the law rail (the third rail of S0-A's re-derivation), the
  ROP pre-scan registering domains with rids, three M5 descriptor twins with their corpus
  cases, the checked calibration derivation, the claimless-module manifest, the address floor
  on the oracle and a pointer-width table with no sub-floor row (and `arm64_32`), the corpus's
  required additional laws, and the two text gates reading their sources as their compilers do.
  S0-E (2026-09-05) landed StreamPack **v4**, the per-resource generation vector (S0-2): an
  append-only record after the trace stream (`n_gens` carved from the header pad at offset 40,
  one `rid/map_gen/data_gen` triple per declared resource in RID order, the header tags pinned
  to the vector's maxima), emitted by every `hydrate`, with the C twin
  (`bcir_sp_for_each_generation`, `bcir_sp_check_generation_vector`,
  `bcir_sp_execute_checked_vector`, the byte-identical re-encode and the DER fast path), the
  ASN.1 projection (`generations [10]`, projection version 2) and the law rail's
  `generations` triples on `bcir.gem.stream_pack`. R11 per resource is one predicate on three
  rails, and the parent's blind spot -- a resource that moved while another held the maximum,
  or one declared after hydration, invisible to the maxima -- was measured RED on each rail
  before it was closed; the mutator keeps the witness (`stale_vector` and its siblings pass
  the maxima-only API and fail the vector). Hand-built packs without a vector stay
  byte-frozen v1–v3; a hydrated vector_add pack grows from 220 to 264 bytes.
  S0-F (2026-09-05) repaired the native measurement rig (GEM+ G7, report P0.3): the strided
  walk `(k * 16) % n` had visited n/16 of a power-of-two buffer (a 2 MiB working set under a
  nominal 32 MiB one) and the rig printed `native microbench (bare-metal)` under any
  hypervisor -- both measured RED on this session's virtualized host before the fix. The walk
  now runs the gcd(stride, n) cosets of the stride, a non-timed census counts the unique
  elements per regime, the rig prints one raw sample per repeat with min/median/max/MAD and
  an attestation of the host (hypervisor flag and nodes, DMI, WSL, container, PMU event
  source, `perf_event_paranoid`, governor, RAPL, clocksource, timer quantum), and derives a
  tenancy from it -- "bare-metal" only with no virtualization signal and an exposed PMU.
  `CalibratedProfile` carries the evidence, re-derives the Q8 ratios from the sample medians
  and refuses a summary or a tenancy claim its evidence does not support;
  `calibrate_native(require_baremetal=True)` refuses every other tenancy with the signals that
  decided it. `strided_order` is the Python twin of the walk, with the parent's n/gcd pinned
  as the witness.
  S0-G (2026-09-05) closed Stage 0's last item, S0-8, the LLVM kernel's runtime-`n` tail
  contract: the single-claim vector kernel had stepped its loop to the runtime `n` itself,
  so any `n` that was not a multiple of the selected width read and wrote past the buffers
  (vector_add at width 16 called with n = 1031 wrote C[1031..1039]; measured RED with the
  new canary harness against the parent's kernel), and a non-divisible compile-time count
  had been legalized to scalar. The kernel now runs its vector loop over `n & -W` and
  finishes the remainder in a scalar epilogue at the selected width, declares
  `epilogue=scalar`, refuses a width the mask cannot express; R12 holds the mask, the
  epilogue and the declaration (an unbounded loop, a missing epilogue and an undeclared one
  are each refused); and the self-check harness -- shared by the AOT, JIT and WASM paths --
  drives every kernel with the planned count, `count + 7`, a sub-width count and zero
  behind 64-element canaries. With S0-A through S0-G landed, Stage 0 of the GEM+/TMSAO
  program is closed and Stage 1 (G1, G3, G11, G5) is unblocked.
  S1-A (2026-09-05) landed G1, the one canonical schedule artifact (the 2026-08-12 report's
  P0.1 and the assessment's rows 6 and 7). The objective had priced fixed greedy conflict
  waves with round-robin affinity bins while the executor ran LPT/EFT placement -- two prices
  for one plan, 51,200 against 25,700 on four independent claims (`pricing.eft.divergence`
  1.9922) -- and both split the sparse GGG tail off BEFORE building any hazard edge, so a
  gather that read what a wave claim wrote started at the phase start (measured RED on both
  rails: the oracle's `schedule_eft`/`execute_tokens`/`price_scheduled` and the law rail's
  `-bcir-schedule-eft`/`-bcir-async`/`-bcir-overlap` all placed the tail at 0 before its
  producer's 5248), and a data-independent claim overlapped a `barriered` fence.
  `gem.schedule.schedule_plan` now places the plan's own step costs by the hazard-honoring
  LPT/EFT dispatch: `concurrency.hazard_predecessors` -- the ONE hazard DAG of the GEM rails,
  RAW/WAR/WAW plus the `barriered`/`volatile` fence edges, the predicate `kbcir.bundle`
  already used -- is built over every claim of a phase before the split, and the tail runs on
  its own stream inside the same event loop. `price_scheduled` reads that placement
  (`.schedule` is the artifact; the divergence row reads 1.0 exactly), `optimize_scheduled`
  sweeps it, both executors return it, and `BCIRSchedule.h` is the law-rail twin shared by
  the four passes (`schedule_hazards.mlir`, emitted by the oracle: identical slots, awaits and
  price on both rails; the gate checks the priced makespan equals the placed one on the
  corpus). The retired wave pricer stays as `price_waves_legacy`, read by nothing but the
  harness witness, which still reproduces the report's 1.9922 on the same fixture. The CSE
  credit had keyed on the op string and the read versions with the barrier guard checked
  afterwards, so a duplicate over a different count, offset or immediate, and an atomic,
  barriered or volatile duplicate, all took the copy credit (RED on both rails);
  `realize.cse_identity` / `cse_eligible` and their `BCIRCostModel.h` mirrors carry the
  complete value identity and the categorical exclusions, and an ineligible claim never
  seeds a match (`cost_model_cse_neg.mlir`, `test_fusion.py`). Durations are exactly the
  plan's step costs, so the serial bound is their sum and R9 holds by construction; the
  re-selection sweep builds the hazard DAG once and places only step-shortening
  alternatives (`optimize_scheduled.slowdown.512` 69.2x -> 6.3x, `optimize_scheduled.512`
  1,148 -> 83 ms A/B on one host, identical assignments to the exhaustive sweep everywhere
  measured). The corpus plans, the matmul 253952 / 761856 overlap, the six-target matrix and
  all 13 performance-audit result digests are unchanged.
  S1-B (2026-09-12) landed G3, the canonical module digest computed once (the report's
  P1.6 and section 5.2). The planner hashed the module, its verifier hashed it again and an
  independent client a third time -- three full FNV chains over a recursively flattened
  item sequence, 65 ms each on this host at 2,048 resources (measured RED: three digests per
  plan-and-verify chain). `provenance.canonical_stream` is now the one iterative walk that
  produces the R13 item sequence and `hash_module` chains it with one reduction per item
  (bit-identical to the recursive flattening on the corpus, 60 generated modules and the
  audit fixture, so no pinned digest moved on either rail); `module_identity` computes the
  digest once per `Module.revision` (bumped by `add_resource`, `add_phase` and the new
  `touch()`) and caches it on the module; `digest_of(module, identity)` is the verifier's
  identity-bound API -- it accepts the identity only when the module's canonical stream is
  exactly what the identity describes (a 3 ms walk against a 37 ms digest) and recomputes
  otherwise, refusing under `strict`. `plan_static_memory` mints the identity once and its
  internal verify validates it; `verify_static_memory_plan` takes an identity from an
  external client; `build_manifest` and `scope_for` read it; R13's `verify_manifest`,
  `replay` and `reproduces` recompute at the trust boundary. The harness gained the exact
  row `static_memory.digests.2048` (3 -> 1) and now measures the three section-5.2 wall
  rows over the audit's own fixture: digest 64.7 -> 37.4 ms, plan 196.1 -> 114.1 ms and
  external verify 100.3 -> 39.3 ms identity-bound on this host, with the audit's 13 result
  digests unchanged. The witnesses: a declared mutation drops the cache, an undeclared
  in-place edit is refused by the content check and reported by the static-memory verifier,
  an appended claim is caught by the census, and module A's identity presented for module B
  is refused.
  S1-C (2026-09-12) landed G11, the plan as bytes (the 2026-09-04 review's binary plan ABI).
  The canonical plan was Python objects: the C twin, the MLIR rail and a resident executor
  could not read it, nothing round-tripped it, and a stale or malformed plan had no bytes to
  be refused by (measured RED: every fixture of every G11 gate failed on the parent -- 26
  corpus plans, 27 reader fixtures, 6 stale and 39 malformed pairs). `gem.execution_plan` is
  now the abstract `ExecutionPlanV1` -- one step per claim carrying the realization and the
  canonical placement G1 made canonical, the static-memory planner's lifetimes, the G8
  movement-edge family (carried and verified, empty until G8 produces edges) and the
  registry's generation vector, minted by `plan_from_realization` bound to the module through
  the S1-B identity API and to the target, read back by `schedule_of` / `realization_of`;
  `abi.execution_plan_abi` is the frozen v1 wire format (64-byte header with the u64 fields
  8-aligned, four length-prefixed record families, CRC trailer, exact consumption, the
  StreamPack's append-only discipline); `runtime/c/bcir_execution_plan.h` is the freestanding
  C twin -- `bcir_ep_verify` applies the same wire laws, `bcir_ep_check_generation_vector` the
  R11 predicate and `bcir_ep_check_pack` binds a pack to its plan by bytes (one segment per
  step with the step's claim, phase, lane and width; identical vectors, an older one is
  `BCIR_ERR_STALE`); `verify_execution_plan` is the oracle's verifier (R9 structure, R13
  binding, the placement re-derived from the plan's own costs, R11, the pack). BCAB gained
  kind 25 / format 13 with the full wire verification in both readers plus the MLIR
  `bcir.artifact.variant` kind, and the `BCIR-ExecutionPlan` ASN.1 module projects the plan
  under DER, OER and JER with the native octets surviving byte for byte. The four harness rows
  read 0 at their bounds (`plan.abi.mismatches`, `plan.readers.disagreements`,
  `plan.stale.accepted`, `plan.malformed.accepted`); on this host the 4,096-claim audit
  fixture's plan is 218 KB against the pack's 713 KB, encodes in 17.8 ms and decodes in
  27.7 ms (the pack: 40.0 / 77.2 ms), and a reader that holds the bytes reads the placement
  without re-running the 37 ms dispatch or re-deriving the digest.
  S1-D (2026-09-12) landed G5, schedule-aware liveness and bounded exact memory (the report's
  sections 6.4 and 6.5), closing Stage 1. RED: the two-phase alias fixture -- A used in phase
  0, B in phase 1 -- planned under phase liveness shares offset 0 and, composed with the token
  placement that runs the two claims at once on different streams, aliases; on a corpus of 500
  seven-resource fixtures of the report's shape first-fit is suboptimal on 40.4% (the report:
  38.6%), worst 1.6x, and the witness that reproduces the report's worst case lays out in 21
  units against a proved 13 (1,344 against 832 bytes at 64-byte alignment). What landed:
  `static_memory.schedule_intervals` derives every resource's half-open liveness interval from
  the canonical placement, `plan_static_memory(schedule=)` computes the plan in that domain and
  binds it to the placement by digest, every plan names its liveness domain, and
  `verify_static_memory_plan(schedule=)` refuses a phase-liveness plan the placement does not
  refine (the alias fixture is REJECTED) while a plan computed from the token placement gives
  the two DISJOINT storage; `exact_layout` is the bounded exact solver behind first-fit (a
  complete branch-and-bound over aligned offsets below the incumbent, budgeted in candidate
  placements), every bank summary records the concurrent-live lower bound, the extent, the
  gap and the stop reason, and the verifier re-runs the solver under the plan's own budget
  rather than trust it. ExecutionPlanV1 gained its first append-only version, v2: the
  liveness byte in the header pad and the tick tail on the lifetime record, the lowest carrying
  version emitted, the alias law by bytes on both rails, and `verify_execution_plan` refusing a
  plan whose lifetimes do not cover its schedule. The three `memory.*` rows read at their
  bounds on the proof rail (0% suboptimal, worst ratio 1.0, 832 bytes on the witness); first-fit
  under phase liveness is byte-identical to the historical layout. The verifier's alias law is
  an exact sweep over the ticks with the live rows' addresses in one sorted list (the recursive
  range-maximum tree it replaces cost 465,000 calls at 2,048 resources): A/B on one host
  against the S1-C commit, `static_memory.verify.2048` 35.0 -> 15.8 ms, `static_memory.plan.2048`
  100.7 -> 84.3 ms, the audit's static-lifetime-planner case 174 -> 142 ms, all 13 result
  digests identical.
  S2-A (2026-09-13) landed G2, incremental delta pricing with identical assignment, the first
  Stage 2 slice. RED: on every corpus program and both harness fixtures the re-selection sweep
  places nothing (no step-shortening alternative exists under any target, Theta or policy), so
  its 6.4x over the serial pass at 512 claims was fixed overhead -- a second placement of the
  artifact it already held and a second fusion pass; and in the general case, reachable with
  the real cost model under ENERGY (a small claim realized vec8 for its large successor's
  locality discount, whose scalar alternative shortens its own step), every trial re-placed the
  whole module: 256 trials x 512 claims, 44.5x the serial pass. What landed:
  `gem.schedule.EftPlacer` records the base placement once per phase (span, entry/exit
  residency, pop order) and prices a trial as the cached prefix, the changed phase replayed
  from the last checkpoint at or before the first pop the change can move, and every later
  phase skipped with its cached span unless it touches a rid the replay moved between streams;
  `_dispatch` became the one-shot form of `_PhaseDispatch.run`, the one dispatch loop the
  executors and the replay share (1,520 placements byte-identical to the parent); the sweep
  reads the candidate map `optimize` built and the artifact the placer holds, and keeps the
  full re-placement as the reference (`delta=False`) the tests hold it to. Outcomes:
  `optimize_scheduled.slowdown.512` 6.4x -> 3.5x (A/B on one idle host: under the 4x bound),
  the general case 44.5x -> 4.7x (605 -> 67 ms), `sweep.replacement.fraction` 1.0 -> 0.0625
  (one phase of sixteen, at the bound); identical assignment, step costs, price and artifact on 1,344
  (fixture, target, Theta, policy) cases; the placer equals `schedule_eft` on 6,840 random
  trials and 1,690 adoptions. Not claimed: a sub-linear replay inside one phase of independent
  claims (the single-phase general fixture goes 44x -> 27x); the `-bcir-overlap-optimize`
  port re-places per trial and still matches the oracle's (makespan, serial) -- the results are
  what parity holds, not the cost.
  S2-B (2026-09-13) landed G4, the bounded exact solvers and the lower-bound stack -- the first
  TMSAO-2 (and TMSAO-1) certificates. RED: the report's section 6.1 corpus was pinned and
  reproduced exactly (all 1,716 nondecreasing six-job multisets with values 1..8; `schedule_eft`
  gives 190 / 1.0078 / 17:15 on two domains and 18 / 1.0013 / 7:6 on three) and nothing in the
  tree could state that distance for a plan -- the four G4 rows were frozen but unmeasured and
  every result was TMSAO-4 by construction. What landed: `gem.exact.exact_schedule`, a
  dependency-free branch-and-bound over one phase's active schedules under the artifact's own
  eligibility rules (the tail stream, the knee), seeded by the heuristic's placement, with
  symmetry breaking on interchangeable streams and identical claims, a budget in node
  expansions and a valid `L` on a budget stop (the least bound over the subtrees never
  entered); the named bound stack (critical path, work over the streams' frontier, bandwidth
  work over the knee, the tail's serial work); `exact_selection` for the section 6.3 sweep;
  and `certify_schedule`, which binds L, U, both gaps, the stop reason, the budget and the
  stack to the `ExecutionScopeV1` digest and lets `certificate_class_allowed` grant TMSAO-1
  when the search closed, TMSAO-2 on a budget stop and TMSAO-4 with the reason when the scope
  is undeclared. The solver is held to oracles that share no code with it: the partition
  optimum on the corpus (1,716 / 1,716 proved on each domain count, at most 3,076 and 6,169
  expansions) and the enumeration of every active schedule on 108 hazard-bearing tiny modules.
  The artifact is never replaced: the placement the executors and the twins read stays
  `schedule_eft`'s. Outcomes: `eft.suboptimal/worst/mean.{2,3}domains` 0 / 1.0 / 1.0 on the
  proof rail (the heuristic's own numbers kept as witness rows), `optimize_scheduled.quality`
  1.0 (the one-sweep selection equals the exhaustive enumeration on 112 corpus cases; the
  report's 55,552 / 55,168 fixture was the retired pricer's), `solver.unproved.fraction` 0,
  `solver.gap.p95` 0. Not claimed: proofs at production scale, the report's remaining bound
  members (roofline, communication cut, queue calculus, occupancy, energy), the memory DP.
  S2-C (2026-09-13) landed G12, the dispatch law, work-unit budgets, resumable search state
  and the plan diff. RED: no solver could resume (every interruption point unavailable: 393
  over the six-job, memory and selection corpora), the exact layout refused a zero budget
  (26 interruption points without an incumbent) and no certificate named the rail that ran
  (12). What landed: `gem.dispatch` -- the law as a table over (region kind, instance size,
  requested class, work budget) to a rail and solver, one runner per region kind, and the
  `DispatchRecord` every `certify_schedule` certificate now carries (rail, solver, units,
  budget, stop reason, bound source, class granted); `exact_schedule`'s budget became the
  module's, consumed phase by phase in topological order, with a content-addressed
  `SearchState` (per-phase frontier: incumbent, placement, open nodes with bounds, expansions)
  so that run(b1) then resume(b2) equals run(b1 + b2) exactly -- 1,118 splits of the six-job
  corpus, multi-phase random modules through three legs; `exact_layout` and `exact_selection`
  gained the same (`LayoutSearchState`, `SelectionSearchState`), the enumeration seeded with
  the sweep's assignment so a budget stop never returns worse than the fast rail; a state
  refuses inputs it was not taken from; `ranked` holds any ranker to a permutation of the
  census and `policy_ranking` orders the portfolio by the L2 gate with every entry kept;
  `gem.diff.plan_diff` names moves, re-selections, re-pricings, re-layouts, the makespans, the
  regret and the bounds. Outcomes: `search.resume.unavailable` 393 -> 0,
  `dispatch.incumbent.missing` 26 -> 0, `dispatch.unrecorded` 12 -> 0; two equal runs are
  identical, states included.
  S2-D (2026-09-13) landed G6, typed regions and the objective registry, affine first (report
  P2, sections 8 and 9). RED: no region concept in the tree -- over the 542-claim region corpus
  no claim was covered by a verified region -- and "semiring" was two attribute names without
  laws. What landed: `kbcir.regions` -- affine regions (maximal runs of consecutive claims of
  one phase whose accesses are 1-D affine maps with static trip counts; the local model is the
  access maps and the dependence distances) and opaque regions (the fallback, with the named
  refusal), each supplying `verify_region`, `expand` (the identity on the carrier: the region
  graph expands to the module claim for claim and the plan selected over the expansion is the
  plan over the module), `region_floor` (the cheapest deforested candidate under the most
  favourable coupling the access maps allow -- never above the plan's score, tighter by exactly
  the discount where no read is shared) and refusal conditions; `kbcir.objectives` -- the typed
  registry (min_plus, max_plus, min_max, boolean, lexicographic, pareto) admitted only through
  `verify_objective` (closure under a declared overflow policy, identities, associativity, the
  commutativity and idempotence of select, distributivity where claimed, the realized strict
  order), with `dag_best_path` reproducing the planner's min-plus path on 200 DAGs and the exact
  scheduler's max-plus critical path on 33 phases; `gem.exact.certify_selection` and the
  dispatch law's `path` kind (the min-plus rail is exact: L == U == the optimum, TMSAO-1 under a
  declared scope, the structural floor and the coupling's price reported). Outcomes:
  `regions.unexpanded.claims` 542 -> 0, `objectives.unverified` 2 -> 0; the native guardrails
  A/B on this host (virtualized) 4.53 -> 4.44x, 15.00 -> 15.56x, 1.315 -> 1.314x, 1.003 ->
  1.004x -- no regression. Not claimed: the other region kinds (opaque today), a polyhedral
  depth above one, a law-rail registry attribute, a silicon certificate.
  S2-E (2026-09-13) landed G13, the workload component W, the measured-candidate corpus and
  the replay-gate integration. RED: no certificate could declare a workload (three workloads
  on one program shared one scope digest on all 36 pairs), the portfolio admitted a
  certificate with one clean episode whatever the log held (24/24 subset promotions), and a
  TMSAO-3 request went to the fast rail whatever evidence existed (12/12). What landed:
  `kbcir.workload` (`Workload`: shapes, batch, concurrency, service level, horizon, the
  expected counts of dynamic claims; validated, digested, held to the module, classed for the
  L2 table; carried as W by `scope_for` and the certifiers), `kbcir.measured` (`MeasuredPlan`
  with raw samples, counters and the host attestation; the append-only chained
  `MeasuredCorpus`, refusing altered, removed, reordered or forged evidence on read;
  `measure_plans`; B1's artifacts adapted; `replay_measured` over every logged episode with a
  corpus-bound `ReplayCertificate`), `PolicyPortfolio.select` over (runtime, workload), the
  dispatch law's measured rail and `solve_measured` (the plan re-derived from the planner,
  stale evidence unused), `gem.exact.certify_measured`, and the ladder's
  `MEASURED_COMPONENTS` and two-target rule. Outcomes: `scope.workload.collisions` 36 -> 0,
  `replay.subset.admitted` 24 -> 0, `dispatch.measured.unavailable` 12 -> 0; every measured
  certificate on this virtualized host is TMSAO-4 with the reason. Not claimed: a silicon
  certificate, W in the cross-rail manifest, a distribution beyond dynamic claims' expected
  counts. Stage 2 is complete.
  S2-D/S2-E follow-up (2026-09-13) closed three findings an adversarial audit of the two landed
  slices raised against itself. (1) The S2-D entry above claimed `dag_best_path` was checked on
  "300 DAGs" and "194 phases"; instrumenting the landed test counts 200 DAGs and 33 phases, so
  the two figures are corrected here (the law held -- an independent sweep reproduced min-plus
  against `semiring.dag_shortest_path` on 2,800 DAGs with no mismatch -- only the counts were
  overstated; commit 5b45fdca's message carries the old numbers and cannot be rewritten).
  (2) `kbcir.regions` listed `Region` in the package export table, which `kbcir.compose` already
  owned: the flattened `_NAME_TO_MOD` kept the later entry, so `bcir.kbcir.Region` silently
  changed from the composite-plan alias to the region dataclass and `compose.Region` became
  unreachable through the package while still advertised in `__all__`. The region dataclass is
  no longer exported at package level, and `test_perf.test_the_export_table_has_no_duplicate_names`
  now refuses any name claimed by two modules across all three lazy packages -- the collision
  class, not the instance (L14). (3) The four `native.*` rows were graded against the report's
  host though they are ratios of two separately COMPILED kernels, which measure the host's gather
  penalty rather than cancelling it; on this machine that tripped a REGRESSION verdict on roughly
  one run in thirty. `Metric.host_dependent` marks them, so they are reported INDICATIVE off the
  baseline host and read as the same-host A/B, which is what the S2-D outcome table always stated.
  S3-A0 (2026-09-17) moved the one freestanding SHA-256 in runtime/c out of the BCAB reader
  into `bcir_sha256.{h,c}` verbatim and added HMAC-SHA256 over it, with FIPS 180-4 and RFC 4231
  vectors (one-shot and byte-at-a-time) as its gate; the BCAB section, the C++ hand-off and the
  thorough tier stayed byte-identically green. It was split out so the control plane's slice
  carries no refactor.
  S3-A (2026-09-23) landed G14, the control-plane record ABI -- the first Stage 3 slice. RED
  (measured on the parent, 544619e2): no codec, no plane, no C twin and no verifier entry point
  existed, so every G14 fixture failed by absence -- 29 corpus records, 82 (malformed variant,
  rail) pairs, 24 stale (fixture, rail) pairs plus the verifier's missing boundary (the trusted
  loader's and context-shard activation's own checks held, run against the parent's modules),
  16 mid-phase pairs, 64 decision pairs, 52 traces. What landed: `ControlRecordV1` (a 64-byte
  header with the `expect` compare-and-swap witness, one fixed body per kind, a lease-scoped
  HMAC-SHA256 and a CRC; eleven wire laws in one order, the encoder refusing what the decoder
  refuses; `docs/kernel/BCIR_CONTROL_PLANE_ABI.md`), `bcir.gem.control` (the values, the shared
  `is_stale` the loader and context-shard activation now express, `registry_digest` over the
  generation vector's own bytes, and `ControlPlane`: a resident state that decides each record
  applied, deferred or refused -- a refusal inert, one pending switch applied exactly once at a
  boundary, eight leases whose ids never recur -- and admits packs and plans against the
  installed registry), `verify_control_record` (R11), the freestanding C twin
  `bcir_control_plane.{h,c}` (statuses 17 and 18, append-only) with a `--api` harness for the
  fail-closed laws no record reaches and a structure-aware libFuzzer target that seals records
  under the plane's key and asserts the plane's invariants, the Python decoder campaign's
  `control` surface (CRC-repaired mutants, so it reaches the field laws), and the
  `BCIR-ControlPlane` ASN.1 module (OID 62596.4; the body a CHOICE whose alternative is the
  kind; DER and canonical OER byte-identical between the compiled source and the hand-built
  model). Outcomes: `control.abi.mismatches` 29 -> 0, `control.malformed.accepted` 82 -> 0,
  `control.stale.accepted` 25 -> 0, `control.deferred.lost` 16 -> 0,
  `control.decisions.nonconforming` 64 -> 0, `control.traces.divergent` 52 -> 0; the C gate
  proves it fires (the deferral law removed turns three rows red). Not claimed: the BCIR UAPI,
  a transport (G15), a signature scheme, capability enforcement beyond the record, replay
  protection beyond sequences, the witness and never-recurring lease ids.
  S3-B (2026-09-23) landed G15, the live SPSC ring and the version-zero triple the driver
  roadmap requires before any D2 driver. RED (measured on the parent, 83c6c015): the only shared
  telemetry ring was the v1 snapshot -- eight records published into four slots handed a reader
  a lap behind the four survivors and no loss count, and a slot read three fields into a rewrite
  came back with the fields of two records and no error -- and no ring, envelope, generated
  table or intake existed on either rail, so every fixture failed by absence (29 / 112 / 140 /
  140 / 22 / 10 / 70 / 104 / 6 across the nine rows). What landed: `TelemetryEnvelopeV0` (source,
  session, generation, signal, sequence, the producer's own loss count and a clock with its
  unit; wire laws in one order on both rails, one spelling per record;
  `docs/kernel/TELEMETRY_ENVELOPE_ABI.md`), the generated 64-byte-row signal table with the
  BCIR/vendor/device ID ranges and the unknown-required-signal law, the host intake (a bounded
  stream table; continuity classified before any refusal by `SequenceTracker`, which the BTLM
  decoder now expresses too; stale generations refused by G14's `is_stale`), and the ring
  (`bcir.gem.ring` and the freestanding C11 twin `bcir_ring.{h,c}`; seven one-writer cache
  lines, a per-slot seqlock under acquire/release publication, BACKPRESSURE or OVERWRITE with
  exact loss accounting, double-buffered accounting, epochs with takeover of a peer proved dead;
  `docs/kernel/BCIR_LIVE_RING_ABI.md`), statuses 19-22 appended. Control records ride a
  BACKPRESSURE control ring, identically on both rails for all 52 G14 scenarios. Outcomes: the
  nine exact `ring.*` rows 0; `ring.throughput` 26x a `memcpy` of the same bytes on the
  reference host (INDICATIVE, and dominated there by thread placement: ~13-49x pinned per vCPU
  pair within one hour, a minimal unchecked queue ~1.1-6x -- so no ring change, including the
  one-writer-per-line layout's early ~190-217 -> ~127-153 ns A/B, is attributable on that
  host). The C gate adds
  ThreadSanitizer (and the race it must report once the atomics are made plain), the
  seqlock-removed mutant it must fail, and -O0 == -O3 == the oracle; `tools/c/check_ring.py` is
  the one grading entry point and `tools/testing/faults/ring.json` the committed fault table.
  Found and fixed in the slice: the live ring's C API claimed three names the v1 emitter emits
  (renamed while still version zero, with a one-program witness), and a 192-byte control slot
  could not carry the control ABI's declared 192-byte bound (a CONTROL ring now needs 256-byte
  slots). The lease-key cache G14 deferred landed with it: a lease's key is derived once at the
  grant, and leased records verify as fast as root-key ones (C ~3.7 -> ~2.3 us). Not claimed:
  MPSC, death detection, blocking or wake-up, authentication of what the ring carries, a
  transport beyond one host, any freeze before the UART and virtio-blk traces.
  S3-C (2026-09-24) landed G16, the data-plane hand-off, and with it the Stage 3 exit gate.
  - RED, measured on the parent (1ee34676) with a real harness over the parent's own seam:
    - `admit(data, len)` admitted 9 of 9 stale packs by default, and 8 of 9 when handed the live
      maxima;
    - `dispatch()` ran every stale pack;
    - a view into a reused buffer dispatched the next step's bytes, and a freed one read freed
      memory (ASan);
    - the dynamic-graph backend was a stub, and there was no manifest-of-shards;
    - with the rails absent, every fixture failed (42 / 90 / 90 / 68 / 80 / 422 / 349 / 392 / 68 /
      115 / 77 / 18 / 75 across the thirteen rows), and `handoff.dispatch.overhead` was 1.95x.
  - What landed:
    - The pack table (`bcir.gem.handoff` and the freestanding `bcir_handoff.{h,c}`;
      `docs/kernel/BCIR_DATA_PLANE_HANDOFF.md`). Write-once slots with epoch-bearing handles that
      never wrap; pins, and retirement instead of reuse under a reader; admission through the live
      plane's `bcir_ctl_admit_pack`, recording the generation and the registry digest; dispatch
      in place as a plane phase; the state digest.
    - The per-step freeze (`bcir_hydrate_generations`, byte-identical to `freeze_claims`).
    - BSHM v0, the manifest-of-shards (`bcir.abi.shard_manifest` and
      `bcir_shard_manifest.{h,c}`; `docs/kernel/BCIR_SHARD_MANIFEST_ABI.md`): the hydrated-layout
      law, canonical sub-packs and a frame, one total reassembly predicate.
    - The C++ RAII seam (`PackArena`, `Reservation`, `PackOwner`, `PackView`, `Borrow`,
      `GraphBuilder`). `Orchestrator::admit` is no longer virtual, the dynamic-graph backend is
      real, and the distributed `cut()` is real.
    - `bcir_ctl_pack_registry_digest`, shared by admission and the manifest.
    - Statuses 23 (`LIFETIME`) and 24 (`SHARD`) appended.
  - The Stage 3 exit flow (`runtime/c/test_stage3.h`, and `run_stage3_python` on the oracle)
    carries one generation through a live control ring, the plane, the pack table, a two-slot
    telemetry ring and the intake. After the switch it offers the old generation at six
    boundaries, and each one refuses it. The evidence reconciles the ring's loss with the intake's
    gaps. It prints 31 identical lines on three rails.
  - Outcomes: all thirteen exact rows are 0, and `handoff.dispatch.overhead` is ~1.01x (the
    whole-pack verification moved from every dispatch to `admit()`, once). The C and C++ gates
    each carry a mutant they must fail, and everything also runs under ASan and UBSan. A
    libFuzzer target covers manifests, shard sets, table scripts and freezes, and
    `tools/testing/faults/handoff.json` injects 34 defects, each caught by its own row.
  - Found and fixed in the slice:
    - the registry binding had no witness, because in one-plane scenarios the generation check
      alone refused every stale dispatch; a restarted plane now witnesses it both ways;
    - the oracle spelled the frame three times, and the public spelling was unmeasured;
    - the grader tracebacked when the oracle could not build its own corpus.
  - Not claimed: a cross-node transport, a pack table shared across processes, and reduction
    across ranks.
  S4-A (2026-09-24) landed G17, the compact planner and its native twin, and opened Stage 4.
  - RED, measured on the parent (731373df):
    - planning the audit's 32,768-claim fixture made 3,419,172 calls (CPython 3.11): a
      `Candidate` and a `CostVector` built twice per claim, twelve coupling calls per edge, and the
      weights re-derived per claim;
    - R9 re-derived the planner's whole offer, and `verify_plan` cost as much as planning (1.07x);
    - with the rails absent, the parity rows read 8,430 and 148;
    - R9, graded over every field a step carries, misjudged 232 verdicts. It raised on a plain-int
      lane and on an unhashable phase, and never bound a step to its claim's phase.
  - What landed:
    - `realize.fused_offer`, the offer as compact rows: one enumeration (`_offer_rows`) over one
      arithmetic (`_base_cost`), the discounts in the same pass, and no candidate objects.
      `fused_candidates` and `result.cand_map` (a lazy `OfferMap`) are views of it.
    - `optimize`: the weights derived once per phase, each realization priced once for both path
      contexts (`_edge_cost_pair`, which `edge_cost` delegates to), and each column relaxed
      against the previous column's cheapest narrow and wide predecessor, over two flat arrays.
    - The pre-G17 planner, kept verbatim as `realize_reference`, is the "before" the parity gate
      holds the compact planner to.
    - R9's single-candidate re-derivation, the phase-binding law, and diagnostics that are total
      over forgeries.
    - The native planner (`bcir_kplan.{h,c}`, freestanding) over the version-zero BKPI and BKPR
      records (`bcir.abi.planner_abi`; `docs/kernel/BCIR_PLANNER_ABI.md`). It is exact in 128
      bits over the declared domain and refuses only a value the record cannot carry, exactly
      when the Python encoder does. `BCIR_ERR_PLANNER` (25) is appended, and
      `bcir_utf8_valid` is now the runtime's one UTF-8 validator.
  - Outcomes:
    - All three exact rows are 0. `planner.calls` is 589,858 (5.80x fewer), and
      `verify.plan.scope.overhead` is ~0.73.
    - The whole K_BCIR->StreamPack chain at scale 8 went from 9.20 M calls to 3.51 M. The
      `audit.kbcir-streampack.scale4` wall row, measured A/B against the parent, went from
      201-219 ms to 117-137 ms, with the same result digest. That is indicative only; the
      chain's unchanged stages (`verify`, hydration, `verify_pack`) now dominate it.
    - The native planner plans the 4,096-claim fixture in ~3.4 ms, against ~33 ms for the
      compact planner and ~73 ms for the reference.
    - The C gate carries a mutant it must fail (the 128-bit carry dropped), and a libFuzzer target
      and two decoder-campaign surfaces landed.
    - `tools/testing/faults/planner.json` injects 29 defects, each caught by its own row.
  - Found and fixed in the slice:
    - The wide-path fixture looked like a witness to the 128-bit arithmetic, but no path in it
      depended on the addition's carry. `carry_case` does.
    - The first sweep missed three faults, and each time the witness was wrong, not the code: a
      corpus that rewrote an operand only once, R9 graded only with the scope that masks its base
      law, and a malformed variant that a later law refused with the same status.
  - Not claimed: the MLIR `-bcir-plan` pass (unchanged), the CXX3 joint solvers (Python only),
    and a certificate produced natively.
  S4-B (2026-09-24) landed G18, the delta chain, and closed Stage 4.
  - RED, measured on the parent (825888e9, the S4-A head):
    - there was no delta: a one-claim edit of the audit's 32,768-claim fixture meant the chain
      from scratch, 10,855,666 calls (CPython 3.11), of which the StreamPack encoder alone is
      7.4 M;
    - over the final corpus, with the mechanisms absent, the rows read 2,064, 2,064, 4,128 and 51.
  - What landed (`docs/kernel/BCIR_DELTA_CHAIN.md`):
    - the declared `Delta` (claim and resource replacements, each addressed by its own id or
      RID). Every rail refuses what v0 does not admit with `DeltaError` before anything moves.
      `apply_delta` is the reference application;
    - `IncrementalOffer`: `fused_offer` with its position indexes, re-deriving the delta's
      dependency cone. The discount is one rule table, `realize._DISCOUNT`, which both
      evaluations of the offer read;
    - `IncrementalPlan`: the one relaxation `optimize` also runs (`realize._relax_column`), a
      cutoff where the next column's input is unchanged, lazy shifts, and the path spliced where
      it rejoins the old one;
    - `PackState`: per-record encodings in chunks. Only the changed records are re-emitted, with
      the records `streampack.step_records` / `double_buffer` build and the encoder's own
      contract functions and writers, which were factored out of `encode` byte-identically;
    - `VerifyState`: the three verifiers refactored into units, byte-identically over 13,288
      modules and 45,195 plan/pack verdicts. What changed is found by object identity, and the
      state keeps an offer of its own;
    - `DeltaChain`, with the fixtures' own reference (`declared`, `reference_chain`) that runs
      on a tree without the mechanism.
  - Outcomes:
    - All four exact rows are 0.
    - A one-claim delta costs 491 calls at scale 8 and at scale 4. The time ratio is ~0.0098 at
      scale 4 and ~0.007 at scale 8, and the build is ~1.45 chains from scratch.
    - The chain from scratch fell to 10.14 M calls (`verify` 1.65 M → 0.58 M). Its wall row is
      neutral in an A/B.
    - `tools/testing/faults/delta.json` injects 31 defects, each caught by its own row.
  - Found and fixed in the slice:
    - `apply_delta` admitted a module declaring a claim id twice when the delta named another
      claim, while the states refused it. One predicate, `_unique_where`, now decides it on
      every rail.
    - A `Delta` whose replacements were not tuples raised a `TypeError`.
    - Sharing the relaxation cost `optimize` a call per column until each phase was weighed once
      per run; `planner.calls` stays 589,858.
    - The first fault sweep missed four defects, and each miss was a finding: a prefetch counter
      no reachable value could observe (removed), a forgery pair the rotations could never
      schedule (coprime rotations), and two verdict paths reachable only through a stale plan
      (forged on every cone round).
    - A profiler window could catch a finalizer, so the call rows now collect first and pause the
      collector.
  - Not claimed: a native or MLIR twin of the incremental chain, deltas that change the module's
    shape or the scope, incremental event laws (rebuilt, counted), and sublinear wall time (the
    per-delta copies are O(n) at C speed).
  S5-A (2026-09-25) landed the G9 remainder: the declared alias facts carried the rest of the way
  to LLVM.
  - RED, measured on the parent (ad4ebff0, the S4-B head), over the final corpus (504 lawful
    kernels, 112 modules to refuse, six emitters, five self-checks per plan):
    - the landed half held (`alias.llvm.false_noalias` 0), but no access carried a scope or a TBAA
      tag (2,646 each);
    - `volatile` was dropped on 1,533 facts, and 742 barriered kernels had no fence;
    - 602 modules the subset cannot lower were lowered, and 336 emitted kernels failed their own
      R12;
    - 1,072 one-fact differentials were silent, 75 self-checks ran three private buffers whatever
      the RIDs said, and 9 correct kernels failed the WASM self-check;
    - inlined into a C caller, the kernel forced 56 redundant accesses to the caller's own data;
    - R12 accepted 1,084 forged kernels (21 kinds of one fact dropped, added or contradicted).
  - What landed (`docs/kernel/BCIR_ALIAS_FACTS.md`):
    - one derivation, `lower.alias_facts.kernel_facts`: the RID partition (exclusive pointers),
      the element type the lowering names and every operand's declared size, the volatility, the
      hazard's fence. It replaced four predicates that disagreed; the hot-shape specialist's
      wrote `restrict` on all three pointers of an in-place claim;
    - the LLVM kernel: `noalias` on exactly the exclusive pointers, one alias scope per resource
      on every access, clang's TBAA tag for the element type, `volatile`, and a barrier's
      `fence seq_cst` first and last. The C kernel, gather form, specialist, ABI header and
      Q-fixed kernel carry `restrict`, `volatile` and `atomic_thread_fence`;
    - refusals where there were miscompiles: `atomic` and unknown hazards, an operand resource
      that is undeclared or declares another element size;
    - R12 holding every fact on both backends (`verify.alias`, total), a false fact and a dropped
      one named apart;
    - every self-check bound one buffer per declared resource (the LLVM AOT/JIT harness, the C
      and Q-fixed self-checks, the WASM node harness).
  - Outcomes:
    - All fifteen rows are 0 (`python tools/perf/check_alias.py --require-llvm`). LLVM's own alias
      analysis proves every declared-disjoint pair from the scopes alone, and clang's IR for the C
      kernel carries the LLVM kernel's facts.
    - The kernel alone is at its bound: its `-O2` code is byte-identical to the parent's (112 of
      112, clang 18 and 23). Inlined into a C caller, the TBAA tags remove one dead store and one
      reload of the caller's data per call site, and a barriered kernel keeps both.
    - Every source the `native.*` rows compile is byte-identical to the parent's (30 of 30).
    - `tools/testing/faults/alias.json` injects 43 defects, each caught by its own row.
  - Found and fixed in the slice:
    - Every emitter lowered `volatile` claims to plain accesses, and ordered hazards to no fence.
      R12 rejected every such kernel, and a volatile claim could never produce one it accepted.
    - Both backends addressed every operand as 4-byte elements whatever the resource declared.
    - The ABI header published "A,B,C are non-overlapping" for every claim.
    - The WASM node self-check computed `+` whatever the claim's operation.
    - The baseline harness could not grade a zero-baseline guard row as a regression.
    - Designing the fault table found two grader gaps: the forgery row accepted any new R12
      message where it needed a finding naming the forged fact, and no module named an undeclared
      resource.
    - A review of R12's reader before landing found it reading a subset of both languages (L4). An
      access spelled without its alignment was invisible to it, so a plain load in a volatile claim
      passed. A fence narrowed to `syncscope("singlethread")` counted as the barrier. In C, a read
      through a cast or an address shed `volatile`, and an early `return` skipped the exit fence.
      Six forgery kinds now witness the rule (252 forged kernels, 196 of which the earlier reader
      accepted).
    - The Q-fixed self-check compares in 64 bits again, as it did before the binding moved into
      the shared harness, so a result that does not fit the lane cannot be narrowed until it
      agrees.
  - Not claimed: kernels beyond the single-claim subset (where scopes carry what `noalias`
    cannot), atomic element operations (refused), the Q-fixed kernel's element size (a
    representation question recorded, not settled), and a law-rail emitter (none exists; the C
    backend is the second path to LLVM, held to the same facts).
  SP-ENC (2026-09-25) compiled the StreamPack encoder's record layouts: S4-B's first
  recommendation, taken up at the user's request.
  - RED, measured on the parent tree (`b1f8bdce`; S5-A does not touch the encoder): one encode of
    the audit fixture's pack at scale 8
    made 7,543,875 calls (74% of the chain from scratch, 10,110,215), and took 66 ms at scale 4.
  - What landed (`bcir/abi/streampack_abi.py`):
    - one function per record kind, and two ways to build a record. A plain record (every field
      of its exact type, no fence names) is packed in one `struct` call through the precompiled
      layout of its shape. Any other record is written field by field and refused exactly as the
      `_Writer` rail refused it;
    - an exact fast path for the encode contract per record.
  - `struct` is not the contract: it packs `True` and any object with `__index__`, which the wire
    refuses, so a layout takes only exact types (L4).
  - The encoder before is kept verbatim as `encode_fixtures.encode_reference`, and
    `streampack.encode.parity` holds the two byte for byte and refusal for refusal: 588 honest
    packs (49 packs, each in 12 wire versions and spellings), 510 packs with one or two fields
    forged, and 1,202 calls handing the forged records to the record functions directly.
  - Outcomes: `streampack.encode.calls` 7,543,875 → 989,245; `streampack.encode` 1.0 → ~0.30;
    `kbcir-streampack.full.calls` 10,110,215 → 3,555,585; the parity guard 0; the C re-encode
    still byte-identical. The delta chain re-emits through the same records: a one-claim delta
    491 → 472 calls. `tools/testing/faults/encode.json` injects 22 defects, each caught by
    its own row.
  - Found and fixed in the slice:
    - the first plain layout read a missing fence array as an empty one (L14);
    - the first sweep charged one defect to the wrong row, because the call floor's refusal hid
      the parity row's finding (L1);
    - the array helpers walked an array twice (checked, then packed), which is sound only for the
      tuple or list the plain layouts' guard admits. The field path, their second caller, gets
      everything else, so an iterable that yields its items once raised `struct.error` where
      `_Writer` packed it (L14). The corpus's "a generator where a sequence belongs" had never
      been built; it now holds a generator, a one-pass iterable and one whose `len()` is short,
      and four faults are caught only by them.
  SP-DEC / SP-REC (2026-10-03) compiled the StreamPack decoder's reads and made the five record
  kinds slotted value classes, for the rows the #786-vs-#797 benchmark left within noise.
  - RED, measured on #797 (`7ae15ac2`): `decode@4` 95 ms and 777,497 calls for 4,096 segments --
    `_Reader._take` 169,730 of them, a method call, a bounds check and a `struct.unpack` per field;
    `hydrate@4` 27 ms, 17 ms of it building four frozen records a step: a frozen dataclass's
    generated `__init__` assigns each field through `object.__setattr__`, 4 microseconds a record,
    and each record carried an instance dictionary the cyclic collector walked beside it.
  - What landed: `decode` reads the body in place (`bcir/abi/streampack_abi.py`) -- the position
    a local, each fixed group of fields through one precompiled layout, each array through the
    layout of its count (cached, bounded), each string through the `bytes` slice -- with every law
    of the field-by-field reader in its order; where a fixed group does not fit, `_Reader`
    finishes the record, so a cut is refused at the field it was refused at. `LaneSegment`,
    `Prefetch`, `Block`, `TraceNote` and `Generation` are `@dataclass(slots=True,
    unsafe_hash=True)`: built by the generated `__init__` in 0.7 microseconds, no dictionary, the
    same equality, hash and repr as the frozen records (a record is a value; `dataclasses.replace`
    changes one). An at-once instance dictionary was tried first and measured out: 613 bytes a
    record against 278, and the collector's share grew with it. `verify_pack` reads the registry
    once per segment where it called `module.resource` per RID.
  - The reader's decoder is kept verbatim as `tests.decode_fixtures.decode_reference`, and
    `streampack.decode.parity` holds the compiled decoder to it pack for pack and refusal for
    refusal over 15,734 items: every honest pack in every spelling, one pack cut at every length
    and every byte of it replaced by seven values, raw and with the CRC remade, and a replaced
    byte beside a cut just past it (a law checked out of the reader's order is found, not only a
    law dropped). The records' equality, hash, repr, `asdict`, `replace`, `copy` and pickle are
    held to a frozen twin (`test_a_streampack_record_is_the_frozen_records_value_and_a_slot_cheaper`).
  - Outcomes (median of 9, the heap collected before each sample, #797 -> now): `decode@4` 95 ->
    37 ms, `decode@8` 909 -> 379 ms, `hydrate@4` 26.8 -> 14.0 ms, `hydrate@8` 341 -> 285 ms,
    `verify_pack@4` 7.3 -> 6.1 ms; every pack, plan and verdict byte-identical; `encode` untouched.
  - Faults: `tools/testing/faults/streampack-decode.json`, 12 -- the empty fence array, the fixed
    group read where it does not fit, the lane and buffer-count laws, the width law checked out
    of order, an array and a trace note read past the buffer, the trailing-bytes law, the UTF-8
    reason, a record compared by identity, a record with a dictionary again, and the verifier's
    written RIDs against no registry.
  PERF-SCHED (2026-10-03) restructured the hot loops of the GEM schedulers, the phase-DAG
  traversals, the executor and the verifier's per-claim laws, for the rows the #786-vs-#797
  benchmark left within noise, with every schedule, order and verdict the historical one.
  - RED, measured on #797 (`7ae15ac2`) over the mixed scheduler fixture (2,048 claims, 64 shared
    resources, 8 domains) and the deep phase DAG (2,048 single-claim phases declared in reverse),
    the min of 21: `sched_waves@4` 3.3 ms, `sched_tokens@4` 3.6 ms, `sched_eft@4` 15.7 ms,
    `dag_exec@4` 2.6 ms, `dag_verify@4` 10.1 ms. The conflict and frontier predecessors sorted
    every claim's predecessor set through a position dictionary's `__getitem__`; the wave
    indices called `max` five times a claim; the fence pass walked every phase to add nothing;
    the dispatch read each duration through a closure (4,096 calls a placement), picked a
    stream in two passes over the eligible streams and built a frozen slot a claim (an
    `object.__setattr__` a field); the traversals rebuilt a tuple frame per dependency; the
    verifier resolved a claim's references into two lists and walked them four times, and
    listed every single-claim phase's sparsity for a pair scan that had no pair to visit.
  - What landed. `gem.concurrency`: the per-resource histories hold positions, so a claim's
    predecessors sort as integers and are spelled as ids once, a claim whose conflicts all lie
    in one history takes it as it is, and the frontier keeps one position until a second one
    joins it; the wave indices compare where they called `max`; a phase without a fence leaves
    the fence pass at its first line; sparsity is read once a claim. `gem.schedule`: `Slot` is
    `@dataclass(slots=True, unsafe_hash=True)` (SP-REC's finding); the dispatch reads every
    duration once into a table, picks the stream in one pass that keeps the running minimum of
    the key `(finish, -score, index)` and scores the running best the first time a later stream
    ties it, and binds the heap and the slot list as locals. `gem.execute`: the ids of a phase
    are spelled once and appended as a list, the telemetry written once. `model.graph`: a frame
    of the two traversals is the phase id and an iterator over its dependencies, resumed where
    a push left it; `Resource.count` is `math.prod`. `verify`: one walk of a claim's resolved
    references for R2 and the three R3 rules, their diagnostics in the order each rule listed
    them (R2 and R3 are separate sections); the pair law returns at a phase of one claim; the
    phase law reads the declared ids from the set it built; an unknown access-pattern shape
    admits no lane through one shared empty set.
  - The code before the slice is kept verbatim in `tests.sched_fixtures` (the two traversals,
    the five concurrency functions, the token plan, the dispatch, the executor and the four
    verifier laws), and `gem.schedule.parity` holds the faster rails to it program for program:
    400 random programs built to reach every branch (shared and private resources, repeated and
    overlapping RIDs, `barriered` and `volatile` fences, GGG and random tails, unpriced and
    zero-cost claims, one to eight domains with and without the locality tie-break, dangling and
    cyclic dependencies, duplicate phase and claim ids, undeclared RIDs, isolated domains, every
    illegal lane, hazard, bound, extent and cost class) x 16 rails and more (the canonical order,
    the cycle verdict, the verdict, the executor with and without kernels, the waves, the token
    plan, each phase's hazard DAG and frontier, the EFT placement on the frontier and on the full
    DAG, the placer) = 6,666 outcomes, 216 of them refusals: 0 mismatches
    (`test_the_restructured_schedulers_are_the_historical_ones_program_for_program`). `Slot` is
    held to a frozen twin (`test_a_slot_is_the_frozen_slots_value_and_a_slot_cheaper`).
  - Outcomes (the min of 21 / the median, #797 -> now): `sched_waves@4` 3.3 / 3.5 -> 2.4 / 2.5
    ms, `sched_tokens@4` 3.6 / 3.8 -> 2.5 / 2.7 ms, `sched_eft@4` 15.7 / 16.3 -> 10.0 / 10.5 ms,
    `dag_exec@4` 2.6 / 2.9 -> 2.2 / 2.3 ms, `dag_verify@4` 10.1 / 10.4 -> 7.0 / 7.3 ms; every
    wave, tail, affinity, fork, await, slot, makespan, order and diagnostic byte-identical.
  - Faults: `tools/testing/faults/gem-schedule.json`, 24 -- a history dropped from a merge, a
    first reader unrecorded, readers kept across a write, a writer position displaced, the fence
    pass skipped over a volatile fence, a read in its writer's wave, the tail reversed, an
    unpriced claim at no cost, both tie-breaks of the stream pick, a slot compared by identity
    and one with a dictionary, the executor's telemetry and a skipped kernel, forks out of
    order, a dangling dependency visited, a cycle missed, a zero-extent count, and five verifier
    rules (an MMIO read, an untouched claim, a two-claim phase, the declared ids, the H lane, an
    exact-extent access): each caught.
  PERF-AUDIT (2026-10-03) removed the Python overhead around the numeric kernels behind the
  TMSAO audit groups the #786-vs-#797 benchmark left within noise, with every floating-point
  operation kept in its order: the audit's `correctness_sha256` at scale 4 is the same digest
  before and after (`b57fc635...`), case for case.
  - RED, measured on #797 (`7ae15ac2`), the median of 9 at scale 4: `bounded-overwrite-ring`
    28.7 ms (378,040 calls: a schema check of seven `getattr`s, seven `_is_i64` calls and two
    generator scans per record, written and read), `q8-q4-blocks` 30.2 ms (456,364 calls: a
    `_quantize_code` and a `_round_half_away` call per value, a per-code validation loop per
    group, a nibble loop per code), `kmeans-knn-scaler-embedding` 40.3 ms (eight distance calls
    a point, two `float` conversions an element), `tiled-matmul` 151 ms (two index products and
    two subscripts per multiply-add), `ols-pca` 4.5 ms, `transformer-block` 9.6 ms, `lstm-gru`
    6.2 ms (a `_row_dot` call per unit per gate, converting its weights per use), `autodiff-adam`
    167 ms (1,437,129 calls: a forward pass per logit for the loss and another inside its
    gradient, a frozen `Node` per interned subexpression, `tape.node` 270,320 times),
    `bounded-mcts` 1.4 ms (a closure and a keyed `min` per simulation).
  - What landed. `telemetry.DataDNA.violations`: the valid record decided in one expression of
    plain-type and range checks, the control-character scan a compiled class, the walk that
    names each violation kept for anything else. `kbcir.quantize`: `quantize_group` converts,
    scans and reduces through `map`, rounds every quotient in one comprehension with the
    rounding chosen once, and saturates only when a clamped scale asks (the band's edge is the
    one case); `QGroup` admits a group of plain in-lane codes from the type set and the extremes
    and walks the rest. `kbcir.lowbit`: the packer pairs codes, the unpacker spells nibbles
    through a 16-entry table. `kbcir.unsupervised`: the nearest centroid over float storage
    (`kmeans_assign` converts its centroids once; `kmeans_fit`'s are floats by construction)
    with the squared distance spelled in place; the finiteness scan through `map`.
    `kbcir.matmul`, `kbcir.ols`, `kbcir.pca`: each dot product walks a row and a column already
    sliced, the same products summed from 0.0 in the same order -- never `sum()`, which 3.12
    made compensated. `kbcir.recurrent`: a cell dots converted rows (`_float_rows`, built once
    per cell call and once per unroll) against the converted input and state, the per-use
    conversions gone; an unroll checks the state lengths once. `kbcir.autodiff`: `Node` is
    `@dataclass(slots=True, unsafe_hash=True)` (SP-REC's finding, again); the traversal's frames
    are ints (a node id, or its complement once expanded); `evaluate` is the forward values read
    at the root; `evaluate_many` is one forward pass for many roots; `grad` takes forward values
    already computed (`fvals`); the training loop computes one forward pass per batch and reads
    each logit from its gradient's value, and predicts a dataset in one pass.
    `kbcir.hardware_rl`: the root-PUCT score in a loop over the sorted candidates.
  - The kernels before the slice are kept verbatim in `tests.kernel_fixtures` and
    `kernel.parity` holds the faster ones to them float for float (`float.hex`) and refusal for
    refusal over random inputs built to reach every branch -- the odd integer, bool, string,
    infinity and NaN, every magnitude, short and long operands, ties, every rounding and width,
    clamped scales, int4 codes past the lane, the padding nibble, dangling and shared tape
    nodes, every op, unbound inputs, every loss, duplicate candidates and empty priors --
    23,419 items, 5,367 refusals of 97 messages: 0 mismatches
    (`test_the_restructured_kernels_are_the_historical_ones_bit_for_bit`); `Node` is held to a
    frozen twin (`test_a_tape_node_is_the_frozen_nodes_value_and_a_slot_cheaper`).
  - Outcomes (the median of 9 at scale 4, #797 -> now): `bounded-overwrite-ring` 28.7 -> 19.9
    ms, `q8-q4-blocks` 30.2 -> 20.1 ms, `kmeans-knn-scaler-embedding` 40.3 -> 30.5 ms,
    `tiled-matmul` 151 -> 84 ms, `ols-pca` 4.5 -> 3.0 ms, `transformer-block` 9.6 -> 7.2 ms,
    `lstm-gru` 6.2 -> 4.4 ms, `autodiff-adam` 167 -> 102 ms, `bounded-mcts` 1.43 -> 0.80 ms;
    with the earlier slices, `iterative-phase-dag` 19.3 -> 13.1 ms, `mixed-wave-token-eft` 32.1
    -> 26.4 ms, `kbcir-streampack` 186 -> 154 ms; `static-lifetime-planner` untouched (172 ->
    178 ms median, 166 -> 167 ms min). Every case's `result_sha256` the same.
  - Faults: `tools/testing/faults/kernels.json`, 24 -- the record's fast path admitting a
    control character, a code point, an unbounded claim id and a utilization above 100; the
    quantizer's saturation skipped and a negative quotient rounded half up; a code past the
    lane admitted by the group and by the packer, the padding nibble kept and -8 admitted by the
    unpacker; a tie to the later centroid; the finiteness scan unconverted; a column walked in
    reverse, a row tile one past its k-tile, a design's last row and a column's first sample
    dropped, a row paired with a shifted vector, an unchecked hidden state, operands pushed in
    reverse, every root read as the first, a batch's forward pass of one logit, the search's
    exploration term over the visits alone, and a node compared by identity or carrying a
    dictionary: each caught.
  PERF-CORPUS (2026-10-03) measured the native twin's CPU over the corpus and found the floor.
  - Measured (the min of 3 passes per file, user + system CPU of the child, Clang -O2, the
    default target and summary mode, 240 `runtime/c/cfront_*.c` inputs): #797 (`7ae15ac2`)
    1,070 ms, the PR's base (CF-CPPZERO in #798) 799 ms, a mean of 3.3 ms per file; this PR
    changes no C source, so the twin's bytes are main's.
  - Where the remaining time is: a process that does nothing (`/bin/true`) costs 1.4 ms of
    the same CPU on this host, 56 page faults -- about 340 ms of the 799 ms is process creation
    the twin cannot reach. The smallest fixture (203 bytes) executes 554,129 instructions, 39%
    of them the dynamic loader, 115 page faults against the empty process's 56: no fixed cost
    of the twin's own is left to cut (the preprocessor's 2.3 MB state is taken zeroed and
    untouched, CF-CPPZERO). The median fixture (3.0 M instructions) and the largest (31 M)
    spend their instructions in the pipeline proper -- the parse (`p_func`, 22%), the canon
    digest of the summary (`canon_walk`, 21-23%), the emit (`emit_unit`, 20-29%, a third of it
    `snprintf`: 1,525 calls at 1,190 instructions each on the largest) and the preprocessor
    (15-19%) -- with no single site worth a byte-exact rewrite of the emitter for the 5% of a
    large file it would return. Found, not changed: the emit's `snprintf` per operand name.
  TC23 (2026-10-04) judged the GEM+ S slices on LLVM/Clang 23 and Node 24, and found the one place they
  disagreed with the release the law rail tracks.
  - Why: every S slice (S0-A through S5-C) was developed against the LLVM 18 / Clang 18 a default container
    ships, and the CI jobs that judge the oracle's lowerings and the C rails (`oracle`, `c-runtime`,
    `c-analysis`, `cfront-ubsan`, `security-assurance`) install Ubuntu's clang 18 too. #752 moved the MLIR
    rail to 23, so `mlir-rail-validate` (22, 23) and `training-llvm-latest` (23) were the only jobs on the
    current release: the AOT/JIT/WASM lowerings, the C twins of G11 and G14-G18, the alias facts (S5-A),
    escape analysis (S5-B) and data movement (S5-C), the fuzzers and the sanitizer sweeps had never been
    judged by it.
  - The toolset: apt.llvm.org is unreachable from the container and conda-forge is not, so micromamba
    2.9.0 came from conda-forge's own package and the `m23` env holds mlir, llvmdev, clang (the `nocfg`
    variant, a distribution-like clang), clangxx, clangdev, llvm-tools, lld and compiler-rt 23.1.2 with
    conda's g++ 16.2 (4.6 GB; `BCIR_LOCAL_FULL=1 bash tools/local/setup_mlir.sh` reproduces it); Node
    24.21.0 through nvm; FileCheck from Ubuntu's `llvm-18-tools` (the scripts accept any major for it).
  - RED, measured on the parent (`d9a3950`, main plus three lines of text) with that toolset first on PATH
    and `LLVM_BIN` set:
    - the MLIR rail on 23.1.2 (`MLIR_MAJOR=23 bash tools/local/check_rail.sh`, bcir-opt built with conda's
      g++): tblgen, the R1-R25 / GEM / optimizer passes, the ODS examples, the bytecode round trip and the
      IRDL corpus all OK -- #752 left nothing for 23.1.2 to find;
    - the thorough tier (`BCIR_THOROUGH=1 BCIR_REQUIRE_LLVM=1 run_all --tier thorough -j 2`): 4,158 passed,
      2 failed, `test_malformed_constants_refused_and_file_scope_forms_lowered_alike` and
      `test_nested_members_of_array_elements_run_as_the_original_on_both_rails`, both the twin's emit failing
      to LINK under clang 23 -- undefined `__atomic_store`, `__atomic_load`, `__atomic_compare_exchange`;
    - `check_alias.py --require-llvm` 0 on every row; `check_runtime.sh` ok (0 FAIL); `sanitize_cfront.sh`
      (ASan/UBSan/LSan) clean; `clang-23 --analyze` over the ten hosted sources 0 diagnostics;
      `fuzz_streampack.sh` (20,000 runs per decoder, libFuzzer + ASan/UBSan) clean; the UBSan witnesses (2,730 witnesses, every one sanitized by clang 23's UBSan, 0 reports; the gate's two test findings are the same two link failures);
      `check_handoff.sh` FAIL at the BCAB C++ wrapper build -- under clang++ 18 and 23 alike, a gate defect (below); the decoder campaign (`--require-c`) PASS on all nine decoders with the C rail, the malformed differential with the 23 `bcir-opt` PASS (6 cases, 0 disagreements, 5 malformed rejected); every row gate 0 (`check_escape`, `check_movement` and `check_volatile --require-cc`, `check_delta`, `check_encode`, `check_handoff.py`, `check_planner.py`, `check_ring.py`), the TMSAO audit's 13 cases, the pass gate once more with its output kept (all passes validate, the assemble-smoke through mlir-translate and llc 23 included), and a clean rebuild of `bcir-opt` on 23.1.2 with 0 errors and 30 warnings -- 28 `-Wmaybe-uninitialized` inside LLVM's own `Hashing.h` under GCC 16 and two in `BCIRCimDvfsPass.cpp` (`w` set but not used, `total` unused; #752 recorded one of them), none a 23 API deprecation; the corpus sweep -- the 240 `cfront_*.c` fixtures through the twin (227 emitted), each fixture plus its emit compiled by clang 18 and by clang 23 to IR at -O2: on the fixed tree all 227 build under both, 0 libatomic calls under either (the corpus holds no file-scope array of structs with an atomic member at a runtime index, which is why two inline test units found it and the witness now adds one), and one diagnostic difference, clang 23's `-Wpointer-bool-conversion` on `cfront_unaryops`'s `if (uo_arr)` in the original and in the faithful emit alike, where 18 is silent (not under any `-Werror` the gates use).
  - The mismatch: both rails reached an `_Atomic` member of an array of structs, or of a member array, at a
    runtime index through byte arithmetic cast at the end, `(*(_Atomic T *)((char *)gt + 0 + (size_t)i *
    8))`. Clang 18 gave that pointer the alignment of `T` and inlined the atomic (`align 8`); Clang 23 gives
    a pointer computed from a *defined* object by `char` arithmetic with a runtime index the alignment of
    `char`, and lowers an atomic access it cannot prove aligned to a libatomic call, which no freestanding
    link defines. Through a pointer parameter both compilers assume the natural alignment, and with a
    constant offset both compute it: the two failing tests were exactly the units with a file-scope array
    and a runtime index. Measured on the saved emit: clang 18 `store atomic ... align 8`, clang 23
    `call @__atomic_store(i64 4, ...)` and `atomicrmw add ... align 1`.
  - What landed:
    - both cfront rails reach the indexed element through its own pointer type from the constant-offset
      base, `(*((_Atomic T *)((char *)base + off) + (size_t)idx * stride/es))` (`atomic_object` in
      `bcir_cfront.c`, `_atomic_object` in `emit.py`), so the alignment is the element's on every Clang and
      GCC; the byte form remains for a stride the element size does not divide (no lock-free atomic has
      one). The CF-ATOMIC lvalue count reads both spellings.
    - `test_an_indexed_atomic_member_of_a_defined_object_is_a_lock_free_atomic_under_clang`: both emits of
      a unit with an array of structs (stride 8 over a 4-byte element) and a member array (stride == size)
      compiled by the host Clang to IR at -O2 carry their atomic instructions and no `@__atomic_*` call;
      RED on the parent (the two link failures under 23, and the parent's emit text under any Clang),
      GREEN under 18 and 23. The nine atomic and RTFP tests pass under both.
    - CI: `oracle-llvm-latest` (the two thorough shards on apt.llvm.org's clang/lld/llvm 23 + compiler-rt
      and Node 24; the anti-vacuity step asks the oracle's own resolver for the major it will use) and
      `c-rails-llvm-latest` (`check_runtime.sh`, the memory-discipline and cfront sanitizer sweeps,
      `clang-23 --analyze`, the 500,000-run decoder fuzz, the UBSan witnesses with the clang engine and the
      decoder campaign under clang 23 + its compiler-rt): the lowering rail and the C rails on the release
      the law rail tracks: 22 minutes of runner time per run (the C rails 14, the two oracle shards 4 each) beside the 61 of the jobs that were there, measured on this PR's run.
    - `tools/local/setup_mlir.sh` `BCIR_LOCAL_FULL=1` (the whole lowering toolset of the major) and its
      README; `mlir/README.md`, the IRDL projection's header and its smoke fixture name 23 as the validated
      major; the cicd skill's job table.
  - Found, not changed: neither rail reads its own indexed atomic emit back as input -- the twin never
    read the byte form (a parse error) and the oracle read it but refuses the typed one ("unsupported base
    expression Cast") -- so both refuse now, in different words. The RTFP subset (`cfront_rtfp.c`) covers
    the constant-offset member form only, which both rails still read.
  - Not claimed: LLVM 24 (trunk); the aarch64 jobs on 23 (they stay on Ubuntu's clang); Windows; the
    training corpus (#768-#786, out of scope by the user's instruction).
  TC23-CI (2026-10-04) cut the wall time of the LLVM 23 C-rails job, and of the workflow's longest job.
  - Measured on the job's first run (`6d17ed2`): 13.8 minutes, serialized -- the UBSan witnesses 5.9, the
    500,000-run decoder fuzz 3.1, the runtime gate 2.2, the rest 2.2 -- where the clang-18 jobs it mirrors
    run as three jobs in parallel (`c-runtime` 5.2, `c-analysis` 6.2, `cfront-ubsan` 8.2, the workflow's
    longest).
  - What landed:
    - `tools/testing/ubsan_witnesses.py --shard I/N`: the Ith stride of the discovered witness list, as
      `run_all --shard` cuts the suite -- the N slices are disjoint and together the whole list, so N cells
      run every witness exactly once; a slice holding no test, an index past its count and a spelling that
      is no I/N are refused as INVALID (L2), and the report and the summary line carry the shard.
      `test_shards_partition_the_witnesses_and_an_empty_or_malformed_shard_is_refused` holds it: the one
      failing helper sits in exactly one of two shards, and each refusal is named. Under clang 23 the two
      halves sanitize the whole set between them (the counts are in the PR's verification table).
    - `c-rails-llvm-latest` is one job of four parallel cells: `runtime` (the runtime gate and the
      memory-discipline sweep), `analysis` (the cfront sanitizer sweep, the analyzer, the decoder fuzz
      bounded to 100,000 runs per decoder -- the deep campaign stays in the clang-18 `c-analysis` job; here
      the question is whether clang 23 builds and runs every fuzzer clean -- the decoder campaign and the
      C++ hand-off) and `ubsan-1` / `ubsan-2` (the witnesses, one shard each). Every cell installs the
      toolchain (cached) and runs the anti-vacuity check.
    - `cfront-ubsan` (clang 18) runs as two shards; the fault table's UW13 anchor follows its command line.
  - Not claimed: fewer runner minutes (the work is the same, spread wider; the cells' installs add about
    a minute each); the measured wall time after the change is in the PR's record.
  BUILD-0 / BUILD-1 (2026-10-04) opened the build program: the C and C++ rails as a CMake project over one
  source manifest, with CTest, presets and a build-parity gate -- the first slice of
  `docs/BCIR_BUILD_ROADMAP.md`'s ladder away from the shell build.
  - Measured first (BUILD-0): 27 shell scripts and 8,600 lines under `tools/`; `check_runtime.sh` alone 4,543
    lines, 116 sections and 223 compiler invocations, its source lists spelled again in the fuzz driver, the
    seam gate, the memory-discipline sweep and a dozen Python harnesses (the #719 trap, patched per pair
    until now); 14 units proved freestanding by hand-written sections; no graph anywhere, so no rebuild of
    only what changed and no two independent sections running at once under the two-worker cap.
  - What landed (BUILD-1):
    - `runtime/manifest.json` (`bcir-build-manifest.v1`): ten class-homogeneous libraries over the 38
      non-main `bcir_*.c` units, five tools, the 32 harnesses, the 19 fuzz targets with the gate's
      `-max_len`, the C++ seam's two libraries and five tests, and the 26 `freestanding_core` units compiled
      `-ffreestanding -nostdlib` at C11 and C23 as part of `ALL`. Per-unit `options` carry what the gates
      pass for a unit (the twin's four GCC warning families, `-ffp-contract=off` for the model kernels),
      scoped `gcc:`/`clang:` so a family one compiler lacks is not an unknown-option error on the other.
    - `CMakeLists.txt` + `cmake/` (compiler-flag policy: C23 spelled `c2x` where CMake must, C++17
      `-Wpedantic`, `-Wall -Wextra -Werror`, Release `-O2` without NDEBUG, `BCIR_SANITIZE`; the
      dependency registry writing `bcir-deps.json`; the manifest reader with closures; the CTest
      registration with labels and `PROCESSORS 2`) and `runtime/{c,cpp}/CMakeLists.txt`, which list
      nothing -- every target comes from the manifest. Fuzz targets compile their library closure into
      themselves so libFuzzer and the sanitizers instrument the code under test. The MLIR law is
      `add_subdirectory()`'d when `BCIR_BUILD_MLIR` is on (default OFF; the `mlir` preset), with
      `LLVM_BUILD_TOOLS` set so `add_mlir_tool` keeps `bcir-opt` in `ALL`.
    - `CMakePresets.json`: `default`/`gcc`/`clang`/`asan`/`ubsan`/`tsan`/`fuzzer`/`mlir`; every build and
      test preset at two workers, which the checker refuses to see changed.
    - `tools/build/manifest.py --check`: twelve rules (schema, shape, files, roles, references, link
      names and options, memory classes, the freestanding set against the gates' `-ffreestanding` lines,
      the gates' compile lines and arrays against the closures, every Python module's literal groups
      (per list/tuple/set, by the AST) against the closures, the seam's two lists, the presets), reading
      the gates and the harnesses out of their own text. `tools/build/build_parity.py`: the CMake-built
      `bcir-cc` against the gate's one-command recipe, every mode over every fixture, stdout + stderr +
      status + pack bytes identical; an empty corpus is INVALID and a missing tool UNUSABLE, never a pass.
    - `bcir/tests/test_build_manifest.py` (repository-only): the tree reconciles; 22 injected violations
      each a finding; the scanners shown to have examined the real gates (the empty-match vacuity of L2);
      the AST grouping; the CMake reader and the checker agreeing on kinds and keys (L12); the parity
      gate's refusals before any compile; the `cmake-build` CI job as the gate's owner.
    - CI: `cmake-build` (gcc + clang): configure with MLIR off, build everything, `ctest -L build`; the
      clang cell builds the `fuzzer` preset and runs its bounded entries.
  - Found while building: the twin is not GCC `-Werror`-clean at `-O2` (misleading-indentation,
    stringop/format-truncation, maybe-uninitialized -- the gates build it without `-Werror`; the manifest
    scopes exactly those families and keeps every other warning an error); `bcir_kplan.c` under
    `-ffreestanding -O2` draws a GCC maybe-uninitialized verdict its hosted `-O2` build does not (the
    gate's sections compile without optimization; so do the checks); `bcir_decode.c` needs libm; a test
    registered before `enable_testing()`'s directory is dropped silently (one fuzz entry registered
    instead of twenty until the call moved above the subdirectories); `add_mlir_tool` leaves `bcir-opt`
    out of `ALL` out of tree; and `tools/c/check_memory_discipline.sh` judged differently under the two
    compilers -- its strict per-unit pass (`-Wpedantic -Werror`) lacked the compatibility warnings its own
    harness builds take, so it passed under clang and failed under gcc on the twin's misleading-indentation
    and format-truncation families, which never showed because the gate prefers `clang` on PATH (L12; found
    the moment the CMake project handed it the configured gcc). The strict loop now takes the same array;
    under clang that adds `-Wno-misleading-indentation` only, so nothing it judged there changes.
    And `.gitignore`'s `build/` matched `tools/build/` too, so the checker and the parity gate sat
    untracked in the working tree: every local check passed over files no checkout would have, and
    the first CI run failed on all of them (L21: an exclusion hides a shipping defect). The directory
    is re-included and a test asserts the tools are neither ignored nor untracked. And the dependency
    index `bcir-deps.json` was not JSON: the registry wrote CMake's `ON`/`OFF` where JSON spells
    `true`/`false`, which no local step had parsed; the cmake-build job's reader caught it. The writer
    spells booleans, and `manifest.py --deps-index` (a `build`-label CTest entry and the job's step)
    holds the index to its schema, with the shipped defect as its witness.
  - Measured: build parity 240 fixtures x 8 modes = 1,920 rows, 0 differ, on GCC 13.3 (CMake `-std=c2x`
    against the recipe's `c11` fallback) and on Clang 23.1.2; gcc configure + build 13 s at two workers;
    `fuzzer` preset 20/20 CTest entries; `mlir` preset: `bcir-opt` built, the four mlir gates 4/4; the
    system clang 18 tree builds with 0 warnings and passes the build label and the fuzz label alike; the
    memory-discipline gate passes under gcc and clang 18 and as the gcc tree's CTest entry.
  - Not claimed: any gate section migrated (BUILD-2 is section by section with a byte-identity proof
    each); install/export; MSVC; a BCIRfile (design only, roadmap S8).
  BUILD-2a (2026-10-04) moved the first six sections of `tools/c/check_runtime.sh` out of the monolith.
  - Each section's own text became `tools/c/sections/<name>.sh` (runtime, artifact_bundle, executor,
    encoder, execution_plan, telemetry_frame), generated from the gate's lines rather than retyped: the
    script takes the harness binaries as arguments, makes its own temp dir and fixtures (the encoder
    section regenerates the executor's `exec.bin` instead of reaching into a previous section's temp),
    and prints the lines the gate printed. The gate keeps its freestanding loops and its compile lines
    and calls the script over the binaries it built, so the gate and the CTest entry run one text.
  - `runtime/manifest.json` gained `sections` (script + harnesses in argument order); the CMake reader
    reads it; `c-section-<name>` CTest entries (label `section`) run each script over the manifest's
    harnesses with `PYTHON` set; the `section` test preset and the `cmake-build` job run them.
  - `tools/build/section_parity.py` (`build-section-parity`, label `build`): for every section it builds
    the gate's recipe for each harness from the manifest closure (`-std=c23 -O2`, the gate's own
    fallbacks for a compiler without the spelling), runs the script over the gate-built and the
    CMake-built binaries and requires stdout, stderr and status byte-identical and at least one PASS
    line under exit 0; an empty section set, a missing binary or a recipe that does not build is
    exit 2, never a pass.
  - Checker rule M13: every section names an existing `tools/c/sections/*.sh` and manifest harnesses,
    every script in that directory is one section, and the gate calls each script (the delegation is
    part of the rule, so a section the gate stopped calling is a finding). Four injected violations
    and a gate-text mutation each fire; `compare_section` is shown to see a differing stream, an
    identical-but-vacuous output and a failing section.
  - Found on CI: the `windows-latest` runner's `bash` on PATH is the WSL launcher stub, which prints a
    UTF-16 "no installed distributions" notice and exits 1 -- an engine that is no engine (L2). The
    parity gate ran it as a shell and the witness read the stub's output as the section's. The gate
    now probes its shell (`posix_shell()`: `BCIR_SHELL` or `bash`, kept only if `bash -c 'echo ok'`
    says ok and exits 0), refuses with exit 2 without one, and the test holds a shell-less host to
    that refusal while a stub that talks and exits 1, and one that exits 0 silently, are both no shell.
  - Measured: gcc tree `ctest -L section` 6/6 in 1.5 s; `build-section-parity` 6 sections identical,
    11 PASS lines, 12 s (gcc's `-std=c23` falling back to `c2x` as the gate does); the gate under
    clang 18 exit 0 with the delegating sections (116 sections, 387 PASS lines); the clang 18 tree
    0 warnings, section 6/6, build 4/4; quick tier 4177/0; the gate's 220 section lines became 6
    delegating lines plus the six scripts.
  BUILD-2b (2026-10-05) moved group 2 of the runtime gate: the sections that build second binaries.
  - `tools/c/sections/{control_plane,ring,handoff,kplan}.sh`, generated from the gate's lines as group
    1 was. Each takes the harness and its second binaries -- the `-O0`/`-O3` builds the optimisation
    parities compare and the fault-injected mutant the "gate fires" leg grades -- as arguments. The
    ring's ThreadSanitizer leg stays in the gate after the script call: it builds sanitizer binaries
    and runs their stress in one step (group 3).
  - `runtime/manifest.json` gained `variants`: ten harness rebuilds, each compiling its harness's whole
    library closure with the variant's options appended after the build's `-O2` (so `-O0` reaches the
    twin, not only the harness main), or with one source mutated. A mutation is spelled once, as exact
    text, and applied once, by `tools/build/mutate.py`: the gate's four `sed` + `cmp` injections now
    call it, and CMake generates each mutant source with it at build time. It writes only when the
    anchor occurs exactly once and the edit changes the source; each mutant was shown byte-identical
    to what the gate's `sed` produced. Mutants build without the warning policy, as the gate built
    them. A section whose mutant needs Python is reported, not registered, on a host without it.
  - Checker rule M14 holds the variants: a manifest harness to rebuild, flag options or a mutation of
    a source in that harness's closure whose anchor occurs exactly once (so a law whose line changes
    is caught at the checker, before a build), the gate generating every mutant through the applier,
    every variant run by a section, no variant shadowing a unit. M13 accepts variants as section
    binaries and checks `varies` declarations.
  - Found while building: the ring section's fault-injection line reports how many concurrent runs
    caught the injected race (3 in one run, 4 in the next over the same binaries), so its output is
    not a function of its binaries and byte-identity between two builds would flake. The section-parity
    gate now masks declared varying values -- a manifest `varies` regex whose one capture group is the
    value, applied line by line (laws.md L6: the value, not the line) -- reports a declaration that
    matches nothing as stale, and on a difference re-runs the gate's own binaries so undeclared
    nondeterminism is reported as such rather than as a build difference. And the manifest was not
    a configure dependency, so `cmake --build` kept the old unit lists after an edit; it is now.
  - The `fuzzer` preset builds the fuzz targets only (`BCIR_BUILD_HARNESSES=OFF`): its CTest label is
    `fuzz`, and building the harnesses and the new variants there was work no entry used.
  - Measured: gcc tree `ctest -L section` 10/10 in 16 s; `build-section-parity` over the ten sections
    identical under both builds with the ring's one value masked; the gate's 4,329 lines are 4,152,
    with the four mutants generated by the shared applier.
  BUILD-2c (2026-10-05) moved group 3a: the two-standard sections and the ring's ThreadSanitizer leg.
  - `tools/c/sections/{x86_interrupt,q8_tables,ring_tsan}.sh`. The first two take the harness built
    under C11 and under C23; the Q8 table's drift check stays in the gate, because it regenerates
    tracked files and a CTest entry must not write the source tree (`test_q8_embed.py` holds the
    committed files to a fresh emission too). `ring_tsan.sh` takes the ring built with
    `-fsanitize=thread` and the same build of a ring whose relaxed atomic stores are made plain, and
    runs the stress the gate's `ring_tsan()` ran; the gate keeps the two compile lines.
  - Variants gained `standard` (C11 here; C23 is the harness's own and is refused as no change) and
    `sanitizer` (`thread`: compile and link). The plain-store injection, a `sed` + `cmp` in the gate
    until now, is a manifest mutation applied by `tools/build/mutate.py`, shown byte-identical to the
    `sed` output.
  - Whether ThreadSanitizer works on a host is one predicate, `tools/build/sanitizer.py`: a trivial
    `-fsanitize=thread` program builds and runs under the same `setarch -R` wrapper the runs use.
    The gate asks it instead of its inline probe, and the configure asks it for the dependency
    index's new `TSAN` row. A variant the tree cannot build -- no runtime, a conflicting
    `BCIR_SANITIZE` such as the asan preset's, a mutant without Python -- is recorded with its reason;
    its section is not registered, the section-parity gate reports it by name (`--unbuilt`) instead of
    failing on a missing binary, and `BCIR_REQUIRE_TSAN=ON`, which the CMake CI job now passes since it
    installs the runtime, turns that into a configure failure (laws.md L2, L12, L14).
  - Checker: M14 holds the new keys (a known standard other than 23, a known sanitizer, the gate
    asking the probe before it builds a sanitizer variant); M13 refuses a section that mixes sanitizer
    builds with plain ones, so a host without the runtime skips whole sections, never half of one.
  - Found while building: the dependency index wrote a row's detail verbatim, so a probe diagnostic
    spanning lines would have made the whole index unreadable JSON; every row's detail is now folded to
    one line.
  - Found by checking the slice against the newest LLVM 23 (23.1.2, the latest `llvmorg-23` tag) and
    Node 24 (v24.21.0): the C11 Q8 variant, built with the warning policy, failed on Clang 23 with
    `#embed is a C23 extension [-Werror]`. The generated header chose `#embed` whenever the toolchain
    had it, and Clang 19 and later offer it to C11 as an extension, so a strict C11 consumer of the
    header could not build it -- and the gate's "fallback self-check under C11" was not what ran on
    those compilers. One probe, `bcir/abi/embed.py`, now admits `#embed` only in C23 mode; the Q8
    header and the inference generator's large weight tables (the same guard, copied from it) both
    spell it through that function (laws.md L14). C11 takes the byte-identical fallback on every
    compiler and C23 keeps `#embed` where the toolchain has it; `test_q8_embed.py` builds both with
    `-Werror` and requires the fallback under C11.
  - Measured: gcc tree with `BCIR_REQUIRE_TSAN=ON`: 0 warnings, `ctest -L section` 13/13 in 15 s
    (`ring_tsan` 5.4 s); each new section fails on the binary it must reject. The gate's 4,152 lines
    are 4,120.
  LATEST-TOOLCHAIN (2026-10-05) made the newest LLVM/Clang/MLIR 23 and Node 24 part of every
  change's validation, after BUILD-2c's C11 `#embed` defect showed what a Clang-18-only pass misses.
  - `.claude/skills/bcir-latest-toolchain/SKILL.md` states the rule: validate on the CI-default Clang
    18 / GCC 13 AND on the newest release of the tracked majors; a failure only 23 shows is a finding
    to root-cause, never a reason to pin back. AGENTS.md (pre-PR step 2), CONTRIBUTING.md (the first
    row of the gate table), the digest and the two existing skills point at it.
  - `tools/local/latest_toolchain.py`: `status` reads the installed clang / llvm-config / mlir-opt
    and the highest Node 24, and compares them with the newest releases read from the sources (the
    highest `llvmorg-23.x.y` tag; nodejs.org's index) -- an unreachable source is UNKNOWN, exit 2,
    never current; `env` prints the PATH exports; `install-node` fetches the newest Node 24 and
    checks the tarball against a checksum list whose signature it verifies against the Node release
    keys before anything is extracted (the data filter refuses members escaping the directory).
  - `tools/local/check_latest.sh` runs the legs CI's 23 jobs own -- toolchain confirmation, a
    `clang23` preset tree with section and build parity, the `fuzzer` preset, `check_runtime.sh`, the
    thorough tier with `BCIR_REQUIRE_LLVM=1` on Node 24, the `mlir` preset -- serially at two workers,
    one verdict line per leg.
  - CI: the `cmake-build` job gained a `clang23` cell (apt.llvm.org's clang 23 + compiler-rt,
    `BCIR_REQUIRE_TSAN=ON`, a version and TSan confirmation step); until now the CMake project was
    built only with GCC 13 and Clang 18. `CMakePresets.json` gained the `clang23` preset it uses.
  - `bcir/tests/test_latest_toolchain.py` holds the tool's decisions offline (mocked upstream), its
    refusals (unsigned, tampered, escaping tarball; unsupported platform; another major), and one
    statement of the rule across the skill, the entry points, the check's legs, the preset and the
    workflow.
  JER-PAIRED (2026-10-05) made the JER SIMD rail's speed floors immune to runner load, and gave
  the multi-byte floor the baseline its claim names.
  - Found on CI: `test_multi_byte_text_does_not_regress_against_the_scalar_rail` read
    `cjk: 0.70x` against its 0.75 floor on one run of a commit that touched neither rail. The
    same job on the same commit passed in its other run. Every speed ratio in
    `test_cpp_jer_simd.py` divided two medians measured one after the other, so a load reaching
    one window and not the other moved the ratio as far as it liked.
  - `_paired_gain` measures the way `tools/silicon/measure_jer_simd.sh` already did for the
    dedicated-host evidence: a baseline round and a vector round back to back in one process,
    the order alternating pair by pair, judged by the median per-pair ratio. All three speed
    tests use it (laws.md L14).
  - Measured: on this host, idle, five estimates each, the paired ratio held to 2.75-2.84x on
    accents where the sequential one ranged 2.09-4.47x. Under a phased load on the
    measurement's CPU (20 estimates), the sequential CJK ratio ranged 1.01-2.74x and the
    paired one 1.65-1.92x.
  - Found while proving it: the floor could not see the regression it named. The driver's
    `scalar` tier is the adapter pinned to scalar, the same alternating walk with byte-loop
    runs, so a cost the walk adds landed on both sides and cancelled. Validating every
    multi-byte run twice more read 1.21x and passed, and eight times more read 1.06x; no
    overhead in the walk could take the ratio below 1.
  - The driver's `bench` now takes `rail`, `bcir_jer_validate_utf8` over the whole document
    with no adapter, and the multi-byte floor is judged against it: the same injection reads
    0.72x and fails (0.65x through the test itself).
  - A witness, `test_the_paired_gain_cancels_a_load_that_moves_between_windows`, runs on a fake
    driver with known costs, so it needs no compiler and runs in every tier on POSIX. A load
    that steps from 1x to 5x breaks the sequential estimator (0.6x on a true 3.0x). The paired
    one reads exactly 3.0x wherever the step falls, reads a true 0.5x regression as 0.5x, honors
    the baseline asked for, and refuses a zero-nanosecond round.
  - Proven by fault injection: the witness fails when the estimator reverts to sequential windows
    or ignores its baseline, and the floor fails on the injected walk overhead that the
    previous test passed.
  BUILD-2d (2026-10-05) moved group 3b: the four sections that drive the C twin and its Python
  parity.
  - `tools/c/sections/{cfront,cfront_abi,cfront_loop,channel}.sh`, generated from the gate's lines as
    the earlier groups were. `cfront` checks the twin's per-fixture summary and structural digest
    against the oracle's, over `test_cfront`. `cfront_abi` checks the `--target` data-model matrix
    over the same binary. `cfront_loop` runs the compile-to-execute loop over `test_cfront_loop`,
    and `channel` the routing decision over `test_channel`. The gate keeps the freestanding probes
    of `bcir_cir.h`, `bcir_plan.c` and `bcir_hydrate.c` and the harness builds, and calls the scripts.
  - No new variants or checker rules: the harnesses were already manifest units. The scripts run the
    oracle beside the harness with `PYTHONPATH` set, as `runtime` does, so a CTest entry running in
    the build tree imports the checkout's `bcir`.
  - Measured: gcc tree `ctest -L section` 17/17 in 16 s; `build-section-parity` 17 sections identical
    under both builds (the new four with 122, 6, 26 and 1 PASS lines); each new section fails on a
    harness that corrupts one answer (a digest, a target's data model, a loop that does not execute,
    a route). The gate's 4,120 lines are 4,030.
  BUILD-2e (2026-10-05) moved group 3c: the first four `bcir-cc` sections, the first sections over
  a tool and a compiler.
  - `tools/c/sections/{bcir_cc,emitlink,recover,writeguard}.sh`, generated from the gate's lines.
    `bcir_cc` runs the driver over the UART and CMSIS GPIO fixtures and `--emit-pack`. The other
    three compile what `bcir-cc --emit-c` emits against the bounds-quarantine runtime:
    `emitlink` the self-contained masked unit, `recover` the recorded two-truth crossing, and
    `writeguard` the out-of-bounds store that must abort, never clamp. `recover` regenerates the
    masked unit `emitlink` used to leave in the gate's temp directory, from `emitlink`'s own
    lines. The gate keeps the `bcir-cc` build, which 69 later lines still use.
  - A section binary may be a manifest tool. The CTest entry passes the CMake-built `bcir-cc`, and
    the section-parity gate builds the gate's recipe from the tool's closure and finds the
    CMake build's copy through `--tool-dir`; a section that takes a tool without one is exit 2.
  - A script that compiles takes the compiler as CC, and refuses to run without one (exit 2). The
    gate passes its own, every CTest section entry gets the configured C compiler, and the
    section-parity gate hands its `--cc` to both runs, so the two builds' sections compile what
    `bcir-cc` emits with one compiler.
  - M9 reads the section scripts as it reads the gates. Until now the moved text only ran
    binaries; these scripts compile runtime units, and moving their lines out of the gate would
    have taken them out of M9's view (laws.md L15). M13 accepts a manifest tool as a section
    binary, and a fuzzer, still, as none.
  - `#atomicring` and `#extentassert` stay in the gate between them: they compile and run the
    runtime header's own contract with no binary to take, like the freestanding probes.
  - Measured: gcc tree with `BCIR_REQUIRE_TSAN=ON`, 0 warnings, `ctest -L section` 21/21 in 23 s;
    `build-section-parity` 21 sections identical under both builds, the new four with 3, 1, 1 and
    1 PASS lines; the clang 18 and clang 23 trees the same. Each new section fails on a `bcir-cc`
    that corrupts one of its answers: a GPIO compile reporting `ok=0`, an emitted unit without
    its runtime include, an unguarded masked read, a store under the read guard. The gate on
    clang 18 prints the same 398 PASS lines as before the move. The section-parity gate now
    builds each binary's recipe once per run (every `bcir-cc` section takes `bcir-cc`): 195 s
    on the gcc tree, against 249 s building it four times. The gate's 4,030 lines are 3,921.
  JER-WARM (2026-10-05) gave the JER SIMD speed floors rounds long enough to start warm.
  - Found by the LLVM 23 thorough leg while validating BUILD-2e: the early-accent floor read
    `27.1x -> 5.3x` and failed. Since JER-PAIRED each pair's vector round follows milliseconds of
    scalar work, and with 32 calls a round on the all-ASCII document lasted about 18 us. Its median
    was bimodal on an x86-64 AVX2 host: p50 576 ns, p90 2,504 ns, about one round in ten 4.3x
    slow, the shape of a vector unit waking up. When the slow rounds reached a majority of a
    trial's pairs, the whole gain read about 5.5x instead of 24x.
  - `_paired_gain` now times 512 calls a round, after as many warm-up calls. Across ten trials the
    early-accent ratio read 0.96 every time, where 32 calls ranged from 0.84 to 4.21. The real
    test failed 1 run in 40 at 32 calls (`19.8x -> 5.2x`) and 0 in 40 at 512. The three speed
    tests take about 7 s together.
  BUILD-2f (2026-10-05) moved group 3d: seven `bcir-cc` contracts that compare the twin with the
  Python oracle.
  - `tools/c/sections/{diag,fallback,r21policy,project,link,effects,linkflags}.sh`, generated from
    the gate's lines. `diag` checks the caret renderer over the `test_diag` harness, whose build
    stays in the gate. The other six take `bcir-cc`, for the fallback decision, the R21 policy,
    project mode, linking, the effect and escape reports, and the derived link flags. `link`
    compiles what the tool emits and so takes CC. The oracle runs under `"${PYTHON}"` with the
    checkout on `PYTHONPATH`, so a CTest entry in the build tree imports the checkout's `bcir`.
  - `#cexpr` stays in the gate. The twin's `--emit-c` puts a declaration right after a `case`
    label, a C23-only form: GCC 13 accepts it silently, and Clang 18 and 23 warn. The warning
    names the section's temp file, which differs between two runs, so the section-parity gate
    would call the section unstable. The emitter fix, on both rails, comes before the move.
  - Measured:
    - gcc and clang 18 trees with `BCIR_REQUIRE_TSAN=ON`: 0 warnings, `ctest -L section` 28/28,
      and `build-section-parity` 28 sections identical; the clang 23 tree the same.
    - The new seven print 22, 6, 10, 5, 6, 2 and 5 PASS lines.
    - Each new section fails on a binary that corrupts one of its answers: a shifted caret, a
      fallback that always exits 0, a reject that exits 2, DIRTY rewritten to CLEAN, missing
      link flags, a dropped escape, a dropped `-lm`.
    - The gate on clang 18 prints the same 398 PASS lines as before the move. The gate's 3,921
      lines are 3,571.
  CF-CASELABEL (2026-10-05) ended every `case` and `default:` label of an emitted switch in a null
  statement, on both rails: the oracle's `_walk` (`bcir/frontends/cfront/emit.py`) and the twin's
  emitter (`runtime/c/bcir_cfront.c`).
  - Both emits had put an arm's first temporary right after its label. A label labels a statement
    (C11 6.8.1), and a declaration is no statement before C23, so nearly every emitted switch was
    C23. Clang 18 and 23 warned by default, GCC 13 only under `-Wpedantic`, and
    `-std=c11 -pedantic-errors` refused it.
  - The null statement keeps the declarations after a label in the switch body's scope, as the
    source's are. A brace around each arm would have ended that scope at the next label, which a
    fallthrough reads past.
  - Witness: `test_an_emitted_switch_is_c11_on_both_rails`. Its unit has five labels, a
    fallthrough, a C23 declaration after a label, and two labels at the switch's end. Each rail's
    emit ends every label in `: ;`, builds under `-std=c11 -pedantic-errors` with Clang and GCC,
    and returns what the original does for 0 to 39.
  - RED: either rail reverted fails the spelling check. With that check removed, the reverted
    rail's emit fails the build ("label followed by a declaration is a C23 extension").
  - `tools/testing/faults/cfront-caselabel.json` holds four faults, a case label and a `default:`
    on each rail, and each fails the witness. CF-ENUMFOLD's EF11 anchors on the new spelling.
  - Found beside it and not fixed here (CF-VLASCOPE): the emit flattens a braced block, so a VLA
    declared in one is still in scope at a later `case` label, a loop's continue label or a
    `goto` target that the source kept outside the block. Neither GCC nor Clang builds such an
    emit, and the oracle's Clang check reports the failed build as `skip:build-failed`, so the
    unit stays CLEAN. `docs/languages/CFRONT_GUIDE.md` records it under Known limits.
  BUILD-2g (2026-10-05) moved `#cexpr`, the constant folds, the last of group 3d, once
  CF-CASELABEL made its emit C11.
  - `tools/c/sections/cexpr.sh`, generated from the gate's lines like group 3d's, takes `bcir-cc`
    and CC.
  - The gate first builds the emit with `-pedantic-errors`, so the section refuses the C23 form
    under GCC too. The `bcir-cc` before CF-CASELABEL fails the emitted build under GCC 13,
    Clang 18 and Clang 23.
  - With the emit clean, the warning that named the section's temp file is gone, and the
    section-parity gate compares the section like any other.
  - Measured:
    - gcc and clang 18 trees with `BCIR_REQUIRE_TSAN=ON`: 0 warnings, `ctest -L section` 29/29,
      and `build-section-parity` 29 sections identical; the clang 23 tree the same.
    - The gate prints 398 PASS lines on clang 18 and 387 on clang 23, as before the move, and
      no C23-extension warning. The gate's 3,571 lines are 3,544.
  BUILD-2h (2026-10-05) moved group 3e: the sections whose programs the gate wrote out itself.
  - The five link-flag rules (#linkflags-fftw, -lapack, -gsl, -sleef, -cerf) had been five probe
    files written from heredocs, one helper copied five times. They are now one harness,
    `runtime/c/test_link_flag_rules.c`: its argument picks a rule's table, each row the probe's own
    edge and comment, and it prints the probe's own `OK linkflags-<rule>`. The gate builds it once;
    `tools/c/sections/linkflags_<rule>.sh` runs one table each.
  - The seven E-series sections (#ols, #pca, #layernorm, #lstm, #classical #svm, #classical #tree,
    #kmeans) compile C the Python oracle emits, a driver `main` appended. A program that exists only
    once the oracle has run needed a manifest kind of its own, `kernels`. Each entry names the
    emitter, its arguments, its driver under `runtime/c/kernels/` and its system libraries.
  - One writer for the unit: `tools/build/emit_kernel.py` writes it for the gate, for CMake and for
    the section-parity recipe. CMake writes it at build time with a depfile naming every oracle module
    the emitter imported.
  - M15 holds each entry to the oracle (the emitter is a function of `bcir/lower/c_kernel.py`, read
    as text), each driver to one entry, each kernel to a section and the gate's own writing.
    `emit_kernel.py` asks the same predicate (`kernel_problems`), so the writer refuses exactly what
    the checker reports.
  - Measured:
    - The gate's text for each kernel is the old heredoc text, the driver's two-line comment aside.
    - On the gcc tree the twelve sections pass under CTest, and `build-section-parity` holds them
      identical, the kernels' recipes written by the same writer.
    - A no-op build re-emits no kernel. Touching `bcir/lower/alias_facts.py`, which the emitters
      import, re-emits all seven; touching an ASN.1 module re-emits none; touching `ols`'s driver
      re-emits `kernel_ols` alone.
    - With M15's call removed, all fourteen injected kernel faults go unreported.
  BUILD-2i (2026-10-05) moved group 3f, the "emit == Clang" family: 52 sections, `#scale` through
  `#stmtexpr`, each running `bcir-cc` over a unit. Fifty of them compile what it emits beside the
  source and a driver and run both, so their scripts take CC; `scale` and `pscale` compare the
  tool's counts with the oracle's and take `bcir-cc` alone.
  - The scripts are generated from the gate's lines, with two substitutions: the binary path became
    the script's argument, and `python3` became `"${PYTHON}"`. None used a variable or function the
    gate defined outside it, which the generator checked before it moved anything.
  - `test_the_scanners_examined_the_real_gates` required 150 compile groups of the runtime gate
    alone. With most of the gate's text in the section scripts, which M9 scans as gate text, it now
    counts the two together (72 and 145) and requires 100 of the sections' own.
  - Measured: `build-section-parity` holds the 52 identical under both builds with gcc and again
    with clang, so no diagnostic names a temporary path, and the gate's 3,086 lines are 1,633.
  BUILD-2j (2026-10-05) moved group 3g, the rest of the gate, and with it BUILD-2 is complete:
  every check `tools/c/check_runtime.sh` makes now also runs as a CTest entry.
  - The ASN.1 twins: `per`, `xer`, `jer`, `asn1_emit`, `per_plan` and `oer` compare their twin at
    `-O0` with it at `-O3`, over twelve new manifest variants; `asn1bench` takes the
    `bcir_asn1_bench` tool and `asn1fast` its harness. The gate's sweeps had turned a failed `-O0`
    or `-O3` build into a SKIP, and the `asn1_emit` sweep into nothing at all, where the CMake
    build of the same variants fails; the gate fails on it now too (laws.md L12, L21).
  - The probe programs became harnesses: `#atomicring`'s heredoc is `runtime/c/test_oob_counter.c`
    and `#extentassert` is `runtime/c/test_extent_assert.c`. Found moving the latter: the gate
    compiled the tampered extent with `-std=c23` alone. GCC 13 refuses that option, so under GCC
    the compile failed on the flag, the assertion was never reached, and the section printed its
    PASS line having judged nothing (L2). The section now compiles the tampered unit in the
    standard the correct one compiled in and requires the assertion's own diagnostic. RED: with the
    macro made a no-op the old check still passes under GCC 13 and the new one fails; a macro that
    fails another way (a negative array) fails the new one too.
  - `#inlineasm`, `#portio` and `#barrier` judge what the Python oracle emits for a C source under
    CC and the gcc and clang on `PATH`, so they take no binary. They are `compiler_only` sections:
    M13 lets a section name no binary only then, and only if its script runs `${CC}`; the
    section-parity gate runs such a section twice and holds the runs identical.
  - The delegated gates: the seven scripts the gate calls that build and judge binaries of their
    own (memory discipline, the StreamPack semantic boundary, the target ABI, the JER index and
    SIMD rails, the C++ hand-off, SYCL) are the manifest's `delegated` entries. CMake registers
    each as a CTest entry of its own, and the gate skips them under `BCIR_SKIP_DELEGATED_GATES=1`,
    which the `c-runtime` entry sets, so `ctest` runs each once. M16 holds the gate's guarded
    calls, each SKIP line naming its entry, the manifest and `cmake/BCIRTests.cmake` in step, and
    refuses a script the gate calls that is neither a section nor a delegated gate. RED: with M16's
    call removed, its eleven manifest faults and seven tree faults go unreported.
  - Found by the CTest runs: `asn1_emit` ended on `ok:` where every other section prints a PASS
    line, so the section-parity gate, which requires one, failed it under every compiler; it
    prints PASS now. Two delegated gates compiled with `-std=c23`, which GCC 13 spells `c2x`: they
    passed under clang and failed as CTest entries under GCC, and now ask the compiler (L12).
  - Measured: on the gcc, clang 18 and clang 23 trees `ctest -L section` runs 106/106 and
    `build-section-parity` holds all 106 identical, the three compiler-only sections over two runs;
    the seven delegated entries pass on each tree. The gate prints 399 PASS lines on clang 18 and
    388 on clang 23, 0 FAIL; its 1,633 lines are 879.
  BUILD-3 (2026-10-05) made the C rails installable: `cmake --install` lays down the manifest's
  libraries and tools, every public header and a version header, and a CMake package an out-of-tree
  project finds with `find_package(BCIR)` (`cmake/BCIRInstall.cmake`, `cmake/BCIRConfig.cmake.in`).
  - One version. The CMake project had said 0.1.0 while the package said 0.2.0; it now reads
    `pyproject.toml`'s version line, `bcir_version.h` is written from it, and a test holds
    `bcir.__version__` to the same number.
  - The public headers are every `runtime/c/bcir_*.h` and the seam's `bcir_*.h`/`.hpp`; the
    cfront fixtures' headers stay out. Each compiles on its own, which the consumer proves.
  - The gate, `tools/build/install_consumer.py` (CTest `build-install`), installs the configured
    build into a scratch prefix and holds the prefix to the manifest: an archive per library, every
    tool, every public header and nothing more, and the package files. It then builds
    `tools/build/consumer` against the prefix with the build's own compilers, from a scratch
    directory holding that project and a copy of the harness, so a quoted include cannot reach the
    tree: every library and tool must be an imported target, every header compiles alone, the
    version header agrees, and `test_runtime` links against `BCIR::*` alone. A consumer asking for
    the next major version must fail to configure, and the `runtime` section must print the same
    output over the consumer's `test_runtime` as over the tree's.
  - Found by the gate's own RED: a version file that accepted every request still saw the next
    major refused -- by the consumer's check that it got the version it asked for, which runs
    after `find_package` -- so the refusal the gate required never tested the version file. The
    gate counts the refusal only when `find_package` itself refuses the package for its version
    (laws.md L11). RED: a public header left out of the install, and a version file that accepts
    every request, each fail the gate; the rules restored, it passes.
  - Measured: `build-install` passes on the gcc, clang 18 and clang 23 trees -- 12 libraries
    (the seam's included), 5 tools, 52 public headers and the package files, the consumer
    linking `test_runtime` against `BCIR::*` alone -- and the `fuzzer` and `asan` presets
    configure with the install rules in place.
  BUILD-4 (2026-10-05) gave the host's dependencies one predicate, `bcir.toolchain`, asked by
  the configure and by every Python harness that needs to know.
  - `optional_library(NAME)` answers for FFTW3F, LAPACKE, GSL, SLEEF and libcerf with the link
    flags that work, or None. Its probe includes the header and calls a function, so the link must
    resolve the symbol. Nine tests had carried a probe of their own: eight copies across the
    link-flag rule modules and one inline in the Area-B red team. The FFTW copy only declared
    `fftwf_execute`, so any `libfftw3` passed it, symbol or not.
  - `host_c_compiler()` is the one compiler pick: `$CC` when it is set, and then only it, else
    clang, gcc, cc. native_bench had ignored `$CC`; the model gates took it unresolved. Each had
    chosen in its own order.
  - The configure runs the predicate (`python -m bcir.toolchain probe-libraries`) in place of
    `find_library`/`pkg-config`, and `bcir-deps.json` records each library's flags. A harness run
    with `BCIR_DEPS_INDEX` at the index -- the CTest Python entries are -- reads the configure's
    answer and probes nothing; an index missing a row, or malformed, raises instead of falling back
    to a probe. D1 asks the same schema predicate the harnesses read the index with.
  - Found and recorded, not fixed here (FFTW-PRECISION): both rails' link-flag rules map `fftwf_*`
    to `-lfftw3`, but FFTW builds its single precision as `libfftw3f`. No CI host has FFTW, so the
    linked path has never run.
  - Found by the first configure that ran the predicate: a probe's detail read "tried -lfftw3f;
    -lfftw3", and CMake keeps the index's rows in a list whose separator is the semicolon, so
    the row split and the join put a raw line feed inside a JSON string. D1 (`build-deps-index`)
    refused the index. The row recorder now maps the semicolon, and any byte outside printable
    ASCII, to a safe one, for every caller (L14); the probe also writes its detail without one.
  - Found by the quick tier: the probe wrote its C source without pinning the line ending
    (`test_line_endings`), so the bytes it compiled depended on the host; it pins `\n` now.
  - Measured: the gcc and clang 18 configures record all five libraries absent on this host, the
    index passes D1, and CTest's `python-quick` -- the quick tier with `BCIR_DEPS_INDEX` at the
    index -- passes on each tree; the quick tier with no index passes (4213 passed), as do the
    thorough tier on LLVM 23 (4213 passed) and the changed modules probing with the host's own
    compiler.
  BUILD-5 (2026-10-05) put the MLIR rail under the top-level project in CI and made the fixtures'
  own RUN lines a test.
  - `mlir-rail-validate` builds `bcir-opt` with the `mlir` preset (MLIR_DIR from the matrix
    major) instead of `tools/wsl/build_mlir.sh`'s tree of its own, and runs the rail's gates as
    CTest entries: `ctest --preset mlir` runs the IRDL round trip, the ODS examples, the passes,
    bytecode and the lit suite. The WSL scripts stay for the WSL route (§7 of the build roadmap).
  - `mlir/lit.cfg.py` runs every `.mlir` under `mlir/` as a lit test against the `bcir-opt` the
    build made and the FileCheck and mlir-opt of its LLVM, each pinned by path through a
    substitution that replaces command words only. Two files are another gate's input and are
    excluded by name, each name naming one file. Until now nothing ran the fixtures' RUN lines
    as written: `check_passes.sh` runs the same files with invocations of its own. A missing
    tool, a path that is not an executable, a tool with no LLVM version and three tools of two
    majors are each a lit fatal, never a smaller suite or a mixed toolset (L1, L2).
  - The configure looks for lit, FileCheck and mlir-opt beside the LLVM the MLIR package belongs
    to, never on PATH, and records a LIT row in `bcir-deps.json`. `BCIR_REQUIRE_LIT=ON`, passed by
    the CI job that installs them and by the local check's mlir leg, makes their absence a
    configure failure instead of a status line (L2).
  - Found by the first run on MLIR 23, fixed before the commit: the config had defined its own
    test format class to select the test directories. lit pickles every test, with its config
    and so its format, to hand it to a worker, and a class defined in the config is not
    importable by name there, so the suite died before its first test. The stub-based unit
    tests had passed: they executed the config but never pickled it. The config now uses lit's
    own `ShTest` and `excludes`, and the unit test pickles the config as lit's pool does, with a
    witness that a config-defined format fails it (L11).
  - Found by the suite's first full run, fixed: `test/irdl/codegen_targets_generic.mlir` checked
    a capability's triple after the lower contract printed below it, and FileCheck matches in
    order, so the file could never pass as written. `tools/irdl/check_corpus.sh` reads the same
    CHECK lines as unordered literal substrings and had passed it. The fixture now checks in
    order, each op's own attributes with CHECK-SAME, which both readers accept.
  - Measured on MLIR 23 (clang 23 tree, `BCIR_REQUIRE_LIT=ON`): `ctest -L mlir` 5/5, the lit
    suite 131/131. RED: a CHECK line broken in a scratch copy of the suite fails it and names
    the fixture; a fixture with no RUN line is UNRESOLVED; the fixture as it was fails; FileCheck
    18 beside `bcir-opt` 23 is a lit fatal; with FileCheck made unfindable the configure fails
    under `BCIR_REQUIRE_LIT=ON` and, without it, leaves `mlir-lit` unregistered and says why. The
    index passes D1 with the LIT row; the quick tier passes (4220 passed).
  BUILD-6 (2026-10-05) landed BCIR Make in the oracle: `bcir/make/` (the BCIRfile grammar, the
  MK0-MK5 laws, the planner) and `bcir-make --dry-run`.
  - The BCIRfile is line-oriented ASCII with one spelling for everything, parsed by its own
    grammar before anything is converted, bounded, and refused at the first departure (MK0).
  - It is judged on the IR: files are resources, targets phases with one dispatch claim each, and
    a reader depends on the writer of what it reads. The anti-cycle law and the plan's order are
    the verifier's own predicates, `phase_graph_has_cycle` and `topological_phase_ids`.
  - MK1 claims, MK2 anti-cycle, MK3 the static footprint (a command's file arguments, a flag's
    attached value included, against its claims), MK4 every input exists, MK5 tools by identity.
    Each law has negative fixtures that fire with its own code.
  - A generation tag is the sha256 of a canonical text of what a target is made from: tools by
    identity, commands, reads by bytes (or by their producer's tag), trees, writes. A test asks of
    every kind of input whether two trees differing only in it differ in the tags, and of a file
    nobody reads and of an `after` edge that they do not.
  - The C rails' build is the first BCIRfile: `tools/build/bcirfile.py` writes it from the
    manifest, 546 targets, every manifest library, tool, harness, seam unit, kernel and buildable
    variant by its manifest name, passing every law with the tools' identities checked.
  BUILD-7 (2026-10-05) landed the runner: `bcir-make` without `--dry-run` runs the plan
  (`bcir/make/run.py`), and the runtime gate's sections run as its tasks.
  - A target whose recorded tag is its tag and whose outputs are there is up to date; otherwise
    the artifact cache restores it when it holds the tag, every file checked against the entry's
    index (a tampered entry is a miss); otherwise it runs and its outputs are stored. The state
    is written after every target.
  - Two workers, in the IR's canonical order as producers finish; a failure stops the run unless
    `--keep-going`, and skips its readers with the reason. A command runs its declared tool by
    path, never by PATH; a tool a command reaches without starting with it (a section's compiler)
    is one the target `uses`, handed to it as `BCIR_TOOL_<NAME>` and counted in its tag.
  - The observed footprint: a target's outputs are removed before it runs and must exist after,
    and a file that appeared or changed beside them which no target claims fails it.
  - Every target that ran or was restored is one TelemetryEnvelopeV0 `datadna` record through
    the live ring (`bcir.gem.ring`), drained in order; `--telemetry` appends them.
  - `bcirfile.py --sections` plans every manifest section as a task: `tools/build/run_section.sh`
    runs the script over the BCIRfile's binaries, and its stdout, stderr and status are the
    task's outputs. A section reads its script, its binaries and, whole, the trees its scripts
    reach, so an edit anywhere in them re-runs the sections: never too few.
  - The gate, `tools/build/make_gate.py` (CTest `build-make`): every target runs and passes, a
    second run executes none, each section's recorded verdict is its direct shell run's, and an
    edit to one runtime unit executes exactly that unit's readers in the lowered graph.
  - Found by the gate's first run with Clang, fixed before the commit: Clang stages an object as
    `<name>.o.tmp` beside it, and a neighbour compiling into the same directory at the same time
    saw the temporary and failed as if it had written a file nothing claims (GCC writes in place,
    so the GCC run passed). The runner now never runs two targets that write into one directory,
    so what appears there is the target's own, and the rails' BCIRfile gives each object and each
    section's verdict a directory of its own to keep two at once. A test stages outputs the way
    Clang does; it fails without the rule.
  - Found by the same run, fixed: a tool was declared by its resolved path, and `clang++` is a
    link to `clang`, whose driver picks C or C++ from the name it is run by, so run by the
    resolved name it linked the C++ seam as C. A tool is now declared where PATH finds it, its
    symlinks kept (its identity hashes the bytes the name reaches either way), and a test links
    a C++ unit through the declared `clang++`.
  - Found by the quick tier, fixed: five writes (the state, the cache index, the telemetry log,
    the gate's edit, the planned list) let the host choose the line ending (`test_line_endings`).
  - Measured: `make_gate.py` passes on gcc (212 s for the 652 targets) and clang 18 (198 s);
    the edit executes exactly its 115 dependents. The quick tier passes (4238 passed), and the
    thorough tier on LLVM 23 (4238 passed).
  BUILD-8 (2026-10-05) landed the C twin of `bcir-make`, its parity with the oracle, the artifact
  cache's manager and the pins, and retired the runtime gate's own builds.
  - `bcir-make` (runtime/c/bcir_make.c, a manifest tool) judges and plans as the oracle does: the
    grammar check by check in the parser's order, the laws target by target, the lowering, the
    IR's three depth-first walks (the cycle check, its witness, the canonical order), the tags, the
    decisions and the waves, and it prints `--dry-run`'s text byte for byte. It plans; the oracle's
    runner runs. A hosted tool: one arena over the host allocator, no recursion.
  - `tools/build/make_parity.py` (CTest `build-make-parity`) generates a seeded corpus: scratch
    trees, BCIRfiles generated as lawful DAGs, two in three broken one way -- every grammar
    refusal, every MK1-MK5 finding and every state the planner refuses, each once before any
    repeats -- and the rails' own BCIRfile; both rails get the same argv and must print the same
    and exit the same. A twin that words one finding otherwise, or schedules a wave one target
    wider, fails it (`test_make_twin`).
  - Laws both rails gained while the twin was written: a `reads-tree` holding a file the grammar
    cannot spell, or a directory that cannot be listed, is an MK4 finding (a tag names every file
    it digests); a relative tool path is under the root for the identity check as for the runner;
    the file is read to one byte past its bound and no further; a recorded state that names a
    target twice is refused.
  - The artifact cache's manager: `--cache-verify` holds every entry to its index,
    `--cache-prune` drops what neither the current nor the two recorded generations name, and
    `--rollback` puts the previous generation back from the cache, every entry checked first, all
    of it or nothing, swapping the two so a second rollback rolls forward.
  - The pins: the configure records every tool the build and the rails' BCIRfile run (cc, cxx, ar,
    python) by resolved path and by the sha256 of its bytes (`tools` in `bcir-deps.json`, held by
    D1); `tools/build/pins.py` (CTest `build-pins`) hashes them again -- a tool whose bytes moved
    since the configure is a finding -- and judges the BCIRfile written from the pins.
  - The runtime gate retired its builds: `check_runtime.sh` runs BCIR Make over the rails' BCIRfile
    with every section a task and shows each recorded verdict in order, then has the C twin plan
    the finished tree, which must be the oracle's plan. Its freestanding compiles became one
    compiler-only section, `freestanding`; its Q8 drift check, which regenerated tracked files, left
    for `test_q8_embed`, which compares without writing. What is left in the gate is glue.
  - M13 holds the gate to showing every section once, running no section script itself,
    building through BCIR Make and keeping the guards below; M14's and M15's gate clauses retired
    with the lines they read, and the rails' BCIRfile is held to mutants through `mutate.py` and
    kernels through `emit_kernel.py` instead. `build-section-parity` and `build-parity` compare the
    CMake build with BCIR Make's, where they compared it with the gate's one-command recipes.
  - The #719 tests (`test_the_gate_and_the_harness_link_the_same_sources`, in four modules) read
    the gate's source list where it is now decided, the manifest closure BCIR Make links
    (`bcir/tests/gate_links.py`): a fixture that compiles a unit outside the closure, or builds
    without the unit's main, fails. The M9 witness, whose fault sat in a link line of the runtime
    gate, now injects it into `check_memory_discipline.sh`, which still links.
  - Found by re-reading the retired gate before the commit, fixed: the gate showed each section's
    verdict as it lay on disk, so after a BCIRfile that broke a law (BCIR Make ran nothing) or a
    build target no section takes that did not compile, the previous run's verdicts stood in for
    this run's and the gate printed `ok`. It now fails unless the laws passed and every target but
    a failing section ran, was restored or was up to date, and it shows a verdict only for a
    section this run's BCIRfile plans. RED in a scratch copy holding an earlier run's verdicts:
    both faults passed the gate as it was, and fail it now.
  - Found by `test_line_endings`, fixed: nine writes in the cache manager and the parity corpus
    let the host choose the line ending.
  - Found while formatting the parity gate, fixed: ruff's `extend-exclude` named `build`, which
    excludes every directory called build, so `tools/build/` -- the build tooling -- was never
    linted or format-checked, locally or in CI. The exclusion is anchored at the root now, and the
    one file it would have flagged, `make_parity.py`, is formatted.
  - Found by the new gate's first run under GCC, fixed: the freestanding compiles, moved verbatim
    from a gate that only ever ran under Clang, asked for `-std=c23`, which GCC 13 spells
    `-std=c2x`. The section now takes the spelling CC accepts, as every compiling section does;
    the gate before BUILD-8 could not run under GCC 13 at all, failing at the same compiles.
  - Found by CI's clang 23 analysis cell, fixed: clang 23's stream checker (`unix.Stream`) flagged
    the twin's two read loops, which read a stream again after a short read -- the end of the
    file or an error, after which C11 leaves the position indeterminate. Each now stops at its
    first short read and asks `ferror`; the parity corpus is unchanged (401 cases, 0 differ). The
    analysis ran only on CI: its unit list lived twice in `ci.yml` and in no local check. It is
    one script now, `tools/c/analyze_hosted.sh`, which both CI steps and a new `analyze` leg of
    `tools/local/check_latest.sh` run (two workers locally); a test drives it with a stand-in
    clang that reports a finding while exiting 0, exits nonzero, or is missing.
  - Measured: on clang 18 the gate takes 149 s, BCIR Make building and running its 639 targets
    in 85 s with two workers, where the gate before took 213 s one command at a time; it prints
    every PASS line the gate before printed (the ring's injected-race line up to its declared
    varying count) and 17 more. Under GCC 13 it passes in 152 s. A second run executes nothing.
    `ctest -L build` passes 8/8 and `ctest -L section` 107/107 on the gcc and clang trees; the
    twin agrees with the oracle on 401 cases (400 generated and the rails' own BCIRfile); build
    parity holds 1920 rows. On LLVM 23 (`check_latest.sh`): the clang23 tree 107/107 and 8/8, the
    gate 406 PASS lines, the thorough tier 4245 passed; the quick tier passes (4245 passed).
  BUILD-8b (2026-10-05): the runner starts the ready targets by measured priority, as the
  roadmap's scheduling design said it would (§8): of the targets ready to start, the one with
  the longest measured path still ahead of it goes first (`bcir.make.run.upward_ranks`, HEFT's
  upward rank over the durations earlier runs measured, kept beside the state and read by
  nothing else; the canonical order breaks ties).
  - The durations order the work and decide nothing else: no tag, law, plan or verdict reads
    them, and a file that cannot be read is no measurement. A test drives a slow target to the
    front once it has been measured, with the same targets run and the same bytes written; with
    the ranks flattened it stays at the back.
  - Measured (a `wall` metric, indicative only), gcc, two workers, interleaved samples: the
    rails' BCIRfile with its sections built from nothing took 96.9 s and 92.5 s with nothing
    measured and 97.9 s and 93.1 s with the first run's durations as priors; an edit to
    `runtime/c/bcir_diag.c` (116 targets re-run) took 44.6, 45.3 and 46.2 s against 46.8, 46.7
    and 45.4 s. No change beyond the noise, and the bounds say why: the full build's work is
    169 s (two workers need 84 s) and the edit's 68 s (34 s), each far above its critical path
    (20 s; for the edit at least the 17 s of `kplan` alone), and the manifest already lists the
    longest sections early (`ring`, `ring_tsan`, `handoff` and `kplan` are its 11th to 14th of
    107), so the canonical order starts them early too. What is left
    above the bound is the runner's own cost per task and the one-writer-per-directory rule over
    `bin/` and `lib/`, which no order buys back. The priority makes an early start of the long
    tasks a property of what was measured rather than of the manifest's listing order.
  - The quick tier passes (4248 passed).
  G19 EXEC-EXACT (2026-10-06): the proved optimum is the plan that runs (audit item 1; the
  first P1 slice of the completion ladder).
  - The defect: G4's exact rail proved every instance of the report's section 6.1 corpus, but
    certified the dispatch's placement without replacing it. `verify_execution_plan` re-derived
    the placement with the canonical dispatch and refused any other, so the solver's shorter
    and equally legal placement could be proved but never carried. On the parent (`d16ea67`) a
    plan carrying the 9-tick optimum of durations (1, 3, 3, 3, 4, 4) on two domains, where the
    dispatch takes 10, is refused by R9 ("the plan's placement is not what the canonical
    dispatch produces"). 190 of 1,716 instances on two domains, and 18 on three, ran
    suboptimal with a certificate saying so.
  - What landed:
    - a plan may STATE its placement: header flag bit 0, `PLAN_FLAG_EXPLICIT_PLACEMENT`, legal
      on an eft plan only. A flag, not a wire version: the record families are unchanged, and a
      reader that predates it refuses the bit as reserved, so none runs the plan as the
      canonical placement;
    - `gem.schedule.placement_violations`, the legality of a phase-barriered placement read
      from where the dispatch reads it: every claim placed once, on a stream it is eligible for,
      for its duration, no earlier than its phase's barrier or its hazard predecessors, one claim
      at a time per stream, and the makespan the last finish. The verifier (R9) holds an
      explicit placement to it, and the canonical one to equality as before;
      `plan_from_realization(placement=)` reads it before minting;
    - `gem.execution_plan.exact_plan` mints the solver's incumbent where it beats the dispatch,
      the canonical plan byte for byte everywhere else, and returns the certificate with it. A
      schedule-liveness memory plan must be bound (`schedule_digest`) to the placement the plan
      carries;
    - the flag on the native codec, the ASN.1 projection (version 4, `flags [15]`) and the C
      twin (`bcir_ep_validate`: any other bit `BCIR_ERR_RESERVED`, the flag on a token plan
      `BCIR_ERR_PLAN`), and in the `execution_plan` gate section, which BCIR Make and CMake
      both run.
  - Found while building it: the first version of `placement_violations` read a slot on stream
    number `domains` -- one past the last -- as the tail, which the dispatch indexes there, so a
    claim the tail would take was accepted on a stream the target does not have. The witness
    pins the exact message list, which is how it surfaced.
  - Gates: `eft.executed.suboptimal.2domains` 11.07% -> 0 and `.3domains` 1.05% -> 0, read
    through each plan's bytes and the verifier. Every witness of `bcir/tests/test_exec_exact.py`
    fails on the parent (the API and the flag did not exist), and so does the section's new leg
    against the parent's C runtime (`BCIR_ERR_RESERVED`). A generated differential over seeded
    random modules with intra-phase hazards and random step costs mints every plan and admits
    it, explicit placements among them. Every corpus program's plan is byte-identical to the
    canonical one: the dispatch is already optimal on all of them.
  - Measured against `98fcec6` (`ab_audit.py`, groups `sp`, `sched` and the new `xplan`, which
    times the plan path -- mint, encode, decode, verify -- that no StreamPack row reached):
    PASS. Every StreamPack and scheduler row's calls and digests are unchanged; `xplan@4`'s
    digest is unchanged and it makes 14 more calls of 1,583,094, the flag checked in each
    `validate_plan` and read by the verifier. That row also shows the plan path costing 19
    times the planner's calls at 4,096 claims, which is the oracle's cost to look at next.
  ASN1-N (2026-10-06): ASN.1 NULL has one abstract value, `codec.NULL`, on every encoding rule
  (audit §4.5, found by ASN1-R's review; a P0 slice of the completion ladder).
  - RED on the parent (`935cb97`), in every witness of `bcir/tests/test_asn1_null.py`:
    - BER/DER took `codec.NULL` and refused `None`;
    - PER and OER took any value at all and decoded `codec.NULL`. They put nothing on the
      wire for a NULL and never looked at the value, so `5` went out as a NULL on both;
    - XER and JER took `None`, refused `codec.NULL` and decoded `None`.
    So a value one rule decoded was refused by another rule's encoder, and `None` -- which
    the value model reserves for an absent component -- was a NULL on two rules.
  - What landed: one predicate, `codec.require_null`, asked by every encoder (DER's message
    now says what it means). Every decoder returns `codec.NULL`. `test_asn1_emit` had pinned
    the disagreement on purpose, so that unifying the spelling would be a deliberate act
    with a visible dependent. It now pins the one spelling, and the write-plan parity test
    drives every emitter from one value instead of two.
  - Found by the latest-toolchain check's thorough tier: `test_c_per_plan`, which needs a C
    compiler and so is skipped by the quick tier, spelled a NULL as `None`. It now spells
    `codec.NULL`.
  - Measured against `935cb97` (`ab_audit.py`, groups `sp`, `sched` and `asn1`): PASS, every
    ASN.1 row's calls and digests unchanged (the measured modules hold no NULL).
  ASN1-T (2026-10-06): a tagged type is a type, so its tags go wherever it goes (audit §4.4,
  found by ASN1-R's review; a P0 slice of the completion ladder).
  - RED on the parent (`94b82df`), in seven of the eight witnesses of
    `bcir/tests/test_asn1_tagged_types.py`. The type model carried tags on components only: a
    tagged assignment's tag was copied onto each untagged component that named it. Every other
    place a tagged type can stand dropped a tag:
    - a component's own tag over a tagged type replaced it. RFC 4120's `ticket [3] Ticket`, with
      `Ticket ::= [APPLICATION 1] SEQUENCE {...}`, went out without the `[APPLICATION 1]`;
    - an element of SEQUENCE OF lost it (`SEQUENCE OF [0] INTEGER`, `SEQUENCE OF Ticket`);
    - a chain of tagged aliases kept the outermost tag at best, and none encoded directly;
    - a direct encode of a tagged assignment carried no tag;
    - `[1] IMPLICIT T` with `T ::= [5] CHOICE {...}` was refused, as if T were untagged.
    The eighth witness pins what the component copy already got right: a SET ordered, and an
    OER CHOICE alternative told apart, by the outermost tag.
  - Found while building it: the write plan's OER emitter wrote an untagged CHOICE
    alternative's INDEX as its tag, on both rails (`emit.py` and the C twin `bcir_emit.c`), so
    `CHOICE { a INTEGER, ... }` under EXPLICIT TAGS went out as `80 01 03`, not `02 01 03`. The
    twin agreed with the Python emitter, so their differential never saw it; the new witness
    compares the twin with the oracle. The twin also admitted tag number 63, which OER's
    one-octet form spells as its escape (X.696 §8.7); both rails now refuse 63 and up.
  - What landed:
    - `Asn1Type.tags`: a type's own tags, outermost first. Every type applies them in
      `encode`, checks and removes them in `decode`, and shows the outermost in `base_tag`;
      `untagged_tag` is the universal tag (`require_tag` checks it). The repr of an untagged
      type is byte-identical to before (`certified.py` digests it), and a tagged one differs.
    - The front end builds an assignment-level or element tag into the type (`_with_tag`, a
      copy), and an IMPLICIT tag over a tagged type replaces its outermost and keeps that
      layer's form (X.690 §8.14.4). It no longer copies an assigned tag onto components.
      §31.2.7's forced EXPLICIT applies to an UNTAGGED CHOICE or open type only.
    - The write plan keeps a tag a member shows on the member, where version 5 put it. Any
      other layer goes in a `tags` line of a version-6 plan, which the twin refuses (fail
      closed) rather than encode without it. A plan with no such layer is version 5, byte for
      byte.
  - Measured against `94b82df` (`ab_audit.py`, groups `sp`, `sched` and `asn1`): PASS.
    - The codec rows are unchanged: `asn1_codec:flat` 49,392 calls and `asn1_codec:recursive`
      16,258, identical digests.
    - The compile rows: `pkix` 14 and `x683` 10 fewer calls (no assigned tag is looked up
      per component); `bcir` 9 more, because `dataclasses.replace` reads one more field of
      each of the nine types it copies. Digests are identical.
    - The first run graded two rows. `asn1_codec:recursive` made 80 more calls, a real cost:
      `Reference` had inherited a `base_tag` that made two calls. It now makes one.
      `verify.plan.scope.overhead` left its band (0.80 to 1.07, confirmed). Its code path
      loads no ASN.1 module, and the re-run measured 1.155 and 1.086, inside the band.
  ASN1-R (2026-10-06): recursive types on every encoding rule, held to one bound, from one
  containment graph (audit §4.2; the completion ladder's second P0).
  - RED on the parent (`bc2c1d7`), in every witness of `bcir/tests/test_asn1_recursion.py`:
    - OER, XER and JER had no case for the front end's forward reference;
    - the JER plan refused a recursive type as an open type, and the write plan as a SET;
    - a recursive CHOICE (`Expr ::= CHOICE { lit INTEGER, neg Expr }`) did not lower at all.
      The front end took "the referenced type is not built yet" for "tag it IMPLICITly",
      behind a `pragma: no cover - guarded`, and an implicit tag over a CHOICE needs the tag a
      CHOICE does not have;
    - nothing bounded a recursion. DER encoded a 256-deep value its own reader refuses past
      64, a 1,000-deep one raised `RecursionError` from the DER and PER encoders, and a crafted
      input of a few kilobytes did the same to the PER, OER, XER and JER decoders;
    - `decode_jer` let a deeply nested JSON text raise `RecursionError` from the parser,
      recursive schema or not.
  - Found by reviewing the fix, in the front end:
    - a recursive X.683 instance of a CHOICE template (`E {T} ::= CHOICE { lit T, neg E {T} }`)
      did not lower, here or imported (§9.8). The first version of the fix read the tag over an
      unbuilt reference from the module's assignments only, and an instance is not one;
    - a type defined only as a reference to itself (`A ::= B` with `B ::= A`, `A ::= [0] A`)
      lowered with no type behind it, and failed only when a value was encoded;
    - a recursive reference to a type with an assignment-level tag lost the tag:
      `next T` inside `T ::= [APPLICATION 5] IMPLICIT SEQUENCE {...}` went out as `30`, not
      `65`, because the tag was recorded after the type was built;
    - a contained subtype naming an alias of a recursive type (`B ::= A` inside `A`) was
      refused as "a contained subtype of itself".
  - The audit's table had PER failing too. That row was the reproduction's own error: it passed
    `decode_per` its arguments in the wrong order. PER already resolved the reference, with a
    private predicate the other rails did not share, so the defect was a mechanism on two
    rails of six (L14). §4.2 is corrected.
  - What landed:
    - `schema.Reference`: the one placeholder, where the cycle closes. `schema.resolve` is the
      one predicate every rail follows it by, and `schema.through` the one guard. A level is
      a context variable, so concurrent codecs do not share a count. The bound,
      `MAX_RECURSION`, is the BER reader's nesting bound.
    - Every rail's dispatcher handles a reference at the END of its chain, so a non-recursive
      value pays nothing. PER's old resolver ran on every dispatch; `asn1_codec:flat` makes
      1.1% fewer calls than the parent.
    - The front end decides §31.2.7's EXPLICIT tag over an unbuilt reference from the
      definition being built: the reference names the lowerer building its target, which
      reads the assignment or the instance's substituted body, through names, templates and
      their actuals. Once the module is complete, every such decision is checked against the
      type built, so a misreading is refused, never encoded (a witness injects one). The same
      pass, over the lowerers of imported templates too, refuses a type defined only as
      itself by its cycle and checks a deferred CHOICE's §29.3 distinct tags. An untagged
      CHOICE that contains itself is refused by name.
    - `lower.containment_graph` / `recursive_types`: the containment graph's strongly connected
      components (iterative Tarjan), computed when asked. A reference carries the cycle it
      closes.
    - The JER plan compiles a recursive type with a `ref` back-edge to the ancestor's node.
      Such a plan is version 2, refused by a version-1 reader; a plan without recursion is
      version 1, byte for byte. The write plan refuses one by naming its cycle
      (`A is recursive (A -> B -> A)`).
  - Measured against `bc2c1d7` (`ab_audit.py`, groups `sp`, `sched` and `asn1`; the `asn1`
    group gains two codec rows): PASS.
    - `asn1_codec:recursive` (DER and PER over recursive values, which both trees carry) makes
      8.1% more calls (15,038 to 16,258): the bound, three context-variable operations per
      level. It takes 1.12x the time over nine interleaved rounds;
    - `asn1_codec:flat` makes 1.1% fewer calls, PER's per-dispatch resolver gone;
    - the compile rows make 0.1% to 0.3% more calls, with identical digests;
    - every harness row, the audit digests and the `sp`/`sched` hot paths are unchanged.
    The first version of the guard was a generator-based context manager at +18.8%. Making
    the containment graph lazy and re-checking only the deferred CHOICEs took the compile rows
    back from +25%.
  ASN1-H (2026-10-06): X.683 parameterization made hygienic, and every name a constraint
  mentions resolved or refused (audit §4.1; the completion ladder's first P0).
  - RED on the parent (`1d4c58a`), in each of the sixteen witnesses of
    `bcir/tests/test_asn1_parameterization.py`:
    - instantiation used dynamic scope. An object-set actual was installed in the module's own
      table under the dummy's name while the body lowered, so a nested template resolved its
      names through the enclosing instance's bindings. The name-keyed caches kept what it found,
      and a module's meaning depended on the order of its assignments;
    - a bound written as a value reference was dropped. Under PER and OER, `INTEGER (0..ub)` with
      `ub INTEGER ::= 255` encoded 200 as `0200c8`, not `c8`;
    - value parameters did not parse (`Bounded {5}`), and a braced actual was looked up as a name
      made of its own text;
    - a table constraint over a union, or over a parameterized or inline object set, was silently
      left untabled;
    - contained subtypes and value-set assignments were dropped or refused, and a template could
      not be imported, because imports carried types only.
  - What landed:
    - `bcir/frontends/asn1/lower.py`: each dummy is classified by governor and spelling (§8.3)
      and substituted structurally wherever it can stand, including nested actuals, table
      constraints, a governing class, value references and object-set elements. Nothing is
      installed under a dummy's name. Inline sets and objects get content-addressed synthetic
      names, and instances are memoised by a structural key, never a `repr`. A template that
      recurses through itself terminates, and an infinite family is refused.
    - §9.8: an imported template lowers in its own module, and each actual is lowered first in
      the module that wrote it (`ast.PreLowered`), keeping that module's tagging environment.
    - `bcir/asn1/constraints.py`: `ValueReference` and `TypeReference`. `require_satisfiable`
      refuses a reference that survives, so an unresolved name is a refusal, never a different
      encoding.
    - The parser reads value references, value actuals, table object-set specs, contained
      subtypes and value sets; the printer writes them back, and the round-trip law holds.
  - Found by reviewing the fix, both older than it and hidden while bounds were dropped:
    - the parser read every `lower-name Type ::= ...` as an information object, so
      `v Count ::= 7` was no value at all;
    - a named value was read against the type that named it, not its own (`x C ::= red`).
    The first version of the fix refused modules the parent had compiled by dropping the bound. A
    governor seen as a type, or spelled with a lower-case letter, now makes a value; a literal
    body is never an object; a value resolves against the type it was assigned; and a value
    defined in terms of itself is refused.
  - Measured:
    - the six tracked ASN.1 modules lower to the same types on both trees;
    - the 52 ASN.1 test modules pass (1,021 tests);
    - the new `asn1` rows of `ab_audit.py` give identical digests in both trees;
    - the front end makes 1.0% more calls on the BCIR modules, 1.1% more on PKIX and 7.4% more on
      the X.683 row. That is the price of the walk visiting every position a dummy can occupy
      and of the structural keys. Classifying each template's dummies once, rather than at every
      instantiation, took the X.683 row down from 11.8%.
  AUDIT-0 (2026-10-06): the GEM+/TMSAO audit of #758–#808 against the original plan
  ([`BCIR_GEMPLUS_TMSAO_AUDIT_2026-10-06.md`](research/BCIR_GEMPLUS_TMSAO_AUDIT_2026-10-06.md))
  and the completion ladder it implies (GEM+ roadmap §9). Every slice the roadmap carried had
  landed; of the 22 items the original plan named, 7 were met, 9 partial and 6 missing, and three
  findings sat where no gate looked: X.683 substitution captured names through the module's
  shared object-set table, recursive types survived only BER/DER, and the cyclic collector was up
  to 68% of the oracle's hot-path time at 32,768 claims while those paths left no cyclic garbage.
  - `tools/perf/ab_audit.py`, the before/after audit each slice of the ladder now ends with
    (roadmap §0.4): the harness and the bounded performance audit in both trees, the touched hot
    paths timed in alternating rounds with each tree in its own process; `exact` rows, call
    counts and output digests graded with zero tolerance, a `ratio` row held to its band, a time
    never graded; a removed metric fails, and a regression lands only with its explanation.
  - RED, on its first run: fifty calls injected into `realize.optimize` in a scratch tree read
    1.04x *faster* on the clock and were refused by the exact call count (73,761 -> 73,811, exit
    1). `bcir/tests/test_ab_audit.py` feeds the grader each kind of change it exists to catch, and
    runs the real child and two-tree paths over a fixture-free row; an unreachable ref comes back
    as a structured UNAVAILABLE report.
  - The tool's first audit of this slice -- no runtime change on either side -- failed twice,
    and both were the instrument: a `ratio` row measured once per tree left its 25% band under
    load (`verify.plan.scope.overhead` 1.02 -> 1.47), and one hot path read as nondeterministic
    between rounds with nothing kept to say what moved. A ratio row that leaves its band now
    earns confirming measurements of its harness group in both trees and is graded on their
    median, every measuring process runs under one hash seed, and a nondeterministic row
    names the quantity that moved; each has a witness that feeds it the case.
  - The moving quantity was a call count, and the instrument was again the defect (audit §4.3).
    `sched_eft@4` counted 34,368 calls in one process and 36,415 in the next, with the collector
    on or off. A per-function diff named one row, `<string>:2:__init__`: `pstats` re-keys
    cProfile's per-code-object entries by (file, line, name), so it merges every dataclass
    `__init__` into one row and keeps one class's count, which one depending on allocation.
    Every call-count row read its count that way, the frozen harness's included. `planner.calls`
    read 589,856 at scale 8 while the planner makes 655,393, and the lost 65,537 were the
    constructor calls its emission floor counts. G17's factor is 5.42×, not 5.80×; its gate of
    5× holds.
    - One counter now, `bcir/tests/call_counts.py`: one entry per code object, with the collector
      paused across the call. Both fixtures count through it, and `ab_audit.py` loads it by path
      from its own checkout, so the two trees of an A/B are read with one instrument.
    - `test_call_counts.py` and three tests in `test_ab_audit.py` were proven RED three ways:
      against the `pstats` total, without the collector pause, and with the fixtures on their old
      counters. Two of them are AST scans that refuse a `pstats` total anywhere in the package or
      in `tools/`.
    - `planner.calls`'s frozen baseline is re-counted over `realize_reference`, the parent's
      planner kept verbatim; counted the old way, that code reproduces the frozen 3,419,172 to
      within four calls. The parent's delta and full-chain baselines keep their numbers, labelled
      as `pstats` totals: lower bounds, so a GAIN graded against them is understated, never
      overstated.
  - The session digest and the systems-engineer skill said everything BCIR emits was TMSAO-4 and
    that G1–G18 were open; both now say what landed and point at the ladder.
  FFTW-PRECISION (2026-10-09): each FFTW precision links its own library, on both rails -- the
  follow-up BUILD-4 recorded.
  - Both rails' link-flag rules mapped `fftwf_*` to `-lfftw3`. FFTW 3 builds one library per
    precision, and the symbol tables of Ubuntu noble's FFTW 3.3.10 packages say so: libfftw3 defines
    `fftw_*` and no `fftwf_` symbol; `fftwf_*` is libfftw3f's, `fftwl_*` libfftw3l's, `fftwq_*`
    libfftw3q's, an MPI entry point libfftw3<p>_mpi's (none for quad), and the threading entry
    points (`init_threads`, `plan_with_nthreads`, ...) are defined alike by libfftw3<p>_threads and
    libfftw3<p>_omp. A single-precision program linked with `-lfftw3` failed on `fftwf_plan_dft_1d`,
    `fftwf_execute` and `fftwf_destroy_plan`; with `-lfftw3f` it links and transforms {1,2,3,4} to
    10 and -2. The B2 wrap is single precision, so no unit with its edge linked with the flags BCIR
    derived. No CI host had FFTW, and the linked tests linked with the probe's flags, not the rule's.
  - Now `fftw_` -> `-lfftw3`, `fftwf_` -> `-lfftw3f`, `fftwl_` -> `-lfftw3l`, `fftwq_` -> `-lfftw3q`,
    `<prefix>mpi_` -> that precision's MPI library, and a threading entry point unknown (no flag): two
    libraries define it, so it names none, the unknown-callee policy's case. The oracle's
    `_fftw_library` and the twin's `fftw_library`, in the same place in the rule order.
  - Witnesses: `test_the_fftw_rule_is_one_table_on_both_rails` reads the C harness's `fftw_edges`
    (twelve rows over every precision, MPI and threads; the `linkflags_fftw` section holds the twin
    to them) and holds the oracle to every row. The linked B2 tests (`test_fftw`, `test_fftw2`) now
    link with the flags the rule derives for the wrap's own callees; against the packages' libraries
    both pass, and the old rule fails them. The oracle job installs `libfftw3-dev`, and a step fails
    the job when the probe the tests use finds no FFTW single precision, so the linked path runs in
    CI rather than self-skipping (L2).
  CF-VLASCOPE (2026-10-09): a block that declares a stack VLA is emitted as a scope of its own, on
  both rails -- the follow-up CF-CASELABEL recorded.
  - Both rails inline every block into its parent. A VLA inlined so put the labels after its block
    -- a later `case`, a loop's continue label, a `goto` target -- inside the array's scope, which C
    forbids a jump to enter (C11 6.8.4.2p2, 6.8.6.1p1). Clang and GCC refused the emit, and the
    oracle's Clang check reported the failed build as `skip:build-failed`, so the unit stayed CLEAN.
  - The oracle's `_block` brackets a block that declares a VLA of its own with `ScopeOpen` /
    `ScopeClose`, which the emitter renders as braces; the twin's `p_block` puts `c.scope` before
    the block's first claim and `c.endscope` after its last. Both are emit-only: no claim the
    digest, the plan or the hydrate reads, so every claim graph is unchanged. An inner block's VLA
    is its own block's, not its parent's.
  - Witness: `test_a_block_that_declares_a_vla_is_its_own_scope_on_both_rails` over
    `runtime/c/cfront_vlascope.c` (a braced case arm before more labels, a `continue` before a VLA
    in a `while` and a `for`, a block a `goto` passes, two VLAs and an inner block in one loop
    body, a loop body whose only VLA is in an inner block, an `if` branch). Each rail emits
    exactly eight scopes, the rails lower the unit to one claim graph on the four targets, and each
    emit builds under `-std=c11 -pedantic-errors -Werror` with Clang and GCC and returns what the
    original does. The fixture also joins the corpus's parity and equivalence check.
  - RED: `tools/testing/faults/cfront-vlascope.json` holds four faults, each rail inlining the
    block or counting an inner block's VLA as its parent's; all four fail the witness. The first
    sweep caught two: the fixture had no block whose only VLA was an inner block's, so a scope
    too many on such a parent went unseen until `vs_inner` was added.
  NDT-GEM (2026-10-09): the native C decoder trainer (#810) wired into GEM+, reviewed and made
  faster without changing one output bit.
  - The trainer's step stood outside GEM+: no lower bound priced it, and nothing but the
    parameter count held the C plan to a second statement of what it allocates.
    `bcir.hosted.models.native_program.training_step_program` states the step from the spec
    alone: every arena span in the C plan's order, the values the spans hold (an overwrite
    starts one, an accumulation keeps the one it reads), and one step per kernel call with its
    reads, writes, GEMMs and counted contraction work. G25/G26 then price it:
    `TrainingStepProgram.roofline` is the hierarchical roofline over the C order, and
    `transient_lower_bound` the concurrent-live bound over the activation and workspace values.
    `bcir_decoder_train.h` names the program, so the oracle inventory lists the C unit as its
    twin under the `decoder_train` section.
  - Four harness rows (`--group train`, graded by `bcir/tests/native_decoder_fixtures.py`):
    `train.plan.mismatch` 12 -> 0 (the program against `bcir_decoder_make_plan` over six specs;
    the parent held only the parameter count to Python), `train.mm.mismatch` held at 0 (the GEMM
    against an independent binary32 emulation of its contract on 20 cases),
    `train.memory.transient` 1.326 (999,424 bytes held over a 753,664-byte bound) (headroom recorded for NDT-MEM: GEM+'s incumbent
    layout reaches the bound on the program, the C ABI's fixed spans do not), and
    `train.step.ms` against a floor measured beside it (G26's compute term at the best rate the
    same kernel reaches on the step's own GEMM shapes: TMSAO-3, a bound on this host's kernel).
  - Profiled (callgrind, two bench steps): 278.2M instructions, 69% in `bcir_tensor_mm`, whose
    B^T path -- every forward linear -- strode through B by k per multiply-add; 16% in the AdamW
    update; 2% in the finiteness scans. Three changes, each bit-identical by construction and
    checked so:
    - the B^T path packs a 32x32 tile of B transposed once per (column, depth) tile and reuses
      it for every row; each output still adds alpha*a times b for k ascending, tile by tile;
    - the update's preflight folds every refusal into one flag (`!(|x| <= FLT_MAX)` is exactly
      "not finite or above FLT_MAX") and the commit is its own loop, so neither branches per
      element; a moment out of float's range is refused before it is converted;
    - the finiteness scan checks 1,024-element blocks branch-free.
    175.0M instructions after (-37%); on this container the step went from 10.0 to 6.9 ms (two runs of seven interleaved rounds: 10.17 / 9.93 -> 6.97 / 6.89 ms), the B^T GEMMs 3.0-3.5x faster and now at the other paths' ~16 GFLOP/s, against a floor of 2.8 ms measured beside them (the gap 3.6x -> 2.5x). Twelve training steps from one
    initial state are byte-identical before and after in every arena and every event, under
    GCC 13, clang 18 and clang 23 alike.
  - Witnesses: the C harness checks the GEMM bit for bit against the ascending-k loop over every
    transpose pair (C11 and C23), a finite gradient whose second moment leaves float's range is
    refused with nothing written, and a non-finite weight or accumulated gradient in a later
    block of its span is refused before the forward pass; the tree's tile offset, a descending
    or fused accumulation, an untransposed A, two plan counts off by a span, the dropped moment
    refusal and a one-block scan are each caught (`test_native_decoder.py`,
    `test_native_program.py`).
  - Found by CI's aarch64 runtime job: the harness's own GEMM reference (`want += av*b`) was fused
    into one FMA by clang's default `-ffp-contract=on` on a target that has one, so the reference,
    not the kernel, left the contract; x86-64 builds have no FMA at the baseline ISA and never
    showed it. Reproduced on x86 with `-mfma`. The reference now rounds each product through a
    volatile, and the manifest gives the harness the trainer's `-ffp-contract=off`, as
    `test_ai_kernels` has; the check passes under on/fast/off with FMA on clang and GCC and still
    fails on the tile-offset fault.
  - Found by the review, fixed: the binding built from the checkout's C sources with no check
    that they exist -- an installed wheel ships none -- so `NativeDecoder.build` failed inside
    the compiler; it now names the missing units. Recorded, not changed: the attention backward
    computes each probability adjoint twice (removing it needs a t-float workspace row, an ABI
    change, for ~1% of the bench step), and `sqrt`'s errno semantics keep the update scalar
    unless the build contract adds `-fno-math-errno`.
  ITS (2026-10-09): the 2015 security-envelope study (Bittl et al.) re-asked with BCIR's ASN.1
  machinery, its conclusion inverted on the same content
  ([`BCIR_ITS_ASN1_BINARY_STUDY.md`](research/BCIR_ITS_ASN1_BINARY_STUDY.md)).
  - The paper found TS 103 097 V1.1.1's binary encoding shorter than ASN.1 UPER in three of four
    envelopes and "significantly faster", but its two columns encoded different data models
    (V1.1.1, and a draft of the ASN.1 successor) through different codecs. `its_security`
    reconstructs V1.1.1's binary rules -- ETSI's server is blocked by this environment's egress
    policy, so the reconstruction is held to the paper's Table VI and reproduces all four lengths
    and the 133-octet authorization ticket they imply -- and transcribes the data model into
    ASN.1 one construct at a time. `ITS-SecuredMessage-TMSAO.asn1` moves V1.1.1's prose rules
    into the schema: field lists as sets of OPTIONALs in canonical order, shaped integers, and an
    X.683 instance per rule family. `its_study.framing` attributes every bit to content or to
    one kind of framing, exactly.
  - Size: the transcription under UPER is 92/212/223/220 octets against binary's 96/222/233/230,
    and the TMSAO PER instance under UPER is 86/203/213/212 -- 97.7 on the paper's Table VII
    average, against binary's 108.6 and the paper's winner (optimized EXI) at 98.4. Every TMSAO
    encoding but DER is shorter than binary in every profile. Binary frames the CAM envelope with
    88 bits (eight type codes and three lengths) around 680 bits of content; TMSAO UPER frames
    the same 680 with 32.
  - Time: `bcir.asn1.cgen` compiles the modules to straight-line C for COER, UPER and the
    V1.1.1 byte rule (`ByteRule`) from one value model, so the binary codec comes from the same
    generator as the ASN.1 ones. TMSAO-COER encodes and decodes every envelope in 0.55-0.74 of
    binary's time under clang 18, GCC 13 and clang 23 (0.60-0.73 of its instructions) and is
    smaller than binary everywhere too; UPER, the smallest, costs 1.06-1.80x, its 32-octet runs
    at arbitrary bit offsets. No generated codec references an allocator, and the deepest stack
    measured is 1,008 octets, where the paper measured 12-20 KB.
  - Witnesses: `test_its_security.py` -- the reconstruction against Table VI, IntX's single
    spelling, every malformed envelope refused, the size and Table VII results, the canonical
    order and the bijection to the TMSAO schema, the exact partition. `test_asn1_cgen.py` --
    octet and value parity with the oracle for every codec and envelope, an oracle differential
    on damaged input (the same under ASan/UBSan, and a libFuzzer campaign reporting a second
    spelling like a crash, in the thorough tier), a cyclic value refused by every encoder, no
    allocator symbol, measured memory, a bench that refuses to time a refusing decoder, and every
    generated file compiling alone with every name prefixed. GEM+ rows under `--group its`.
  - Recorded, not built: an ECN replica of V1.1.1's rules. X.692 can state IntX (§19.7's `TO
    BITS` mapping exists for self-delimiting integer codes), but BCIR's ECN rail writes fixed
    encoding spaces and refuses §21.2.7's `self-delimiting-values`; the byte rule is a
    `cgen.ByteRule` over the ASN.1 module instead.
  OER-EXT (2026-10-09): X.696 extensibility on every OER rail, found by the ITS study's schema,
  the first extensible SEQUENCEs the OER rails had been given.
  - `oer.py`, the E1 emitter, its C twin (`bcir_emit.c`) and the plan-driven C decoder
    (`bcir_oer.c`) wrote no §16.2.2 extension bit ahead of an extensible SEQUENCE's presence
    bitmap, and spelled §16.4/§16.5 extension additions and §20.2 extension alternatives as root
    ones. All four agreed, which is what a round trip cannot see. The preamble now leads with the
    bit; additions follow the root behind a length-prefixed bitmap as open types, a version
    bracket being one addition (its members a SEQUENCE, X.691 §19.9's structure); an extension
    alternative is an open type; the C decoder gained `bcir_oer_decode_sequence_ext`. The vectors
    were cross-checked against asn1tools 0.169.0's OER codec, and the one disagreement (it
    flattens a version bracket into separate additions, which its own PER codec does not) is
    stated in the test.
  - Found on the way: the native bench's OER and PER field builders read attributes plan
    version 3 had removed, so every field was described as unconstrained, and the bench timed
    decodes that failed; each case is now decoded once before it is timed. With SIZE ranges
    declared, the C PER plan decoder read a SIZE(2..10) length two too long (11.9.3.3's offset
    from the lower bound, passed as 0) and applied 17.6's more-than-two-octets alignment rule to
    length-determined strings, which 17.8 aligns whatever their count.
  C-PERF (2026-10-09): the C ASN.1 rails faster with identical outputs.
  - The plan-driven PER decoder reads bit fields an octet at a time and skips a string body
    instead of reading it 64 bits at a time and discarding it: a 200-octet record 1.9 -> 0.36 µs
    ALIGNED, 2.0 -> 0.27 µs UNALIGNED.
  - The generated codecs: word-at-a-time bit I/O, each function's cursor in a local with only
    the fields a call can change synchronized, single byte-swapped loads for fixed widths. GCC's
    inliner reported `inline-unit-growth limit reached` on the generated files and left the bit
    writer out of line, so every field went through memory -- UPER ran 1.8-3.8x slower under
    GCC 13 than under clang; the per-field primitives are now forced inline behind a
    `__GNUC__` guard. One bounds test serves the bit reader's fast path and its truncation
    check, and the slow-path window takes values rather than the reader, whose escaping address
    had kept clang's decoder cursor in memory (the same instruction count, 32-38% slower).
    Interleaved before/after: TMSAO UPER encode 0.30-0.34x and decode 0.56-0.62x under GCC 13,
    0.64-0.68x and 0.57-0.60x under clang 18; the binary codec, generated too, gained alike.
  - Found and fixed in the generator: its helper selection could keep a byte-order `#if` and
    drop its `#endif`, and dropped the cursor typedef from a file of fixed octets alone (neither
    compiled); its macros leaked unprefixed (`P_BSWAP16`, `P_NATIVE_WORDS`). Each has a witness
    that fails on the old generator. The stack harness read a freed pointer (GCC 13's
    `-Werror=use-after-free`). `gemplus_baseline.py --group`'s help listed fewer than half of
    the groups; it is now generated from the registry.
  - RED: `tools/testing/faults/its.json` holds 13 faults across the oracle, both C twins, the
    generator and the harness, each caught by its own witness. The first sweep missed one: a bit
    reader over-reading its input, refused only by the final length check and so as MALFORMED,
    which the accept/refuse differential cannot tell from TRUNC. Every cut of every encoding is
    now held to TRUNC. Phase H's selector, given the paper's objective, selects canonical
    UNALIGNED PER at exactly the study's sizes (`test_its_security.py`).
  REVIEW-812 (2026-10-10): an adversarial review of the ITS / OER-EXT / C-PERF diff before it
  merged. Ten findings, each verified against the tree before it was fixed, one more that the new
  witnesses found themselves, and one on the last re-read.
  - The generator (`cgen`). A variable-size INTEGER's decoder skipped the value constraint its
    encoder checks (COER's length-prefixed forms, UPER's semi-constrained and unconstrained ones,
    the byte rule's), so a decoded value could fail its own re-encode. A UPER preamble wider than
    one bit read's 57 bits went out in one `P_put`: after five bits of another field, a shift by
    a negative count. An extensible INTEGER with an open root bound printed `None` into the C. A
    one-enumerator ENUMERATED, a single-value INTEGER and their kin were taken to encode in a bit,
    so a SEQUENCE OF them refused its own valid encodings as truncated; such a count is now
    bounded by its SIZE, and refused at generation without one. The byte rule ignored a variable
    string's SIZE. A byte-rule vector moved at every doubling and abandoned each copy (33 elements
    took 124 slots); it now grows in place while it is the arena's last allocation (64), and a
    vector of one-size elements is counted by division (33, as COER's). The witnesses, each
    compiled from a small schema, also found a function that never names its cursor leaving it
    unused, which `-Werror` refuses. The last re-read found the arena rounding its fill OFFSET up
    to an element's alignment, where it needed the address: caller memory starting off an
    8-octet boundary got misaligned arrays (undefined behaviour, and a bus error on a core that
    traps misaligned loads). The arena aligns the address now.
  - The oracle. CANONICAL-OER accepted a set padding bit in the root preamble -- the C plan
    decoder reports it as non-canonical and the generated codecs refuse it -- and, in the
    additions' bitmap, set unused bits and an addition sent equal to its DEFAULT. Each is a second
    spelling of one value, now refused under CANONICAL; BASIC still reads them. The V1.1.1 binary
    decoder copied the data's prefix for every vector element and read an element of no octets
    forever; it now threads the vector's end through and refuses both. The binary rules and the
    byte rule hold a SIZE alike.
  - The harness read hex with `sscanf("%2x")`, which takes one digit, a sign or a `0x` prefix as
    an octet. GEM+'s four ITS timing ratios divide one compiled codec by another (or by a
    `memcpy`), so like the `native.*` rows they measure the host and the compiler; they are
    host-dependent now, INDICATIVE off the baseline host.
  - The memory table's V1.1.1 arena (1,200 / 1,928 octets) is what its count-less vectors
    reserve; the same structs need 684 / 1,180, what the transcription's COER codec takes. Sizing
    them exactly costs a counting pass (1.5-1.8x the decode's time) or a give-back per vector
    (3-9% of its instructions), so the time-optimal decoder keeps the reservation and the study
    reports both numbers.
  - Re-measured on the final code, fifteen interleaved repetitions per compiler: TMSAO-COER at
    0.55-0.74 of binary's time (0.60-0.73 of its instructions, unchanged), UPER at 1.06-1.80x. An
    A/B of two builds showed byte-identical codecs timed up to 17% apart -- code placement, with
    the instruction counts unmoved -- so the study now states that band and rests its ratios on
    the instruction counts.
  - Recorded, not changed: the Python OER rail checks no INTEGER value constraint on either side
    (`encode_oer` writes 9 for an `INTEGER (0..7)`), where the generated codecs refuse it. No ITS
    schema has such a constraint, so no number moves.
  - RED: 16 more faults in `tools/testing/faults/its.json` (29 in all), each caught by its own
    witness.
  REVIEW-812b (2026-10-10): Codex's four findings on the ready PR, each verified against the tree
  and fixed with a witness, and the family each belonged to closed on every rail that had it.
  - The defaults of an absent version bracket. X.680 §25.12 gives an absent DEFAULT component its
    default, and a bracket's members are components like any other. PER and OER filled the
    defaults of single additions only (PER, with the extension bit clear, on a path of its own),
    so a value of `SEQUENCE { a INTEGER, ..., [[ c BOOLEAN OPTIONAL, d INTEGER DEFAULT 3 ]] }`
    decoded without `d` on both while JER and XER returned it. Both read one predicate now,
    `schema.addition_defaults`.
  - One spelling under CANONICAL-OER. The oracle's CANONICAL decoder read every length
    determinant as BASIC -- a long form for a short length, leading zero octets -- so the
    extension bitmap's and each open type's lengths, and a string's and a quantity's, had second
    spellings at the digest boundary; so did a variable-size INTEGER's contents, an ENUMERATED in
    the long form, a CHOICE tag's long form and a SET OF out of order. `decode_length` takes the
    caller's rules and every site passes them, and §31.4's fewest-octets rule is one predicate
    (`_require_fewest`) for the INTEGER, the ENUMERATED and the quantity. The plan-driven C
    decoder had the same hole: it read a variable-size INTEGER as BASIC, rightly, but reported it
    canonical; `canonical` now says what the oracle refuses, field for field. A present addition
    its encoder omits is refused under CANONICAL too, by the encoder's own `_supplied`: on OER a
    version bracket of nothing but absent or default members (the earlier §31.9 fix looked at
    single additions only), on PER any addition equal to its DEFAULT (§19.5) or such a bracket
    (§19.9).
  - Fixed sizes from 64K under UPER. X.691 §17.8 gives a SIZE-fixed OCTET STRING of 64K octets or
    more a length determinant, fragmented at that size; the generated UPER codec wrote and read
    the octets bare. Generation now refuses the type in both directions, at exactly the size
    where the oracle's encoding changes form; COER is unaffected (X.696 §14.1).
  - The TMSAO mapping's value space. A shaped integer is extensible, and BCIR reads an extensible
    constraint by its root (the frontend keeps no AdditionalElementSetSpec), so the TMSAO module
    admits integers V1.1.1 cannot spell, and `from_tmsao` mapped them to transcription values no
    V1.1.1 encoder writes. Both directions now hold V1.1.1's range through the binary rules' own
    bounds, and the module header and the study say the bijection is between V1.1.1's values and
    their images. Keeping the additional set in the constraint model decides BCIR's posture for
    every extensible constraint, not this module's, and is a follow-up of its own (closed by
    ASN1-EXT, below).
  - Recorded, not changed: the typed BER/DER rail treats a version bracket as one OPTIONAL
    component of its own (a nested SEQUENCE named `[[n]]`), so a value with the bracket's members
    flat -- the form PER, OER, JER and XER read -- is refused by its encoder and decoded without
    them. A follow-up of its own (closed by ASN1-EXT, below).
  - RED: 19 more faults in `tools/testing/faults/its.json` (48 in all), each caught by its own
    witness.
  ASN1-EXT (2026-10-10): REVIEW-812b's two follow-ups -- version brackets on the typed BER/DER
  rail, and the additional set of an extensible constraint.
  - A version bracket is flat on every rule but PER and OER. X.690 has no notion of a bracket:
    §8.9.2 encodes one value per component type the definition lists, and a bracket lists its
    members as component types of the enclosing SEQUENCE or SET; only X.691 §19.9 and X.696 wrap
    it into one open type. The typed rail encoded `[[ c, d ]]` as a nested SEQUENCE named `[[n]]`,
    refused `{"a": 1, "d": 3}` by name and decoded without the bracket's DEFAULT, and SET did the
    same. One predicate, `schema.flat_components`, lists a SEQUENCE's or SET's components with
    each bracket's members in its place: JER and XER had a copy each, PER and OER expanded the
    members inline for their unknown-name check, and the typed rail had none. The ITU texts could
    not be fetched here (the organization's proxy policy refuses itu.int), so the DER is checked
    against two independent codecs, asn1tools 0.169.0 and pycrate 0.8.1, which agree on every
    bracket vector, X.691 Annex A.4's record included. Where they part -- a root component after
    the extension-marker pair, which asn1tools moves before the additions -- this rail keeps the
    definition's order, with pycrate and §8.9.2.
  - A bracket is absent whole or present with its mandatory members: the lowering makes the
    bracket OPTIONAL, and X.691 §19.9 encodes a present one as a SEQUENCE of its members. JER's
    and XER's copies of the flattening dropped that optionality, so `SEQUENCE { a, ..., [[ d
    INTEGER, e BOOLEAN OPTIONAL ]] }` without its bracket -- a value PER, OER and both independent
    codecs read -- was refused there. `flat_components` lists a mandatory member as OPTIONAL, and
    `schema.bracket_violation` holds a carried bracket to its mandatory members on the typed rail,
    JER and XER, on encode and on decode; a member at its DEFAULT does not carry its bracket
    (canonical XER writes it out for an absent one). The bracket's pseudo-name (`[[2]]`) is no
    component on any rule: PER and OER let it through, ignored. JER's X.697 §19.2.3 test, which
    tells an UNWRAPPED choice's object-producing alternatives apart by a mandatory member name,
    reads the same flat view: a bracket's mandatory member separates nothing, so an object that
    is one alternative with its bracket absent and another without an OPTIONAL member is
    reported ambiguous, where it was read as the second without a word.
  - A bracket's members are extension additions of the enclosing type -- X.680 §25.1 lists a
    bracket among the ExtensionAdditions -- and the flattening now says so. It listed each member
    with its own `extension`, False inside a bracket, so JER's ARRAY (X.697 §27.2.1: the root,
    then the additions) put `SEQUENCE { a, ..., b, [[ c, d ]], f }`'s `c` and `d` among the
    root (`[a, c, d, b, f]`) and a root component after the marker pair behind a bracket;
    canonical XER (X.693 §9.6.2: a SET's root sorted by tag, then the additions in the
    definition's order) sorted a tagged bracket member into the root, so one addition went out
    in two orders with and without its bracket; and X.697 §14.2 let a NULL bracket member under
    ARRAY through, where it refuses the same plain addition. These were older than this slice.
  - The flat view keeps a member's identity. JER files an encoding instruction by the object it
    is assigned to, and assigns a NAME to a component (X.697 §9.9); the first cut of this slice
    listed a mandatory member as a fresh OPTIONAL copy on every call, so a NAME on it was looked
    up under another object and dropped, and the decoder agreed, so the round trip passed. The
    flat members are made once per bracket (`Component.flat`), a copy leads back to its member
    (`Component.origin`), and `JerInstructions` files a copy's instructions under that member.
  - Recorded, not changed: a mandatory extension addition outside any bracket (`SEQUENCE { a,
    ..., b INTEGER }` without `b`) is accepted by PER and OER and refused by DER, JER and XER --
    the same question outside brackets. Whether a value of an earlier version is a value of the
    type is a posture for every rule, and a follow-up of its own (closed by ASN1-MAND, below).
  - An extensible constraint keeps its additional set, and the codecs keep the relay posture
    (LangRef §17.3). Both parser sites dropped the AdditionalElementSetSpec, so `(0..7, ...,
    8..255)` and `(0..7, ...)` were one constraint to the model, to the printer (the module did
    not round-trip) and to every structural key built from a constraint (two instances of a
    parameterized type that differed only in their additions were one), and a name in the
    additions was never looked up. `Extensible.additions` carries it, the printer writes it,
    reference resolution resolves it or refuses it, and `known` answers this version's value
    set. No encoding moves: PER encodes a value of the additions as one beyond them (X.691
    §13.1), OER sees no extensible constraint (X.696 §8.2.2 g), and every rule, the generated C
    codecs included, encodes `(0..7, ..., 8..255)` exactly as `(0..7, ...)`. Enforcing the root
    and the additions was the alternative, and it is refused: `(0..7, ...)` would then have to
    refuse everything outside its root, and a relay could not re-encode a later version's value.
    The TMSAO module's shaped integers state V1.1.1's range in the schema now: each one's root and
    additions are exactly that range.
  - RED: `tools/testing/faults/asn1-ext.json` holds 28 faults (BK1-BK19 on the brackets, AD1-AD9
    on the additional set), each caught by its own witness. `its.json`'s RV17 -- the shared
    predicate giving a bracket's DEFAULT members nothing -- is re-anchored on `flat_components`,
    which `addition_defaults` now reads, and still fires (the table gate found the drift).
  ASN1-MAND (2026-10-10): ASN1-EXT's recorded split -- a value without a mandatory extension
  addition -- decided once for every rule.
  - The posture is the relay's (LangRef §17.3): an extension addition may be absent, mandatory
    or not, because a value without it is a value of the version before it was added, and
    X.691 §19.8 and X.696 §16.4 carry the absence with one presence bit per addition. PER and
    OER read and wrote such values; the typed BER/DER rail, JER and XER refused them both ways,
    so a relay that decoded one from a PER peer could not re-encode it under DER. Every rule now
    encodes and decodes it. Each addition is judged on its own, as its presence bit is, so
    `{a, c}` keeps the later `c` without the mandatory `b`; a bracket stays one addition (absent
    whole or present with its mandatory members); a mandatory root component stays mandatory.
  - One predicate, `schema.may_be_absent`, read through `flat_components`: an addition that is
    mandatory as written reads as an OPTIONAL stand-in, made once (`Component.flat`) and leading
    back to the component (`Component.origin`), so JER's instructions still reach it. No rail
    has a check of its own: every missing-component check on the typed rail, JER, XER, the JER
    plan's `required` flag and §19.2.3's mandatory names read the flat view.
  - Checked against independent codecs, not round trips: pycrate 0.8.1 takes the same posture
    on every rule and writes the same octets, `{a, c}` included; asn1tools 0.169.0 decodes it
    on every rule and encodes it under BER, DER, PER and OER, but refuses it under JER and XER,
    and its binary encoders drop `c` after the absent `b` without a word.
  - RED: `asn1-ext.json` gains MA1-MA3 (the flat view reading a mandatory addition as written,
    the predicate letting a root component go absent, a stand-in filing an instruction under
    itself), and BK5/BK18 are re-anchored on the rewritten `_as_component`.
  QUAL (2026-10-10): qualification on a concrete target -- one BCIR-native model, built from the
  checkout, run through the whole stack on real hardware (`docs/BCIR_QUALIFICATION.md`).
  - QUAL-1 made a decoder a BCIR program: `decoder_program` writes the greedy generation as one
    claim per decoder operation per token over a RID ABI (the activations, each layer's weights
    and cache rows), with its planned work and traffic (`program_traffic`); K_BCIR plans it and
    GEM hydrates it like any module, and every projection (module JER; plan binary, DER, OER,
    JER; pack native, DER, OER, JER) round-trips.
  - QUAL-2 executed it through GEM. `bcir_decoder_gem.c` is the C interpreter `bcir_sp_execute`
    dispatches to, `decoder_gem.py` its oracle twin; both generate what the monolithic decoders
    do, bit for bit, on tied and untied heads and on a model wide enough that a reassociated
    product shows. Found by its own witnesses: a run started from the activations the last run
    left, FINAL aliased onto the activation the attention norm writes, and a claim table sized
    from the pack header before the semantic walk read the body -- each fixed, each a fault in
    `tools/testing/faults/qual.json`. A libFuzzer target (`fuzz_decoder_gem`) holds the
    interpreter to its allocation and run-isolation contracts on resealed packs.
  - QUAL-LIN made GEM linear. On a hydrated decoder pack the R10 walk and the executor rescanned
    every earlier record per record -- quadratic, so a 2,900-claim request ran 5-14x the
    monolithic runner. Running maxima with exact fallbacks (shortlex names, cursor searches that
    wrap) and a run-ordered dispatch fast path keep every verdict and the dispatch order
    identical (a differential over eleven bases in every order) and make both linear (a scaling
    witness); the ratio is now 1.06.
  - QUAL-M built the model from the ground up: `build_lab_model.py` runs one recipe
    (`bcir.lab_model.v1`) through BCIR's corpus preparation, BPE, native C trainer and BCIRQ8
    export, and a card ties the artifacts together by SHA-256. BCIR-Docs-860K: 857,216
    parameters (4 layers, width 128, GQA 4/2, SwiGLU, RoPE, tied), 1,200 steps on the tracked
    docs, validation perplexity 63.0. Found on the way: the builder appended a newline to the
    tokenizer's canonical JSON, which the reader refuses, so the first model shipped a tokenizer
    nothing could load; the file is the canonical JSON exactly, and the harness checks that the
    card, the file and the BCIRQ8 metadata name the same tokenizer.
  - QUAL-3, the harness (`run_qualification.py`) and CI's `qualification` job (x86-64 and native
    AArch64, callgrind required): every check gated, every number in one report
    (`bcir.qualification_report.v1`). On this host: the same 48 tokens on five rails, C GEM and
    the C monolithic runner bit-identical, C and the oracle 0 apart, planned == counted traffic
    (57,539,312 octets read, 4,236,480 written), GEM/monolithic 1.055, 918 tokens/s. The pack's
    JER (913,010 octets) needs 100,001 nodes and the bounded JER reader refuses it by design, so
    the JSON chain is qualified on a 12-token request of the same model, which reproduces the
    headline's prefix.
  - The simulated traffic is one run of each rail, read in callgrind's own format. The first CI
    run collected it with `--toggle-collect` on each rail's entry point and scraped
    `callgrind_annotate`'s human-readable function table: on AArch64 the line it took for a rail
    was no function's row and the parse died on it (`int('')`), and on any host a `.` (zero)
    column would have shifted every later event onto the wrong name. bcir-qualify's callgrind
    build (`-DBCIR_QUALIFY_CALLGRIND`) now brackets each rail's first run with callgrind client
    requests, a profile part per rail, and the harness reads the parts by the format's grammar:
    exactly one part per rail, every cache event counted, ASCII decimal counts, a refusal for
    anything else. Profiling two repetitions counts the same instructions and data references
    per rail as profiling one, and the unbracketed build measures nothing rather than zero. The
    header comes from the valgrind on PATH (`-idirafter` its include directory), since a
    compiler with its own sysroot -- the local clang 23 -- does not search the system's.
  - RED: `tools/testing/faults/qual.json` holds 28 faults (QL1-QL15 on the linear walk and
    dispatch, QG1-QG14 on the interpreters and the planned work), each caught by its own
    witness. QG12 was dropped as an equivalent mutant: the oracle's cache is a list, which
    cannot hold the hole the C fault leaves. `qual-sim.json` holds 18 more (QS1-QS18) on the
    profile reader, the derived rows, the brackets and the callgrind build.
  NDT-PERF (2026-10-11): the native-training half of the 2026-10-10 assessment's next update
  (#814), every output bit unchanged, and the 9.84M decoder trained on BCIR and on PyTorch side
  by side (`docs/research/BCIR_NATIVE_TRAINING_PERFORMANCE_2026-10-11.md`).
  - Reviewed first: every table of the 2026-10-10 assessment regenerates from its `results.json`,
    and its gaps were of scope. Its GEMM rows were forward shapes only, its attention forward
    only, the production AdamW unmeasured, and the decoder evidence ran BCIR on vendor BLAS,
    never on its own kernels. At the comparison's shape the base kernel spent 11.6 s of a step
    in GEMMs, 86.2% of it in the backward's shapes (dx = dy W with beta, dW += dy^T x with A
    transposed), which ran at 4.0 GFLOP/s against the forward's 12.5.
  - `bcir_tensor_mm` became a register-blocked kernel. A KCxNR tile of B is packed once per
    (column, depth) block and reused by every row, and an MRxNR block of C stays in registers
    across the tile's depth. There is a portable C instance everywhere, and on x86-64 Linux with
    GCC or Clang AVX2 and AVX-512 instances chosen per call from the CPU (`bcir_tensor_mm_variants`,
    `bcir_tensor_mm_variant`; `BCIR_TENSOR_NO_DISPATCH` / `BCIR_TENSOR_MAX_VARIANT` cap the
    choice). The contract is unchanged: each element starts at its beta-scaled value and adds
    (alpha*a)*b for l ascending, one rounded multiply and one rounded add per term, so every
    instance is bit-identical to the ascending-k loop. Per step: 11.6 s -> 2.98 s portable,
    1.99 s AVX2, 1.24 s AVX-512 (5.2 -> 48.8 GFLOP/s).
  - Attention runs its products through the same kernel with a mask over the matrices' own
    indices (store only j <= i; take term l only when l <= i, or only when l >= i). A masked term
    is skipped, never added as a zero, so nothing outside the causal mask reaches a result, a
    non-finite value included. The new `bcir_tensor_attention_backward_inplace` rewrites the
    layer's probabilities with their score adjoint and needs no workspace. dq, dk and dv receive
    the reference's terms in its order (heads in order for the shared GQA rows), so they are
    bit-identical to `bcir_tensor_attention_backward`, which stays as the reference. Per layer:
    forward 18.5 -> 8.3 ms, backward 53.4 -> 15.5 ms.
  - AdamW's two passes carry the same instances (the build's `-fno-math-errno` lets `sqrt`
    vectorize, changing no result). The first pass records the largest magnitudes, from which
    `adamw_bounded` proves that the preflight would admit every element (monotone rounding, half
    of FLT_MAX in reserve) and skips it; anything else still runs it. 147 -> 65 ms.
  - A step of the 9.84M decoder at B=8, T=128: 12,297 -> 1,702 ms on BCIR's own kernels
    (7.23x), 1,322 -> 958 ms with OpenBLAS behind the CBLAS seam, the training state the steps
    leave identical. Twenty steps end in one state on GCC 13, clang 18 and clang 23, each with
    all three instruction-set variants -- but clang's builds were 20-59% slower (NDT-CLANG,
    below).
  - The comparison trained one model from identical weights, batches and learning rates on four
    backends: BCIR's own kernels, BCIR with OpenBLAS, eager PyTorch and compiled PyTorch, one
    thread each. On the repository's documentation as a byte-level language model, all four end
    at 1.6319 nats per byte on the test split (compiled PyTorch: 1.6318), 1,000 steps. On a copy
    task, all four copy every one of 2,000 held-out strings. Per step: 1,726 / 963 / 774 / 712 ms,
    so time to any validation loss is 2.2x eager PyTorch's on BCIR's own kernels and 1.24x with
    OpenBLAS; BCIR's processes peak at about half of PyTorch's resident memory (544 against
    1,056-1,140 MiB). The harness's planned peak rate of 2e-3 left every structured task on its
    plateau, so all runs use a quarter of it, and the abandoned configurations are recorded.
    The step breakdown puts the gap in the GEMM: MKL runs a step's 60.4 GFLOP in 476 ms, BCIR's
    no-FMA kernel in 1,329 ms, and everything else together is faster on BCIR (162 against
    229 ms).
  - At 1,024 tokens per step from T=64 to T=1024 (the `context` stage), BCIR's own kernels stay at
    2.0-2.1x eager PyTorch's step and the losses agree at every context. BCIR's attention, which
    stores every probability, goes from 0.88x to 1.87x PyTorch's flash-attention path per layer,
    and the arena grows by those probabilities (323 -> 503 MiB). On four threads PyTorch's step is
    6.1x faster than BCIR's single-threaded kernels and 2.1x faster than BCIR with a four-thread
    OpenBLAS.
  - Witnesses: the C harness checks every instance the host runs against the ascending-k loop
    and refuses an instance the host cannot run, with nothing written. It checks attention
    forward and the in-place backward against the reference loops, finite and non-finite, and the
    update against the reference update on both paths. The decoder_train section runs it as C11,
    C23, portable-only and AVX2-capped builds. RED: `ndt-perf.json` NP1-NP9 are each caught;
    `ndt-gem.json`'s NG1, NG2 and NG4 were re-anchored on the new kernel and are still caught.
    `clang --analyze` cannot see that every caller passes a block no wider than its tile, so it
    is told so under `__clang_analyzer__` alone, and the object code is identical with and
    without the hint on all three compilers.
  NDT-CLANG (2026-10-11): the native trainer's GEMM as fast under clang as under GCC, every output
  bit unchanged.
  - Found by NDT-PERF's identity stage: twenty steps leave one state under GCC 13, clang 18 and
    clang 23, but clang's builds ran 30-34% slower with AVX-512, 20% with AVX2 and 58-59% as
    portable C. `NativeDecoder.build`, and every Python harness through `host_c_compiler`, picks
    clang before gcc, so the default build was the slow one wherever clang is installed.
  - The cause was in the generated code. GCC realigns the stack to 64 bytes for the AVX-512
    instance; clang keeps the ABI's 16, so the packed tile's offset within a cache line was left to
    the caller's stack depth and ASLR, and every 64-byte load of a misaligned tile read two lines.
    Across five runs of the same kernel, a step's AVX-512 GEMMs took 1.86-2.80 s under clang 18 and
    1.21-1.31 s under GCC. A store-back bounded by the masks gave clang a variable index into the
    accumulators, and clang 18 kept the block on the stack across the depth loop. The packing loop
    tested layout and padding per element.
  - The fix: the tile is `_Alignas(64)`, the store-back runs to the block width and tests each
    column, and the packing loop is split by layout, each source row read contiguously and the
    padding written once. A fourth change, forming the A operand's address once per block, was
    measured and dropped: it slowed GCC's AVX-512 instance and clang's portable one.
    Against main (`fb8d3a0`), interleaved on one host, the step's AVX-512 GEMMs went
    1,905 -> 1,175 ms under clang 18, 1,604 -> 1,396 under clang 23 and 1,234 -> 1,161 under GCC
    13, and a training step 3,214 -> 1,594 ms (2.02x), 2,187 -> 1,774 (1.23x) and 1,683 -> 1,602
    (1.05x). Twenty steps now take 1.57-1.59 s each on all three compilers with AVX-512, and end in
    the same state as before the fix on all ten builds. The GEM+ row `train.step.ms` (clang 18,
    the host's default) reads 3.40 ms against a 0.78 ms floor, from 4.58 against 1.34. Clang's
    portable instance on x86-64, which runs only without dispatch, is still 1.5-1.7x GCC's.
  - Witnesses: the C harness passes under GCC 13, clang 18 and clang 23, as C11, C23,
    portable-only and AVX2-capped builds and under ASan+UBSan; `clang --analyze` is clean on 18
    and 23. RED: `ndt-clang.json` NC1-NC3 (a ragged block's last column unstored, a transposed
    B's last column or an untransposed B's last depth row dropped from the tile) are each caught;
    NG1, and the same mutant in `test_the_rows_fire_on_wrong_kernels_and_plans`, are re-anchored
    on the new packing loop; NG1-NG7 and NP1-NP9 are caught again.
  VPS-STABLE (2026-10-11): `verify.plan.scope.overhead` timed in paired rounds of the thread's CPU
  time, so a load on the host no longer moves its verdict.
  - NDT-CLANG's audit was the third to flag the row over code it does not touch. AUDIT-0 read
    1.02 -> 1.47, ASN1-T 0.80 -> 1.07, and NDT-CLANG's six runs of identical code 0.49 to 1.03. The
    harness timed three `optimize` calls, then three `verify_plan` calls, the first of them cold,
    and divided the medians: two windows that a load could reach one at a time. In twelve processes
    on this container with nothing else running, it read 0.68 to 1.47. That is two false gains and
    a false regression against the frozen 1.07. Under a phased load on the measuring CPU (one
    process pinned to it, busy and idle in phases of 20-300 ms), it read 0.54 to 2.19: three false
    gains and four false regressions in twelve.
  - `paired_ratios` (`tools/perf/gemplus_baseline.py`) is JER-PAIRED's estimator for callables in
    one thread. Each callable runs once untimed. Then sixteen rounds time the planner, the verifier
    and the floor back to back, in reverse order every other round, and the row is the median of
    the per-round ratios. On the wall clock, rounds alone read 0.88 to 0.99 with nothing else
    running. Under the phased load they still read 0.72 to 1.02, because its phases split rounds
    faster than the median discards them.
  - `thread_clock` times the rounds in the thread's CPU time, which stands still while the thread
    is descheduled. Under the same load it read 0.92 to 0.99, and with nothing else running the
    two clocks agree (medians 0.950 and 0.953, ten processes each). Windows advances thread time
    once per scheduler tick, so the clock's step is measured, not read: a clock coarser than
    10 us, one that never steps, or one that fails gives way to the wall clock.
  - The final instrument, measured alternately with the old one: 0.90 to 0.99 in twelve processes
    with nothing else running, and 0.93 to 1.01 in twelve under the phased load, every verdict
    NO-CHANGE. The quantity, the frozen baseline (1.07) and the 25% band are unchanged. Measuring
    the verifier group now takes about 1.9 s, up from 0.5 s.
  - It still sees a regression. A `verify_plan` that does its work twice reads 1.93-1.96 against
    0.94-0.97 clean, REGRESSION in every process. `ab_audit.py` grades it 0.99 -> 1.90, confirmed,
    next to the exact call count it moves (`kbcir-streampack.full.calls`). An R9 that re-derives
    each step's cost twice adds 13%: 0.93-0.96 -> 1.05-1.09. That is inside the band, so it is
    graded NO-CHANGE as the band intends, but its readings no longer overlap the clean ones.
  - Witnesses, in `bcir/tests/test_gemplus_baseline.py`, use callables of known cost and a fake
    clock, so every tier runs them:
    - a load that steps from 1x to 5x reads 4.5x on a true 0.9x through the old estimator, and
      exactly 0.9x in rounds wherever it arrives or leaves;
    - a drift of 30% a call reads 1.17 in one fixed order, and within 4% of 0.9 reversed;
    - the refusals: an odd or too-small count of rounds, nothing to compare, a call that measured
      no time or NaN;
    - the clock choice;
    - the row reads the estimator over the planner, a clean verify and the floor.
  - `tools/testing/faults/paired.json` injects 15 defects, and each is caught by its own test.
  GAIN-CONFIRM and NATIVE-PAIRED (2026-10-11): the before/after audit confirms a gain the way it
  confirms a regression, and the native and ring ratio rows are timed in pairs.
  - Found by VPS-STABLE's RED audit. Between two trees with the same compiled kernels,
    `ab_audit.py` graded three rows GAIN: `native.blocked-reduction` 19.8 -> 24.8,
    `native.dense-parity` 0.813 -> 1.108 and `ring.throughput` 38.7 -> 25.2. It re-measured a
    ratio row only when its first reading was a regression, so a gain the host made was never
    checked. The roadmap's §0.2 owes every gain an analysis.
  - GAIN-CONFIRM: `confirm_ratios` re-measures a ratio row that leaves its band in either
    direction, alternating the trees, and grades the median of every measurement. The first
    reading fixes only the row's direction. A gain can come out a gain, NO-CHANGE or a
    REGRESSION, and the last fails the audit. `test_a_ratio_gain_is_confirmed_before_it_is_graded`
    runs all three outcomes, on a row where lower is better and on one where higher is.
  - A ratio row is held to the band it declares. The harness publishes each row's `noise` band,
    and the audit grades and confirms a ratio row against it, falling back to 25% for a table
    without one. The ITS codec floors declare 50%, because their denominator, a memcpy of 216
    octets, is the noisiest number in the run. Held to 25%, `its.coer.decode.floor` read a
    confirmed GAIN of 14.8 -> 9.9 on a path no change in this PR touches, inside the row's own
    band. `tools/testing/faults/ab-audit.json` injects 8 defects, each caught.
  - NATIVE-PAIRED. Ten harness runs of identical code read `native.dense-parity` 0.65 to 1.60,
    `native.gather-avoidance` 3.9 to 6.2, `native.direct-stride` 1.18 to 1.40 and
    `ring.throughput` 15.9 to 39.8. The two dense-parity kernels compile from the same C, yet they
    ran as two programs in two processes. The other three drivers timed every call of one kernel,
    then every call of the other.
    - Every driver in `bcir/bench.py` now times both kernels in one program over the same buffers
      (`_paired_rounds_c`). Each round runs three calls of each kernel back to back, and the order
      alternates round by round. A round's ratio is the ratio of its two blocks' best calls, and
      the program reports the median over rounds (`paired_milli`). Each row
      (`measure_native`) is the median of three programs. A random-access kernel's cost follows
      the pages its process draws: one `gather_reduce` program read 23.6, the next 31.7.
    - `ring.throughput` reads each side's best of 16 repeats (`best_ratio`), not the ratio of the
      medians of five. In one process, pinned or not, the two-thread handoff cost 193 to 786 ns a
      record from one repeat to the next, while the one-thread memcpy held still. Pairing ring and
      memcpy cannot cancel that. Over twelve runs of one build, the medians' ratio read 16.3 to
      44.9 and the best repeats' 15.0 to 19.0.
    - After, ten harness runs per tree, alternated with the old: `native.dense-parity` 0.97 to
      1.01, `native.gather-avoidance` 4.47 to 5.05, `native.direct-stride` 1.23 to 1.31, and
      `ring.throughput` 15.8 to 30.2 (nine of the ten from 15.8 to 19.8).
    - `native.blocked-reduction` did not tighten: 23.1 to 35.7, against 20.4 to 29.6 before. Its
      random gather follows the memory system's latency, which moves over seconds on a shared
      host and reaches the two kernels unequally, so no pairing cancels it. GAIN-CONFIRM is its
      safety net.
  - Found while building it: the first timing loop named its block counter `k`. The strided
    driver's calls saw the counter instead of their stride, so the kernel ran at strides 0, 1 and
    2, and `native.direct-stride` read 4.5 where the old driver read 1.24. Every name in the loop
    is now prefixed and scoped. Every driver checks its outputs after the timed calls as well as
    before, so a timed call that computes something else fails the run.
  - Witnesses:
    - in C on a fake clock, `test_the_paired_rounds_cancel_a_load_and_keep_the_drivers_names`.
      A 1x-to-5x load step reads 4.5 on a true 0.9 through the windowed estimator, and exactly 0.9
      through rounds wherever it lands. A true 2.0 reads 2.0. A 5%-per-call drift reads 0.958 in
      one fixed order and 0.902 alternated. Every call sees the program's own `k`, `r`, `a`, `b`;
    - `test_the_native_rows_are_the_median_of_three_paired_programs`;
    - `test_the_throughput_row_is_the_best_repeats_ratio`.
    `tools/testing/faults/native-paired.json` injects 13 defects, each caught. Among them are the
    counter named `k` and a strided driver whose timed calls run another stride than its
    checked ones.
  - The audit of record, `origin/main` against the slice's head, passes. Four ratio rows left
    their band on the first reading and came back NO-CHANGE: `native.blocked-reduction`,
    `native.dense-parity`, `native.gather-avoidance` and `verify.plan.scope.overhead`. It grades one
    GAIN, `ring.throughput` 22.9 -> 15.8, the median of three measurements per tree. That gain is
    the instrument's: the ring's code did not change, the before tree reads the medians' ratio
    and the after tree the best repeats'. The frozen 26x is the medians' number, so a reading
    below it on the reference host is the instrument's too, until the baseline is re-read there.
  - Cost: the native group takes about 5.6 s instead of 2 s, and the ring bench about 1 s more.
  NOW-NS64 (2026-10-11): every timing driver the oracle emits reads one clock, in 64 bits.
  - The drivers of `bcir/bench.py` (five emitters) and of `bcir/clang_compare.py` (one harness for
    eight) each defined their own `now_ns`, computed in `long`: `(long)t.tv_sec * 1000000000L`.
    The nanoseconds since 1970 need 61 bits, and a `long` has 32 on LLP64 hosts (64-bit Windows)
    and on ILP32 ones, where the product overflowed: undefined behaviour, and a reading that is
    garbage where it is not a trap. Run on wasm32, the old clock read 2026-10-11 as -696082667.
    Every reading was held in a `long` and printed with `%ld` as well. No CI cell compiled a driver
    where a `long` has 32 bits: Windows runs the quick tier, which hides the compilers.
  - One clock now: `bcir.bench.NOW_NS_C`, an `int64_t` from `(int64_t)t.tv_sec *
    INT64_C(1000000000)`, after one header block, `DRIVER_HEADERS_C` (<inttypes.h> for `PRId64`).
    Every driver includes both, keeps its readings in `int64_t` with `INT64_MAX` as the sentinel,
    and prints them with `PRId64`. `clang_compare`'s seven best-of loops are one `_best_of`. What
    the drivers print, and how `bench` and `clang_compare` parse it, did not change. The gather
    drivers' `long` indices belong to the kernel ABI, are bounded by the problem size, and stay.
  - Witnesses, in `bcir/tests/test_bench.py`:
    - `test_every_driver_reads_the_one_64_bit_clock` (no compiler): all fourteen driver sources
      include the headers and define `now_ns` once, as `NOW_NS_C`, and an emitter the witnesses do
      not build fails it;
    - `test_every_driver_compiles_where_long_has_32_bits`: every driver type-checked for
      `x86_64-pc-windows-msvc` and `wasm32` against stand-in C library declarations, with a
      reading narrowed to 32 bits (`-Wshorten-64-to-32`) or printed with `%ld` (`-Wformat`) an
      error. Where a `long` has 64 bits, both compile silently;
    - `test_the_clock_reads_the_epoch_in_64_bits_where_long_has_32`: `NOW_NS_C` built for wasm32
      with `-fwrapv` and run under node, against a clock that answers what it is handed: 2026-10-11,
      2^31 seconds, the last instant an `int64_t` of nanoseconds holds, and a nanosecond field
      alone. Each reads back exactly.
    `tools/testing/faults/now-ns64.json` injects 10 defects, each caught, and each witness is the
    only one to catch some of them. Only the wasm32 run catches the old clock's arithmetic. Only
    the type check catches a reading kept in a `long` or printed with `%ld`. Only the first
    witness catches a driver with a clock of its own.
  - Every driver builds with the same warnings as before under GCC 13, clang 18 and clang 23
    (`-Wall -Wextra`), and the native rows read as before: dense-parity 1.004, gather-avoidance
    4.544, direct-stride 1.293 and blocked-reduction 32.57.
  TRAIN-0 (2026-10-11): the training-acceleration ladder, and the corpus as the IR's training data.
  - `docs/research/BCIR_TRAINING_ACCELERATION_ROADMAP.md` reads the 2026-10-11 report's breakdown
    as a budget. Of BCIR's 1,632 ms step (B=8, T=128, 9.84M parameters, one thread), 1,329 ms are
    GEMMs under contract v1, whose one rounded multiply and one rounded add per term cap the kernel
    at half of an FMA peak (48.8 GFLOP/s is 73% of that cap; MKL reaches 127 with FMA); 102 ms are
    libm's `expf`; and the whole step runs on one thread where PyTorch's runs on four. The levers,
    and the ceiling each leaves: a second GEMM contract (`fused`: one IEEE fma per term, ascending
    k, emulated exactly by the oracle, selected by the plan and recorded in the state's digest), a
    deterministic thread pool whose partition keeps every element's reduction order, a BCIR-owned
    exponential, then the GEM+ plan driving the step's dispatch. The budget they add up to: about
    875 ms on one thread against PyTorch's 762, about 270 on four against 257. Each slice names the
    exact row it gates on before its code, and what it may not do.
  - The one-way corpus law (its §5). `training/` was "never a build dependency of BCIR" and
    "neither may depend on the other". The amendment is one-way and data-only: the IR's `train`
    rail (TRAIN-RAIL) reads a pinned selection of the corpus as training text the way the lab
    recipes read `docs/`; nothing under `bcir/`, `mlir/` or `runtime/` imports or compiles
    anything from `training/`; the corpus reaches the IR only through the lazy edges its standard
    names; a change to the IR's semantics still must not require a corpus change, and a change to
    the corpus changes what the rail trains on, never whether the IR builds. Stated in `AGENTS.md`
    §2, the structure doc's table, the corpus README's scope note, `CONTRIBUTING.md` and
    `training/ROADMAP.md`.
  PY315 (2026-10-11): the oracle judged on the newest CPython, 3.15.0, beside the 3.11 floor.
  - CI ran Python 3.11 (the floor job) and 3.12 (ubuntu-latest's interpreter: every other job and
    the host matrix); no cell ran 3.13, 3.14 or 3.15, so `requires-python = ">=3.11"` was a claim
    about three interpreters nothing exercised, and a 3.15-only break (what 3.13 and 3.14
    deprecated is gone; UTF-8 mode is the default, PEP 686) would have shipped green. On 3.15.0
    (conda-forge; github.com's and python.org's downloads were both proxy-denied in the container)
    the quick tier passed 4551/4551, and again with every DeprecationWarning, PendingDeprecationWarning
    and SyntaxWarning promoted to an error; the three security gates and the wheel build and
    import passed; the thorough tier on 3.15.0 with LLVM 23.1.3 and Node 24.21.0 passed. No source
    change was needed: the tree was a 3.15 tree already, unproven.
  - The proof lives where the LLVM 23 / Node 24 policy already lives (L14). `pyproject.toml`
    declares classifiers 3.11-3.15; `python-newer` runs the quick tier, the security gates and the
    wheel on 3.13, 3.14 and 3.15, its first step reading the running interpreter against the
    classifiers as the floor job reads `requires-python`; `host-portability` runs 3.12 and 3.15 on
    ubuntu and windows (the Windows 3.15 cell is where UTF-8 mode changes behaviour);
    `oracle-llvm-latest`'s thorough shards run on 3.15 and confirm it, so the newest LLVM, Node
    and Python are judged together. `test_every_declared_python_version_is_a_ci_cell` holds the
    classifiers, `requires-python`, `PYTHON_SERIES` and the three matrices to one gapless list.
    `hosted-model` stays on 3.12: its pinned torch 2.13.0, numpy 2.2.6 and safetensors 0.8.0
    publish no 3.15 wheels, and the training report's PyTorch numbers are tied to those pins.
  - Locally, `latest_toolchain.py status` reports a `python 3.15` row -- the highest `v3.15.x` tag
    of python/cpython through the one `release_tags` predicate LLVM's query now shares; an
    interpreter from `BCIR_PYTHON_LATEST`, PATH, the BCIR cache's conda-forge envs, uv or pyenv; a
    pre-release's `--version` is not a release -- `env` exports `BCIR_PYTHON_LATEST`, and
    `check_latest.sh` gains a `python` leg (quick tier + security gates on 3.15), confirms the
    interpreter in `confirm`, and runs its `thorough` leg on 3.15.
  BPE-FAST (2026-10-12): the byte-BPE encoder in O(n log n), the same tokens.
  - `BytePairTokenizer.encode` applied every merge as one greedy left-to-right pass over the
    whole sequence, in order: O(merges x bytes). The train rail's 40-step smoke spent 568 s, 22 of
    them training; encoding the 5.3 MB corpus through 763 passes was most of the rest. The
    encoder now merges by rank: the adjacent pairs that are merges sit in a heap keyed by (merge
    index, position) over a linked list of the original positions, so the lowest-ranked pair
    present is merged first and, within one merge, the leftmost occurrence first, as the pass
    did. A merge can only create pairs of a higher rank than its own (its token is newer than any
    token in an earlier merge's pair), so the heap pops ranks in the order the passes applied
    them, and an entry whose pair changed under it is dropped when popped. The pass encoder stays
    as the definition (`_encode_by_passes`); the rank table is computed once per tokenizer.
  - Measured on 319 KB of `docs/` in 60 sections with a 764-merge tokenizer: passes 21.13 s,
    ranks 0.61 s (34.7x), identical tokens. BPE training (the 768 KB sample to 1,024 tokens:
    about 44 s per 262 KB here) is now the smoke's largest cost; the trainer's pair recount per
    merge is the same O(merges x bytes) shape and stays for a later slice.
  - `test_the_rank_ordered_encoder_is_the_pass_encoder`: the two agree on overlapping
    occurrences, on a pair one merge creates for a later one, on multi-byte, CR/LF and NFC text
    and on random text over small alphabets (where merges nest deepest); the tokenizer's own
    `encode` is the rank-ordered one; and a rank table that reverses the merge order disagrees,
    so the differential can fail.
  NDT-KBLOCK (2026-10-12): the blocked fused GEMM contract (v3) and the admitted provider.
  - Found while reading NDT-MEASURE's GEMM gap: OpenBLAS's `sgemm` (scipy_openblas32, the
    SkylakeX target) is bit-identical to BCIR's fused contract on every step shape whose depth
    is one K block, and to a K-blocked form of it on the rest -- its level-3 driver's split (Q
    448, unroll 16: whole blocks while two remain, then half the remainder rounded up to 16,
    then the remainder), each block's fused chain from zero, c = beta*c + P1 + P2 + ... in order
    -- on all 15 shapes, 14 further depths, beta 0 and 1, 1 and 4 threads, alpha 1 (its
    small-matrix path, below about 10^6 multiply-adds, keeps other orders).
  - Declared as a third GEMM contract: `bcir_tensor_kblock_split` (the rule, one predicate on
    both rails), `bcir_tensor_mm_kblock` (`_pool`, `_variant`, `_variants`): per column block
    and K block the block's tile packed once and every row group's partial kept in registers
    across the whole block, then c = c + alpha*P -- the fused instances' widths and fusion; the
    state's `BCIR_DT_CONTRACT_KBLOCK` (4, never with the fused bit), `NativeDecoder(contract=
    "kblock" | "kblock+exp")`. `use_cblas` under it runs the admission differential over the
    trainer's own GEMM shapes at beta 0 and 1 and refuses a provider that differs on any case,
    leaving the state on BCIR's kernels; `provider_admission` records the verdict.
  - Evidence. The C harness's `gemm_kblock_contract`: every instance against the K-blocked
    `fmaf` chains over 64 cases (the v2 shapes and five deeper ones crossing the rule's three
    cases), serially and on a pool of three, equal to v2 exactly on the 8 cases the contract
    says (one block, beta 0, alpha 1) and apart on 56; the rule on ten depths; `pool_identity`
    under all six contracts, each its own state, each identical at 1..4 threads; the undefined
    bit 8 and the pair fused|kblock refused. `train.mm.kblock.mismatch` 0 of 40 against
    `mm_reference_kblock` (exact rationals per term, two binary32 operations per block) with
    the C and Python split rules agreeing on every depth; `train.mm.provider.mismatch` (since PR817-P0 `train.mm.provider.differences`, beside the graded `train.mm.provider.unsound`) 0 of 30
    (OpenBLAS on the comparison decoder's shapes, where the wheel is importable). The admission
    witness: BCIR's blocked kernel behind a CBLAS callback admitted, the v1 kernel refused on
    30 of 30 cases, the fused kernel on the 15 beta-1 cases, an empty differential refused, and
    OpenBLAS refused on the threads bench's small products and admitted on the comparison
    decoder's -- the admission is per shape set. `ndt-kblock.json`: 6 defects, 6 caught (the rule
    without its halving, a chain from c, a fused add, the trainer and the one-thread pool entry
    on the fused kernel, an admission that does not read its differential); NF5 re-anchored.
  - Measured ([measurements/2026-10-12/kblock/](research/measurements/2026-10-12/kblock/)): the
    1,000-step `kblock+exp` run ends in one state (`d7fc618c…`) on BCIR's own kernel and on
    OpenBLAS admitted, at one and at four threads -- validation 1.5001, test 1.6319 on all four;
    655 ms a step with the provider on four threads (753 on BCIR's kernel; §4's `fused+exp` 758),
    1,066 on one (1,425). BCIR's own v3 kernel is faster than its v2 kernel at one thread (the
    step's GEMMs 1,081 to 952 ms: the partial in registers across a whole block) and equal at
    four; the provider's rate is now the ceiling the kernel is measured against (NDT-BLOCK).
    The step by kernel under the provider found a fifth of its four-thread step in contention:
    OpenBLAS's workers spin 2^28 cycles after each call, burning three cores while the pool runs
    the attention, the update and the norms between two GEMMs (the attention 57 to 115 ms);
    `OPENBLAS_THREAD_TIMEOUT=4` before the library loads (the harness's `_openblas` sets it)
    puts them to sleep at once -- the 1,000-step run's median 616 ms a step on four threads
    (655 spinning), the step by kernel 527 (632), the same state. The update's passes were
    measured next ([update-passes.json](research/measurements/2026-10-12/update-passes.json)):
    41 of its 53 four-thread ms are the first pass (16 the gradient's index-order chain, 25 the
    weight and moment streams' maxima and finiteness), 12 the pooled commit (33 serial) --
    NDT-NORM's row in the roadmap now carries that expectation (about 34 ms with the three
    streams on the pool, the chain kept) instead of a guess.
  ATTN-FMA (2026-10-18): the attention's products under a fused contract, one `fmaf` a term.
  - The scores, the context and the backward's products (the probability adjoints, their dot sum,
    dq, dk, dv) were v1's arithmetic, a rounded product and a rounded sum a term. `bcir_tensor.h`'s
    `BCIR_TENSOR_ATTENTION_FUSED` (through `bcir_tensor_attention_contract_scan_pool` and
    `bcir_tensor_attention_backward_contract_pool`) makes each multiply-add one IEEE fusedMultiplyAdd
    over the same terms in the same order; the softmax and the score adjoint stay v1's. Portable
    `fmaf`, AVX2+FMA and AVX-512 instances (the v1 register blocks written once as always-inline bodies
    and instantiated twice, the multiply-add a constant of each copy); the trainer's
    `BCIR_DT_CONTRACT_ATTN` (bit 32, any GEMM contract or none, no ABI change), Python `attn` and
    `<contract>+attn` ([measurements/2026-10-18/attnfma/](research/measurements/2026-10-18/attnfma/),
    the report's §33).
  - The C harness holds the entries to `fmaf` reference loops (192 cases, 180 apart from v1), the
    runtime gate's portable and AVX2-capped builds hold those instances to the same loops, the
    thread-state test holds three `+attn` contracts to one state at every thread count, and
    `tools/testing/faults/ndt-attnfma.json` holds 17 faults. Its first sweep caught 14 of 15: a wrong
    contract at the trainer's backward call passed the trainer's probe, which recorded the contract
    the trainer meant to pass, so the probe moved into the tensor unit's jobs (every job since the
    last read must have run one contract) and two faults at the calls joined the table. Eight entries
    of `ndt-fma`, `ndt-attn`, `ndt-fuse`, `ndt-lanes` and `ndt-perf` anchored in the restructured
    blocks were re-anchored and fire again; the GEMM unit's `mm_run`, which only the attention's
    no-panel tiles called, went with the change, and a duplicated `CONTRACTS` line in `native.py`.
  - Measured: one layer at four threads 1.88 -> 1.37 ms (the backward x1.45), at one thread x1.42; in
    place the attention 14.5 -> 11.3 ms of the four-thread step; the step 80.0 -> 77.6 ms under
    `bf16+exp+reduce` (13,199 tokens a second) and 176.8 -> 169.4 under `kblock+exp+reduce`. One
    state per contract through the tile, AVX-512, AVX2 and portable builds at one and four threads;
    the contracts without the bit keep their states (`4bf4e057…`, `e91ae701…`).
  LM-ATTN (2026-10-18): the fused attention over 1,000 steps, five seeds paired.
  - LM-PAIRED's five seeds ran five arms each at four threads in one session, the order rotated: BCIR
    under `bf16+exp+reduce+attn` and `bf16+exp+reduce`, under `kblock+exp+reduce+attn` and
    `kblock+exp+reduce`, and compiled PyTorch under autocast, with LM-BF16's analysis fixed before the
    runs ([measurements/2026-10-18/attnfma/lm/](research/measurements/2026-10-18/attnfma/lm/), the
    report's §34).
  - BCIR's fastest contract trains at 2.015 times compiled PyTorch bf16's tokens a second (95% interval
    1.926-2.107, faster on 5 of 5: 13,157-14,999 tokens a second against 6,592-7,802), its test
    cross-entropy 0.00016 nats below PyTorch's (inside the ±0.01 margin). The attention bit buys 1.032
    times under bf16 (0.993-1.073) and 1.026 in FP32 (0.999-1.053) at -0.00036 and +0.00000 nats. The
    arms without the bit end in LM-BF16's states on all five seeds. Peak RSS 540 MB against
    1,132-1,148.
  BF16-KERN + BF16-PANELS-1 (2026-10-18): the bf16 GEMMs' kernel and packs, every state kept.
  - BF16-ATTR took the bf16 entry apart at the 9.84M decoder's 13 shapes: the packs a quarter of its
    time at four threads, the compute a quarter of the tile unit's peak; oneDNN's bf16 matmul through
    PyTorch takes 7.38 ms from binary32 operands where BCIR's entry takes 3.58 with this commit
    ([measurements/2026-10-18/panels/](research/measurements/2026-10-18/panels/), the report's §35).
  - BF16-KERN: the AMX kernel loads its B tiles as streamed (`TILELOADDT1`) and prefetches the next
    block's C for writing (`PREFETCHW`), the best of seven candidates held bit for bit to the kernel
    before it (the 13 shapes' compute x1.110 at four threads, x1.118 at one).
  - BF16-PANELS-1: the SwiGLU block's outputs -- the hidden activation, the gate and up adjoints -- are
    written by their producers as the bf16 GEMMs' panels (A and B, A and A^T, in the binary32 slot)
    and read pre-packed (`bcir_tensor_mm_bf16_group_panels_pool`, `bcir_tensor_silu_mul_panels_pool`,
    `bcir_tensor_swiglu_backward_panels_pool`), when every GEMM they feed is whole 32-tiles and the
    slots are 64-byte aligned.
  - Measured with three libraries in one process, a step each in turn: the step 99.57 -> 96.36 ->
    92.73 ms at four threads (x1.063, 95% interval 1.049-1.076) and 329.9 -> 300.3 at one (x1.085),
    every round's losses equal; the comparison decoder's states unchanged: `08da09e8…`
    (`bf16+exp+reduce+attn`) through the tile, AVX-512, AVX2 and portable builds, `e91ae701…`
    (`bf16+exp+reduce`) through the tile and AVX-512 builds, `4bf4e057…` (`kblock+exp+reduce`, a
    path the panels do not take) on the default build, each at one thread and four.
  - `ndt-panels.json` holds 34 faults. Its first sweep caught 31: an ulp's slip in a writer hides
    behind the bfloat16 conversion, so the harness gained values built at a conversion midpoint and a
    tensor-unit probe of the values each writer's job computed. Review found two defects before the
    first green run: the trainer's predicate left out the model's width (a model off the 32 grid there
    would have left the down projection unwritten), and an A panel at alpha other than 1 would have
    dropped alpha; both are refused now, each with its case and its fault.
  - The native AArch64 C runtime job then failed the harness's first version, which held the NaN
    elements of the SwiGLU VJP's panels bit for bit against the binary32 entry's values converted:
    which NaN an operation on two NaNs propagates is the compiled code's operand order, and for a NaN
    gate the sigmoid's NaN has the opposite sign, so the writers' copy of a kernel and the entries' can
    hold different NaNs. The harness holds every element bit for bit and a quiet NaN where the
    reference holds one (`same_panel`, as `same_bits_or_nan` holds the GEMMs), naming the first
    difference with its case; `bcir_tensor.h` says the same of the panels (the report's §35). PN34,
    the portable writer storing a NaN as a signaling NaN, is the rule's witness: a comparison that took
    any NaN passes it, and a writer whose NaNs differ only in sign fails the first version and passes
    this one ([measurements/2026-10-18/panels/nan/](research/measurements/2026-10-18/panels/nan/)).
  LM-PANELS (2026-10-18): the bf16 GEMMs' kernel and panels over 1,000 steps, five seeds paired.
  - LM-PAIRED's five seeds ran three arms each at four threads in one session, the order rotated: BCIR
    under `bf16+exp+reduce+attn` on 5804ad5 and on d5bc457 (before BF16-KERN and BF16-PANELS-1), and
    compiled PyTorch under autocast, with LM-BF16's analysis fixed before the runs
    ([measurements/2026-10-18/panels/lm/](research/measurements/2026-10-18/panels/lm/), the report's
    §36).
  - BCIR trains at 2.282 times compiled PyTorch bf16's tokens a second (95% interval 2.061-2.526,
    faster on 5 of 5: 14,654-14,925 tokens a second against 5,853-6,963), its test cross-entropy
    0.00054 nats above PyTorch's (inside the ±0.01 margin). The two changes buy 1.078 times
    (1.054-1.102, 5 of 5), the two BCIR arms ending in one state, LM-ATTN's, on every seed. The session
    ran slower than LM-ATTN's (the same code x1.051, PyTorch x1.105), so LM-ATTN's pair read 2.117
    here against 2.015 there. Peak RSS 534 MB against 1,141-1,146.
  PANELS-WRITER, PACK-ATTR, GEMM-OVERLAP (2026-10-18): the writers' second panel and the step's packs,
    measured and not kept.
  - A strip writer that never reads the A panel back -- 32 rows by a strip of binary32 values in L1,
    the second panel converted from them by the same conversion -- is slower than the shipped writer at
    every strip and store kind (x0.546-0.990), its panels identical; streamed stores for the second
    panel read x0.978-1.061 at four threads and below 1 at one, and the panels' GEMMs follow within the
    step ([measurements/2026-10-18/panels/writer/](research/measurements/2026-10-18/panels/writer/), the
    report's §37).
  - The shipped writer's time split: the second panels are 17% (SiLU) and 31% (VJP) of it at four
    threads, their interleave and transpose rather than the read of the A panel.
  - PACK-ATTR and GEMM-OVERLAP: a probe build timed the packs inside the four-thread step -- 8.15 ms of
    wall against 23.6 ms of tile compute, the weights' B packs 42% of the packs' thread time (about
    3.4 ms) -- and the B pack's loops swapped (its tasks 9% faster, the step x1.019, 0.997-1.036) and the
    packs overlapped with the compute per call (x1.002) and per strip (x1.008) were within the noise:
    not kept ([measurements/2026-10-18/panels/packs/](research/measurements/2026-10-18/panels/packs/)).
  FACTS-RUN + MULTI-STEP (2026-10-18): a run call's later steps trust the call's own commits.
  - A step scanned all 39 MB of weights before its forward, because a caller may have written them.
    Inside one `bcir_decoder_run` call no caller code runs between a step's commit and the next step,
    and a commit stores only admitted candidates (no larger than FLT_MAX, NaN excluded, by the
    preflight or by `adamw_bounded`'s proof), so the steps after a call's first now skip that scan
    (`DT_COMMITTED`); a call's first step, and every lone call, still scans.
  - Witnesses: the C harness's `facts_run` (a run of three steps against three calls of one, state
    and events, four contracts, one and two micro-batches, serial and pooled) with a scan probe
    (`BCIR_DT_SCAN_PROBE`) counting one scan a run and one a call, and
    `test_a_run_scans_its_weights_once` through the Python binding, where a NaN written between two
    calls is the next call's refusal before any step. `ndt-facts.json` FR1-FR3, 3 of 3 caught.
  - Measured at four threads, four arms rotated in one process
    ([measurements/2026-10-18/facts/](research/measurements/2026-10-18/facts/)): the scan's removal
    alone x1.035 (95% interval 1.014-1.057), a ten-step call against ten one-step calls x1.048, the
    calls themselves nothing (x0.998); all four trainers end in one state. A 16-round first screen
    resolved none of the three; the 48-round run is the record.
  - The comparison harness's `train --steps-per-call N` runs a bcir backend N steps a call, never
    across an evaluation, its record and name carrying N (`-kN`): the same states, losses and
    evaluations as one step a call, its own arm.
  BF16-ATTN + ATTN-SOFT (2026-10-19): the attention's products on bfloat16 operands, and the binary32
    softmax's scale folded into its exponential.
  - `BCIR_DT_CONTRACT_ATTNBF16` (bit 64, with a bf16 GEMM contract alone: refused without one and with
    the fused attention) puts the attention's six products -- the scores S = Q K^T, the context Y = P V,
    the backward's dP = dY V^T, dQ = dS K, dV and dK -- on bfloat16 operands, each element the bf16 GEMM
    contract's over the depth tiles the causal mask reaches from its 32-row block: every channel tile for
    S and dP (their blocks past the diagonal never computed), a query block's key tiles 0 .. its own for
    Y and dQ, a key block's query tiles from its own on for dV and dK, computed transposed (dV^T = dY^T P,
    dK^T = Q^T dS: a product of two bfloat16 values is exact either way round, so the same terms reach the
    same lanes). The scaling, the softmax (v1, or the BCIR exponential under EXP), the dot and the score
    adjoint stay v1's binary32; the future scores and adjoints are +0.0; a KV head's dk and dv take its
    query heads in order, the first from +0.0. The entries `bcir_tensor_attention_bf16_scan_pool` and
    `..._backward_pool` run one task per (batch, KV head) -- its keys and values packed once for its
    query heads, a query block's scores, softmax and probability strip (its dP, dot, score adjoint and
    dS strips) done while its rows are in the core's cache -- in regions of the trainer's workspace,
    one a worker where the pool has no more workers than tasks, else one a task
    (`bcir_tensor_attention_bf16_workspace`, pool-independent); the backward reads p and never writes
    it, its dS in the workspace. Portable, AVX-512 (MXCSR FTZ/DAZ) and AMX instances, one result;
    the AMX kernel split into begin/blocks/end so a task configures its tiles once. Python names it
    `<bf16 contract>+attnbf16`; `bcir_decoder_workspace_bytes` sizes the workspace for the attention
    too.
  - ATTN-SOFT: the softmax's row maxima are taken over the raw scores and scaled once (rounding is
    monotonic and the scale positive, so the maximum's element is the same and its scaled value the
    same float), and the scale is folded into the transposed exponential's argument (`blk*scale - max`,
    the same two roundings): the separate pass that scaled every score is gone, under both the binary32
    and the bf16 attention, every value as it was.
  - Witnesses: the C harness's `attention_bf16_reference` against a reference written apart from the
    unit (each product from `ref_bfv`/`ref_lane`/`ref_flush`, the bf16 GEMM model) -- 912 cases on 13
    shapes (one row and channel to the 330-token backward, head groups of one to four), finite, with an
    infinite value and a NaN adjoint, and with the key the softmax hides (each score of a key -inf, its
    probabilities 0, every output finite: 288 cases, the verdict refusing it from the key itself), under
    both exponentials, every instance the host admits, serial and on pools of two, four and 64 (a region
    a task); the backward leaving p; every refusal writing nothing. `trainer_attnbf16`: one state at one
    to four threads under three bf16 contracts with the bit, another without it, the attention probe
    recording the bf16 entries for every job, the arena's probabilities as the forward wrote them, the
    workspace at least the GEMMs' and the attention's. The writers' verdict test runs the trainer's
    hidden key, overflowing logits and a hiding SiLU under the bf16 attention too (its key weight
    bfloat16's largest finite value: FLT_MAX is +inf as a bfloat16 operand, and its products with the
    other tokens' zeros NaN). `test_the_bf16_attention_is_a_contract_of_its_own` through the binding (the
    names, the probe, one state per thread count and contract, the workspace the larger of the GEMM
    groups' and the attention's at 40 and 96 tokens, the budget, a resume). Every manifest build of the
    harness passes under GCC 13, clang 18 and clang 23. `ndt-attnbf16.json` AB1-AB36, 36 of 36 caught.
  - Re-anchors: twelve entries of five tables anchored in the restructured code fire again -- AF12,
    AF14, AF16, AF17 (the trainer's attention calls, now `attention()`/`attention_backward()`), NF5 (the
    undefined bit is 128 now), NL7 (the max subtraction is in the exponential's argument), SW5-SW7 (the
    verdict in `att_verdict`, shared), SW12 (two fallbacks now), NT2 and NT11 (one claim loop for both
    task kinds).
  - Measured ([measurements/2026-10-19/attnbf16/](research/measurements/2026-10-19/attnbf16/)): one
    layer's attention x1.280 (1.245-1.299) at four threads and x1.230 (1.197-1.243) at one, every
    round, its backward x1.378; the four-thread step x1.029 (1.018-1.042) against the binary32
    attention on one library and x1.004 (0.989-1.024) at one thread, about the six layers' 2.5 ms
    timed alone, of an 80.5 ms step. ATTN-SOFT (its attention alone x1.018 and x1.017 against the parent's library) and
    the AMX kernel's split read inside what two copies of one library read apart: up to about 3% in
    one process (A/A controls), and over eight processes, the slots swapped every other one, the
    parent's bf16 GEMM over this tree's x0.997 and x1.002 beside the A/A's x0.996 and x1.004. One
    state, `fcd6b43b…`, through the AMX, AVX-512, AVX2 and portable builds at one and four
    threads. The harness built without the x86 instances, as the aarch64 job builds it, passes in
    25.7 s (clang 18) and 27.9 s (GCC 13) of its 60 s cap, 33.2 s and 44.9 s of 300 under UBSan.
  - Screens, measured and not kept ([measurements/2026-10-19/screens/](research/measurements/2026-10-19/screens/)):
    FACTS-PASS -- the first pass reading only the gradient inside a run call, the commit handing the
    next step the weights' and moments' maxima -- has a ceiling of x1.024 (95% interval 0.996-1.041)
    at four threads and x1.037 (1.010-1.063) at one, an A/A pair of two builds of the tree reading
    x0.994 and x1.009: the first pass's w, m1 and m2 pointed at the zero page, every value as it
    was, the first pass already at its streams' bound (NDT-UPDATE); WEIGHT-PANELS -- the update
    writing the weights' bf16 panels -- is bounded by the weights' B packs, about 3.4 ms of the
    step's 8.15 ms of packs (PACK-ATTR), before the update writes two panel layouts (39 MB more a
    step), one of them (W^T as B, the forward's) a layout no writer has.
  LM-MULTI (2026-10-19): the bf16 attention over 1,000 steps, five seeds paired, against PyTorch's bf16.
  - LM-PAIRED's five seeds, four arms a seed in one session, the order rotated: BCIR
    `bf16+exp+reduce+attnbf16` and `bf16+exp+reduce+attn` a hundred steps a run call, the latter one
    step a call too, and compiled PyTorch under autocast; the analysis fixed before the runs
    (`paired_multi.py`), the ±0.01-nat margin on every pair but the one that is one state
    ([measurements/2026-10-19/attnbf16/lm/](research/measurements/2026-10-19/attnbf16/lm/), the
    report's §41).
  - Quality: the bf16 attention's test cross-entropy +0.00006 nats from PyTorch bf16's (95% interval
    -0.00157 to +0.00169) and -0.00117 from the binary32 attention's (-0.00298 to +0.00064), inside
    the margin. Speed: 2.315 times PyTorch bf16's tokens a second (2.194-2.444, five of five); the
    binary32 attention one step a call 2.276 (LM-PANELS's 2.282). The in-process gains are not
    resolved across processes -- the bf16 attention x1.023 (0.991-1.056), a hundred steps a call
    x0.995 (0.958-1.032) -- inside the A/A controls' 3% placement band. A hundred steps a call end in
    one-step calls' states, and those in LM-PANELS's, on every seed; peak RSS 496-534 MB against
    PyTorch's 1,128-1,151.
  BF16-ATTN-COST (2026-10-19): the bf16 attention's witness at a third of its cost, every verdict kept.
  - This PR's three CMake cells reached the job's 25-minute timeout (gcc cancelled at 25.3 min, clang
    23.9 and clang 23 22.9 against main's 20.7, 15.3 and 9.3): the `decoder_train` section ran 76 s
    against main's 17 and its ThreadSanitizer leg 50 against 15, and `build-make` and
    `build-section-parity` run them again. Where the bf16 instances are the portable loops (the
    CI runners, the aarch64 job), `attention_bf16_reference` was 15.0 s of the harness's 21.9.
  - The reference rounds each operand to bfloat16 once, before the products read it, instead of
    at each of the t or d terms it enters, and its lane step returns binary32's rounding for a
    result no less than 2^-126, where the 24-bit rounding with the exponent unbounded is no less
    either: the same function, 10.9 -> 1.4 s. The pools of two and four run on the finite pass
    under libm's exponential, the other passes and the BCIR exponential serially and on the pool
    with more workers than tasks (a task and its region are the shapes' and the pool's, whatever
    the values), and a ThreadSanitizer build runs the finite pass under libm's exponential on every
    pool and instance: the test 15.0 -> 4.1-4.7 s. The AVX2-capped harness 21.9 -> 10.7 s (main
    6.8), the build without x86 dispatch (the aarch64 job's) 17.0 s (27.9 before, main 12.3), the
    four builds of the section 48.1 s here (main 30.6), the clang ThreadSanitizer leg 23.8 (19.6),
    GCC's 158.9 (149.5); every fault-table anchor in the harness unchanged.
  - `cmake-build`'s timeout 25 -> 40 minutes: the job ran 9.3-20.7 minutes on main, and the same
    commit's identical steps ran 1.3 times slower on one runner than another (the compile 75.7
    against 57.2 s), so the gcc cell had no headroom before this PR added a minute.
  MAKE-GATE-TREES (2026-10-19): the make gate refuses an untracked file under a tree it reads.
  - `check_latest.sh`'s cmake leg failed `build-make` over this commit's working tree: the change
    step copies the tracked tree, a target that reads a tree whole is tagged from every file on
    disk under it, and the new records under `docs/` were not yet added, so the train task's tag
    differed in the copy and it re-ran there -- where its corpus law could not ask git for an
    index -- and the step executed 2 targets instead of `bcir_diag.c`'s 119 dependents.
  - `tools/build/make_gate.py`'s `untracked_tree_files` names every file `walk_tree` yields under
    the trees the BCIRfile reads that git does not track and no target writes, and the gate stops
    UNUSABLE before any run when there is one, as it did for an untracked read by name. Witness
    `test_the_gate_names_an_untracked_file_under_a_tree_a_target_reads`; `make-gate.json`
    MG1-MG5 (the read predicate's entry new too), laws.md L12's instance.
  RED-VACUITY (2026-10-19): a fault the compiler rejects is not a fault the test caught.
  - The native trainer's C harness, its library and bcir-qualify build under `-Werror`, so a mutant that
    left a parameter, a variable or a static function unused stopped the build before any test ran;
    `check_tests.py` named the failing test and `red_sweep.py` scored RED. NL7's old mutant was the first
    seen (re-anchoring it after ATTN-SOFT); a scan of all 913 committed faults whose path is a C
    source -- each mutant compiled as its check compiles it, against a baseline of the clean unit under
    the same flags: clang's diagnostics and the object's symbols, then GCC compiling for real (its
    `-fsyntax-only` skips an unused static function) over the 304 whose checks build with `-Werror` --
    found fourteen in nine tables on the committed head: twelve under clang, the checks' first compiler
    (`AF11`, `NE4`, `NZ1`, `PN12`, `PN13`, `PN19`, `PN24`, `PK1`, `NO2`, `NL7`, `NK1`, `QG1`), two under GCC alone (`NA13`, `NZ2`);
    none in the C frontend's tables (the twin builds without warning flags) or behind the C gates (whose
    harness failures are their own `build` finding). Each is re-crafted to compile (the expression
    replaced, a dropped use consumed: `(void)u;`) and swept again: all fourteen are caught by their tests.
  - PK1, compiling, came back NOT CAUGHT: the AMX kernel streaming one B tile into both is withdrawn by
    the instance's admission differential and never runs, so no value could show it. The C harness now
    holds the differential to passing wherever the build and the host offer the instance
    (`bcir_tensor_bf16_amx_offered`, under `BCIR_DT_PANELS_PROBE`), and PK1 is caught there.
  - The instrument: the builds a test runs before it tests anything -- the trainer's library, harness and
    copies, the C frontend's twin, bcir-qualify (`BUILD_FAILED`), the executor's and the ASN.1 codec
    drivers, the planner's and the ring's harnesses -- raise `bcir.toolchain.BuildFailed` (a RuntimeError,
    as `NativeDecoder.build`'s failure always was, its first line the build and the compiler's first
    error; `build_units` runs one); `check_tests.py` reports a test that raised it as the finding `build`,
    the C gates' finding for a harness that did not build; `red_sweep.py` scores a fault whose check did
    not fire beside a `build` finding DOES NOT BUILD, a miss. The committed head's vacuous PK1, NK1 and
    QG1, swept through it, come back DOES NOT BUILD. Witnesses in `test_red_sweep.py`,
    `test_toolchain.py`, `test_native_decoder.py` and `test_c_cfront.py`; `red-vacuity.json` RV1-RV9,
    9 of 9 caught; laws.md L25's RED-VACUITY instance.
  RING-RACE (2026-10-18): the ring harness's late kill made to happen, and its killed peer parked.
  - CI failed `test_ring.c --procs kill-producer bp` once with `killed_at=20000 ... violations=2`,
    every record delivered: the supervisor, descheduled between seeing the producer pass half the
    stream and killing it, killed it only after it had published the rest and detached, and the
    successor's takeover of a detached producer is refused (`BCIR_ERR_STALE`: a takeover needs the
    dead epoch still attached) -- two refusals, the successor's and its exit status. A `kill()` held
    100 ms reproduced it on every run, `bp` and `ow` (the drain-window run 3: its window never opened).
  - The killed peer now parks, alive and attached, three quarters of the way through the stream until
    the SIGKILL lands (`park`): a producer that could finish would detach, and a consumer that drained
    the stream left its successor nothing to take over. A run may declare a kill delay (`late_us`,
    after the window); `STRESS_RUNS` adds a late kill of each peer, which must report its delay and
    its peer parked or it is a violation (L2), so the interleaving happens on every run. Every late run
    kills at 15,000 of 20,000 with no violation; the runs without a delay still kill mid-stream.
  - `ring.json` gains four faults -- a producer and a consumer that never park, a supervisor that
    ignores its delay, a fixture that never passes it -- 29 of 29 caught.
  RING-PARK (2026-10-19): a late kill holds until its peer has parked, not for a fixed time.
  - This PR's aarch64 oracle job (thorough tier, four workers on four cores) failed
    `test_every_g15_row_is_zero_on_both_rails` with `ring.concurrent.violations: 2`, while the same
    commit's push run passed: RING-RACE's late runs held the supervisor 100 ms after the mark and took
    the peer to have parked by then, but a peer reaching the park from the mark moves 15,000 (a
    producer) or 25,000 (a consumer) of the G15 run's 60,000 records through a 32-slot ring, and on a
    loaded host that took longer, so both late kills found their peers short of the park and the L2
    check scored both runs. A peer stopped 300 ms at the mark reproduces it on every run.
  - The supervisor of a late run now waits until the peer reports it has parked (`await_park`,
    bounded: a peer that ends unparked, or has not parked in 30 s, is a refusal), whatever the host's
    speed; `--procs` takes `late` as a flag and an optional `stall_us` after it, and `STRESS_RUNS`
    adds a stall run for each peer -- the killed peer stopped 300 ms at the supervisor's mark, three
    times the old hold -- which must report the whole stall (`stalled_us`) and its peer parked, or it
    is a violation (L2).
  - `ring.json`: the delay fault, whose code is gone, becomes five -- a fixed hold for each peer
    (RING-RACE's 100 ms), a supervisor that never waits, a peer that never stalls, a fixture that
    never passes the stall -- 33 of 33 caught; laws.md L2's RING-PARK instance.
  GEO-3 (2026-10-19): per-layer caching coupled with packing-aware buffer planning -- the slice Codex's
    PR818 geometric-head report names as its next opportunity -- in three parts, every state the
    all-kept trainer's at the exact need, bit for bit ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md),
    the measure note's §42, [measurements/2026-10-19/geo3/](research/measurements/2026-10-19/geo3/)).
  - WS-PLAN: `bcir_decoder_workspace_need` counts what the step's bf16 GEMM groups pack by the
    tensor unit's own rule (`bcir_tensor_mm_bf16_group_panels_workspace`, new), the groups as the
    step issues them -- operands two calls read alike packed once, a SwiGLU panel operand not at
    all -- where the parent summed every call packed whole: 4,522,048 bytes for the 9.84M decoder
    against 11,534,592, 1,933,376 for Codex's attention model at B4/T256 against 2,752,768.
    `bcir_decoder_workspace_panels` is the least at which the SwiGLU block keeps its panels (a call a
    panel feeds runs whole, so `panels()` now asks it), `bcir_decoder_workspace_floor` one 32-row
    strip of A and one 32-column strip of B at the step's deepest call; `valid()` admits any
    workspace from the floor up (below: `BCIR_DT_CAPACITY`), every product returns whether it ran,
    and a refusal fails the step with `BCIR_DT_CAPACITY`, nothing committed -- where the parent
    ignored the GEMMs' returns, safe only while admission demanded the whole conservative sum. The
    bf16 attention's regions are counted for the state's pool (`bcir_tensor_attention_bf16_pool_workspace`:
    one a worker where the pool has no more workers than tasks, else one a task), its entries
    admitting the pool's own. The harness probe now records the need with the panels named: the
    parent's recorded a group without them, so two panel operands -- NULL addresses -- read as one
    shared operand, a figure neither path packs (the 9.84M decoder: 6,553,664 against 4,522,048
    with the panels and 10,747,968 without them).
  - REMAT (state ABI 5): `bcir_decoder_state.recompute`, bit l for layer l (a bit past the last
    layer refused). The kept layers' cache slots come first in layer order and the recomputed ones
    share one slot after them (`bcir_decoder_activation_count`, the arena the state must lend, the
    logits its last floats); the backward re-runs a recomputed layer's forward from its residual
    input up to the hidden activation, the last value its backward reads -- the highest such layer
    free, its cache still in the shared slot -- and every partition being the shapes', the values
    are the forward's: one state under every set, thread count, contract and micro-batch count.
    The 9.84M decoder's arenas go from 351.4 MB (binary32, every layer kept) to 220.3 MB with every
    layer recomputed, five layer forwards a step re-run.
  - MEM-PLAN: `memory_plan` and `NativeDecoder(recompute="auto", max_memory_bytes=...)` return the
    first plan that fits in a declared order, TMSAO-4: the fewest recomputed layers, each with the
    workspace at the need and then at the panels' need, the floor after every set; a set or a size
    outside its C type is refused before ctypes can wrap it (2^64 would reach the C as the empty
    set). Accumulation, which changes the state, stays the caller's. The GEM+ program models the
    sets (`training_step_program(..., recompute)`: the shared slot and the re-run work) and the new
    exact row `train.plan.remat.mismatch` holds its arenas to the C count (0).
  - MEM-PLAN-2: the planner's first order walked every set at the need before the panels' tier, so a
    budget a byte short of the kept plan recomputed two layers where the panels' tier would have
    done. The frontier priced the panels' tier below one re-run on every measured shape, and the
    order follows it; the planner's test now holds the whole walk, the scenario driver's
    `plan_within` the same order.
  - The frontier (ABBA rounds of whole steps; every run the reference's state): a re-run layer frees
    25 MiB of the 9.84M decoder for x1.012 (0.965-1.059) of a step under bf16 with the bf16
    attention at four threads and x1.057 (1.031-1.084) under binary32, and 10.5 MiB of Codex's model
    for x1.090 and x1.087; every layer recomputed frees 125 MiB for x1.090 and x1.188, and 21 MiB for
    x1.214 and x1.221. The workspace at the panels' need frees 1.3-2.1 MiB for no cost the rounds
    resolve (x1.006, x1.032 at one thread, x1.044); at the floor 1.7-4.1 MiB for x1.32-1.75.
  - Codex's attention model at its 32 MiB envelope: inside 32 MiB the planner recomputes every layer and
    trains the B4 state bit for bit -- 29.55, 31.40 and 31.76 MiB under binary32, bf16+attn and
    +attnbf16, for x1.158, x1.180 and x1.208 of an update kept -- where two accumulated B2
    micro-batches (Codex's attention32) cost x1.011, x1.135 and x1.155 and train another state. The
    9.84M decoder in 256 MiB recomputes five layers for x1.131 against accumulation's x1.095. At the
    parent's workspace figure the bf16 plan would need 32.18 MiB: the fit is the two slices'
    coupling.
  - Witnesses: the C harness's `trainer_workspace_plan` (every workspace from the floor up one state
    on two shapes, two contracts, one thread and four, the panels exactly from their groups' need,
    the probe's need the plan's, a byte below the floor refused with nothing written, a refused
    product the step's CAPACITY -- in the forward, and in the backward alone where the whole forward
    runs) and `trainer_recompute` (nine sets, five contracts, one and four threads, one and two
    micro-batches, the floats past the count untouched, the logits last, a bit past the last layer
    and an arena a float short refused); `test_recompute_and_the_workspace_keep_every_state`
    and `test_the_memory_plan_fits_its_budget_and_keeps_the_state`. `geo3.json` GW1-GW12 and RM1-RM6,
    18 of 18 caught; `geo3-py.json` GP1-GP9, 9 of 9 -- the planner's
    first test missed its order fault until it walked a contract with three distinct tiers (under
    the bf16 attention at four workers its regions are both the need and the floor), and GP3, GP6
    and GP9 now hold the frontier's order; GW6, GW8 and GW12 went uncaught until the harness reached
    them (a refusal in the backward alone, a shape whose panel groups are the step's largest), and
    GW5 and GW7 were respelled where the compiler rather than the test refused them (RED-VACUITY: an
    unused variable under -Werror is no catch). The C harness built as the aarch64 CI job builds it
    passes in 17.9-18.3 s against its 60 s cap and 24.7-36.8 s under UBSan against 300 s, 1.7-4.1 s
    more than the parent's built and timed alike.
  - Re-anchors: eighteen entries of eight older tables (`ndt-attnbf16`, `ndt-attnfma`, `ndt-bf16`,
    `ndt-fuse`, `ndt-pages`, `ndt-panels`, `ndt-scanw`, `ndt-threads`) whose anchors GEO-3's edits
    moved were re-anchored to the new code and caught again; the 79 older entries whose injections
    land in the files GEO-3 changed were re-swept on the final tree with `geo3-py.json`: 88 of 88 caught.
  JVM-STDOUT (2026-10-19): the stackify test's JVM runner keeps the VM's own output off the program's stdout.
  - The runner reads stdout whole, and in the quick tier a JDK 21 VM that found its perf-data file
    (`/tmp/hsperfdata_<user>/<pid>`) locked by another process logged `[warning][perf,memops] Cannot use
    file ... because it is locked by another process` to stdout, ahead of the program's `20.0`. The
    runner's VM now keeps no perf-data file (`-XX:-UsePerfData`, taken by every HotSpot) and, where the
    VM has unified logging, logs to stderr (`-Xlog:disable -Xlog:all=warning:stderr`). JDK 8 will not
    start with `-Xlog` (and already writes its VM warnings to stderr), so the runner asks the VM once.
  - Reproduced on purpose -- a shell that holds the file for its own PID, then execs the VM, which keeps
    the PID -- the bare launch's stdout failed the parse, and each flag alone kept it clean. Witnesses:
    VM logging set to stdout ahead of the runner's flags (`JAVA_TOOL_OPTIONS`) and a VM asked to save its
    perf data, each against a bare launch that shows the provocation fires; the probe against stand-in VMs
    that refuse and take `-Xlog`. Checked on JDK 21 and on JDK 8 (Zulu 1.8.0_472). `jvm-stdout.json`
    JS1-JS3, 3 of 3 caught. The first sweep caught the logging witness asking the code it tests whether
    to run at all (JS2 made it skip itself); its own bare launch decides now (laws.md L11).
  BASH-BOUND (2026-10-19): the script-running tests bound bash by the harness's child bound, not a minute.
  - `test_latest_toolchain`'s refusal test failed once on windows-latest with Python 3.15: `check_latest.sh
    --bogus`, refused before any leg runs, hit its 60 s bound -- Git Bash's start-up on a loaded runner.
    The tests that run a tool's shell script (that one, the analyzer leg's, the train rail's refusal
    test) now bound each launch by the harness's child bound (`BCIR_TEST_CHILD_TIMEOUT`, 300 s by
    default) and ask one probe for their shell, `run_all.posix_bash()`, in place of three copies of an
    unbounded `echo ok`. The probe has the same bound, and a probe that runs out raises rather than
    turning into a skip (laws.md L21).
  - Reproduced with a stand-in bash 61 s slow to refuse `--bogus`: the old test failed at 60 s with the
    CI run's `TimeoutExpired`, the new one passes in 61 s. A wedged shell hung the old probe until it was
    killed; it fails the new one in `BCIR_TEST_CHILD_TIMEOUT` seconds. Witness
    `test_posix_bash_is_a_shell_probed_within_the_child_bound` (a WSL launcher's notice, a shell, a wedged
    one); `posix-bash.json` PB1-PB3, 3 of 3 caught.
  PX-1 (2026-10-19): the native step's launches held to the launches its step program states.
  - The gap Codex's Triton comparison names (its 2026-10-19 update, draft PR #821): the native trainer's
    direct C loop issues no StreamPack claim per tensor operation, and nothing showed which decisions reach
    a call. Every decision the trainer's slices made -- a linear's two VJPs as one pool job and the linears
    reading one input as one (NDT-SCHED), the SwiGLU block writing the GEMMs' bf16 panels (BF16-PANELS), the
    recompute set's shared slot and re-run forwards (REMAT), the scans a step's facts leave out (NDT-SCAN,
    FACTS-RUN), the gradients' first writers (NDT-ZERO) -- landed bit-identical, so the state cannot show
    whether the C runs it.
  - `native_launch.run_launches` states every launch one `bcir_decoder_run_accum` call issues -- each call
    to the tensor unit, a provider or the pool, the job it shares, its entry (v1, fused, blocked, bf16 alone;
    a pair's, a group's or a bf16 group's; a provider's), its shapes and scalars, and the (arena, offset,
    bytes) of every operand -- from `training_step_program`'s buffers and the trainer's decisions alone:
    contract, recompute set, workspace and the panels' threshold, the arenas' alignment, a provider,
    steps and micro-batches. The update's preflight runs only when the first pass's maxima fail to bound
    the update, a fact of the values, so it is planned optional. A build defining `BCIR_DT_TRACE_PROBE`
    interposes every dispatch in `bcir_decoder_train.c` and records it, 48 words a record, in words the
    caller lends; a record past them is counted. The library is untouched: without the define the probe
    is three empty macros.
  - `train.plan.executed.mismatch` 221 -> 0: the 221 calls of the trace corpus (the plan corpus under six
    contracts, three recompute sets, one step and two steps of two micro-batches; the panels' bench at the
    workspace's three tiers at one thread and four; a provider's GEMM, and its GEMM and SiLU; an update
    whose bound fails) issue exactly their planned launches, every record kind, every GEMM entry and both
    panel operands reached. The trace build trains the library's state on 27 decision sets (12 distinct
    states), and a trace lent half its words holds the first half and counts the rest.
  - What the trace sees that the state does not: each of `px-trace.json`'s eight C faults -- the linears
    ungrouped (217 of 221 calls leave the plan), a linear's VJPs unpaired (217), the panels never written
    (21), the final norm's 1/rms moved to `dk` (221), the loss rows' terms moved to `tmp` (221), the top
    recomputed layer re-run (144), the deferred gradient scan run anyway (221), an untied model's first
    fill widened to every gradient (73) -- trains the clean library's bits on all 27 decision sets.
    `px-trace.json` PX1-PX8, PT1-PT4 (the probe: drops uncounted, every call its own launch, a recorded add
    run twice, the workspace placed outside the state) and PM1-PM7 (the launch model and the
    comparison): 19 of 19 caught, each by its own check (PT1
    only by the capacity check, PT3 only by the state check). The C harness's C11, C2x and UBSan builds
    compile the probe in, never begun: warning-free under -Wpedantic -Werror on GCC 13, clang 18 and
    clang 23, and the harness's 29 checks hold with it.
  - The ladder: NDT-SCHED's remaining item -- the step reading its plan -- is the training roadmap's PX
    rows: the step as a StreamPack a C interpreter executes (PX-2), the per-call choices carried by the
    plan and one intervention measured (PX-3), the transients placed by lifetime (PX-4, NDT-MEM's row),
    GF(2) layout maps held to the bf16 packers (PX-5), and GPU lowering and synchronization, gated on
    hardware this host lacks (PX-6).
  PX-2a (2026-10-20): the native step run from the launch program its planner states.
  - PX-1 held the launches the C step issues to the ones its program states; the step still ran its own
    fixed sequence. `bcir_decoder_run_program` (bcir_decoder_train.h) runs a program instead: PX-1's
    records -- kind, the job a call shares, flags, eight dimensions, ten operands as (space, byte offset,
    bytes) -- with the host's in place: a CHECK where the step checks its forward, its products and a
    micro-batch's loss, and the update's pass record as UPDATE. Each kernel record runs its call through
    the step's own helpers (the embedding, the loss rows and the scatter became helpers of their own,
    moved from the step unchanged: the 27 state digests are the parent's), so a program's trace is the
    program and its bits are the step's. `native_launch.step_program` states the program for a state
    and the planner's choices -- the linears grouped or not, a linear's VJPs paired or not, the panels
    taken or not, the run's facts used or every scan made, any recompute set the arena holds -- and
    `NativeDecoder.run_program` runs it (`NativeRunStopped`: the status and the steps a stopped run
    completed): an `array('Q')` is lent to the C as it is, and a list's words are checked by one set of
    their types and converted in C (`array('Q')`, its range check), 1.87 ms a call of ten steps where the
    first binding's Python loop and ctypes construction took 12.8 ms. The state ABI stays 5.
  - The whole program is checked before anything runs, and a refused one writes nothing, not even the
    completed count: each record's grammar and canonical form; each operand inside a space its use
    admits (kernels read the weights, gradients, activations and scratch and write the last three),
    4-byte aligned, of the size its dimensions name and apart from every operand its job writes (one
    GEMM chain's shared C excepted); each GEMM job one the state's kernels take (the contract's entry, a
    pair's two roles, a group's chain and the rest, a panel only on whole 32-tiles and one call's, the
    bf16 workspace whole), alpha 1 and beta 0 or 1, the state's eps and attention flags; a provider's
    call only with the provider; the weights read only once a scan of all of them or an update has made
    them known finite (FACTS-RUN); one loss a micro-batch, an update after exactly `accumulation` of
    them, `steps` updates; the program outside the state, the events and the count. The check holds each
    call to its kernel's contract and the run to its step's shape; that the operands' places compute the
    step's gradient is the program's, which the witness holds.
  - `train.plan.program.mismatch` 356 -> 0: of PX-1's trace corpus, each of the 221 calls run from the
    program that takes the C step's own decisions, and its 27 state cases under each of five other
    choices (the linears ungrouped, a linear's VJPs unpaired, no panels, every scan made, every layer
    recomputed): every run trains the C step's state and events bit for bit, runs to its end and traces
    exactly its program. Each choice changes some program and states what it names; the programs reach
    every record kind but the trace's own three.
  - 48 malformed programs are refused with no byte written: 37 edits of a v1 program of two steps of two
    micro-batches, one at each rule of the check and the bound checks at their edges, a program short of
    its last record, a valid program lying inside the state's second moment, three programs for another
    state (another state's kernels, BCIR's SiLU under a provider's, a cache the arena cannot hold), five
    of the bf16 contract's rules and a panel on partial tiles. A run that fails stops where the step
    stops, every byte as the step leaves it: weights not finite (NUMERIC, nothing written), a forward
    that overflows on finite weights (NUMERIC at its check, before a gradient is written), an update the
    preflight refuses (NUMERIC after a step), a product refusing its workspace (CAPACITY before any
    commit, ahead of the gradient's verdict when both fail).
  - A new trust boundary, fuzzed: `fuzz_decoder_train.c` runs any program on a two-layer state under
    eight contracts, with and without recompute, and asserts no access out of bounds, that a refused
    program wrote nothing and that a run is a function of its input; `native_decoder_fixtures.fuzz_seeds`
    seeds it with the planner's programs and near misses of the check (a seed refused or a miss admitted
    fails `test_the_fuzz_seeds_start_inside_the_grammar`). 20,000 runs under clang 18 and 100,000 under
    clang 23, with ASan and UBSan, no finding.
  - The runner costs nothing this host resolves ([measurements/2026-10-20/px2/](research/measurements/2026-10-20/px2/README.json)):
    four call paths rotated on one trainer of the 9.84M decoder -- every path trains the same bits, so no
    ratio compares two placements of a state -- read the program x1.0035 of the fixed sequence (95%
    interval 0.977-1.030) under bf16 at four threads, x1.0070 (0.991-1.023) under binary32 and x1.0170
    (0.984-1.051) at one thread, each interval the width of the A/A control's beside it. The check walks a
    step's 223 records in about 12 us; every run ends in the state of a trainer that ran each of its calls
    through `bcir_decoder_run_accum`, bit for bit, over 1,000 steps.
  - `px-program.json` PG1-PG25 (the check's rules, the run's verdicts, shares and counts, a program inside
    the state) and PM8-PM19 (the program model, the binding, the fuzz seeds, the header's constants):
    37 of 37 caught. The first sweep caught two spellings the compiler judged before the tests did
    (RED-VACUITY): PG12 left a parameter unused, so the build refused it, and PG16 left a counter set but
    unread in the library build. Both were respelled, and every C entry of the table compiles warning-free
    under GCC 13, clang 18 and clang 23, with the trace probe and without. PX-1's PX5, PM1 and PM3
    re-anchored to the moved code (the loss rows as a helper, the launch model's job builder), and
    `px-trace.json` re-swept: 19 of 19 caught.
  - The ladder: PX-2b carries the same program as BCIR's own artifact -- a StreamPack of claims over RIDs,
    one a launch, their places an `ExecutionPlanV1`'s, executed through `bcir_sp_execute` -- whose
    lifetimes PX-4 then packs.
  PX-2b (2026-10-20): the native step run from a StreamPack and its ExecutionPlanV1.
  - PX-2a ran the step from a launch program the C reads as words. Here the same program reaches the
    trainer as BCIR's own artifacts and through them alone. `native_pack.training_pack` states it as a
    `Module`: one claim per kernel call, one phase per launch, each phase after the one before, so GEM's
    dispatch order (phases in order, a phase's claims by id) is the program's record order. Its
    resources are the program's values: each parameter tensor of the four arenas, the bf16 workspace
    and the micro-batch's tokens and targets live through the whole program; a write of a span of the
    activations or the scratch starts a new value and ends every value of a span it meets, and a read
    takes the span's current value -- a read of a span no live value holds is refused by the builder, so
    a program that reads bytes another value left cannot be stated. The C step's reuse of its buffers
    (the scratch every layer, the RMS VJP's 1/rms in the head of dq, the loss rows' terms in the head of
    cur, the slot every recomputed layer shares) is then many values on the same bytes at disjoint
    times. `pack_artifacts` plans the module with K_BCIR, hydrates it with GEM into a StreamPack and
    mints its ExecutionPlanV1, whose lifetimes place every value (its bank, offset, size and phases).
  - `bcir_decoder_lower_pack` (bcir_decoder_pack.h, a new hosted unit) takes the two artifacts and the
    state and nothing else. It verifies them with their own verifiers -- `bcir_ep_check_pack`: the
    pack's provenance and range laws, the plan's wire laws (the alias law among them: no two lifetimes
    of one bank live at once on overlapping bytes), the pack the plan's lowering, one generation vector
    -- then lowers each claim `bcir_sp_execute` dispatches, in its order, to the record that runs it. A
    claim's name comes from a closed vocabulary (25 names and the GEMM grammar
    `train.gemm.<alone|pair|chain|rest>.<nt|nn|tn>[.acc][.pa|.pb]`); every dimension and flag is
    derived from the state and the operands' sizes; every operand's place is its value's lifetime, which
    must be live at the claim's phase (the alias law's half-open interval) -- with the alias law, no
    claim can write a byte another value still holds. Every claim is lowered and checked before any
    record is written, so a refused pack writes nothing; a refusal names the verifiers' status or one of
    ten reasons of its own. `NativeDecoder.lower_pack` returns the array('Q') `run_program` runs as it
    is (`PackRefused` on a refusal), and `run_pack` is the two.
  - `train.plan.pack.mismatch` 356 -> 0: the 356 program runs of PX-2a, each from its pack, lower to
    exactly their program's words and train the C step's state and events bit for bit. A plan that
    places every activation and scratch value elsewhere (the buffers in reverse order, a placement the C
    step never uses) lowers to the same program at the new places and trains the same bits: the
    operands' places are the plan's. The module verifies under every R-law and its plan under the plan's
    laws against the module, the target and the pack.
  - Twenty-three classes of pack or plan are refused at their reason with nothing written -- a name outside
    the vocabulary or out of its grammar's order, a value no lifetime places, a claim before or past its
    value's lifetime, two live values on one span (the alias law), a bank the state has not, the tokens',
    the targets', a plain operand's or the bf16 workspace's value in another bank, operands too many,
    too few or an in-place one split, a run of tensors not contiguous, a size no shape divides, a
    block's count or base not the claim's, a claim id out of range, the plan of another pack or of
    another generation, lifetimes in the schedule's ticks, a corrupt pack -- as are a v1 pack on a bf16
    trainer, a plan header declaring more lifetimes than its bytes hold (before any figure is lent) and
    short buffers (CAPACITY).
  - A new trust boundary, fuzzed: `fuzz_decoder_pack.c` lowers any pack and plan on the training
    harness's state under eight contracts and asserts no access out of bounds, that a refused pack
    wrote nothing, that the lowering is a function of its input and that what it lowers to runs under
    the runner's contract; each input is driven raw and resealed (both CRCs recomputed), so mutations
    reach the laws behind the CRCs. `fuzz_pack_seeds` seeds it with the planner's packs and near misses
    of the lowering's refusals (each seed held to its verdict and run at its shape byte's shape).
    20,000 runs under clang 18 and 100,000 under clang 23 with ASan and UBSan, no finding -- on the
    final lowering (the first campaign, before the one walk below, the same: no finding).
  - The lowering's cost led to its verifiers. The plan walk's claim check rescanned the steps before
    every step (quadratic in them: 17 ms of a ten-step training plan's 2,231 steps on this host); it
    now rescans only a claim at or below the greatest so far -- QUAL-LIN's rule for a pack's
    segments, on the plan -- so GEM's ascending plans walk it in linear time, every verdict the
    same: each order of a plan's steps with a claim stepped twice at, below or right after the
    greatest is refused on the C twin as the codec refuses it, and the walk's cost grows with the
    steps, not their square (`test_execution_plan`'s two new witnesses; `test_execution_plan.c
    --time`). And the lowering walked the plan's laws twice, in `bcir_ep_check_pack` and again to
    read the lifetimes: `bcir_ep_check_pack_lifetimes` hands the lifetimes over as its one walk
    verifies them, its status `bcir_ep_check_pack`'s on every input, and a status refuses ahead of
    any reason a lifetime gives the lowering (a plan breaking a law and naming a bank the state
    has not is refused at the law). Together, a ten-step pack's lowering 121 -> 48 ms on this host.
    The lowering against the steps a pack carries ([lower.json](research/measurements/2026-10-20/px2b/lower.json)):
    4.7 ms for one step, 49.6 for ten, 143.8 for twenty, of which the plan's laws are 23.8 and 79.8 --
    the alias law, the one part left quadratic (each lifetime against every earlier one, heap-free).
  - The lowering's cost ([measurements/2026-10-20/px2b/](research/measurements/2026-10-20/px2b/README.json)):
    no cost this host resolves for lowering a pack inside every call -- pack / program x1.0195 (95%
    interval 0.997-1.042) under bf16 at four threads and x1.0037 (0.972-1.037) under binary32, beside
    A/A controls of x0.9979 and x0.9923; a ten-step lowering 47.3 and 54.5 ms (2.1% and 1.3% of a call
    here), nothing per call once lowered and lent; the artifacts made in 0.34-0.37 s; every run the
    reference trainer's state over 75 calls of ten steps. The container restarted at about 03:13 onto a
    2.8 GHz Xeon with AVX-512F and neither AMX nor AVX512-BF16, so the bf16 step here (413 ms) is not
    PX-2a's.
  - `ep-walk.json` EW1-EW6 (the claim check's tie, rescan and linear cost; the one walk's handover;
    the lowering's status and its precedence): 6 of 6 caught, each by its own check.
  - `px-pack.json` PK1-PK33, PM1-PM13, PN1-PN3 and PF1-PF5: 54 of 54 caught, each by its own check
    (the lowering's refusals and records, the module's values and names, the binding, the comparison and
    the fuzz seeds); after the one walk, the ten entries in the lowering's moved lines (PK9, PK14,
    PK16-PK18, PK29-PK33) re-swept: 10 of 10 caught. Every C entry of both tables compiles warning-free under
    GCC 13, clang 18 and clang 23, at -fsyntax-only and at -O3. `px-program.json`'s PM16 re-anchored
    (the pack seeds repeat its line) and re-swept: 1 of 1.
  - The ladder: PX-3 takes the pack's per-call choices from the measured candidates, and PX-4 packs
    the lifetimes the plan now carries.
  PX-4 (2026-10-20): the native step's transient values placed by their lifetimes.
  - PX-2b ran the step from a StreamPack whose ExecutionPlanV1 placed every value where the C step
    keeps it: one span a buffer, the C plan's, 999,424 bytes of activations and scratch at the bench
    shape against G26's 753,664-byte concurrent-live bound on the step program's values
    (`train.memory.transient` 1.326, NDT-MEM's headroom since 2026-10-11). The pack already states
    every value with its lifetime, so PX-4 places them by it.
  - `native_pack.place_by_lifetime` lays every activation and scratch value of a pack out in one bank
    (the activations) with GEM+'s static memory planner, each value its own bytes, 64-byte aligned,
    live from its first phase through its last. The values split into parts where none is live across
    a phase -- in phase order, a part ends where the next value starts at or past the furthest end so
    far -- each part laid out by `exact_layout` (the incumbent portfolio, then the bounded branch and
    bound), and the parts share addresses, since no two of their values are ever live together. The
    extent is the largest part's; the bound is G26's alignment-aware concurrent-live bound over the
    same values. At the bench shape -- under each contract family, several steps of several
    micro-batches, a recompute set and the panels' bench alike -- the extent is the bound: a proved
    optimum (TMSAO-1). Across the trace corpus's 356 placed runs the extent is the bound in 254,
    each proved optimal, and 32 bytes (66 runs) or 52 bytes (36) above it in the other 102, where
    the exact search stops at its budget without a proof: each gap is read beside the row
    (`place_mismatch`'s fourth value), never counted in it.
  - At the bench shape that is 754,176 bytes, 245,248 (24.5%) under the C plan's 999,424, and
    `train.memory.transient` reads 1.0. The bound sits 512 bytes above the step program's: the pack
    makes the loss rows' terms (an f64 a token), which the C keeps at the head of the scratch, a value
    of their own, as it does the norms' 1/rms the C keeps in `dq`.
  - `bcir_decoder_run_program` (and through it a pack's run) takes the activation and scratch arenas
    at the lengths the caller lent -- their capacities, any length -- and holds every operand inside
    them (PX-2a's check, by those lengths) and the program, the tokens, the targets, the events and
    the completed count apart from them. The fixed step still needs the plan's spans and refuses a
    shorter arena (BCIR_DT_CAPACITY). No ABI change: the capacities were the state's already.
    `NativeDecoder(transient=(activation floats, scratch floats))` lends arenas of those lengths -- a
    placement's `floats`, and none for the scratch -- and counts them in the memory it reports; it
    refuses them with `recompute="auto"`, since a placement is one program's. The update a program's
    UPDATE record runs checks the state as the run's own admission did (`update_with(..., placed)`):
    it read the fixed step's spans at first, so every placed trainer was refused at its first update
    (BCIR_DT_CAPACITY) -- the corpus witness's first run found it, every pure check having passed.
  - `native_pack.placed_words` states the words the lowering of a placed pack must write, each
    record's operand slots bound to their values (`TrainingPack.bindings`): an oracle of the C's
    relocation that never reads the C.
  - `train.plan.place.mismatch` 356 -> 0 (a new GEM+ row): the 356 pack runs of PX-2b, each placed by
    lifetime and run on a trainer lending exactly the placement's arena, lower to their programs'
    words at the placed places, place no two live values in one byte (none below its bound) and train
    the C step's state and events bit for bit; each layout's gap to the bound is read beside the row,
    never counted in it. A placed trainer refuses the fixed step, an arena one float short and a value
    moved past the arena's end, each with nothing written; one lending more than the placement trains
    the same bits; a run whose tokens lie inside the lent arenas -- past the placement's end, bytes no
    value is placed in -- is refused.
  - Fuzz: both trainer harnesses (`fuzz_decoder_train.c`, `fuzz_decoder_pack.c`) lend the transient
    arenas at what a program reaches, or one float short, each as the tail of its own allocation, so
    an overrun lands past it where ASan sees it; seeded with the placed programs and packs of four
    configurations and a short one. A placement holds the scratch values in the activation arena, so
    it can reach past the plan's activations: each harness allocates room for the plan's whole
    transient span and lends a run that is not placed exactly what the trainer lends (the recompute
    set's activations). A replay build of each (BCIR_FUZZ_REPLAY: a driver in libFuzzer's place
    records every input's statuses) holds every seed to the verdict its Python twin holds it to
    (`test_the_fuzz_harnesses_hold_their_seeds_to_their_verdicts`) -- the twin and the C had capped
    the placed reach at the plan's activations alike, so two configurations' placed seeds started
    outside the run, and only the C's own replay sees such a drift. Fuzzed under ASan and UBSan --
    20,000 runs of the program harness and 20,000 of the pack harness under clang 18, 100,000 and
    100,000 under clang 23 -- no finding.
  - `px-place.json` (PL2-PL6, PA1-PA7, PD1-PD4, PC1, PC2, PC4), `px-place-corpus.json` (PL1, PW1,
    PW2, PC3: the faults only the corpus witness catches, swept apart so the others never pay for
    its 356 runs) and `px-place-fuzz.json` (PH1-PH3, PC5: the harnesses' room and lending, held by
    the replay): 19 of 19, 4 of 4 and 4 of 4 caught, each by its own check. The first sweeps let
    four through, each past a witness that never reached it, now reached: PL4 (the fixed step's
    admission taking a lent arena) behind the run's own capacity check -- a forward, which only that
    admission guards, now runs on an arena one float short of the plan's spans; PA2 (a part ended
    while a value is live) and PA6 (the alignment ignored) on a stated set with no value written at
    the phase its predecessor is last read and no size off the alignment -- the set now has both;
    PH2 (a run that is not placed lent the whole room) with no seed between the recompute set's
    arena and the room -- the kept layers' program now runs on the recompute set's trainer. The
    px-place and corpus tables were swept again in a worktree of their own, their first run having
    shared its tree with another sweep. PX-2a's PG25 and PG18 re-anchored (the program's admission
    and update lines moved), and `px-pack.json`'s PF2 and PF5 on the shared helpers they now reach
    (one comparison for the pack and the placed runs, one training pack for both harnesses' seeds).
    With them, the 47 older entries inside the constructs PX-4 changed, or held by the witnesses it
    changed, re-swept: 47 of 47 caught.
  - The cost ([measurements/2026-10-20/px4/](research/measurements/2026-10-20/px4/README.json)): no
    cost and no gain this host resolves for running on the placed arena -- placed / C spans x1.0022
    (95% interval 0.987-1.018) under bf16 at four threads and x1.0086 (0.981-1.037) under binary32,
    beside A/A controls of x1.0107 (0.998-1.023) and x1.0193 (0.991-1.048); the ten-step pack placed
    in 123-166 ms (the median of 20 calls), its 1,640 (bf16) and 1,460 (binary32) transient values
    in 10 parts each, one a step, at 166,199,296 and 176,685,056 bytes against the C plan's
    194,019,328 (0.857 and 0.911 of it), each at its bound, proved; the trainer's whole memory 355.9
    -> 328.1 MB and 351.4 -> 334.1 MB; every trainer's state one state, bit for bit, over 24 rounds
    of 10-step calls. The placed arena against the C plan's at one step a call, by recompute set:
    binary32 0.911 (none), 0.878 (layers 0-2), 0.725 (all six); bf16 0.857 (none), 0.833 (layers
    0-2), 0.725 (all six) -- each at its bound, proved.
  - The ladder: PX-5 states the layouts the kernels keep, and PX-3 takes the pack's choices from
    measured candidates.
  PX-5 (2026-10-20): the trainer's kernel layouts stated as GF(2) maps and held to their packers.
  - The gap (Codex's Triton comparison, `docs/research/BCIR_TRITON_COMPARATIVE_ANALYSIS.md` §3a,
    item 1): the native trainer's packers keep their layouts as index arithmetic -- the AMX tile and
    panel order of the bf16 GEMMs, the VNNI pair interleave, the AVX-512 blocked GEMM's B tile and
    its 16 x 16 register transpose -- and nothing stated them, so a second spelling of the same
    arithmetic was the only check. Triton's linear layouts treat a layout as a linear map over GF(2)
    on the bits of the indices; PX-5 adopts that algebra beside the integer-affine maps
    (`kbcir/layout.py`'s G3 pivot is unchanged) and states the trainer's layouts in it, before any
    layout is generated.
  - `bcir.kbcir.linear_layout` (dependency-free): `LinearLayout`, a map from named input dimensions
    to named output dimensions, each a power of two, given by the image of every input bit --
    `apply` is the XOR of the set bits' images; `compose`; one Gaussian elimination over GF(2)
    (`rank`, `injective`, `surjective`, `kernel` -- the inputs a map sends to zero, a broadcast's --
    and `preimage`: one input and the kernel's span); `inverse`, `convert_to` (B^-1 . A, one storage
    to another), `product`, `rename`, `reorder_outs`; `from_function` reads a map from its images of
    the unit inputs and checks it on every input, refusing one that is not linear. `TiledLayout`
    puts a GF(2) tile under an integer mixed-radix grid (radices need not be powers of two: three
    16-lane vectors make a 48-wide block) with a shape, so a tail past the operand is padding the
    statement names. Every rule refuses what breaks it (ValueError): a value missing, extra, outside
    its dimension or not an integer is never packed into a neighbour's bits.
  - `bcir.hosted.models.native_layouts` states, in that algebra: the bf16 A tile (16 rows x 32
    depths, the AMX tile register's 64-byte rows) and B tile (the VNNI interleave -- offset bit 0
    the depth's low bit, bits 1-4 the column, 5-8 the depth's other four), the packs and panels as
    those tiles under the C's grids (a 16-row block's depth tiles in order, rows and depth padded to
    32), the conversion a panel writer makes from its A panel to its second panel, the AVX-512
    blocked GEMM's B tile (a depth row of three 16-lane vectors, the columns past the block zero),
    the register transpose and each of its four stages (unpack, shuffle, two 128-bit lane exchanges
    -- each a 3-cycle of bits; composed, the transpose), and the A broadcast (`_mm512_set1_ps`: the
    kernel is all four lane bits -- a statement and its kernel, not held to a packer: nothing packs
    it).
  - Held to the C bit for bit (`test_native_layouts`), on operands whose elements carry their
    coordinates, in a build of the tensor unit defining BCIR_DT_LAYOUT_PROBE (one build a process):
    the bf16 packs of every instance the host has (the portable C and AVX-512 here; AMX where
    admitted) on ragged shapes, A and B as stored and transposed -- every element at its offset,
    every padding offset +0.0, every instance the portable C's bytes; the panel writers by every
    instance (the new probe `bcir_tensor_probe_panels`: the library's entries run only the host's
    widest) and by the library's entries -- every element where the statement puts it, each second
    panel the first converted, every instance the same bytes; the blocked GEMM's B tile at every
    block width from 1 to 48 and both orientations, the columns past the block zero; the register
    transpose. The algebra's laws on seeded random maps against every input (`test_linear_layout`:
    inverse, composition, the kernel and every preimage against the map's own table, conversions,
    the linearity refusal, the tiled round trip and its mask, and each refusal).
  - What is not observable is not claimed: the register blocks' accumulators never leave the kernel,
    so their layout is not held; the AMX instance does not run on this host (no AMX), so its packs
    and panels are held where a host admits it.
  - Fault sweeps: `px-layout.json` (the algebra and the statements its pure witnesses hold): 31 of
    31 defects injected, each caught by its own witness (LA1-LA28, LS1-LS3); `px-layout-c.json` (the
    C packers, the panel writers, the B tile and the transpose, and the statements only the C
    witnesses hold): 21 of 21 caught (LS4-LS11, LC1-LC13). Every C entry compiles warning-free under
    GCC 13, clang 18 and clang 23, at `-fsyntax-only` and `-O3`, with the layout probe and without
    (RED-VACUITY): 13 C faults and the clean unit, 16 builds each (GCC 13, clang 18 and clang 23 at
    C11, clang 23 at C23; -fsyntax-only and -O3; the probe defined and not), 0 compiler findings.
  - The ladder: PX-3 takes the pack's choices from measured candidates; PX-6 (GPU) is a documented
    skip on this host.
  PX-3 (2026-10-20): the step program's per-site choices taken from measured candidates.
  - The ladder's row: the pack's per-call choices taken from measured candidates, and one
    intervention measured end to end at the same state. Every choice the planner made until now was
    one boolean over the whole step (NDT-SCHED-0/1's grouping and pairing, BF16-PANELS' panels),
    taken the same way at every layer, and none had been measured where it runs.
  - Choice sites (`native_launch.choice_sites`), seven a layer and one for the head: each layer's
    q/k/v and gate/up projections (one pool job, or a call each), its SwiGLU block's bf16 panels (or
    the block plain), and the VJPs of its down, o, gate/up and q/k/v projections (one job, a job a
    projection, or a call each); the head's VJP. `step_program(sites=...)` takes any site's
    alternative in place of the booleans' (`default_sites`: the booleans' program is the one its
    default sites state, word for word) and refuses a site or an alternative the program does not
    have; `step_program(labels=...)` names the site each launch belongs to -- the site's GEMMs, and
    the SwiGLU kernels for its panels site. Every alternative trains the same bits; what differs is
    the time.
  - The job clock: a build defining BCIR_DT_TIME_PROBE (`NativeDecoder.build(probe="time")`, a
    library of its own; the library a caller runs has no clock) adds the nanoseconds from each job's
    dispatch to its return into the word of its first record's index, in words the caller lends --
    ISO C11 `timespec_get`; a job past the lent words is counted; lent none, the clock writes and
    counts nothing -- so a measurement reads every site's launches in place, on the state it trains.
  - `bcir.hosted.models.native_choose`: `measure` runs the trainer's default program and, for each
    kind of site and each alternative that states a program of its own, the program with every site
    of that kind switched, in rotated rounds on one trainer (they train the same bits, so they take
    turns on one state), reading the launches the kind's scope labels -- the panels' scope is its
    block and the two VJPs that read its panels. `estimate` turns the paired rounds into the
    alternative's time over the default's: the geometric mean and its 95% Student-t interval, in Q8.
    The result is a `ChoiceTable`: frozen, generation-tagged, one canonical JSON spelling, naming
    the shape it was measured at (the decoder, the call, the threads, the recompute set, the
    workspace and the contract) and the host's attestation (information, never a verdict). `choose`
    takes, per kind, the alternative whose whole interval lies below 256 with the lowest mean (the
    narrower interval on a tie); every other site keeps its default; no table is the defaults, and a
    table measured at another shape is refused. The measured prices rank; they never decide
    legality: the chosen program is stated as a pack (`training_pack(sites=...)`), planned by K_BCIR
    and verified under the plan's laws (R9 among them) like any other, and the C lowers it to the
    program it states.
  - `train.plan.choose.mismatch` 351 -> 0 (a new GEM+ row): the trace corpus's 27 state cases, each
    under every kind switched alone to each alternative the C step does not take and three seeded
    draws of an alternative a site, run from their programs on the trace build, trace exactly the
    chosen programs and train the C step's state and events bit for bit; 283 of the 351 run a
    program other than the C step's own (the rest are choices a contract or a shape leaves one way:
    a provider's GEMMs, the panels outside bf16). Every alternative and seeded mixes, on a bf16 case
    whose shapes take the panels and on a kblock case, also train the C step's bits from their packs
    -- planned by K_BCIR, hydrated by GEM, lowered by the C to exactly the program's words.
  - The measurement and the intervention
    ([measurements/2026-10-20/px3/](research/measurements/2026-10-20/px3/README.json)): on the 9.84M
    comparison decoder at four threads (this host: AVX-512F, neither AMX nor AVX512-BF16), 24
    rotated rounds of four-step calls on one trainer of the clocked library -- 458 s under bf16, 235
    s under binary32. Under bf16 every one of the ten alternatives reads slower at its sites than
    the step's default, each interval wholly above 1.0: a call each for the q/k/v and gate/up
    projections x1.094 and x1.047, the SwiGLU block plain x1.074, a call each for the down, o,
    gate/up, q/k/v and head VJPs x1.055, x1.039, x1.078, x1.156 and x1.062, a job a projection for
    the gate/up and q/k/v VJPs x1.035 and x1.102: NDT-SCHED-0/1's grouping and pairing and
    BF16-PANELS' panels are the measured winners at every site. Under binary32 none of the nine
    reads either way (x0.965 to x1.070, every interval holding 1.0). So `choose` keeps the defaults
    under both. End to end, in 32 rounds of four trainers of the library without the clock, the
    program the measurement prices slowest (every kind at its slowest alternative: 893 launches a
    four-step call against the default's 577) runs x1.0641 (95% interval 1.043-1.086) of the default
    under bf16, slower in 29 of 32 rounds, where its readings predicted x1.096 for the jobs it
    switches; under binary32 it reads x0.9995 (0.975-1.025), its readings having predicted x1.017.
    The chosen program is the default's, so its arm is a third trainer running the default program:
    x0.983 and x0.966 against the first, beside the A/A control's x0.983 and x0.990. Trainers
    running one program differ by up to 3.4% on this host, an offset the rounds' interval does not
    hold -- which is why the choice rests on one trainer's readings in place; the slowest program's
    x1.064 under bf16 lies beyond it (x1.083 against each of the other two). Every trainer's state
    is one state, bit for bit.
  - Fault sweeps, on the tree with PX-4 and PX-5 beneath: `px-choose.json` (the sites, the labels
    and scopes, the estimate, the table, the choice and the frozen row's count: PS1-PS14, PC1-PC18,
    PR3), 33 of 33 defects injected, each caught by its own witness; `px-choose-c.json` (the job
    clock, the clocked build, the measurement, the decisions and shape read off the trainer, the
    corpus row's runs and an alternative training other bits: PT1-PT4, PB1-PB3, PM1-PM6, PR1, PR2,
    PI1), 16 of 16. The two older entries whose lines PX-3 rewrote (`px-program.json` PM13,
    `px-trace.json` PM1) were re-anchored to the new lines, and the 17 older entries inside the
    constructs PX-3 changed (`px-program.json` PG16-PG18, PG20, PG23, PG24, PM9, PM10, PM12, PM13;
    `px-trace.json` PM1, PM2, PM4, PM6, PM7; `px-place.json` PL6; `red-vacuity.json` RV6) re-swept
    on the combined tree: 17 of 17 caught. Every C entry compiles warning-free under GCC 13, clang
    18 and clang 23, at `-fsyntax-only` and `-O3`, plain, with the clock and with the trace probe
    (RED-VACUITY): the 4 C faults and the clean unit, 24 builds each, 0 compiler findings. The C
    harness's C11, C2x and UBSan builds compile the clock in, never begun -- warning-free under
    -Wpedantic -Werror on GCC 13, clang 18 and clang 23, the harness's 29 checks holding with it --
    so the harness's own entries keep their verdicts: the clock never runs there.
  - Not claimed, and open (PX-3b): the kernel instance and the pool's thread split as choice sites
    (each call's instance is the host's widest; the split the pool's), and the measured prices
    carried into K_BCIR's own candidates on its three rails (a BKPI table beside the target's
    constants) instead of choosing beside them. The measuring host is this container's (no AMX, no
    AVX512-BF16): a table is the host's that measured it, and names it.
  - The ladder: PX-6 (GPU lowering and synchronization) is a documented skip on this host.
  PX-3b-1 (2026-10-21): the kernel instance and the pool's thread split as choices of the step program.
  - PX-3 left both fixed by the host: every GEMM and attention ran its family's widest instance, every
    job every thread of the pool. They are words of the program now -- 41, the instance a job's GEMMs
    or attention products run (one bit of `bcir_tensor.h`'s: 1 portable, 2 AVX2, 4 AVX-512, 8 AMX,
    that the job's family admits; 0 the widest, as before), and 42, the workers it runs on (0 all the
    pool's threads, as before) -- alike in every record of a job. A program of zeros is the program
    before PX-3b, word for word.
  - `bcir_tensor`: `bcir_tensor_pool_cap` (a job's workers: the workers past a cap report every
    generation without claiming a task, a cap of one runs on the caller in order) and the
    `*_variant_pool` entries of the GEMM, its pair and group jobs and the attention and its VJP, each
    returning the instance it ran (0: nothing written). Every instance of a family is the contract's
    bits and a pool's partition is a function of the shapes alone, so every dispatch trains the same
    bits.
  - `bcir_decoder_run_program` holds the words to the host before anything runs -- an instance the
    family lacks, two at once, an instance on a kernel without one or on a provider's GEMM, more
    workers than the pool's threads, workers on a CHECK, a job whose records disagree: each refused
    with nothing written -- and leaves the pool's cap as it found it, however the run ends. The trace
    records the instance a job ran (the entry's own return) and its workers, and the C harness's
    instance probe (BCIR_TENSOR_INSTANCE_PROBE) holds every entry to the kernels that ran.
  - Python: `native_launch.step_program(dispatch=...)` states a dispatch per kind -- a GEMM's role
    (each linear's forward and VJP, the head's), an attention, every other kernel -- and labels each
    launch's kind (`dispatch_labels`); `dispatch_alternatives` names what a host admits,
    `parse_dispatch` the grammar, `program_launches` the defaults as the trace records them.
    `native_choose` measures the dispatch like any site (`dispatch:<kind>`, a table naming the host's
    instance masks: its times are of that host's defaults, so another host's table is refused) and
    `choose_dispatch` takes only alternatives resolved faster.
  - `train.plan.dispatch.mismatch` 1,153 -> 0 (a new GEM+ row): the 27 state cases under every
    dispatch kind switched alone to the portable instance, to its family's widest named and to each
    worker count below the case's threads, and three seeded draws of one a kind, run from their
    programs on the trace build, trace the dispatched programs as the host resolves them and train
    the C step's state and events bit for bit; 1,063 of them run a program other than the C step's
    own. Every admitted instance of every family is the plain entry's bits at caps one, two and all.
  - The measurement ([px3b/](research/measurements/2026-10-21/px3b/README.json)), on this host
    (AVX-512, AVX512-BF16 and AMX) at four threads: under bf16 none of the 93 alternatives reads faster
    at its sites -- AVX-512 in place of AMX x4.74-x10.11 on the GEMMs, the portable instance
    x274-x646, one to three workers x0.92-x3.82 -- so the default dispatch is kept (a third trainer on
    it x0.9875 of the first, the A/A control x1.0012); under binary32 the embedding gradient's scatter
    at one worker reads x0.848 (0.734-0.980) at its sites and is chosen, x1.0101 (0.990-1.030) end to
    end beside the A/A control's x0.989: not resolvable. Every GEMM and attention on the portable
    instance runs x215 under bf16 (18.8 s a step against 79.5 ms) and x62.5 under binary32 end to end,
    as the sites' readings predict (x240 and x64.5). The two slowest arms first shared the 32 rounds of
    the others -- at 18.8 s a step, hours of them -- so they take six rounds of their own beside a
    default trainer of their own (`--slow-rounds`).
  - Fault sweeps: `px-dispatch.json` (PY1-PY28: the alternatives, the grammar, the host, the program,
    the resolution, the table, the choice and the row), 28 of 28 caught, each by its own witness;
    `px-dispatch-c.json` (PC1-PC4, PI1-PI6, PK1-PK10, PR1-PR7, PT1-PT3, PB1-PB4: the pool's cap, the
    instance an entry runs, the check, the run, the trace, the binding and the measurement), 34 of 34
    -- four first stopped a strict build (`-Wunused`: two in the sweep's own build, two more under GCC
    13's `-Wall -Wextra -Werror`), were rewritten to build warning-free under GCC 13, clang 18 and
    clang 23 and caught. The 20 entries of
    eight older tables whose lines PX-3b rewrote (`geo3.json` GW8, GW9; `ndt-attnbf16.json` AB8;
    `ndt-attnfma.json` AF10, AF11, AF14, AF16, AF17; `ndt-kblock.json` NK5; `ndt-threads.json` NT9,
    NT10, NT14, NT16; `px-choose.json` PC14, PC17, PC18; `px-program.json` PM8, PM11; `px-trace.json`
    PT2, PM3) were re-anchored to the new lines, each keeping its defect, and re-swept: 20 of 20
    caught (GW8 first stopped the build, `-Wparentheses`, and was rewritten), with GW12, inside the
    construct GW8 anchors in, 1 of 1.
  - Not claimed, and open (PX-3b-2): the measured prices inside K_BCIR's own candidates on its three
    rails, so the plan rather than a choice beside it takes the dispatch. PX-6 (GPU) stays pending: it
    continues in a local session on the hardware.
  - Found by CI's aarch64 oracle after the push: the witness that the trainer states its own
    decisions required each of them, the instances its families admit among them, to take more than
    one value over its cases -- but an aarch64 host's families all admit the portable instance
    alone, so there the instances take one value whatever the contract. They are required to vary
    only where the host's families differ, and otherwise held to the masks the cases spell from each
    contract's name; the old witness fails, and the new one passes, against the trainer built with
    `BCIR_TENSOR_MAX_VARIANT` 1 on this host, which admits the portable instance alone as aarch64 does.
  PX-3b-2 (2026-10-21): the measured prices inside K_BCIR's own candidates, on its three rails.
  - PX-3b-1 chose the dispatch beside the plan: `choose_dispatch` read a table and the program
    stated its pick. A target carries the table now -- `cost.MeasuredPrices`, frozen and
    generation-tagged: entries of (op, count, alternative, Q8 1..255), the alternative an instance
    (`i<bit>`) or a worker count (`t<k>`), in one spelling (strictly ascending by op, count and
    alternative, at most seven a key, no op spelled as an opcode -- the law rail's name for an op-less
    claim) -- and `realize._offer_rows` offers a claim of a priced (op, count), after each of its own
    realizations, that realization run each priced alternative (`<name>@<alternative>`: the same
    lane, width and operands, its compute and memory axes scaled by the entry's Q8 after the
    discount, floored). Control ops and the barrier are never measured (`UNMEASURED_OPCODES`). The
    planner prices the rows like any candidate, R9 re-derives the same offer and `hash_target` folds
    the table; a target with no table, or with one no claim reads, plans every module, writes every
    record and hashes as before, byte for byte.
  - The C planner: BKPI and BKPR version 1 ([the planner ABI](kernel/BCIR_PLANNER_ABI.md)) -- the
    measured section (`n_measured` at @40; an entry's op, count, code and Q8, the code the instance's
    bit or 0x8000 | k) and a step's code -- each version-1 law refused on both rails
    (`planner.malformed.accepted` 188 -> 0, 71 + 23 variants). `bcir_kplan.c` offers the measured rows
    and writes the compact planner's bytes, and the native hydrate's packs carry the measured names
    (`planner.measured.parity` 456 -> 0, a new row: 114 version-1 records -- every module of the fixed
    corpus and two native-trainer training packs under every target, each with a table drawn over
    its claims, and the seed module under a table naming its capital-led op -- each held to R9, the
    plan's price, the native planner's bytes and the native hydrate's pack). The decoder walks the op
    table forward once to name every entry's op, so a version-1 record stays linear.
  - The law rail: the capability's `measured_gen`, `measured_ops`, `measured_alternatives` and
    `measured_values` attributes and their verifier (one spelling, as the oracle's; the structural
    corpus's `measured.*` cases on both rails), `BCIRCostModel.h`'s `appendMeasured`, `-bcir-plan`'s
    `kbcir.plan_alternative`, R9 matching it and R13 folding the table.
    `differential --emit-measured` writes `plan_measured.mlir` (the oracle's plan, alternatives and
    m_target on every example program under two targets and a table no claim reads) and
    `verify_measured.mlir` (the verifier's refusals), drift-gated in CI; a measured path's symbol is
    quoted where it is not bare (`@"path_2_tile@i4"`).
  - The trainer: `native_choose.measured_prices` keys the picks of a dispatch table on the (op,
    count) of each kind's jobs' claims, a key shared by two kinds priced at the dearer of their
    prices; `carried_kinds` leaves out, and names, a pick whose keys another kind's claims hold at
    another pick or at none, so a carried price never moves a job the measurement did not price.
    `bcir_ep_check_pack_walk` hands the lowering each plan step in the walk that verifies the plan,
    and `bcir_decoder_lower_pack` writes a step's alternative into its job's dispatch words (41 and
    42), refusing any other spelling (`BCIR_DP_ALTERNATIVE`); `bcir_decoder_run_program` holds the
    dispatch to the host, so a measured plan another host cannot run is lowered and then refused
    with nothing written. `native_pack.plan_dispatch` reads the plan's alternatives as the lowering
    does.
  - `train.plan.measured.mismatch` 54 -> 0 (a new GEM+ row): the 27 state cases of the trace corpus,
    each under two seeded tables that measured the dispatch, their picks carried into K_BCIR's
    candidates, the default program planned on the AVX-512 profile carrying them, hydrated, placed and
    lowered by the C, write the program `step_program` states under the carried dispatch and train
    the C step's state and events bit for bit.
  - Fault sweeps: `px-measured.json` (MO1-MO20: the offer, the planner's slots, the table's spelling,
    the hash, the records, the law rail's text, the trainer's prices, the carried kinds and the
    plan's dispatch), 20 of 20 caught, each by its own witness -- MO6, the planner's slots sized
    without the measured rows, first read the same plans (every corpus module left the slots slack
    to spare) and was caught once a witness planned a one-claim module whose key carries the
    table's most entries, which leaves the slots none;
    `px-measured-c.json` (MC1-MC6 the native planner, ML1-ML5 the lowering and the check walk), 11 of
    11. The older entries PX-3b-2 moved (`ep-walk.json` EW5, EW6; `hydrate.json`'s vector width) were
    re-anchored, each keeping its defect, and re-swept.
  - On this host's own tables ([carry.py](research/measurements/2026-10-21/px3b/carry.py)): under
    binary32 PX-3b-1's one pick -- the embedding gradient's scatter at one worker -- is carried as one
    entry (`train.scatter`, 393,216, `t1`, Q8 217), and the comparison decoder's four-step pack plans
    its four scatter claims at `gather@t1` among 893, lowers to exactly the program the chosen dispatch
    states and trains that program's state bit for bit; under bf16 there is no pick to carry and the
    plan is the default's.
  - Not claimed: a gain. The row's tables are seeded and this host's one pick is not resolvable end
    to end (PX-3b-1), so the slice shows the prices reach the plan and the plan the C, not that a
    measured plan runs faster; a table prices only the host that measured it.
  ALIAS-SWEEP (2026-10-21): the plan's alias law in O(n log n), every verdict the same.
  - PX-2b left the alias law the one quadratic check of the plan walk: each lifetime against every
    earlier one, re-walking the validated prefix with no heap -- more than half of a ten-step
    lowering (the plan's and the pack's laws, the alias law among them, 36.5 of its 51.2 ms on the
    PX-3b host), and on the Python codec 2.7 s at 8,000 lifetimes, growing with their square. Both
    rails now judge it by a sweep. Visiting the lifetimes' ticks in order -- at one tick the ends
    before the starts, the intervals being half-open -- keeps the live lifetimes ranked by (bank,
    offset) in a Fenwick tree; until a pair is found the live lifetimes of one bank lie apart, so a
    lifetime that starts aliases a live one only if it aliases one of its two neighbours. The
    verdict names the first lifetime aliasing an earlier one, as the pairwise law did, refused ahead
    of any later lifetime's own law: the prefixes are monotone, so that lifetime is a binary search
    of sweeps, run only once the whole is known to hold a pair -- a plan the law admits costs one
    sweep.
  - The C twin sweeps over scratch the caller lends -- `bcir_ep_verify_lent`, and
    `bcir_ep_check_pack_walk`'s new scratch arguments: 62 bytes a lifetime and 4
    (`bcir_ep_sweep_bytes`), little-endian bytes at any alignment, no heap -- and judges the law
    pairwise without it, `bcir_ep_verify` among them; the lowering lends its executor's bytes, which
    the walk is done with before the executor takes them (`work` grows only where the sweep needs
    more). The walk reads each record's own laws first, then the alias law over the records before
    the first that breaks one, then hands its callback the records before the verdict -- the
    lifetimes one walk checking both record by record would have handed over. The Python codec's
    `first_alias` is the same sweep, and `validate_plan` names the same pair as before.
  - Witnesses: the codec's sweep against the pairwise law on 3,000 crowded draws (equal ticks,
    touching spans, three banks) and each draw moved apart, its message the pairwise law's, the
    boundaries pinned; the C twin's lent and unlent verdicts and the codec's on 400 crowded plans, a
    third with a lifetime breaking its own law, the lifetimes handed over with scratch and without
    the same, no byte past the lent scratch written (canary bytes, an odd address); both rails' cost
    n log n in the lifetimes (16x the lifetimes under 64x the time) and the lent verdict under a
    quarter of the pairwise one's time at 32,000; the lowering's check walk handing over the same
    steps and lifetimes with the executor's scratch, on every refusal of `test_native_run_pack`. The
    StreamPack fuzz harness aborts where the lent verdict, or the lifetimes the lent walk hands
    over, differ from the pairwise ones.
  - Measured ([alias-sweep/](research/measurements/2026-10-21/alias-sweep/), the 9.84M comparison
    decoder's packs, binary32, one thread, the PX-3b host): a ten-step lowering 51.2 -> 26.8 ms
    (x1.91; the tree before by its own driver), twenty steps 149 -> 51.2 ms; the plan verifier at
    ten steps 20.7 -> 1.5 ms lent (x13.9), at twenty 70.4 -> 2.8 (x25.2); the lowering's walk 22.4
    -> 4.8 and 78.7 -> 9.5 ms, the rest of it the records' own laws and the pack's claim checks; the
    codec's law 82.2 -> 6.2 and 312 -> 12.6 ms, and over GEM's reuse shape 2,712 -> 24.2 ms at 8,000
    lifetimes, 111 ms at 32,000 (the pairwise law, about 43 s by its square, not run). The scratch:
    104,536 bytes for a ten-step plan's 1,686 lifetimes.
  - Fault sweeps: `alias-sweep.json` (AP1-AP8: the codec's sweep -- the order of ends and starts at
    one tick, either neighbour unlooked-at, the rank without its bank, the first alias one past, an
    ended row left live, the message's pair, the pairwise law kept), 8 of 8 caught, each by its own
    witness; `alias-sweep-c.json` (AC1-AC10: the C twin's sweep, its tree and its k-th rank, the
    scratch's size, the lent scratch never swept, the walk's hand-over), 10 of 10, every fault
    compiling warning-free under GCC 13 and clang 18 and 23 (RED-VACUITY); `ep-walk.json`'s EW5 and
    EW6, which the walk's new arguments moved, re-anchored, each keeping its defect, and the older
    entries over the code ALIAS-SWEEP touched (EW4-EW6, ML5, PK33) re-swept, 5 of 5.
  OT/BG start (2026-10-21): PR #821's backend-optimality experiments and gates, executable.
  - Draft PR #821 proposes sixteen optimality experiments (OT-01 to OT-16: semantic graphs,
    compilation and native kernels; GPU code generation and resident execution; movement, telemetry,
    distribution and OS services) and eight entry gates (BG0, the measurement contract, to BG7,
    integrated promotion), as prose with no executable test. `tools/perf/ot_contract.py` is BG0 as
    code: the comparator routes, the kinds of gap a result reports and the host capabilities an
    experiment needs, as closed vocabularies; the record every experiment writes, held by
    `validate_record` to one shape (measured: its routes, semantic and target IDs, an independent
    reference, versions, results and raw records, memory scopes, unsupported and fallback labels and
    the gaps it reports; unavailable: the capabilities this host lacks; blocked: the prerequisite
    the repository lacks) and every other shape refused by its own rule; the sixteen as a registry
    and the eight as requirements over evidence -- a record measured, a GEM+ row at its bound, a
    host capability, a driver-ladder milestone attested by a report the repository holds, another
    gate -- stated by `evaluate_gates`: PASS only when every requirement is met, UNAVAILABLE when
    one needs a capability the host lacks (never a pass, never simulated), OPEN otherwise, and never
    ahead of a gate it requires. BG3 reads the first half of the driver ladder's M2 (the UART at
    D0-D2), BG4 all of M2 and M4. JSON it reads holds every key once.
  - `tools/perf/ot_run.py` writes all sixteen records and states the gates. What runs here, on
    BCIR's own rails: OT-01 (the first common block and its backward on BCIR's kernels against
    NumPy's float64); OT-03 (the direct C step against its program and its pack); OT-04 (one plan
    intervention); OT-06 (the arena frontier); OT-14 (the trace observer's cost). Unavailable here:
    OT-02, OT-05, OT-07, OT-08, OT-09, OT-10, OT-11, OT-12, OT-13, OT-16 (PyTorch and Triton in the
    interpreter that ran, a GPU, a multi-device host, QEMU, a device rig); blocked: OT-15 (no IPC
    adapter to decompose). An experiment whose capabilities a host has and whose runner this tree
    lacks is recorded blocked, never a failed run: under the model-lab interpreter (torch 2.13.0,
    Triton 3.7.1) OT-02, OT-05, OT-10 and OT-11 are.
  - Results: OT-01: every element of the block and its backward within its binary32 bound on all 9
    shapes of the proposal's CPU grid, the worst at 0.366 of its bound; the zero-size product a
    no-op; strided operands unsupported, labelled so. OT-03: 27 of 27 state cases one state and one
    event sequence, bit for bit, from the direct loop, the program and the pack. OT-04: the q/k/v
    and gate/up projections a call each -- 145 launches become 163, the state kept bit for bit --
    x0.9543 (95% interval 0.915-0.995) of the default step on the comparison decoder, beside the A/A
    control's x0.9689 (0.927-1.013), not resolvable beside its control. OT-06: the bench step's
    transients x1.326 of G26's bound in the C plan's spans, x1.000 placed by lifetime (a proved
    optimum). OT-14: the launch probe x1.0159 (95% interval 0.973-1.060) of the plain step, beside
    the A/A control's x0.9772 (0.941-1.015), not resolvable beside its control; the same state,
    5,550 records and 0 dropped. Gates on this host: BG0 PASS; BG1 PASS; BG2 UNAVAILABLE; BG3 OPEN;
    BG4 UNAVAILABLE; BG5 UNAVAILABLE; BG6 UNAVAILABLE; BG7 UNAVAILABLE.
  - Witnesses (`bcir/tests/test_ot_contract.py`): the registry against the proposal; each record
    rule's refusal by its message; the gates' semantics, including every gate passable with complete
    evidence and none passing while a gate it requires is only unavailable; OT-01's checks firing a
    thousandth past their bounds; the CLI's refusals (a key twice, a malformed file); every
    experiment without a runner recorded on every host, blocked where the host has its capabilities
    (it first failed the run there: a host with PyTorch); a smoke run of the runner, its timed
    experiments each beside an A/A control. `ot-contract.json` (OC1-OC15), 15 of 15 caught, each by
    its own witness.
  - Not claimed: any comparison with PyTorch, Triton or a device. Their routes need what this host
    lacks, and their records say so; the driver-ladder gates (BG3-BG6) wait on milestones their
    reports must attest; BG2 waits on PX-6 (GPU), pending in a local session on the hardware.
  GEO-2 (2026-10-22): Codex's PR818 geometric-head study rerun from its bundle, bit for bit, as
    uploaded and on current main ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §5,
    [measurements/2026-10-22/geo2/](research/measurements/2026-10-22/geo2/)).
  - The five study bundles reached main on 2026-10-21 (`docs/research/{gp-global,wedge-next,
    wedge-integrated,wedge-query-aware,wedge-pr818}`, 1,370 files, 1.77 GB). GEO-0: every bundle's
    manifest against the bytes main holds -- PR818's 442 of 448 files byte-identical (its 66 selected
    checkpoints among them), three more equal but for the trailing newline GitHub's web rename adds,
    three absent (an evaluation protocol its evaluation rewrites; two LICENSE copies, this
    repository's by hash); the earlier bundles likewise, short only of `wedge-next`'s and
    `wedge-integrated`'s `protocol.json` and `wedge-next`'s generated `build/` directory.
  - The host had moved: an Intel Xeon @ 2.80 GHz (family 6, model 85: AVX-512F and VNNI, no AMX),
    so the bf16 GEMMs ran the AVX-512F instance Codex's AMD EPYC ran, and the AMX timing GEO-2
    planned stays open.
  - GEO-2a, as uploaded: Codex's `build_upgrade.py` regenerates its 17 overlay sources byte for byte
    and the frozen protocol holds every source; the drivers ran unchanged (`launch.py` answers the
    protocol's two cgroup-v2 reads on this cgroup-v1 host). All 51 binary32 runs (41,400 updates)
    and 15 bf16 runs (12,600) equal Codex's records in every field -- state digests, histories, every
    update's loss and gradient norm, the budget policy's decisions -- and all 66 evaluations (633
    rows, 138 counterfactual groups): AMD's results on Intel, the bf16 T2048 ordered-recall gain
    (89.06% to 95.31%) example for example.
  - GEO-2b, current main: `build_main.py` makes `build_upgrade.py`'s graft from origin/main's trainer
    (the state checks, the workspace functions, the GEMM dispatch, the attention entry, the SwiGLU
    blocks, the optimizer and run tail; not the PX-2 runner, the transformer's step) with main's
    tensor unit (ABI 5); every refusal main's products return is counted and held at 0; three driver
    lines change for ABI 5 and the protocol carries the port (eleven entries re-hashed, 37 held).
    All 66 runs and their evaluations equal Codex's again, field for field. The policy's decisions
    are Codex's in every run; its bytes moved only where main's accounting did (the state record's
    8-byte `recompute` word, the bf16 attention's exact WS-PLAN need). Every change since PR818 --
    FACTS-RUN, BF16-ATTN, ATTN-SOFT, GEO-3, PX-1 to PX-5 -- keeps the study's bits under both
    arithmetic families.
  - Against PyTorch (`geo_compare.py`: the study's three models at batch 4 on four threads, twelve
    rounds in rotated order, B-DIRECT twice as an A/A control): BCIR trains every model faster, 2.1
    to 7.5 times the bundle's own Torch equations (P-EAGER) and 1.9 to 3.3 times `torch.compile`
    (P-IND), and runs both geometric heads' inference 3.2 to 4.2 times faster than either. The one
    loss is the attention baseline's inference at T2048: 1.74 (95% interval 1.52-1.98) times
    P-EAGER's time and 1.59 (1.36-1.87) times P-IND's, at parity at T1024. Its attention writes every
    head's t x t probabilities (268 MB a layer at B4/T2048) even with no backward to read them;
    PyTorch's CPU flash kernel never forms them. That is the next inference update.
  - Corrected by COMPILE-LIMIT (2026-10-24): seven of the fifteen P-IND readings -- the attention
    model's training at T512 and every inference case -- ran eagerly past Dynamo's recompile limit.
    Rerun with each case compiled at its own shapes (the session's later host): BCIR trains in
    x0.33-0.83 of Inductor's time and runs the geometric heads' inference in x0.77-0.90 of it, 1.1 to
    1.3 times its speed rather than 3.2 to 4.2; the attention baseline's inference took x1.39 of
    Inductor's time at T1024 and x1.67 at T2048. The P-EAGER readings stand.
  - Not claimed: AMX timing (no AMX here); a rerun of the four earlier studies (two protocols absent);
    any learning result beyond what the bundle's own records already state.
  OT P-routes (2026-10-22): PyTorch's and Triton's routes of PR #821's experiments, run.
  - `tools/perf/ot_torch.py` runs PyTorch's routes in the OT records: P-EAGER beside B-DIRECT in
    OT-01; the capture cost of P-FX, P-AOT and P-IND (OT-02); P-EAGER and P-IND in the block
    portfolio (OT-05), beside B-DIRECT's block under each GEMM contract and under the trainer's own
    (kblock+exp), whose T-DSL rows come from a session of PR #821's Triton-CPU pilot in its own
    interpreter, admitted only over this tree's `bcir_tensor.c`; the frontier from
    train_comparison.py's runs over one recipe (OT-10); Inductor's cache lifecycle (OT-11). Each
    compiled step runs in its own process with a fresh cache; Triton is a capability only with a
    usable backend; every input file is read before an experiment runs, and every measured record
    names the PyTorch it ran (2.13.0+cu130, AVX512, BLAS mkl) and the threads it ran at. Measured on
    this host: OT-01, OT-02, OT-03, OT-04, OT-05, OT-06, OT-10, OT-11, OT-14; gates: BG0 PASS; BG1
    PASS; BG2 UNAVAILABLE; BG3 OPEN; BG4 UNAVAILABLE; BG5 UNAVAILABLE; BG6 UNAVAILABLE; BG7
    UNAVAILABLE ([ledger](research/BCIR_OT_BG_LEDGER.md),
    [measurements/2026-10-22/ot/](research/measurements/2026-10-22/ot/README.json)).
  - Results: OT-01's P-EAGER within its binary32 bound beside B-DIRECT (worst 0.351 and 0.366 of the
    bound). OT-02: the decoder step's first compiled call 3.5 s (P-FX), 6.8 s (P-AOT) and 44.3 s
    (P-IND) as one whole-loss graph, the module form breaking at its input checks (2 graph breaks);
    B-PLAN's step program and its words 4.5 ms (its TrainingPack 9.7 ms more, the StreamPack and
    plan lowered in 67 ms) over a library built once in 11.5 s. OT-05, the block as the trainer runs
    it (kblock+exp), forward and backward, against PyTorch's: at four threads faster in sixteen of
    eighteen comparisons (1.32-5.45 times), at parity in two (512 x 768 x 1024); at one thread
    faster in ten (1.07-1.72 times), at parity in five, slower in three: 512 x 768 x 1024 (1.37 and
    1.38 times eager's and Inductor's time) and 512 x 257 x 1025 against Inductor (1.10 times its
    time); against Triton-CPU its GEMM 2.2-3.4 times and its RMSNorm 2.6-10.0 times faster, the
    SwiGLU product under the BCIR exponential at parity at a million elements. OT-10: the 9.84M
    decoder's 300 updates at 272 ms a step on B-DIRECT against 533 (P-EAGER) and 390 (P-IND),
    validation CE 2.3525, 2.3643 and 2.3730, peak RSS 489, 1118 and 1059 MB. OT-11: Inductor's first
    call 43.6 s cold, 7.9 s on its kept cache, 80.1 s with dynamic shapes; B-PLAN's program a shape
    at most 5.3 ms.
  - A reading corrected before merge: the day's first run (`e81fb74`, pushed in `6e11a62`) compared
    PyTorch's block with B-DIRECT's GEMM contracts under v1's libm SiLU passes alone, read 1.17-1.44
    times PyTorch's time at one thread on the 128- and 512-row blocks, and put it down to MKL's
    GEMMs. The block taken apart at one thread (`block_parts.py`,
    [block-parts/](research/measurements/2026-10-22/block-parts/README.json)) showed v1's passes at
    3.4-7.3 times the BCIR exponential's time -- the trainer runs the exponential's -- so the
    portfolio gained the trainer's arm and the run was repeated; the exponential's passes take the
    block to 0.72-0.94 of the libm arm's time at one thread. What remains is GEMM time: the
    k-blocked forms keep pace with MKL's but on the largest block, whose A (1.5 MiB) is larger than
    the core's 1 MiB L2 cache (the nn and tn forms 1.35 and 1.45 times MKL's), and in the weight
    adjoint's transposed form (1.04-1.53 times at seven of nine shapes) -- the next training
    update's target.
  - Witnesses (`bcir/tests/test_ot_contract.py`): the PyTorch records keep every route that ran and
    every failure; OT-05's arms each run their GEMM contract's and elementwise contract's entries,
    declared as `bcir_tensor.h` declares them, and every phase's ratios pair a round with the same
    round; a Triton session is evidence about this tree alone, every BCIR implementation a case
    times correct on both sides (the first run's reader assumed one implementation a case: the smoke
    run against the real session found it); Triton a capability only with a usable backend; OT-10's
    frontier from one recipe; OT-01's verdict every route's; the runner's inputs refused before it
    runs (a rows file it could not read failed the run only after its last experiment, until it was
    read first); a record naming the PyTorch and the threads it ran. `ot-contract.json` OC1-OC48,
    each caught by its own witness.
  - Not claimed: P-TUNED (no budgeted autotuning lifecycle ran), T-GLUON and every GPU route (no
    GPU), B-MLIR on the block, T-DSL inside OT-11 (the interpreter that runs PyTorch has no Triton
    with a usable backend).
  INFER-ATTN (2026-10-22): the attention forward for inference without its probabilities, every
    value as it was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §6,
    [measurements/2026-10-22/infer-attn/](research/measurements/2026-10-22/infer-attn/README.json)).
  - The loss it answers: GEO-2's comparison on the study's attention model, whose inference at T2048
    read 1.74 times eager PyTorch's time while the attention wrote every head's t x t probabilities
    -- 256 MiB a layer at B4/T2048 -- that no backward reads.
  - `BCIR_TENSOR_ATTENTION_SCRATCH` (8u) makes p the forward's scratch: each (batch, query head)
    runs its rows a strip at a time -- scores, softmax, context -- in its own slot of whole rows at
    the start of p, about 64K floats of scores a strip (at least 32 rows, a multiple of 16), so a
    strip's scores, probabilities and the context's reads of them stay in the core's cache;
    `bcir_tensor_attention_scratch_floats(batch, time, heads)` is all of p it writes. Every context
    value and verdict is the stored attention's bit for bit: a score is its channels' products added
    in order from +0.0, a context element its keys' in order from +0.0, a row's softmax its own --
    the AVX-512 blocks start at the strip's first row (`att512_scores_from`, `att512_context_from`),
    the v1 kernel's rows run as a rectangle and a triangle under the causal mask with the terms in
    their order. The bf16 attention takes the bit alike: its 32-row query blocks in the slot, the
    future columns its packed key tiles reach filled. The backward ignores the bit.
  - `bcir_decoder_forward` passes it: no backward reads its cache, and the backward runs its own
    forward, which keeps its probabilities, as REMAT's recompute and the PX-2 program runner do. The
    tensor unit's probe records the bit.
  - Measured on the committed tree against the one before it (24 or 12 rotated rounds, each beside
    an A/A control): the kernel at B4/T2048, the GEO model's attention, x0.677 (0.646-0.710) of the
    stored forward at four threads and x0.679 at one, with p 256 MiB down to a 4 MiB slot; B1/T8192
    x0.730 (p 1 GiB); B1/T4096 with 8 heads of 64 x0.785; the 9.84M decoder's T128 x0.890.
    `bcir_decoder_forward` end to end at the GEO model's dimensions: x0.822 at T1024 and x0.786 at
    T2048 (x0.770 on one thread, x0.883 under bf16); the 9.84M decoder at T128 unchanged (x0.999,
    its attention a small part of its forward). The GEO study's attention model itself, on level 3
    (GEO-2b's overlay with the tensor unit swapped, `build_level3.py`): x0.786 (0.709-0.871) at
    T1024 and x0.748 (0.717-0.780) at T2048 of the build before; against PyTorch, at T1024 x0.632 of
    eager's time and x0.910 (0.780-1.062) of Inductor's, at T2048 x1.173 and x1.355. The T2048 loss
    narrows (this run's build before read about x1.56 of eager's time) but stays.
  - Codex's trained attention checkpoints (the bundle's lm and kv_single models) give byte-identical
    logits from both builds at T1024 and T2048, and one forward from a zeroed arena writes 49.3M
    nonzero floats on the stored build against 27.2M on the scratch build: the scratch path ran.
  - What remains at T2048 is the attention's own rate: after INFER-ATTN it is about 85% of that
    forward (three layers' 59.7 ms calls of 210 ms), about 72 GFLOP/s on four cores at head
    dimension 32. That is the next inference update; an online softmax would be a new arithmetic
    contract, not these bits.
  - Witnesses (`runtime/c/test_decoder_train.c`): `attention_scratch` -- 398 binary32 and 133 bf16
    cases over shapes, contracts, instances and thread counts, each context and verdict the stored
    attention's bit for bit, nothing written past the scratch extent (a canary after it), every slot
    at least 32 rows or all of them; `trainer_infer` -- 7 contracts at two contexts and two thread
    counts, the forward's logits the backward's forward's, the probe naming the bit after the
    forward and not after the backward. `infer-attn.json` IA1-IA17, each caught by its own check,
    every one compiling warning-free under GCC 13, clang 18 and clang 23; nine earlier faults (AB14,
    AB17, AF1, AF2, AF10, AF16, SW15, PI5, PR2) re-anchored to the lines INFER-ATTN moved, each
    keeping its defect, caught again.
  - Not claimed: training's attention memory (the backward still stores P; the exact-recompute tiled
    attention is the training update the roadmap names); a smaller decoder arena (the plan still
    sizes each layer's probability region for training, so the forward leaves it untouched rather
    than unallocated); AMX timing; a GPU.
  ATTN-RATE (2026-10-22): the attention's exponential and context at their rate, every value as it
    was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §7,
    [measurements/2026-10-22/attn-rate/](research/measurements/2026-10-22/attn-rate/README.json)).
  - The loss it answers: INFER-ATTN's residual -- the GEO attention model's inference at T2048 still
    x1.17 of eager PyTorch's time, the attention about 85% of that forward. Timed a phase at a time on
    one core (an instrumented copy of the unit, TSC cycles a call at B4/T2048), the scratch kernel ran
    its scores in 169M, its softmax in 196M and its context in 224M: the softmax and the context, more
    than the products' rate.
  - EXP-LANES: the BCIR exponential's polynomial is a chain of seven dependent multiply-then-add steps
    under -ffp-contract=off, and its loops ran at the chain's latency. The exponential's own loops (its
    instances and the softmax's rows in place) now take exp1 over a block of elements, each step over
    the whole block before the next: four vectors in a row on AVX-512 (64 elements) and two on AVX2
    (16; four spill its sixteen registers), whichever compiler builds them. GCC's unroll pragma copies
    a vector's whole chain after the previous one's (measured no change), and its AVX2 instance did not
    vectorize exp1 at all (8.7 ns an element against clang's 1.8). The portable instance keeps its
    loop, and a call whose y partly overlaps x runs the element loop whole. The SiLU, the SwiGLU
    product and its VJP keep exp1 an element under clang's interleave pragma, four vectors an
    iteration (a block there holds too many arrays to stay in registers: measured slower at 1M
    elements); GCC builds them as before.
  - CTX-WIDTH: the AVX-512 context ran a 48-channel block as three 16-lane vectors whatever the head's
    width, so at head dimension 32 the third vector, every lane masked off, was loaded and multiplied
    anyway. `att512_ctx_block` runs the vectors a block's channels reach -- one, two or three -- each
    element's terms the same, in the same order.
  - Every value is the same: `attn_hash.c`'s 88 cases (y, and p when stored) and `elem_hash.c`'s 181
    lines (each exponential instance, the SiLU, the SwiGLU product and VJP, serial and pooled, over
    lengths that end mid-block and inputs with +-0, +-inf, NaN and the thresholds) byte for byte before
    and after under clang 18 and GCC 13; the GEO model's trained checkpoints' logits byte-identical at
    T1024 and T2048.
  - Measured on the committed tree against the one before it, the kernels and the training step with
    both builds in one process (A, B and B again as the A/A control, each round's outputs held equal),
    the forward in rotated processes. The exponential x0.75 (clang) and x0.76 (GCC) at 16K elements,
    x0.76 and x0.81 at 1M; GCC's AVX2 instance x0.47-0.49; clang's SwiGLU product x0.71 and its VJP x0.67
    (x0.91 at 1M). The attention at B4/T2048 x0.78 (scratch) and x0.84 (stored, the training forward)
    under clang, x0.86 and x0.90 under GCC; B1/T4096 with 8 heads of 64 x0.83; on one core the softmax
    196M -> 169M cycles and the context 224M -> 178M, the scores unchanged. `bcir_decoder_forward` at the
    GEO model's dimensions x0.899 at T1024 and x0.876 at T2048 (x0.896 on one thread); the 9.84M decoder
    at head dimension 48 unchanged (x0.978), as is its training step (clang x0.967, 0.931-1.006). GCC's
    decoder step read x0.924 (0.897-0.952), more than its exponential and context account for (under
    1% of that step), so this record does not attribute it. The GEO study's attention model on level 3,
    built with GCC as the study builds it: x0.878 at T1024 and x0.920 (0.872-0.971) at T2048 of the
    build before; against PyTorch, at T1024 x0.607 of eager's time and x0.777 (0.685-0.882) of
    Inductor's -- faster than both -- and at T2048 at parity with eager (x1.053, 0.961-1.155) and
    x1.130 of Inductor's.
  - What remains at T2048 is three phases at about a third each (scores 168M, softmax 169M, context
    178M cycles on one core). Candidates within this contract, none measured yet: the softmax's
    maximum taken where the scores are stored, not in its own transposed pass; a head's key panels
    packed once for all its strips (each strip packs every panel up to its last row: 32 times the
    stored forward's packs at T2048); the context's eight broadcasts a key. An online softmax would be
    a new arithmetic contract.
  - Witnesses (`runtime/c/test_decoder_train.c`): the context widths 32, 33 and 48+33 added to the v1
    and fused attention references; every exponential instance on the edges and on a call whose y
    lies past x and overlaps it (y = x + 1, 9, 17 and 60), element by element as the loop runs it --
    both hold on the tree before too. `attn-rate.json` CW1-CW6 and EL1-EL6, each caught by its own
    check and compiling warning-free under GCC 13, clang 18 and clang 23; NE3 and five INFER-ATTN-era
    faults (IA2, NA5, NA15, NU1, NZ3) re-anchored to the lines this moved, each keeping its defect,
    caught again; `ndt-exp.json` 6 of 6 and NL11 caught on the final tree.
  - Not claimed: the portable instance (every non-x86 host's, untimed here) and GCC's SiLU and SwiGLU
    loops are unchanged; the decoder's training step, at head dimension 48, gains nothing resolved
    under clang; AMX timing; a GPU.
  BWD-WIDTH (2026-10-22): the attention backward's transposed products at the head's width, every value
    as it was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §8,
    [measurements/2026-10-22/bwd-width/](research/measurements/2026-10-22/bwd-width/README.json)).
  - The loss it answers: the GEO attention model's training update at long context. Paired with PyTorch
    on ATTN-RATE's tree (the study's update -- forward, backward, clipping, AdamW -- of 4 x T tokens, 12
    rotated rounds), BCIR took 94, 281 and 1107 ms at T512, T1024 and T2048 (medians) against eager
    PyTorch's 219, 364 and 783 and Inductor's 165, 292 and 739, paired: faster than both at T512 (x0.42
    and x0.55 of their times), faster than eager's and at parity with Inductor's at T1024 (x0.80;
    x0.994, 0.964-1.025), x1.43 of eager's time and x1.51 of Inductor's at T2048. BCIR's update grows as
    T squared there (x3.9 from T1024 to T2048, PyTorch's x2.2-2.5): the stored attention, whose forward
    took 66 ms a layer at T2048 on four threads after ATTN-RATE and its backward 206 ms -- three
    quarters of it.
  - The change: the backward's two transposed products -- dV = P^T dY and dK = dS^T Q,
    `att512_transposed_any` -- ran a 48-channel block as three 16-lane vectors whatever the head's
    width, the waste CTX-WIDTH removed from the context: at head dimension 32 a third vector of masked
    lanes was loaded, multiplied and stored for every term. `att512_tr_block` runs the vectors a block's
    channels reach -- one, two or three -- each element's terms the same, in the same order.
  - Every value is the same: `bwd_hash.c`'s 34 backward cases (17 shapes whose head widths meet the
    16-lane boundaries, v1 and fused: dq, dk, dv and the score adjoints left in p) and ATTN-RATE's 88
    forward cases byte for byte before and after under clang 18 and GCC 13; the GEO model's trained
    state (weights, both moments, the step) the same digest from both builds after three updates at each
    context.
  - Measured on the committed tree against the one before it, both builds in one process (A, B and B
    again as the A/A control, each round's outputs held equal). Each row whose interval held 1 or met
    its A/A interval, but the two controls at head dimension 48, ran again with 24 rounds, declared
    before it ran; both runs are recorded. The backward at B4/T2048, head dimension 32, x0.850
    (0.815-0.888) on one core; on four threads x0.886 (0.841-0.934) at T2048, x0.817 at T1024 and x0.859
    at T512 (24 rounds; the 12-round run x0.890, 0.857 and 0.863, each meeting its A/A interval); GCC
    x0.839 at T2048 and x0.868 at T1024; B1/T4096 with 8 heads of 64 x0.894; the 9.84M decoder's
    attention, head dimension 48 -- three vectors either way -- unchanged (x1.033, 0.918-1.164). The
    training step at the GEO model's dimensions x0.912 (0.873-0.953) at T2048 under clang and x0.905
    under GCC; at T512 and T1024 x0.938 and x0.911 in the 24-round run (x0.943 and x0.925 in the first),
    each meeting its A/A interval; the 9.84M decoder's step unchanged (x0.963, 0.927-1.001). The GEO
    study's attention model on level 3 (built with GCC, as the study builds it): x0.914 (0.879-0.950) of
    the build before at T1024 and x0.920 (0.892-0.948) at T2048 over 24 rounds (the 12-round first run
    x0.958 and x0.985, its intervals holding 1), x0.915 at T512 meeting its A/A interval; the trained
    state the same digest. Against PyTorch the update is faster than eager's and Inductor's at T512
    (x0.36 and x0.49 of their times) and at T1024 (x0.77 and x0.92), and at T2048 x1.35 of eager's time
    and x1.42 of Inductor's.
  - What remains at T2048, timed a phase at a time on one core (an instrumented copy of the unit, TSC
    cycles a call at B4): of the backward's 2026M, dV 564M and dK 528M -- the two transposed products
    still more than half, at about 8 cycles a vector multiply-add against the context form's 3.4 for the
    same multiply-adds (dQ, 231M) -- the dP tiles 300M and their second evaluation 209M (past T512 they
    take the GEMM kernel: the values' transposed panel holds 16384 floats; past the first 256 columns
    pass 1 evaluates them again), the row dots 121M and the score adjoints 66M. Each transposed product
    walks P down its columns, a row of P a term: at T2048 an 8 KB stride, a new page every term. Walking
    P a band of rows at a time, each output row's terms still in ascending order, is the next training
    update.
  - ATTN-SOFTMAX, measured and not taken: the first of ATTN-RATE's candidates -- the softmax's maximum
    taken along the stored rows (order-free: NaN skipped, and exp1(+-0) = 1 makes the zero's sign
    irrelevant), its exponential in row order and one transpose for the in-order sum -- every value the
    same (the 88 cases, both compilers), at parity: the inference attention at B4/T2048 x0.970
    (0.915-1.028) of its time on one core, x0.973 (0.929-1.020) on four, x0.950 (0.886-1.018) under GCC.
    The transposes it removed were spent again in the pass it added; the softmax's remainder is its
    exponential and its division, both fixed by the contract.
  - Witnesses: the v1 and fused attention references already hold the backward at the widths the vector
    boundaries meet (1, 15-17, 24, 32, 33, 47-49, 65, 81, 130); `bwd-width.json` BW1-BW6, each caught by
    its own check and compiling warning-free under GCC 13, clang 18 and clang 23; NA8, NA9, NA16 and NZ5
    re-anchored to the lines this moved, each keeping its defect, caught again.
  - Not claimed: the portable and AVX2 instances (the change is the AVX-512 kernel's); the 9.84M decoder
    at head dimension 48, three vectors either way; AMX timing; a GPU.
  BWD-LONG (2026-10-22): the attention backward at long context, every value as it was
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §9,
    [measurements/2026-10-22/bwd-long/](research/measurements/2026-10-22/bwd-long/README.json)).
  - The loss it answers: BWD-WIDTH's residual. Timed a phase at a time on one core at B4/T2048 (TSC
    cycles a call), the backward spent 494M and 491M of its 1823M on its two transposed products, dV =
    P^T dY and dK = dS^T Q, at about 7 cycles a vector multiply-add where dQ -- the same multiply-adds
    in the context's form, walking P along its rows -- took 3.2; and 266M and 181M on its
    probability-adjoint tiles, which past T512 took the GEMM kernel.
  - BWD-BAND: each transposed-product block walked its terms down P's columns, a row of P a term -- at
    T2048 an 8 KB stride, a new page every term. Once a row of P spans a 4 KB page
    (`BCIR_ATT_BAND_FROM`, 1024 floats) the terms run a band of 32 rows of P at a time
    (`BCIR_ATT_BAND`): every 8-row output block the band reaches takes the band's terms -- its diagonal
    terms first when the band holds them -- before the next band starts, its sums stored between bands,
    so a band's rows are read along their columns. Each element takes its terms in ascending order
    either way; below the threshold one band holds every term, the loop as it was (banded at every
    context, the 9.84M decoder's attention at T128 read x1.074, 1.028-1.121).
  - DP-PANEL: the probability-adjoint tiles read the KV head's values transposed once into the task's
    16384-float panel (NDT-ATTN-5), so past it -- T512 at head dimension 32 -- they took the GEMM
    kernel. Each tile's values are now transposed into the same buffer as the tile is evaluated (d x 64
    floats, head dimensions to 256), for the register blocks: the GEMM kernel's arithmetic for these
    operands, the same bits.
  - Every value is the same: `band_hash.c`'s 48 backward cases -- BWD-WIDTH's 17 shapes and 7 past the
    threshold, T1024 to T2048, whole bands and a context ending inside one, a GQA group, head widths of
    one to three vectors, v1 and fused: dq, dk, dv and the score adjoints left in p -- and ATTN-RATE's
    88 forward cases byte for byte before and after under clang 18 and GCC 13; the GEO model's trained
    state the same digest from both builds after three updates at each context.
  - Measured on the committed tree against the one before it, both builds in one process (A, B and B
    again as the A/A control, each round's outputs held equal), 24 rounds every row. The backward x0.597
    (0.554-0.644) of its time on one core at B4/T2048; on four threads x0.641 at T2048 and x0.707 at
    T1024 (GCC x0.635 and x0.687); B1/T4096 with 8 heads of 64 x0.695; T512 (x0.994) and the 9.84M
    decoder's attention at T128 (x0.977) unchanged. On one core the phases at T2048: dV 494M -> 211M, dK
    491M -> 199M, the tiles 266M -> 166M and their second evaluation 181M -> 116M, the rest unchanged;
    the backward 1823M -> 1077M. The training step at the GEO model's dimensions x0.804 (0.783-0.825) at
    T2048 under clang and x0.797 under GCC, x0.852 at T1024, x0.958 at T512 (0.910-1.009); the 9.84M
    decoder's step read x0.899 (0.860-0.940), more than its unchanged attention accounts for, so this
    record does not attribute it. The GEO study's attention model on level 3 (built with GCC, as the
    study builds it): x0.789 (0.769-0.810) of the build before at T2048 and x0.904 (0.877-0.931) at
    T1024, unchanged at T512 (x0.984); the trained state the same digest. Against PyTorch the update is
    faster than eager's and Inductor's at T512 (x0.38 and x0.55 of their times) and T1024 (x0.68 and
    x0.85), and at T2048 x1.09 (1.051-1.131) of eager's time and x1.23 of Inductor's -- from x1.35 and
    x1.42 after BWD-WIDTH.
  - The design runs, recorded with the slice: bands of 16 measured the same as 32 and bands of 64
    slower; banded at every context, short contexts paid (the T128 decoder's x1.074), hence the
    threshold; the tiles' panels over the bands alone x0.842 on one core at T2048 and x0.826 at T1024.
  - BWD-FUSE, measured and not taken: each 32-row strip of P taken through every pass while it is in
    cache -- dV's band of P's rows, the tiles and dots, the score adjoints, dK's band of dS's rows, dQ's
    rows -- every value the same (48 cases, both compilers), at T2048 x0.927 (0.842-1.021) on one core
    and x0.930 on four, parity under GCC and at T4096, and x1.092 (1.017-1.172) at T1024: a strip's
    bands keep all of dV's and dK's accumulators live, a larger working set than the separate passes.
  - What remains at T2048, on one core: of 1077M, dQ 215M, dV 211M, dK 199M, the tiles 166M and their
    second evaluation 116M, the row dots 110M, the score adjoints 53M -- every product now at about 3.1
    cycles a vector multiply-add, two vector operations a term under v1's separate multiply and add.
  - Witnesses: the C harness builds once more with both paths at every context (`BCIR_ATT_BAND_FROM` 16,
    `BCIR_ATT_WHOLE` 0: bands from 16 terms, every tile on its own panel), so the v1 and fused attention
    references hold them -- contexts of 33 to 330 run two to eleven bands, a context ending inside a
    band among them; `bwd-long.json` BB1-BB6 and DP1-DP4, each caught by its own check and compiling
    warning-free under GCC 13, clang 18 and clang 23; NA9, NA13, NZ5, AF4, NT8, BW1 and BW2 re-anchored
    to the lines this moved, each keeping its defect, caught again.
  - Not claimed: the threshold and the band's height are this host's choice, and no value depends on
    them; the portable and AVX2 instances; the bf16 attention's own transposed products; AMX timing; a
    GPU.
  ATT-CONTIG (2026-10-23): the attention's tasks on contiguous copies of their heads' rows, every value
    as it was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §10,
    [measurements/2026-10-23/att-contig/](research/measurements/2026-10-23/att-contig/README.json)).
  - The loss it answers: not the traffic in P that BWD-LONG named. Four ways at that traffic were built
    on the session's earlier host, every value the same in each, and none moved the attention on four
    threads at B4/T2048: BWD-RECOMP (the forward without its probabilities, the backward recomputing
    each 32-row strip of P exactly) x0.989 of the attention, its forward x0.680 and its backward
    x1.192; row fusion (dQ, dK and dV in the strip loop) x0.968-x1.019 of the backward; ATT-PACK (the
    context's keys and the bands' rows packed) x1.013; ATT-PREFETCH (P's lines asked for ahead of the
    walks) no distance resolved. The layout did move it: the same buffers read as B x H batches of one
    head ran the attention at x0.845 at B4/T2048 (the backward x0.801) and x0.617 at B1/T4096 with 8
    heads of 64. A row of q, k or v holds every head's channels, so a head's rows lie heads x dim floats
    apart -- the context's walk alone took 2.10-2.24 cycles a vector multiply-add at a 128-float stride,
    1.56-1.66 at 32 -- and four threads' heads share their output rows.
  - ATT-CONTIG: from `BCIR_ATT_CONTIG_FROM` positions (1024), where a row holds more than one head, a
    task lent a region of a caller's workspace copies its head's rows in, runs the head as a batch of
    one head of one row stride, and copies its results back -- a forward task its query head's q and
    its KV head's k and v, its context out; a backward task its KV head's k, v, dk and dv, then each
    query head's q, dy and dq, the group's later heads adding to dk and dv as they add in place (the
    head's backward takes an accumulate flag). Values move; none is computed. A region is seven t x d
    copies, each to a 64-byte boundary; one a worker where the workers are no more than the forward's
    tasks, else one a task (`bcir_tensor_attention_pool_workspace`, at most
    `bcir_tensor_attention_workspace`); with less, or none, every task runs on the rows in place. The
    entries `bcir_tensor_attention_contract_variant_scan_ws_pool` and
    `bcir_tensor_attention_backward_contract_variant_ws_pool` take the workspace; the entries without
    one run them with none.
  - The trainer lends the binary32 attention its state's workspace under every contract:
    `bcir_decoder_workspace_need` covers the regions (alone under a binary32 GEMM contract; under bf16
    the most of the GEMM groups' packs and the regions), a lent workspace is held apart from the state,
    every arena and a caller's buffers under every contract, and any size trains the same state. PX-1's
    trace, the program executor's check, the pack's claims and their lowering state it as the
    attention's operand on both of its launches wherever the state lends bytes; the Python planner lends
    one only where the step packs or copies into it.
  - Every value is the same: the entries without a workspace give BWD-LONG's 48 backward and 88 forward
    cases byte for byte before and after under clang 18 and GCC 13, and the workspace entries the
    entries without one in 564 cases (the context, the verdict, the stored probabilities, dq, dk, dv; a
    non-finite key among them), built as committed and built to copy at every context; the GEO model's
    trained state the same digest from both level-3 builds at every context.
  - Measured on the committed tree against the one before it, on the host the session moved to on
    2026-10-23 (an Intel Xeon @ 2.30 GHz with AMX, 2 MiB of L2 a core, a 260 MiB L3), both units in one
    process beside the tree after it run in place, the contiguous layout and an A/A control, 48 rounds a
    row on four threads: the GEO shape at B4/T2048 x0.934 (0.906-0.962), the backward x0.883 and the
    forward unchanged (GCC x0.945, its interval meeting the control's; the contiguous layout x0.922 and
    x0.912); at T1024 x0.975 and on one thread x0.984, neither resolved; 8 heads of 64 at B2/T1024
    x0.850, B1/T2048 x0.716 and B1/T4096 x0.593 (GCC x0.599), against the contiguous layout's x0.792,
    x0.683 and x0.576 (GCC x0.588) -- within its interval at T2048 and T4096; the in-place path
    x0.98-x1.02 everywhere. The threshold: copying from T1024 read faster on this host (8 heads of 64 at
    B1/T1024 x0.842 in the run that set it), from T512 not (x1.009, x1.014, x0.983, none resolved). The
    training step at the GEO dimensions x0.912 (0.899-0.925) at T2048 and x0.953 (0.934-0.973) at T1024
    under the study's contract, x0.956 at T2048 under v1, unchanged at T512 and on the 9.84M decoder at
    T128 (GCC x0.961 and x0.968, each meeting its control). The GEO study's attention model on level 3,
    built with GCC as the study builds it and lent its attention's regions (`contig_level3.py`): x0.934
    (0.923-0.945) of the build before at T2048 (x0.948 against the control's arm), unchanged at T1024
    and T512; at T2048 now x1.016 (0.990-1.043) of eager PyTorch's time and x1.08 of Inductor's, from
    x1.086 and x1.15 on this host.
  - The design runs, recorded with the slice (on the earlier host): copies of every operand x0.895-0.907
    of the attention at B4/T2048 (the outputs alone x0.898, the inputs alone x0.885); B1/T4096 with 8
    heads of 64 x0.656 (the contiguous layout x0.619); T512 slower (the backward x1.10); the task order
    remapped batch-fastest nothing once the tasks copy; the GEO training update with a thread-local
    prototype x0.904 of the build before.
  - What remains: the copies' own cost (x0.934 against the contiguous layout's x0.922 for the GEO shape
    at T2048, x0.850 against x0.792 for 8 heads of 64 at T1024), which a layout the step keeps -- its
    projections writing each head's rows contiguous -- would close; the GEO shape at T1024 and on one
    thread; under GQA the forward's K and V copied once a query head.
  - Witnesses: the tensor entries with a workspace against the entries without one, bit for bit -- the
    context, the verdict, the stored probabilities and the score adjoints left in them, dq, dk, dv and
    the scratch forward's context -- lent the pool's need from a 64-byte boundary and from the address
    after one, a byte short of it and none, every byte outside the regions untouched, on every instance
    and contract, finite and not (a key no context reaches among them), serially and on pools of two,
    four and 64 workers; the threshold's own context in every build, every context in the C harness's
    new -contig build; the regions' size rule; the trainer's need against the GEMMs' packs and the
    regions exactly, the binary32 refusals and the overlap with a caller's buffer; a T1024 binary32 case
    through the trace, the program executor and each of the planner's choices; the manifest's
    ThreadSanitizer builds of the harness copy at every context (`BCIR_ATT_CONTIG_FROM` 1), so the
    workers' regions run under the sanitizer -- pools of two, four and 64 workers, no race, the injected
    partition fault still reported (the default build's TSan run skips the one context that copies).
    `att-contig.json` AC1-AC23 and `att-contig-px.json` AX1-AX8, each caught by its own check and
    compiling warning-free under GCC 13, clang 18 and clang 23; `geo3-py.json` GP10 (the planner lending
    a workspace where nothing packs or copies); GW9, IA16, AB4, AF10, AF14, AF16, AF17, NZ4, PI5, PT4,
    SW6, PK24 and PM5 re-anchored to the lines this moved, each keeping its defect, caught again.
  - Not claimed: the threshold is the measuring hosts' choice, and no value depends on it; the bf16
    attention (it packs its own operands); a KV head's rows already contiguous (one KV head) are copied
    as well; AMX timing; a GPU.
  FWD-STRIP (2026-10-23): the stored training forward a strip of rows at a time, every value as it
    was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §11,
    [measurements/2026-10-23/fwd-strip/](research/measurements/2026-10-23/fwd-strip/README.json)).
  - The loss it answers: against PyTorch's scaled_dot_product_attention on the GEO attention model's
    shape (B4, 4 heads of 32, T2048, four threads), BCIR's backward ran at parity with the flash
    kernel's (39.5 ms against 38.0) and its stored forward at twice the flash forward's time (32.5 ms
    against 16.6); PyTorch's math kernel, which stores the probabilities as BCIR does, took x4.9 of
    BCIR's attention. Timed a phase at a time, the forward's fill took 15% of it, the scores 29%, the
    softmax 33% and the context 20%: each phase took the head's whole t x t probabilities, 16 MB at
    T2048, in turn. Per-task cycles were the same on one thread as on four -- a core's own traffic, not
    the pool's.
  - FWD-STRIP: from `BCIR_ATT_STRIP_FROM` positions (1024) a stored forward runs each (batch, query
    head) a strip of rows at a time in place -- a strip's rows' future columns zeroed first, then
    INFER-ATTN's `att_strip` (its scores, softmax and context) on the stored probabilities' own rows
    while they are in the core's cache, a strip `att_strip_rows` rows (256 KB). `att_strip` computes a
    strip's rows bit for bit as the head-at-a-time forward does and every future column ends +0.0 either
    way, so the stored probabilities, the context and the verdict are as they were; a task on
    ATT-CONTIG's copies strips them too. A probe of the unit (`bcir_tensor_attention_strips_probe`, the
    harness's builds) counts the heads that ran in strips: the path the values cannot show.
  - Every value is the same: BWD-LONG's 48 backward and ATTN-RATE's 88 forward cases byte for byte
    before and after under clang 18 and GCC 13, and again with the tree after built to strip at every
    context; 288 stored forwards at T1016 to T2050 (p after a NaN fill, every float, future columns
    included, the context and the verdict; GQA, head widths 4 to 64, every contract, one thread and
    four, with and without a workspace, finite and not) byte for byte before and after; the workspace
    entries the entries without one in ATT-CONTIG's 564 cases; the GEO model's trained state the same
    digest from both level-3 builds at every context.
  - Measured on the committed tree against the one before it, both lent the pool's need, 48 rounds a
    row: the attention's forward at B4/T2048 x0.868 (0.839-0.898) on four threads and x0.831 on one, GCC
    x0.921, GQA (4 heads, 2 KV heads) x0.923; at T1024 x0.925 under clang (meeting its control) and
    x0.924 under GCC; the whole attention x0.962 on four threads (meeting its control) and x0.931 on
    one; T512 and 8 heads of 64 at T1024-T4096 at parity. The training step at the GEO dimensions read
    x0.966-0.977 at T1024 and T2048 under both compilers and contracts, none resolved at 48 rounds; the
    T2048 rows, run again with 64 rounds as declared before they ran, resolved: x0.949 (0.935-0.963)
    under clang and x0.951 (0.934-0.967) under GCC. The GEO study's attention model on level 3, built
    with GCC as the study builds it and both builds lent their regions, read x0.970 at T2048 and x0.968
    at T1024 in 32 rounds (each meeting its control), and x0.951 (0.939-0.963) at T2048 in the declared
    64 -- now x0.990 of eager PyTorch's time (parity) and x1.086 of Inductor's.
  - The design runs, recorded with the slice: the strips with plain stores x0.898 of the forward, with
    streaming stores for the zeros x0.847 and the streaming zeros alone x0.929 at T2048; strips of 32K,
    128K and 256K floats no better than 64K; the streaming zeros against plain stores directly x0.959 on
    four threads (meeting its control) and x0.930 on one -- not taken, since the trainer runs four
    threads and the stores would need a fence and its own witness.
  - What remains: the stored forward at x1.7 of the flash forward's time -- its probabilities written
    out, half of them the future's zeros, each line read before it is written; 8 heads of 64, where the
    scores' products outweigh the probabilities' traffic.
  - Witnesses: a harness test at the threshold's own context and either side of it (T1023, T1024, T1031)
    against the reference loops under every contract, p NaN-filled first, finite and with an infinite
    key, serially and on a pool of four, with and without a workspace, with the probe; the harness's
    -long and -contig builds and the manifest's ThreadSanitizer builds strip at every context, so the
    attention references and the sanitizer hold the strips at every shape. `fwd-strip.json` FS1-FS9,
    each caught by its own check and compiling warning-free under GCC 13, clang 18 and clang 23.
  - Not claimed: the threshold is the measuring host's choice, and no value depends on it; the scratch
    forward (INFER-ATTN's strips already); the bf16 attention; AMX timing; a GPU.
  P-LOWER (2026-10-23): the stored training forward leaves the probabilities' future columns unwritten,
    every value read as it was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §12,
    [measurements/2026-10-23/p-lower/](research/measurements/2026-10-23/p-lower/README.json)).
  - The loss it answers: FWD-STRIP left the stored forward at x1.7 of the flash forward's time, its
    strips zeroing every row's future columns before the scores -- half of the head's t x t
    probabilities, 8 MB at T2048, written only to be ignored. The binary32 backward reads no
    probability past a row's diagonal under any contract: its register blocks read a few columns past
    it inside their diagonal tiles and discard them under their masks, and the GEMM kernel's masked
    products skip masked terms.
  - P-LOWER: `BCIR_TENSOR_ATTENTION_LOWER` (16) is the caller's word that it reads each row's
    probabilities j <= i alone; under it a stored forward may leave p's future columns as p held them.
    From `BCIR_ATT_STRIP_FROM` the strips skip their fill (the score blocks still store +0.0 in the
    future lanes of the 32-wide key blocks they reach); below it the head-at-a-time forward writes every
    column as before; y, the verdict and every probability j <= i are the same bit for bit. The trainer
    runs its stored binary32 forward under the bit (`attention_forward_contract`); PX-1's program
    check, the pack's lowering and the Python launch plan state it on the binary32 forward's launch;
    the trainer's header says what a backward now leaves in each probability region (the score
    adjoints at j <= i, the rest +0.0 or what the region held). A scratch forward and the bf16 entry
    ignore the bit: the bf16 backward reads the future +0.0s as its tiles' terms.
  - Every value read is the same: 208 cases of the stored forward under the bit and the backward from
    it (y, the verdict, p at j <= i, dq, dk, dv; GQA, head widths 2 to 64, every contract, one thread
    and four, with and without a workspace, p NaN-filled first) byte for byte against the tree before
    (which ignores the bit) under clang 18 and GCC 13, and with the tree after built to strip at every
    context; the paths without the bit -- BWD-LONG's 48 backward cases, ATTN-RATE's 88 forward cases and
    FWD-STRIP's 288 long-context stored forwards (every float of p) -- byte for byte before and after; the
    workspace entries the entries without one in ATT-CONTIG's 564 cases; the GEO model's trained state
    the same digest from both level-3 builds at every context.
  - Measured on the committed tree against the one before it, the bit in both arms' word, both lent the
    pool's need, 48 rounds a row: the attention's forward at B4/T2048 x0.817 (0.788-0.848) on four
    threads, x0.826 on one, GCC x0.847, GQA (4 heads, 2 KV heads) x0.849, v1 x0.859; at T1024 x0.851
    under clang and x0.878 under GCC; 8 heads of 64 x0.934 at T1024 and x0.92-x0.93 at T2048 and T4096
    (meeting their controls); T512 at parity, below the threshold. The whole attention x0.941 on one
    thread and x0.944 under GCC; on four threads under clang x0.915, its interval meeting its control's
    by 0.001, and in the 64 rounds declared before they ran x0.949 (0.920-0.979). The training step at
    the GEO dimensions at T2048 x0.937 under GCC and x0.953 under clang's v1 contract; under clang's
    kblock+exp+reduce x0.970 beside a control that drifted to x0.972, and in the declared 64 rounds
    x0.948 (0.931-0.966); T512, T1024 and the 9.84M decoder (T128) at parity. The GEO study's attention
    model on level 3, built with GCC as the study builds it, both builds lent their regions and stating
    the bit on their stored forward: x0.960 (0.942-0.977) at T2048 in 32 rounds -- now x0.964
    (0.944-0.983) of eager PyTorch's time and x1.063 of Inductor's; T1024 x0.981 (meeting its control),
    T512 at parity.
  - Against PyTorch's scaled_dot_product_attention on the same inputs (B4, 4 heads of 32, four threads,
    24 rounds), the attention as the trainer runs it: at T2048 BCIR 60.6 ms against the flash kernel's
    56.0, flash x0.968 (0.903-1.039) of BCIR's -- parity, from x0.82 before FWD-STRIP -- BCIR's backward
    the faster (38.3 ms against 39.2) and its forward the slower, flash's x0.813 (0.758-0.872) of BCIR's
    (17.0 ms against 21.8); at T1024 parity too (flash x1.072, 0.993-1.157); PyTorch's math kernel,
    which stores the probabilities as BCIR does, x5.5 of BCIR's time.
  - The prototype first: the strip forward's fill removed, its stored forwards' y and verdict and the
    backward's dq, dk and dv from p NaN-filled before the forward, 176 cases a build on the AVX-512, AVX2
    and portable instances under clang and GCC, each the committed tree's; timed at x0.833 of the
    forward and x0.938 of the whole attention at B4/T2048 in 32 rounds.
  - What remains: the stored forward at x1.23 of the flash forward's time -- the probabilities j <= i
    written out for the backward to read, where the flash kernel recomputes them; the head-at-a-time
    forward below the threshold, which still writes its future columns; the bf16 forward, whose backward
    reads them; 8 heads of 64.
  - Witnesses: the harness's FWD-STRIP test runs each forward once more under the bit, p filled with a
    NaN canary -- y, the verdict and every probability j <= i the reference loops', every future column
    +0.0 or the canary, the canary still in the last column of every row whose strip ends before it, and
    the backward from the canaries' probabilities the reference's; `attention_lower` holds the same on
    the reference shapes under every contract, finite and not, serially and on a pool, with and without
    a workspace (the harness's -long and -contig builds strip at every context: 4,187,904 canaries kept
    there); `trainer_lower` trains steps from activations, scratch and workspace whose every float is a
    NaN canary and gets the zeroed arenas' loss, gradients and logits bit for bit, the canary left
    exactly where a binary32 forward's strips leave it -- every other float of the arena the step
    writes. The tensor unit's probe records the bit; the trainer's probe checks expect it on every
    stored binary32 forward. `p-lower.json` PL1-PL5 and `p-lower-px.json` PL6-PL9, each caught by its
    own check and compiling warning-free under GCC 13, clang 18 and clang 23; FS1, FS8, IA12, IA15, IA16
    and PI5 re-anchored on the lines the slice rewrote and caught again.
  - Not claimed: below the threshold, where the head-at-a-time forward writes every column; the bf16
    attention; AMX timing; a GPU.
  BWD-KEEP (2026-10-23): the attention backward keeps every probability-adjoint tile of a row block,
    every value as it was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §13,
    [measurements/2026-10-23/bwd-keep/](research/measurements/2026-10-23/bwd-keep/README.json)).
  - The loss it answers: timed a part at a time with the trainer's job clock, the GEO step at T2048
    spent 41% of its time in the attention's backward, 23% in its forward and about a quarter in the
    GEMMs. The backward evaluates a row block's probability-adjoint tiles (dP = dY V^T, 32 rows by 64
    columns) once for the rows' dots and again for their score adjoints; it kept the first 256 columns'
    tiles on its stack (32 KB) and evaluated every tile past them twice -- three quarters of a row
    block's tiles at T2048.
  - BWD-KEEP: given a region of the lent workspace (ATT-CONTIG's, from `BCIR_ATT_CONTIG_FROM` positions
    where a row holds more than one head), a backward task keeps every tile of a row block there -- 32
    rows of the context's width in whole tiles of 64 columns, after the seven copies -- so its second
    pass reads every tile its first evaluated and none is evaluated twice. A kept tile holds the bits
    its second evaluation would have, so dq, dk and dv are as they were; without a region (no
    workspace, or less than the pool's need) a task keeps the stack's first 256 columns and evaluates
    the rest twice, as before. The region grows by the kept tiles (at T2048 and a head width of 32, a
    seventh), and the trainer's need with it.
  - Every value is the same: 208 cases of the stored forward under LOWER and the backward from it, and
    BWD-LONG's 48 backward cases, byte for byte against the tree before under clang 18 and GCC 13, and
    with the tree after built to keep its tiles at every context; ATTN-RATE's 88 forward cases and
    FWD-STRIP's 288 long-context stored forwards byte for byte before and after; the workspace entries
    the entries without one in ATT-CONTIG's 564 cases; the GEO model's trained state the same digest
    from both level-3 builds at every context.
  - Measured on the committed tree against the one before it, both lent the larger of their two needs,
    48 rounds a row: the attention's backward at B4/T2048 x0.905 (0.866-0.946) on four threads and
    x0.865 (0.843-0.888) in the 64 rounds declared before they ran, x0.874 on one thread, v1 x0.854,
    GQA (4 heads, 2 KV heads) x0.868, GCC x0.877; at T1024 x0.898 under clang and x0.877 under GCC;
    8 heads of 64 x0.874-x0.875 at T2048 and T4096 (x0.902 at T1024, meeting a control that drifted to
    x0.958); T512 at parity, below the threshold. The whole attention x0.916 on one thread, x0.915 with
    GQA, x0.924 under GCC and x0.931 at T1024; on four threads under clang x0.953, meeting its control's
    interval by 0.006 beside a forward -- which the slice does not touch -- reading x1.034 inside its
    own, and in the declared 64 rounds x0.914 (0.896-0.933), the forward x1.002. The training step at
    the GEO dimensions at T2048 x0.944 under GCC and x0.955 under clang's v1 contract; under clang's
    kblock+exp+reduce x0.955, clearing its control by 0.0002, and in the declared 64 rounds x0.937
    (0.919-0.957); T512, T1024 and the 9.84M decoder (T128) not resolved. The GEO study's attention
    model on level 3, built with GCC as the study builds it, both builds lent their regions: x0.946
    (0.931-0.961) at T2048 in 32 rounds -- now x0.911 (0.892-0.930) of eager PyTorch's time and x1.005
    (0.983-1.029) of Inductor's, parity; T1024 x0.967 (meeting its control), T512 at parity.
  - Against PyTorch's scaled_dot_product_attention on the same inputs (B4, 4 heads of 32, four threads,
    24 rounds), the attention as the trainer runs it: at T2048 BCIR 57.5 ms against the flash kernel's
    55.4, flash x0.994 (0.888-1.113) of BCIR's -- parity -- BCIR's backward the faster (35.3 ms against
    38.0) and its forward the slower (22.3 against 17.2, flash's x0.796); at T1024 BCIR the faster, flash
    x1.222 (1.168-1.279) of its time, the flash backward x1.364; PyTorch's math kernel, which stores the
    probabilities as BCIR does, x5.8 of BCIR's time at T2048.
  - The prototype first: every tile kept on the stack (a 256 KB frame, which no library may take on a
    caller's thread), BWD-LONG's 48 backward cases byte for byte against the tree's; timed in 32 rounds
    at x0.909 of the backward at B4/T2048 on four threads, x0.873 on one and x0.924 at T1024.
  - What remains: the stored forward at x1.26 of the flash forward's time -- the probabilities j <= i
    written out for the backward to read, where the flash kernel recomputes them; below the copies'
    threshold, or with one head a row, no region is lent and the tiles past 256 columns are evaluated
    twice as before; the GEMMs, about a quarter of the step.
  - Witnesses: `attention_keep` runs each backward three ways -- lent none, a byte short of the pool's
    need and the need -- at the copies' threshold and 33 positions past it (neither a whole row block
    nor a whole tile), under v1's and the fused products, serially and on a pool of four: the same bits
    all three ways, and the unit's probe (`bcir_tensor_attention_tiles_probe`) counts each path's tile
    evaluations. ATT-CONTIG's size pins hold the region's layout, a context past a whole number of tiles
    among them; the harness's -contig build keeps the tiles at every context, so the attention
    references and ATT-CONTIG's cases hold the kept path to the reference loops. `bwd-keep.json`
    BK1-BK7, each caught by its own check; AC15 and NA1-NA3 re-anchored on the lines the slice rewrote
    and caught again.
  - Not claimed: below the copies' threshold or with one head a row, where no region is lent; the bf16
    attention (its own backward); AMX timing; a GPU.
  ATT-BOUND (2026-10-23): the attention and the GEMMs against the host's measured rates; four ways at the
    attention's forward measured and not taken; the GEO model's inference read against PyTorch
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §14,
    [measurements/2026-10-23/att-bound/](research/measurements/2026-10-23/att-bound/README.json)). Records and
    instruments; no change to the runtime.
  - Inference: the GEO attention model's one forward of 4 x 2048 tokens takes 100.2 ms against eager
    PyTorch's 92.5 and Inductor's 83.2 -- x1.057 (0.969-1.152) of eager's time, x1.190 (1.116-1.269) of
    Inductor's; at T1024 x0.714 of eager's and at parity with Inductor's (x0.992). Two thirds of it is the
    attention's forward, the training forward's kernels.
  - The forward a phase at a time (TSC cycles, B4/T2048, 4 heads of 32, EXP, one core): the scores 2.07
    cycles a probability, the context 1.83, the softmax 2.26 (its max 0.25, exponential and sum 1.46,
    division 0.53). The host's rates, measured (`fma_peak.c`): 217 GFLOP/s a core in fused multiply-adds and
    102 in v1's multiply-then-add (856 and 399 on four cores) -- 94 and 44 flops a TSC cycle, the cores near
    3.4 GHz under AVX-512 load while the TSC counts 2.3. At v1's measured rate a score's or a context term's
    32 multiply-adds take 1.45 TSC cycles: the scores run at about 70% of it, the context at about 79%. The
    exponential-and-sum pass is mostly BCIR's exponential, whose sequence (about 38 vector operations for 16
    lanes) the EXP contract fixes. Corrected the same day: the first text took the TSC's rate for the
    core's and called the products' 2.0 TSC cycles their bound.
  - Measured and not taken, every value the same in each (the stored forward under LOWER with the backward
    from it, 208 cases; 88 forward cases; 288 stored forwards; clang and GCC; and with copies and strips at
    every context), 32 rounds beside an A/A control: FWD-STREAM, each stored strip computed in the region and
    streamed out with non-temporal stores -- the attention x1.12 at B4/T2048 and x1.15 at T1024 (the 260 MB
    L3 holds a layer's probabilities for the backward; streaming stores send them to memory), parity where
    they outgrow it (8 heads of 64 at T4096; the step); the strip at a padded stride copied out -- the
    forward x1.04-x1.12; SCRATCH-PAD, the scratch forward's strips padded with no copy -- parity (x0.98),
    so a power-of-two context's L1 set conflicts are not a cost; SOFTMAX-DIV, the softmax's division in 16
    lanes where the compilers divide in 4 -- no row resolved in two runs.
  - What the fused attention contract (ATTN-FMA, in the tree; not the GEO study's) would buy, one library,
    interleaved beside an A/A control: the attention x0.822 (0.781-0.865) at B4/T2048 on four threads, the
    backward x0.778 and the forward x0.895, the scratch forward x0.874; the GEO step under
    kblock+exp+reduce+attn 252.5 ms against 290.5. Every attention value differs under it (another
    arithmetic): the study's choice, recorded as a reading.
  - The GEMMs: 25.8 GFLOP in 74.3 ms of the step at T2048, 347 GFLOP/s on four cores -- 41% of the 856 the
    cores sustain in fused multiply-adds, the largest room the step has left (OT-05 found BCIR's k-blocked
    GEMMs level with MKL's on most of these shapes).
  - Not claimed: counter-based figures (this VM has no PMU; the cycles are TSC); the fused contract's
    effect on the study's trained results; AMX; a GPU.
  MM-ROWS (2026-10-23): the k-blocked GEMM's tasks, chunks, edge blocks and job order, every value as it was
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §15,
    [measurements/2026-10-23/mm-rows/](research/measurements/2026-10-23/mm-rows/README.json)).
  - The loss it answers: ATT-BOUND left the step's GEMMs at 41% of the cores' measured FMA rate. A shape at a
    time, one core ran the step's products at 46-57% of its rate and at 85-87% where A's panel fits in L2. The
    AVX-512 nest runs a 48-column block's tile over every row of its range before the next block, so each block
    reads A's rows again -- from L3 once the panel outgrows L2 (the step's 8192-row operands are 4 MiB); the
    pool's task, one block by 128 rows with the column fastest, had three cores pull the same rows of A from L3
    at once; a 128-wide product's last block is 32 columns, computed as 48; and in the step's paired and grouped
    jobs the weight adjoints -- 128 or 256 rows over a depth of 8192, three to six tasks of some 50 million
    multiply-adds -- were claimed after the input adjoints' many small tasks and ran with threads idle around them.
  - MM-ROWS: under the blocked contract a pool task spans as many whole column blocks as leave its job sixteen
    tasks (`BCIR_MM_TASKS`) -- every column from 2,048 rows -- so it reads its rows of A once for all its blocks;
    the serial nest walks 128-row chunks, every block of a chunk before the next, where A is stored by rows, its
    panel 2 MiB or more and a chunk's slice at most 512 KiB; an AVX-512 edge block computes the vectors its columns
    reach; and a pair or group job whose second part's tasks are the larger starts with them -- the pool's task t
    runs the job's task (t + lead) mod total, under every contract. The partition stays a function of m, n and the
    contract, so a chain's calls share one tiling; each element's K blocks run in order, each from zero, then
    c + alpha*P.
  - Every value is the same: `bitcheck.c` -- every k-blocked entry (the serial instances, the pool at four and two
    threads and through each instance, pair jobs, group jobs whose chains include a call of depth zero) over 712
    cases, 6,081 hashes -- before == after under clang 18 and GCC 13, and clang's lines GCC's; the GEO model's
    trained state the same digest from both level-3 builds at every context.
  - Measured on the committed tree against the one before it, in one process beside an A/A control: on four
    threads the GEO step's 8192-row products x0.734-x0.814 (all seven), 4096 and 2048 rows x0.793-x0.883, the
    paired weight adjoint x0.823 and the grouped ones x0.791 and x0.874; on one thread x0.760-x0.887, the weight
    adjoints x0.943-x0.953. The training step at the GEO dimensions under clang x0.935 at T512, x0.905 at T1024
    and x0.939 at T2048 -- x0.930 (0.915-0.946) in the 64 rounds declared before they ran -- under GCC x0.935 and
    x0.938, on one thread x0.952 (x0.956 declared); its GEMM claims, read a part at a time, 72.9 -> 61.0 ms at T2048
    (x0.837) and 252.9 -> 219.1 on one thread. The GEO study's attention model on level 3 (GCC, 32 rounds): the
    training update x0.918 at T512, x0.928 at T1024 and x0.968 (0.952-0.984) at T2048 -- x0.872 of eager PyTorch's
    time and x0.919 of Inductor's at T2048 in the same run; its inference forward at T2048 x0.945 (0.916-0.974),
    meeting its control's interval, x1.17 of Inductor's time.
  - The 9.84M decoder (1,024 rows): its step at parity (x0.988 in 64 rounds), its GEMMs within noise at four
    threads but for three small products -- the head x0.935 and 1,024 x 1,024 x 384 x0.971, and the 192-row
    weight adjoint x1.063, whose path the slice leaves as it was but for the code's placement. On one thread that
    adjoint and the 1,024 x 384 x 384 product read x1.03 and x1.02: the kernel's 8 x 48 loops now start 32 bytes
    into a cache line where they started on one, the same 40 instructions, and with both trees' loops at 64-byte
    boundaries the four such shapes read x0.999-x1.011.
  - The design runs first, every prototype bit for bit (14 of 14; the first draft, whose partition read the
    call's depth, 213 of 6,081 lines apart -- a chain's call of depth zero tiled apart from its chain): whole rows
    from 1,024 rows, tasks of whole blocks for 32 tasks and for 16, timed in one process beside an A/A control
    (separate processes moved by up to x1.6 between runs here) -- alike at the GEO shapes and within noise at the
    decoder's 1,024 rows, the 16-task rule taken; chunks at every depth to 1,024 cost the decoder's 192-row weight
    adjoint x1.080 on one thread, so they wait for a panel that outgrows L2, and a transposed A's 1,024-row
    adjoint x1.04-x1.07 even with loops aligned, so they wait for A stored by rows (the first measured commit,
    6ea3244, chunked it; its runs are kept as superseded); ordering took the step's paired jobs from x0.96 to x0.79
    of their time.
  - What remains: the weight adjoints -- 128 to 260 rows over a depth of 8192, about half the core's rate with
    their K blocks' tiles in L2, and three to nine tasks -- whose next step is a split over the depth with the K
    blocks folded in order, which needs the trainer to lend the products' room; the kernel's loop placement, which
    moves its rate by a few percent with any edit of the unit; the bf16 contract's GEMMs (their own partition).
  - Witnesses: the harness's `gemm_kblock_rows` -- 2,100, 1,030 and 1,300 rows: the serial entry and each
    instance, the pool and each instance on it at two to four threads, the pair both ways round, a chain with a
    depth of zero beside two weight adjoints, edge blocks of one, two and three vectors, all against the K-blocked
    fmaf chains -- where no case before reached a blocked GEMM of 1,024 rows on a pool: without it `mm-rows.json`'s
    MR1, MR2 and MR8 pass every test. `mm-rows.json` MR1-MR8, each caught by its own check; NK7 re-anchored on the
    moved row loop (in its table and in `test_the_rows_fire_on_wrong_kernels_and_plans`) and caught again;
    ndt-kblock (18), ndt-threads (18), ndt-fma (5) and PX-3b's PI2 swept whole against the new tree.
  - Beside it (`246dbfe`): CI's CMake build job's hang guard 40 -> 60 minutes -- on 535eac7 its gcc cell ran 21
    minutes on one runner and reached 40 on two others, its build-parity step 970 seconds against 1,882.
  - Not claimed: timing on AVX2-only hosts (their instance takes the tasks, the chunks and the order, not the edge
    width); the bf16 contract's GEMMs; AMX; a GPU.
  SCORES-T (2026-10-23): the inference forward's 16-row groups with their rows in the lanes, every value as it was
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §16,
    [measurements/2026-10-23/scores-t/](research/measurements/2026-10-23/scores-t/README.json)).
  - The loss it answers: after MM-ROWS the GEO model's inference at T2048 took x1.17 of Inductor's time, two thirds
    of it the attention's scratch forward. The scores landed with the keys in the lanes (8-row x 32-key blocks into
    the strip's rows) while the softmax needs each row's chain in a lane of its own, so it transposed every 16 x 16
    block in registers once for the maximum and twice for the exponential and sum, and the score blocks packed each
    32-key panel afresh for every 32-row strip.
  - SCORES-T: on the AVX-512 instance under the exp contract a scratch strip runs a 16-row group at a time with its
    rows in the lanes -- the group's queries transposed once into a d x 16 panel, its scores 16 x 16 blocks with one
    accumulator a key (each score its channels' products in order from +0.0), the softmax on the blocks as they
    stand, the context reading a key's eight rows' probabilities side by side; heads to 256 channels, strips ending
    on a whole group; every other case, the libm softmax and the stored forward keep the row path. A probe counts
    the groups run in the lanes.
  - Every value is the same: `scratch_hash.c` (the scratch forward's context and verdict, the stored forward's every
    byte of p under LOWER too, 1,248 cases with a pool of four and the workspace lent) and the tree's fwd, lower,
    attn and band hashes, before == after under clang and GCC, clang's lines GCC's.
  - Measured on the committed tree against the one before it, in one process beside an A/A control: the scratch
    forward at B4/T2048 x0.927 on four threads (GCC x0.917) and x0.920 on one, x0.884 at T512, x0.848 at T256, x0.891
    at T128, x0.859 under the fused attention contract, x0.889 for 8 heads of 64; `bcir_decoder_forward` at the GEO
    dimensions x0.927 at T1024 and x0.906 at T2048. The GEO study's attention model on level 3 (GCC, 24 rounds):
    inference x0.945 at T512, x0.916 at T1024 and x0.924 at T2048 -- now x1.056 (0.991-1.126) of Inductor's time
    at T2048, at parity, and x0.977 of eager PyTorch's; x0.839 of Inductor's at T1024. The 9.84M decoder's forward
    (T128, its GEMMs most of it) at parity; its attention x0.883.
  - The prototype's phases at B4/T2048 (TSC cycles a probability): the scores 1.96 -> 1.62, the softmax 2.37 -> 1.82
    (the maximum 0.25 -> 0.18, the exponential and sum 1.54 -> 1.13, the division 0.54 -> 0.44), the context 1.70
    -> 1.56.
  - Measured, not taken: the stored training forward on the same path, its groups transposed out to p's rows
    (every byte of p the same). The transposes out cost 0.79 cycles a probability, which the row path spends inside
    its other phases: x1.014 at T2048 and x0.940 at T1024, meeting its control.
  - The backward a phase at a time (B4/T2048, four threads, 36.2 ms a layer), against v1's 1.45 cycles a product:
    dV 1.92 (75%), the dP tiles 2.02 (72%), the dots 0.68, the score adjoints 0.46, dQ 1.55 (93%), dK 1.76 (82%), the
    copies and the rest 1.15 (12%); prefetching the copies read 0.78, the call's time within noise.
  - Witnesses: the C harness's `attention_lanes` -- the scratch forward against the reference loops under the four
    contracts over head widths 16 to 257, contexts 16 to 80 on every instance and 1030 and 1040 (past the strip
    threshold) on AVX-512, a pool of four, the workspace lent, finite and not, the probe's count of the groups in the
    lanes; `scores-t.json` ST1-ST12, each caught by its own check, ST10 (the probe's count) and ST12 (a 17-channel
    context) only by the new case. The long contexts ran on every instance at first; on a contended CI shard that
    pushed the harness's C11 build past its 60-second limit, and on the other instances they repeat the row path
    `attention_scratch` and `attention_strips` already cover, so they run on AVX-512 alone and the limit is 120 seconds.
  - Not claimed: AVX2-only and non-x86 hosts (the row path); the stored forward; the bf16 attention; a GPU.
  SOFTMAX-T (2026-10-23): the exponential's single scaling and the softmaxes in one pass, every value as it was
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §17,
    [measurements/2026-10-23/softmax-t/](research/measurements/2026-10-23/softmax-t/README.json)).
  - The loss it answers: after SCORES-T the scratch forward's softmax took 1.82 TSC cycles a probability, 1.13 of them
    the exponential and sum and 0.44 the division, a pass of its own. The AVX-512 exponential and v2 SiLU forms were
    exp1's and the sigmoid's scalar code compiled for sixteen lanes -- 2^k two exponent-field scalings, each chain a
    vector at a time; ATTN-RATE's block forms took clang's SiLU forms to x0.67-x0.91 and left GCC's at x0.97-x1.02 --
    and the softmaxes scaled, exponentiated and summed in separate passes over their blocks.
  - SOFTMAX-T: on AVX-512 the exponential applies 2^k by one vscalefps -- exp1's (p * 2^k1) * 2^k2 rounds once, the
    first product exact, as vscalefps rounds p * 2^k once -- and the exponential, SiLU, SiLU times u and the SwiGLU VJP
    run as AVX-512 instances, each step over four vectors before the next (EXP-LANES' order), so both compilers
    interleave the chains; the scratch forward's lanes softmax and the stored forward's transposed blocks scale,
    subtract, exponentiate, mask and sum in one pass, four columns at a time, each sum in column order; and the lanes
    path divides each key's probabilities inside the context's first pass -- rows 0-7's pass the keys to R+7, rows
    8-15's the eight after -- once, before the context reads them. Every other instance and the libm softmax as they
    were.
  - Every value is the same: over every binary32 the exponential and the three SiLU forms (apart, in place and
    overlapping) against their scalar definitions under clang and GCC, 0 mismatches (the tree before the same, the
    instrument's control), and before against after byte for byte, 0 apart; `scratch_hash.c` and the fwd, lower, attn,
    band and elementwise hashes before == after under clang and GCC, clang's lines GCC's. At a NaN input the SiLU forms
    return NaNs, as before; under GCC 50,331,642 of those NaN outputs -- three for each of the 16,777,214 NaN inputs --
    carry another sign or payload than before, under clang none: the compiler orders a product's operands, and which NaN
    is not part of the contract, which the header now says.
  - Measured on the committed tree against the one before it, in one process beside an A/A control: on one thread the
    exponential x0.733 (clang) and x0.678 (GCC) at 16,384 elements, x0.769 and x0.745 at 2^20; SiLU times u x0.905 and
    x0.882 under clang, x0.695 and x0.748 under GCC; the SwiGLU VJP x0.860 and x0.940 under clang, x0.654 and x0.782
    under GCC -- GCC's forms now at clang's rate; the AVX2 instance, which the slice leaves, x1.003 and x1.008. The
    scratch forward at B4/T2048 x0.914 on four threads (GCC x0.909) and x0.938 on one, x0.927 at T512, x0.899 under the
    fused attention contract; at T1024 x0.947 and for the 9.84M decoder's attention x0.950, each meeting its control, at
    T256 x0.937 holding 1; heads of 64 and 128 channels at parity. The stored forward x0.965 (0.937-0.994), meeting its
    control. `bcir_decoder_forward` at the GEO dimensions x0.908 at T2048 under GCC; under clang x0.965 in 64 declared
    rounds (0.946-0.983), at its control's edge. The training step within noise but at T512 (x0.974, 0.954-0.995):
    x0.971-x0.998 elsewhere. The GEO study's attention model on level 3 (GCC, as the study builds it), its logits and
    trained state the same: inference x0.931 (0.908-0.954) at T2048 in 48 declared rounds (x0.926 in 24) -- x1.005 of
    Inductor's time, at parity, and x0.899 of eager PyTorch's -- x0.960 at T1024, meeting its control, and x0.976 at
    T512, holding 1; its training update at parity at every context, x0.982 of Inductor's time at T2048.
  - The design runs, each level against the tree before (the scratch forward at B4/T2048, four threads, 24 rounds): the
    single scaling and the four-chain forms x0.942 under clang and x0.937 under GCC; with the one-pass softmaxes x0.994
    and x0.922; with the division inside the context x0.953 and x0.870, and x0.981 and x0.946 against the one-pass level
    alone. Its phases (TSC cycles a probability): the softmax 1.80 -> 1.15 (the exponential and sum 1.12 -> 0.93; the
    division's 0.44 now 0.28 inside the context, 1.56 -> 1.84), the call 5.44 -> 4.96. A first draft wrote each vector's
    chain as one expression, which clang scheduled a chain at a time: its SiLU x1.06 slower than the tree's, the stored
    forward x1.042 (holding 1); its runs are kept as superseded.
  - Witnesses: the C harness's `exp_scalef` -- every binary32 where the result rounds, [-104, -87] (the subnormal
    results) and [88, 89] (the overflow edge), 2,359,298 inputs through every instance against bcir_tensor_expf bit for
    bit, apart and in place, every eighth run of 4,096 of them through the v2 SiLU forms against their scalar
    definitions, serially and on a pool, and NaNs their sign and payload; 0.5 s on this host. `softmax-t.json` SM1-SM16,
    each caught by its own check (SM15's first edit left the maximum unused, which the -Werror build refused; it keeps
    it named and was swept again). Without the new case all sixteen are still caught -- the exponential's contract case
    (every 256th binary32 of [-104, 89] through every instance, the thresholds and their neighbours) and the attention
    cases reach each -- and on its own it catches nine of the ten in the exponential and the SiLU forms, all but the
    overlapping output's order (SM8, the contract case's), and is the first to stop SM6 (a NaN's payload), which
    otherwise reaches attention_strips' byte comparison. ndt-lanes.json NL8 and NL11 and scores-t.json ST5, ST6 and
    ST12, whose anchors the slice rewrote, re-anchored with the same defects and caught again.
  - Not claimed: AVX2-only and non-x86 hosts (their instances as they were); the bf16 attention; a GPU.
  COMPILE-LIMIT (2026-10-24): every P-IND case audited for how it compiled; the comparisons' protocol corrected and
    the affected ones rerun; records and docs only ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §18,
    [measurements/2026-10-24/compile-limit/](research/measurements/2026-10-24/compile-limit/README.json)).
  - The fault: `geo_compare.py`, `train_compare.py` and `infer_compare.py` ran their cases in one process, each case's
    P-IND module through `oracle.torch_model_next`'s one forward code object. Dynamo compiled a process's first case
    at its own shapes and every later one under automatic dynamic shapes, the tokens' time dynamic; past its recompile
    limit (8) it compiled nothing and ran the module eagerly, warning once. Of the 64 P-IND cases in the branch's
    records 20 compiled at their own shapes, 37 dynamic in time, and 7 -- GEO-2's attention training at T512 and all
    six of its inference cases -- eagerly: their P-IND is eager PyTorch's time. `tools/perf/ot_torch.py` already
    compiled each shape from a fresh Dynamo state, so the OT ledger's rows stand.
  - The protocol: `torch._dynamo.reset()` before every case, `torch.compile(dynamic=False)`,
    `fail_on_recompile_limit_hit` (a fallback is a failure, never a timing), each compile's shape symbols and graphs
    recorded and checked. `witness.py` shows the mechanism -- the second compile dynamic in time, the ninth eager, the
    code object eager from then on -- and, from a fresh state with the flag set, the ninth call raising.
  - The dynamic rows stand: on SOFTMAX-T's level-3 build the dynamic-time compile against the fixed-shape one in one
    process took x0.998, x0.975 and x0.969 of its time in training at T512-T2048 and x1.016, x1.003 and x0.992 in
    inference, every interval holding 1. Against the fixed-shape compile BCIR's update took x0.608, x0.795 and x0.838
    of Inductor's time (x0.893 in 32 declared rounds at T2048, where SOFTMAX-T's record read x0.982: PyTorch's time
    moves between processes by more than a run's interval), its inference x0.593, x0.853 and x0.995 -- parity at T2048.
  - GEO-2's comparison rerun with every case compiled at its own shapes, on the session's later host (the AMX Xeon):
    level 2 trains the three models in x0.33-0.83 of Inductor's time (attention T512 x0.831, where GEO-2's eager row
    read 0.50) and x0.15-0.61 of eager PyTorch's; the geometric heads' inference x0.77-0.90 of Inductor's time (global
    T2048 not resolved) and x0.32-0.36 of eager's -- GEO-2's 3.2-4.2 times faster than P-IND was against eager
    PyTorch; the attention baseline's inference x1.39 of Inductor's time at T1024 (GEO-2: parity) and x1.67 at T2048.
  - Not claimed: PyTorch's run-to-run variation as a property of either rail (the paired rounds hold within a process);
    a compile of the geometric heads at other than their fixed shapes.
  GEO-AMX (2026-10-24): the PR818 study's bf16 arm timed on AMX, against PyTorch's bf16; records and docs only
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §19,
    [measurements/2026-10-24/geo-amx/](research/measurements/2026-10-24/geo-amx/README.json)).
  - The open item it closes: GEO-2 reran the study's bf16 runs bit for bit on a host without AMX, so their timing on
    the unit the contract was shaped for stayed open (the review note's open item 1).
  - On the AMX Xeon, the study's three models (level 2's overlays linked with this branch's tensor unit, `build_amx.py`
    checking every entry's prototype in both headers), 24 rotated rounds beside an A/A arm: the bf16 contract (26, the
    GEMMs on bfloat16 operands through the AMX instance) x0.676-x0.888 of binary32's time for the attention model at
    T256-T2048 and x0.719-x0.784 for the geometric heads; with the bf16 attention too (90) x0.503-x0.585 at every
    context. Codex's AMD host, without AMX, read x1.14 and x1.04. Every arm deterministic; the first update's loss
    within 1.3e-4 of binary32's.
  - Against PyTorch's mixed precision (`torch.autocast` to bfloat16, oneDNN's AMX matmuls), eager and compiled at fixed
    shapes (COMPILE-LIMIT's protocol), 16 rounds: BCIR's bf16 trains the attention model in x0.244-x0.727 of Inductor's
    bf16 time and the geometric heads in x0.267-x0.373. Autocast speeds PyTorch's attention model at T1024 and T2048
    only (x0.85 and x0.78 of its binary32 time); BCIR's bf16 GEMMs alone, the attention in binary32, take x1.14 of
    P-IND-BF16's time at T2048.
  - Not claimed: bf16 quality beyond the study's own records (GEO-2 holds them bit for bit); a GPU.
  BWD-T-0 (2026-10-24): the attention backward's gap placed; BWD-VT measured and not taken; records and docs only
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §20,
    [measurements/2026-10-24/bwd-vt/](research/measurements/2026-10-24/bwd-vt/README.json)).
  - The step at T2048 after SOFTMAX-T: 117 ms of its 280 ms of jobs the attention backward (41.9%), 69 the attention
    forward, about 65 the GEMMs. The backward a phase at a time (B4/T2048, TSC cycles a probability): dV 1.97-2.35
    (61-73% of v1's measured rate), the dP tiles 1.98-2.08 (69-73%), dQ 1.53-1.68 (86-94%), dK 1.76-1.91 (76-82%), the
    dots, score adjoints and copies 2.3-2.6 together; one thread the same. Their two kernels on cache-resident operands
    (`tr_micro.c`, `dp_micro.c`, interleaved repeats, clang and GCC) run at medians of 95-109% of v1's rate at every row
    stride tried: dV and the dP tiles lose their quarter to moving their operands, not to their arithmetic.
  - BWD-VT, measured and not taken: V's slot in the backward's contiguous task holding the values transposed, so no dP
    tile transposes them again (every hash the same, clang and GCC): the backward x0.946 at B4/T2048, its interval
    meeting the A/A control's; at parity at one thread, T1024, heads of 64, the fused contract and under GCC.
  - Not claimed: counter-based figures (no PMU on this VM); the next slice -- it starts from these readings.
  GEO-NOW (2026-10-24): the study's models on this branch's tensor unit; three levers measured and not taken; records
    and docs only ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §21,
    [measurements/2026-10-24/geo-now/](research/measurements/2026-10-24/geo-now/README.json)).
  - The study's three models on this branch's unit against main's (level 2), the same overlays and headers, in one
    process with every trained state and logit the same bytes: training x0.85-x0.96 at T128-T512, inference x0.78-x0.85
    at T1024 and T2048. Against Inductor at fixed shapes the geometric heads train in x0.31-0.54 of its time and run
    inference in x0.67-0.75 of it (global at T2048 at parity); the attention baseline's level-2 overlay, without the
    scratch forward, x1.35 at T2048 (the level-3 build is at parity: COMPILE-LIMIT).
  - THP-0, not taken: the attention's buffers on transparent huge pages, the backward x0.985 at B4/T2048 in 48 rounds.
  - WADJ-ROWS-0, not taken: the weight adjoints in row ranges small enough for sixteen tasks, x1.19-x1.56 slower (each
    task packs the deep operand again); in the step each runs beside its input adjoint at x0.95 of the forward
    products' rate already.
  - KBK-0, not taken: the k-blocked kernel's 448-term block in L1-sized sub-ranges (the same bits) x0.974-x1.017 --
    the block runs at 188-200 GFLOP/s with sixteen row blocks to a tile, 87-92% of the core's rate; one block called
    alone reads 115-120.
  GEMM-NEST-0 (2026-10-24): where the k-blocked nest loses its block's rate; two prototypes not taken; records and docs
    only ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §22,
    [measurements/2026-10-24/gemm-nest/](research/measurements/2026-10-24/gemm-nest/README.json)).
  - The nest runs M x 128 x 128 at 174-178 GFLOP/s on one core while A and C fit in L2 (to 1,024 rows) and at 141-142
    from 4,096: A's and C's movement from L3.
  - NOPACK-0 (an untransposed B read in place, not packed a chunk) at parity on the step's products and x1.49-1.54
    slower where B's rows are 4 KiB apart; GEMM-PF-0 (prefetches of the next block's A and the block's C) x0.94-0.95 on
    one thread at 8,192 rows but x1.27-1.30 slower for the decoder's 1,024 x 384 x 384 on four threads. Every value the
    same in both (bitcheck.c's 712 cases); neither taken.
  P-BANDS (2026-10-24): the stored attention's probabilities in 16-row bands with the rows in the lanes, every value as
    it was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §23,
    [measurements/2026-10-24/p-bands/](research/measurements/2026-10-24/p-bands/README.json)).
  - The loss it answers, placed first (BWD-NEXT): the backward a phase at a time on the tree before, B4/T2048, four
    threads, 10.1 TSC cycles a probability -- dV 2.30, the probability-adjoint tiles 2.16, the dots 0.87, the score
    adjoints 0.48, dQ 1.61, dK 1.92. The transposed products walked 32 rows of P at once and the dots transposed every
    16 x 16 block; eight band and prefetch variants of the row walk moved the backward within its runs' spread at four
    threads, while the same product over P in 16-row bands ran 1.51 TSC cycles a probability resident and 1.84 streamed
    against the walk's 1.78 and 2.17.
  - P-BANDS: `BCIR_TENSOR_ATTENTION_BANDS` (32) lays a stored forward's probabilities out in 16-row bands, row i's
    probability of key j at (i/16)*16*T + j*16 + i%16 of its head's region, a band's keys past its last row +0.0 or,
    under LOWER, as p held them. It takes effect where the host's widest instance of the products' family is AVX-512,
    T is a multiple of 16 and the head at most 256 wide -- the host's and the shapes' choice, never a call's instance.
    On AVX-512 the stored forward computes SCORES-T's 16-row groups in the band itself and the backward walks dV and dK
    over the bands, then a group at a time evaluates its probability adjoints rows-in-lanes with the dots fused in,
    writes the score adjoints with masked stores and takes dQ from the band; the portable and AVX2 instances run the
    layout through scalar paths. The scratch forward, the bf16 entries, a portable build and an AVX2-capped build ignore
    it. The trainer's stored binary32 forward runs under LOWER|BANDS and its backward under BANDS
    (`attention_forward_contract`, `attention_backward_contract`); the pack's lowering, the step program's words and
    the Python launch plan state the bit; `bcir_tensor_attention_bands_probe` counts the heads each direction ran
    banded.
  - Every value the same: the row paths' hashes (the scratch forward's 1,248 cases, FWD-STRIP's 288, P-LOWER's 208,
    ATTN-RATE's 88, BWD-LONG's 48) before == after under clang 18 and GCC 13; `bands_check.c`'s 6,220 cases over every
    pair of forward and backward instances and across the compilers, every probability at its band address, 0 failed;
    every timed arm's outputs the same bytes in every round; the trainer's state the same digest with the bit on and
    off; the GEO model's trained state and logits the same from both level-3 builds.
  - Measured on the pushed code commit against the one before it, in one process beside an A/A control, 48 rounds a
    row: the attention at B4/T2048 x0.841 (0.817-0.866) on four threads -- its backward x0.807, its forward x0.901 --
    x0.834 under GCC and x0.866 on one thread; x0.755-x0.829 at T128 to T1024; x0.831 under the fused contract and
    x0.904 under v1's exponential; GQA x0.875, 8 heads of 64 x0.890 at T2048 and x0.925 at T4096; heads of 128 x0.950,
    meeting its control (64 rounds declared before they ran). The training step at the GEO dimensions x0.892
    (0.870-0.915) at T2048, x0.887 under GCC, x0.917 on one thread, x0.911 at T1024 and x0.861 at T512; the 9.84M
    decoder x0.962. The GEO study's attention model on level 3: training x0.808 at T512 and x0.905 at T2048, inference
    at parity; against PyTorch at fixed shapes its update now takes x0.780 of eager's time and x0.797-x0.824 of
    Inductor's at T2048, x0.537 of Inductor's at T512. Against `scaled_dot_product_attention` the attention as the
    trainer runs it is the faster at T2048 (flash x1.188 of BCIR's time, its forward at parity with BCIR's) and at T1024
    (x1.422).
  - Witnesses: the C harness's `attention_bands` (12 shapes from T7 to T1040, heads of 3 to 257, the four contracts
    with and without LOWER, the instances 0, 1, 2 and 4 with the backward on the same or the widest, serially and
    pooled, with and without the workspace, and a pass of special values) and `trainer_bands` (three steps under five
    contracts, one thread and four, the digests the same across the harness's builds); `p-bands.json` PB1-PB17 and
    `p-bands-px.json` PBX1-PBX5, each caught by its own check; AX3, IA12, IA16, AF14, AF17, PL2, PL5, PL7, PL8, PL9 and
    PI5 re-anchored on the lines the slice rewrote and caught again.
  - Not claimed: hosts whose widest instance is AVX2 or aarch64, where the bit takes no effect; the bf16 attention;
    counter-based figures (no PMU on this VM).
  POOL-AFFINITY (2026-10-24): the tensor pool deals each job's tasks in per-worker blocks, every value as it was
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §24,
    [measurements/2026-10-24/pool-affinity/](research/measurements/2026-10-24/pool-affinity/README.json)).
  - The loss it answers, placed first (NEXT-0): the GEO step a part at a time after P-BANDS, the GEMMs 54% of it at
    T256 and 46% at T512, the attention 21% and 34%; on one thread the T512 step took 109.3 ms and on four 36.5, where a
    quarter is 27.3, the elementwise jobs scaling worst (the RMSNorm x1.37 its one-thread speed, its VJP x1.68, SwiGLU's
    VJP x2.13, the attention backward x3.18), each reading rows the job before wrote on other cores: an RMSNorm over
    2,048 rows took 45 µs back to back on four threads, 56 after a pool job wrote its operands and 161 cold.
  - Measured first and not taken: FWD-BANDW-0 (the forward x1.022 and x1.009), BWD-SWEEP-0 and BWD-PF-0 (the backward
    x0.966 and x0.963, none resolved), WADJ-TILE-0 (a lone weight adjoint x0.774, the chains x1.085-x1.207 and the
    8,192-row pair x1.081 slower), POOL-YIELDS-0 (x0.962 at T256 and parity at T512 for idle workers yielding 200,000
    rounds after every job); EXP8 (the resident softmax x0.93, about 2% of a group's phases) is left for a slice of
    its own.
  - POOL-AFFINITY: worker w's block of a job's n tasks over W workers starts at n/W*w + min(w, n%W); where the compiler
    has atomics each block's counter sits on a 64-byte line of its own, a worker claiming its block's tasks in ascending
    order and then the other blocks' leftovers; a job's `head` leading tasks -- a pair or group job's led weight
    adjoints, every task of a job with no more tasks than workers -- are claimed in turn before the blocks; without
    atomics a worker runs the head tasks w, w+W, ... and then its block. The counters live in the caller's pool block.
  - Every value the same: `bitcheck.c`'s 6,081 cases and the attention's five hash programs before == after under clang
    18 and GCC 13, clang's lines GCC's; every timed arm's weights, logits or outputs the same bytes; the study's trained
    states and logits the same.
  - Measured on the pushed code commit against the one before it. The elementwise jobs on four threads at 2,048 rows:
    the RMSNorm 43.5 -> 26.9 µs back to back and 55.2 -> 29.8 after a pool job, the residual add 22.1 -> 10.7 and 30.0
    -> 15.3. The training step at the GEO dimensions alone in its own process, steps back to back on one pool: x0.824
    (0.778-0.872) at T128, x0.944 (0.898-0.991) at T256, x0.932 at T512, parity at T1024; in one process beside an A/A
    control x0.92 under GCC at T256, x0.919 on two threads, x0.893 under v1's contract, x0.962 for the 9.84M decoder,
    parity at T1024 and T2048. The inference forward x0.837 at B4/T256 and x0.89-x0.91 at B1. The study's models (level
    2): the geometric heads' training x0.66-x0.77 of its time and inference x0.80-x0.85, every row resolved, now
    x0.22-x0.35 of Inductor's time in training; the attention model x0.87-x0.94 in training at T128-T512 and at parity
    at level 3, where its update at T2048 takes x0.793 of Inductor's time.
  - The design a part at a time: without stealing the step x1.05-x1.10 slower and the pair and chain jobs x1.06-x1.40;
    without the led head the step at parity in one process and x0.946 at T256 alone in its own process, four pair and
    group jobs x1.14-x1.20 slower alone. The head stays; its trade against the input adjoints' rows is left open.
  - The rigs: `ab_gemm.c` and `ab_bands.c` share one set of buffers across their arms and always run the second
    library's arm after the first's pool, so under POOL-AFFINITY the tree after against a byte copy of itself reads the
    lone nt products x1.395 and x1.618 slower -- single GEMM jobs there say nothing of this slice. `ab_bands.c` also made
    every arm's pool with the first library, and the tree after's attention on the tree before's pool waited forever on
    reports its workers never made; `ab_bands_own.c` makes each arm's pool with its own library.
  - Witnesses: the C harness's `pool_blocks` (every task once at each cap of a four-thread pool, the pool inside its
    bytes from every max_align_t offset of a line, a held first task's block taken by the other workers; with
    BCIR_TENSOR_POOL_PROBE each worker's own block one contiguous run of its share's length, the last worker taking
    every other block's rest, led jobs every task once; under ThreadSanitizer too); `pool-affinity.json` PA1-PA11, each
    caught by its own check; NDT-THREADS' NT2 and NT11 re-anchored on the block claim and caught again.
  - Not claimed: hosts with more cores or another cache topology; the bf16 contract's jobs (dealt in blocks, not
    timed); counter-based figures (no PMU on this VM).
  LOSS-ONCE (2026-10-24): the trainer's loss evaluates each logit's exponential once, every value as it was; the
    study's geometric heads take their gate's exponential a span at a time
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §25,
    [measurements/2026-10-24/loss-once/](research/measurements/2026-10-24/loss-once/README.json)).
  - Placed by measurement (NEXT-1, `survey.json`, on the tree before): after POOL-AFFINITY the k-blocked GEMM runs the
    T256 step's products at 480-613 GFLOP/s back to back on four threads, x2.7-x3.7 their one-thread rates, the
    workers 88-93% of the 1,024-row A Bᵀ product's job inside its tasks (the plain product's 75-91%, the three-task
    weight adjoint's 59-64%); they sleep 2.2-8.6 times a step from T128 to T512 (84 jobs a step), a job after a 1 ms
    pause 39-57 µs slower; the step spends 1.9% of its time outside its jobs. A sampling profiler (SIGPROF on the
    process's CPU time; no PMU on this VM) placed the rest: the loss's task and libm, its two `exp` calls a logit,
    5.7% of the T256 step's samples, and in the study's geometric heads' inference at T1024 the gate, its scalar
    exponential `bcir_tensor_expf` called an element (`wedge_gate_fwd_task` and it, 16.6-19.2%).
  - LOSS-ONCE: a row keeps its first 1,024 exponentials in double on its task's stack and its adjoints divide the kept
    values; a wider row evaluates its tail again. The same call on the same argument returns the same double, and the
    sum, the term and both divisions keep the loop's order, so every adjoint, term and loss is the two-evaluation
    loop's bit for bit. The loss task alone, one thread, 1,024 to 8,192 rows of 260 logits: 9.6-10.1 ns a logit to
    6.4-6.5 under clang (x0.650-x0.672), 9.0-9.4 to 6.6-6.7 under GCC (x0.712-x0.742); 512 rows of 1,500, the tail
    evaluated twice, x0.797 and x0.840. In the step on four threads, by the job clock, the loss job takes x0.62-x0.65
    of its time (0.67-0.70 ms to 0.44-0.45 at T256, 1.33-1.38 to 0.84-0.85 at T512), some 1.6-1.9% of the step; alone
    in its own process the step reads x0.977-x1.008 from T128 to T1024 and x0.988 for the 9.84M decoder, each holding
    1, every run's weights the same digest on both trees.
  - The study's builds: `once_level3.py` makes the same edit to the PR818 study's three trainer forks and gives the
    geometric heads' gate its exponential a task's 512 elements at a time through `bcir_tensor_exp`, every instance
    of which equals `bcir_tensor_expf` on all 2^32 inputs (`exp_span_check.c`, clang and GCC). Every trained state
    and logit is the fork's. Against level 2 in one process: inference x0.849-x0.869 for `global` and `query_full` at
    T1024 and T2048, each resolved; training x0.90-x0.94, `global` at T512 resolved (x0.943), the rest meeting their
    controls or holding 1; the attention model at level 3, where only its loss changes, at parity. Against PyTorch
    at fixed shapes the geometric heads now run inference in x0.45-0.61 of Inductor's time (x0.52-0.67 in
    POOL-AFFINITY's records) and train in x0.22-0.33 of it.
  - Witnesses: the C harness's `loss_once`, through `bcir_decoder_loss_probe` (BCIR_DT_LOSS_PROBE, the harness's
    builds only), holds every adjoint, row term and their sum byte for byte to the two-evaluation loop written apart
    at nine vocabularies to 2,049, five row counts about a task's sixteen, four kinds of row and targets in the
    buffer and past it, on no pool and pools of two and four; `loss-once.json` LO1-LO7, each caught by its own check;
    NDT-REDUCE's NR4 and NR9, beside the new task, caught again.
  - Not claimed: a vocabulary past the buffer gains on its first 1,024 logits only; the study's own pair kernels and
    the scans after its gate, measured here and left the study's; counter-based figures (no PMU on this VM).
  GP-LANES (2026-10-24): the study's geometric heads a row a lane, every trained state and logit the fork's; with
    NEXT-2's survey; records and docs only ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §26,
    [measurements/2026-10-24/gp-lanes/](research/measurements/2026-10-24/gp-lanes/README.json)).
  - NEXT-2's survey: the cores sustain 221-230 GFLOP/s of FMA a thread with every operand in registers on one thread
    as on four; four threads each running the k-blocked GEMM on operands of its own reach x0.86-x0.98 of four times
    one thread on the T256 shapes, and the step's GEMM jobs as the trainer issues them take x0.86-x1.20 of four such
    copies' time. A job of trivial tasks costs the pool 1.2-4.8 µs back to back and 34-35 µs once its workers sleep;
    timed a worker at a time through the step, the workers enter every job within 0.9 µs at T256 (but the step's
    first) and are 66-97% busy in it. The GEMMs take 56%, 48% and 37% of the pool's time at T256, T512 and T1024, the
    attention 21%, 34% and 50%.
  - Not taken: GROUP-ROWS-0 (the forward's group jobs a row range at a time) x0.965-x1.015 from T128 to T512, each
    holding 1; COMMIT-BLOCKS-0 (the update's commit in four times as many chunks) holding 1 at T128 and T256.
    ROPE-TABLE-0, the rotation reading a table of the same cosines and sines, takes x0.22-x0.48 of
    `bcir_tensor_rope_pool`'s time, every value the same -- some 2% of the step -- and is left for a slice of its own,
    the table needing a span in the plan on both rails.
  - GP-LANES: `gp_lanes_overlay.py` runs the forks' pair kernels (`gp_pair_forward`, `gp_pair_backward`, GP_RANK 22)
    and query_full's query frame (`gp_query_contract_pair`, `gp_query_contract_vjp`) sixteen rows at a time, a row a
    vector lane, each scalar operation the one intrinsic that performs it in the fork's order, so every feature and
    adjoint is the fork's bit for bit; a row the fork sends to its FP64 fallback is recomputed by the fork's kernel,
    and each group prefetches the next group's rows. The kernels take x0.26-x0.80 of the fork's time a row.
  - In the models, 48 rounds in one process with every trained state and logit the same, against once_level3.py's
    builds: training x0.915-x0.972 (`global` at T256, `query_full` at T256 and T512 resolved), inference x0.938-x0.990
    (both at T2048 resolved); the heads' GP kernels from 8.5-13.9% of the training's samples to 3.9-6.5%. Against
    Inductor the heads now train in x0.19-0.33 of its time and run inference in x0.41-0.57 of it.
  - Witnesses: `gp_lanes_check.c` (the lane kernels against the fork's, copied verbatim, every output byte for byte
    over six kinds of row, 8,189 of 24,576 through the fallback, clang and GCC), `gp_tasks_check.c` (both builds'
    entries side by side on row counts 1 to 1,024, serial and pooled: 128 outputs, none apart), and `red_lanes.py`
    (five faults, each caught; three mutations recorded as unobservable, with the reason).
  - Not claimed: the overlay is the study's code, offered to it as a measurement, not BCIR's runtime; a host without
    AVX-512F runs the forks' tasks; a perturbation of a double intermediate below float's rounding is out of the
    witnesses' sight, so the bit-identity rests on the statement-for-statement correspondence; counter-based figures
    (no PMU on this VM).
  ROPE-TABLE and ROPE-BOUND (2026-10-24): the rotation reads a table of the same cosines and sines, a span of the
    plan on both rails, and both rotations turn a head of one range of pairs at their loop's rate under clang; every
    value as it was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §27,
    [measurements/2026-10-24/rope-table/](research/measurements/2026-10-24/rope-table/README.json)).
  - Placed by NEXT-2's per-job probe: RoPE's jobs took 34-36 µs each at T256, twelve a step at the GEO dimensions
    (the forward's queries and keys and the backward's adjoints of both, in each of three layers), each evaluating
    libm's `cos` and `sin` for every position and pair -- 8,192 calls a rotation at T256, on the same arguments in
    every rotation of the step.
  - ROPE-TABLE (`a6f8feb`): `bcir_tensor_rope_table_pool` fills time x head-dim floats once a forward with the
    expressions the rotation evaluates -- each pair's frequency the same `pow`, each angle the same product in
    double, each entry the same libm call rounded to float -- and `bcir_tensor_rope_lent_pool` rotates with them, the
    sine times -1 under inverse, then the same two rounded products and one rounded sum or difference an element:
    every rotated value is `bcir_tensor_rope_pool`'s bit for bit. The table is the scratch arena's last span in the
    C plan and the step program alike (32 KiB at the GEO dimensions at T256), filled after the embedding and read by
    every rotation of the step. A program's ROPE record reads it as its second operand and a ROPE_TABLE record (kind
    24, the pack's claim `train.rope.table`) writes it; the checker holds the one to time x dim floats it may read
    and the other to time x dim floats it may write, an even dim and no unused word. `train.plan.dispatch.mismatch`
    counts 1,171 runs, and `train.memory.transient` reads 1,001,472 / 755,712 and 756,224 / 756,224 bytes, the
    table's 2,048 at the bench shape.
  - ROPE-BOUND (`b86bfc4`): measured on its own, the first commit's table rotation took x0.47-0.65 of the per-call
    one's time under clang, where ROPE-TABLE-0's prototype read x0.22-0.31. The two loops differ only in their bound:
    the prototype's ran to the head's half, the distance between the two halves the loop pairs; the commit's to the
    range's count, the same count for every head of at most 64 pairs. `bound_cause.c` shows that is the whole of it:
    under clang 18 and 23 the forms bounded by the halves' distance take 18.3-19.1 µs at T256 on one thread against
    54.4-56.4 for the others, and under GCC all five run alike; with one head a row they run alike under clang too, its
    vector code entered past one overlap test for all of a row's heads, whose halves interleave from head to head, so
    that the test fails. The per-call rotation had the same loop. Both now turn
    a position's pairs through `rope_turn`, bounded by the head's half wherever the head fits one range (every head up
    to 128 wide); the harness's output is the first commit's byte for byte.
  - Measured on the committed tree against the tree before the slice. Under clang the table's rotation takes
    x0.28-0.29 of the per-call one's time on one thread and x0.29-0.48 on four (GCC x0.31-0.39 and x0.38-0.71); the
    per-call rotation itself takes x0.57-0.81 of its time before ROPE-BOUND on one thread; the fill costs x0.69-0.76
    of one per-call rotation. In the step, a worker at a time, the rotations and the fill take 190 µs a step against
    447 at T256 (x0.43), 336 against 815 at T512 and 764 against 1,596 at T1024: some 1.9% of the step at T256. In one
    process the step reads x0.958 (0.931-0.985) at T256 beside an A/A control of 1.003 (0.980-1.027), the intervals
    meeting, and x0.970-x0.973 at T128, T512 and T2048, parity at T1024 and for the 9.84M decoder, each holding 1;
    alone in its own process x0.948 (0.905-0.992) at T128, resolved, and parity at T256. The inference forward holds
    1 at B4, and at B1/T256 reads x0.930 (0.888-0.973), meeting its control. Every round's weights and logits are the
    same bytes on both trees.
  - Not taken: RMS-LANES-0, the RMSNorm's per-row passes sixteen rows at a time, a row a lane, every operation the
    unit's in its order: alone with their operands in cache the jobs take x0.66-0.80 of their time, in the step
    x0.953-x1.023 from T128 to T1024, each holding 1 -- the norms' jobs wait on where their operands sit, not on their
    arithmetic -- and on non-finite input one to three elements carry another NaN than the unit's.
  - Witnesses: the C harness's `rope_table` (201,372 table entries, each the per-pair reference's own libm call; 504
    rotations byte for byte the reference's and `bcir_tensor_rope_pool`'s over seven shapes, four bases, forward and
    inverse, serial and pools of two and four; a table of another base and one read a position late both seen),
    every other line of the harness as before, under clang 18 and GCC 13 and under ThreadSanitizer; six new program
    refusals; `rope-table.json` RT1-RT16 and `rope-table-px.json` RTX1-RTX16, each caught by its own check -- RTX14,
    first aimed at the step program's forward rotations alone, could not be seen, the table's lifetime being its last
    read's, the backward's; PX-DISPATCH's PY27 re-anchored, NDT-ROPE's NO1-NO7 and NDT-THREADS' NT6 caught again.
  - Not claimed: heads wider than 128 keep the ranges' loop at clang's slower rate; the branch is a rate, not a value
    (sending every head down the ranges' path changes no output, so only the timings witness it); the v1 contract's
    step at x0.935 at T256, resolved by the rule but more than the rotations' whole time a step (0.45 ms of 25)
    beside an A/A control of 1.025, and the first commit's B1/T256 forward at x0.734 beside one of 1.105 -- the rig's
    bias at those sizes; counter-based figures (no PMU on this VM).
  KSPLIT (2026-10-25): the blocked contract's weight adjoints split over their depth into the workspace the state
    lends, folded in K order, every value as it was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §28,
    [measurements/2026-10-25/ksplit/](research/measurements/2026-10-25/ksplit/README.json)).
  - Placed by NEXT-3's re-probe after ROPE-TABLE: the GEMMs take 60% of the GEO step's pool time at T256 and their
    rate falls from 454 to 389 GFLOP/s by T1024, and of the GEMM forms the weight adjoint (dW = dY^T X, a blocked call
    with A transposed whose depth is the batch's rows) stood furthest from MKL's: at a depth of 4,096 on four threads
    x1.16-2.26 of `torch.mm`'s time. A blocked call's task spans all of the depth, so every task walks all of A, which
    from a few K blocks deep a core's L2 no longer holds. KSPLIT-0, the prototype, ran such a call a task a K block:
    the GEO step x0.928-0.981 at T1024 and T2048 in its six runs, every byte equal.
  - KSPLIT (`bb69e4f`): a blocked-contract call with A transposed and B not, at least four K blocks deep, runs a task
    a K block where the room it is lent holds its blocks' partials (blocks x m x n floats): each task computes its
    block's product over all of C with the instance's own register kernel, the accumulators stored as they are (a new
    epilogue of the portable, AVX2 and AVX-512 blocked kernels, `BCIR_MM_RAW`), and a fold job, sixteen rows of C a
    task, gives every element its blocks in order with the blocked kernels' epilogue: the first block's beta scaling,
    then c + alpha*P block after block. Each block's partial is the nest's fmaf chain and the fold is the nest's
    epilogue in the nest's order, so every element of C is the nest's bit for bit. A job's calls take the room in call
    order while their partials fit, a group's independent calls and never its chain; the rest run as before. The
    trainer lends the state's workspace -- under a binary32 contract the attention's copies (ATT-CONTIG), idle while a
    GEMM job runs -- to every blocked-contract GEMM job with a weight adjoint (one predicate, `ks_lends`, on the fixed
    step and the program executor); no plan grows, so where the state lends none nothing changes. Every record of such
    a job states the workspace, the checker requires it there and nowhere else under the contract, and the trace, the
    launch plan and the pack (its claims and its lowering) carry it. The Python trainer's exact need on four threads
    holds all 22 of the GEO step's weight adjoints a step at T1024 and T2048, and one a layer of the 9.84M decoder's.
  - Measured on the committed tree against the tree before. Alone, the GEO step's weight adjoints over 4,096 and 8,192
    rows take x0.56-0.78 of their time on four threads and x0.66-0.88 on one, x0.74-1.31 of MKL's time at 4,096 rows
    where the nest took x0.98-2.03. A worker at a time the GEO step's GEMM claims that split take 21.3 -> 17.9 ms a
    step at T1024 (x0.84) and 46.0 -> 36.9 ms at T2048 (x0.80). In one process the step reads x0.950 (0.930-0.970) at
    T1024, resolved (gcc x0.941, meeting its control; at the trainer's exact need x0.968), and x0.959-0.971 in five
    64-round runs at T2048, each meeting its control's interval; the 9.84M decoder's step at T1024 x0.938
    (0.912-0.966), resolved, where every weight adjoint is lent room, and x1.008 at its exact need. Where nothing
    splits -- the GEO step at T512, the v1 contract, the 9.84M decoder at T128 and T512, the forward -- every reading
    holds 1. Every round's weights and logits are the same bytes on both trees.
  - The residual: the split's rate falls as the weight adjoint widens -- at 8,192 rows and n 384 x1.11-1.24 of MKL's
    time from m 256 to 640 on four threads and x1.68-1.80 at 1,024 -- as a K block's panel of A (448 x m floats, 1.75
    MiB at m 1,024) outgrows what a core's L2 holds beside B's. On one thread that takes the 9.84M decoder's 1,024-wide
    weight adjoint at 8,192 rows to x1.00-1.14 of the nest's time (five of six recheck runs resolved slower), where
    four threads still gain (x0.905-0.926); the trainer lends it no room on one thread (1.44 MiB against its 28.5 MiB
    of partials), and lent one its one-thread step reads x0.957 (gcc, resolved) and x0.975.
  - Witnesses: the C harness's `gemm_kblock_split` (990 weight adjoints through the single, pair and group entries
    byte for byte the K-blocked fmaf chains, 3 to 10 K blocks, every instance's row blocks and edge widths, serially
    and on pools of two and four; 432 split, the room written exactly where a call qualifies and its partials fit,
    never past what was lent); the harness's -contig and ThreadSanitizer builds split every blocked weight adjoint
    whatever its depth, and the -contig build's trainer digests are the default build's; a long-context case through
    the C step's trace, every program and pack run of the planner's choices and a weight adjoint run as a group's
    chain of one, each the C step's state and the state a never-splitting build trains; the step program's loan under
    each job shape and contract; five new program refusals. `ksplit.json` KS1-KS19 and `ksplit-px.json` KX1-KX14,
    each caught by its own check (KS9, KS16 and KX7 rewritten to build: their first forms stopped the build);
    GW8, MR5, NK5 and NT14 re-anchored and caught again; clang 23's analyzer's finding in the executor's pair fixed
    (KX5 re-anchored).
  - Not claimed: the split where the state lends no room (the blocked contract below 1,024 positions; the 9.84M
    decoder's wider weight adjoints at its need) and the wide weight adjoints' rate, the next slices' (KSPLIT-ROOM,
    KSPLIT-ROWS); T2048's gain by the rule; the fuzz harness's state lends no workspace under the blocked contract, so
    the checker's new rule is the witnesses' and not the decoder campaign's; counter-based figures (no PMU on this VM).
  KSPLIT-ROOM (2026-10-25): the blocked contract's planned workspace holds the room its weight adjoints split in,
    every value as it was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §29,
    [measurements/2026-10-25/ksplit-room/](research/measurements/2026-10-25/ksplit-room/README.json)).
  - Placed by NEXT-4's re-probe after KSPLIT, on the tree before: at three K blocks (GEO T256) the split does not pay
    alone (x1.01-1.69 of the nest's time on four threads, x0.97-1.13 on one), so the threshold stays at four; lent the
    room the plan now
    holds, the step read x0.917-0.922 at the corpus recipe's decoder (B8/T256, the train rail's), x0.932-0.937 at the
    GEO dimensions' B4/T512 and x0.925-0.939 at the 9.84M decoder's T512, every byte equal. KSPLIT-ROWS -- a K block's
    task over a range of rows, for the widest weight adjoints' panels and the five-block calls' four workers -- took
    some weight adjoints alone to x0.62-0.89 of the nest's time but read x0.933-0.951 in the step against the
    committed split's x0.917-0.937, and x0.994 at the 9.84M decoder's T1024: measured, not taken.
  - KSPLIT-ROOM (`8d212f9`): the tensor unit states the room a call takes by the rule its entries take a slice by
    (`bcir_tensor_mm_ksplit_bytes`: the call's K blocks' partials, blocks x m x n floats, to the next 64 bytes, from
    `BCIR_MM_KSPLIT_BLOCKS` blocks; one shared step, so the plan and the split cannot disagree), and the trainer's need
    under the blocked contract is the most of the attention's copies and the backward's largest GEMM job's room, the
    sum of its weight adjoints' slices -- the head's pair, the gate and up group, or the q, k and v group (the down and
    output projections' pairs take no more). `bcir_decoder_workspace_unsplit` is the need without the room, and
    `native.memory_plan` offers each recompute set at the need, then without the room, before the next set. The
    corpus decoder now lends 2.5 MiB where it lent none, GEO T512 1.25 MiB, and the 9.84M decoder 30 MiB at T512 and
    57 MiB at T1024 (5.75 MiB before); every weight adjoint of those steps splits (29, 22 and 43 a step; at T1024 six
    before).
  - Measured on the committed tree with one library, the state lent the need without the room against the need --
    the room alone, every instruction the same: the step x0.906 (clang, resolved) and x0.923 at the corpus decoder,
    x0.922-0.923 at GEO T512, x0.939-0.941 at the 9.84M decoder's T512 and x0.928-0.949 at its T1024, each resolved
    but gcc's corpus and T1024 rows (meeting their controls); one thread x0.947 (clang, resolved) and x0.972 (gcc,
    meeting its control) at the corpus decoder and x0.945-0.984 elsewhere, holding 1; where the two figures are equal
    (GEO T1024 and T2048) x0.967-1.040, holding 1. Between the two trees the same rows read x0.918-0.953, but that run's own controls read x0.905 and
    x0.938 at T2048, where both trees lend and split the same; with the trees' places swapped they hold 1
    (x0.973-1.023) -- the run's swing at that shape, not a build's, and why the one-library run carries the slice's
    reading. Every round's weights are the same bytes on both trees and both arms.
  - Witnesses: the C harness's `ksplit_room` (the unit's rule against a statement written apart from it at five
    depths and its refusals; the need, the need without the room and every other contract's need at three specs,
    each of the three jobs that can be the largest once; a step lent the need, a byte short of the room and none, the
    same state bytes, splitting 15, 13 and 0 weight adjoints under the probe, `bcir_tensor_ksplit_probe`);
    `test_the_blocked_contract_plans_the_room_its_weight_adjoints_split_in` (the trainer's loan at a short context
    four K blocks deep, `memory_plan`'s order at a long context whose copies fall short of the room, one state under
    every loan at one thread and four, the probe's counts); the KSPLIT launch witness's short case through the C
    step's trace, every program and pack run and a never-splitting build. `ksplit-room.json` KR1-KR15, each caught by
    its own check (KR8 rewritten to build); KSPLIT's KS12 and KS13, ATT-CONTIG's AC20 and NDT-BF16's NB14 re-anchored
    and caught again.
  - Not claimed: a gain at three K blocks (the threshold's), the widest weight adjoints' rate (KSPLIT-ROWS, measured
    and not taken), the room under the bf16 contract (its GEMMs do not split), any figure on another host;
    counter-based figures (no PMU on this VM).
  Q8-LANES (2026-10-25): the C BCIRQ8 decoder's matrix kernels with their outputs in vector lanes, every token and
    logit as it was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §30,
    [measurements/2026-10-25/q8-lanes/](research/measurements/2026-10-25/q8-lanes/README.json)).
  - Placed by NEXT-5's re-probe after KSPLIT-ROOM, on the tree before: the train rail's binary32 step (the corpus
    recipe's decoder, B8/T256, four threads) is at its bounds -- every large GEMM job 94-98% busy (NEXT-4's 81% was a
    mean over the small fold jobs), the GEMMs at their own isolated rates, the attention at 87-92% of four times its
    one-thread rate, the loss libm's exponential most of its time, 0.6 ms of 37.9 outside the pool -- and LOSS-MAX
    (the loss's row maximum in lanes, every value as it was) took the loss to x0.889 with eight lanes and x1.066 with
    sixteen, some 1% of the step: measured, not taken. The C decoder the rail's model is qualified on ran its BCIRQ8
    matvec and row dots a scalar double term at a time: 95% of the qualification's instrumented request.
  - Q8-LANES (`b444807`, `ee6bac1`): `bcir_ai_kernels.c`'s matvec and row dots on three instances -- 1 the portable loops, as
    they were; 2 AVX2; 4 AVX-512 -- chosen at run time on x86-64 Linux (GCC, Clang), each output's chain the portable
    loop's own, term for term in its order, the multiplies and the add never fused: the matvec a row's columns in
    the lanes, a run of one group at a time (AVX-512: 64 columns' sums in registers over every input row where the
    columns are a multiple of 8 and the group a power of two of at least 8, a scale computed where the group
    changes), the head's row dots eight rows a vector over their codes transposed eight by eight; a group's scale
    ldexp's value, built from its bits for every normal double. `bcir_ai_q8_instances` and the `_instance` entries
    name an instance (a refused one writes nothing); the prevalidated entries run the widest; the Python binding's
    `q8_matvec` / `q8_rows_dot` take `instance=`. Under valgrind the AVX2 instance runs (AVX-512 hidden from it).
  - Measured on the committed tree against the tree before (one host; clang 18 and GCC 13 at the
    qualification's -O2): the matvec at the 860K decoder's shapes x0.17-0.21 of the tree before's time on AVX-512 and
    x0.21-0.27 on AVX2, the head's row dots x0.22-0.28 and x0.25-0.33; the standalone C decoder's generated token
    x0.280 (clang) and x0.316 (gcc) over 200 tokens, 1.05 -> 0.30 ms, and the 200-token request, its load included,
    x0.313 and x0.372; the qualification's request on the lab model 52.10 -> 12.81 ms monolithic and 55.97 -> 17.11 ms
    GEM, a generated token 0.972 -> 0.234 ms (791 -> 2,828 tokens/s), the logits one SHA-256 on both trees and 0 from
    the oracle's, 922 M -> 231 M instructions under callgrind (the AVX2 instance); every token and final logit of 7
    builds of the tree after (clang and gcc at each instance cap, clang with the dispatch compiled out) the tree
    before's bytes over 2 models, 3 prompts and 4 lengths. The first commit's kernel A/B read the portable instance at
    x1.17-1.28 of the tree before's loops under clang: inside the dispatching entry they lost their registers (clang
    reloaded three bases a term), which a decoder capped at 1 hid, its compiler dropping the vector code. The second
    commit (`ee6bac1`) makes them the tree before's prevalidated entries byte for byte, in functions never inlined:
    their instructions are the tree before's under both compilers (168 and 152 under clang, 187 and 151 under GCC),
    clang's instance holds 1, and GCC's reads x1.05-1.23 at its default loop alignment and x0.93-1.06 with loops
    aligned to 32 bytes -- the same instructions placed differently.
  - Witnesses: the kernel harness's `q8_lanes_test` (every admitted instance against the portable loops over 22
    shapes -- the decoder's, the block's with its tail blocks, groups straddling rows and vectors, a group not a power
    of two, groups under 8 -- at four exponent ranges, the format's, a narrow one, ldexp's subnormal scales and past the
    largest double, the code 0x80 among them: 352 comparisons on an AVX-512 host; non-finite outputs zeroing every
    output in the last block's last lane and a block's middle row; refused instances and inputs writing nothing; the
    prevalidated entries on the widest instance under the probe, `bcir_ai_q8_instance_probe`), run by
    `test_native_ai_c_harness_c11_and_c23` under C11, C23, C23 capped at AVX2 and with the dispatch compiled out;
    `test_native_q8_instances_match_the_increasing_coordinate_oracle` (each admitted instance against the Python
    oracle's loop, nine shapes); the decoder's parity tests, the qualification (C logits 0 from the oracle's, GEM and
    monolithic one SHA-256, counted == planned traffic) and the memory-discipline gate, whose AddressSanitizer build
    found GCC 13 giving a 64-byte-aligned local array 32 bytes on its fake stack (the kernels keep no such array).
    `q8-lanes.json` QL1-QL22, each caught by its own check (QL2 and QL11 rewritten to build; QL21's first form, a
    term's product rounded before its power-of-two scale, differs only under double rounding below a subnormal grid,
    so it was replaced); the witnesses before Q8-LANES (the harness's cases without `q8_lanes_test`, one
    build, and the matrix kernels' oracle test) catch 5 of the 22 -- the AVX-512 paths the decoder's shapes reach --
    and the other 17 only the new witnesses.
  - Not claimed: the attention row and the rotation (the request's next costs after the kernels), threads, any other
    host's figures; counter-based figures (no PMU on this VM). The GEM ratio row rises (1.07 -> 1.34 under its 1.5
    band) because GEM's trust boundary is a fixed cost the faster kernels no longer hide -- NEXT-6's.
  CRC-SLICE (2026-10-25): zlib's CRC-32 eight bytes a step for every C unit that checks one, every value, verdict
    and wire byte as it was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §31,
    [measurements/2026-10-25/crc-slice/](research/measurements/2026-10-25/crc-slice/README.json)).
  - Placed by NEXT-6 on the tree before, after Q8-LANES left the qualification's GEM/monolithic row at 1.336: on the
    qualification's own pack (610 KB) a CRC-32 pass took 1.27 ms (the byte table, one dependent lookup a byte), a GEM
    run paid three of them (`bcir_sp_execute` validates, then `bcir_sp_verify_semantic` and the segment walk validate
    again) -- some 3.8 of GEM's 4.3 ms over the monolithic runner -- and a GEM load four; the BCIRQ8 loader computed
    its body's and every tensor's CRC with the bitwise loop, eight dependent steps a byte: 8.4 ms (clang) and 15.8 ms
    (GCC) of a load of the lab model's 0.9 MB. The 2026-10-10 note had recommended a portable slicing-by-8/16 CRC with
    the polynomial, conventions and wire bytes kept (its P1).
  - CRC-SLICE (`ce86ead`): `runtime/c/bcir_crc32.h` holds zlib's CRC-32 as eight tables and a step of eight bytes --
    table 0 the bitwise loop's state after eight steps from each byte, table k table k-1 carried through one more zero
    byte, the bytes assembled little-endian (the same step on every byte order), freestanding -- read by
    `bcir_runtime.c`'s `bcir_crc32` (every StreamPack, plan, record, bundle and frame digest on the C rail) and the
    BCIRQ8 loader; the byte table and the bitwise loop retired.
  - Measured on the committed tree against the tree before (one host; clang 18 and GCC 13 at the qualification's
    -O2): a CRC pass x0.24-0.26 (clang) and x0.19-0.21 (GCC) of its time from 16 KiB to 8 MiB, about 2.1 ns a byte
    before and 0.4-0.5 after; the lab models' load 8.2 -> 1.31 ms (clang) and 16.1-16.4 -> 1.07-1.11 ms (GCC); a GEM
    load's four passes and walks 5.7-5.9 -> 1.8-1.9 ms, `bcir_sp_execute` with its kernels doing nothing 4.3-4.4 ->
    1.35-1.43 ms; a whole request of 1 new token x0.38-0.41 (clang) and x0.20-0.21 (GCC), of 64 x0.71-0.72 and
    x0.56-0.58 (the load's difference alone: the decoder's step untouched). In the qualification GEM's load 6.19 ->
    2.22 ms, the trust boundary 2.93 -> 1.00 ms, GEM's own overhead 4.17 -> 1.57 ms, GEM/monolithic 1.336 -> 1.116
    (1.32-1.35 -> 1.11-1.13 over three runs more of each tree), the first token 5.82 -> 2.99 ms; the logits and the
    native pack one SHA-256 on both trees, 23 of 23 checks on each, and the monolithic runner's request the same
    210,858,806 instructions under callgrind. The 2026-10-10 note's own CRC stage reads `bcir_crc32` at x6.8-11.4 of
    Python's zlib before and x1.9-2.2 after.
  - Witnesses: `test_the_runtime_crc_is_zlibs` (the header's 2,048 entries against the recurrence, `bcir_crc32`
    against `zlib.crc32` over buffers around the eight-byte step and a NULL buffer of no bytes, an entry flipped in
    each of the eight tables caught by the CRCs); the standalone C decoder's parity test for the loader (its models
    load, a flipped byte is a CRC error); `crc-slice.json` CS1-CS10 -- a table entry flipped, the words' bytes swapped
    or the running state left out, two lanes on each other's tables, the step advancing seven, the tail a byte short
    or on the wrong table, the final complement dropped, the loader's running CRC restarted on every span or a byte
    short -- each caught by its own check.
  - Not claimed: fewer passes (a GEM run still validates the pack three times, now 0.25-0.32 ms a pass: TRUST-ONCE);
    zlib's own speed (its `crc32` reads x0.34-0.51 of the new step's time here); any other host's figures;
    counter-based figures (no PMU on this VM).
  DECODE-ROW (2026-10-25): the C decoder's attention row eight key rows and eight coordinates at a time, its rotation
    a table a position, every value as it was ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §32,
    [measurements/2026-10-25/decode-row/](research/measurements/2026-10-25/decode-row/README.json)).
  - Placed by NEXT-7 after CRC-SLICE: in Q8-LANES' per-operation record the attention was 14.4 ms (24%) and the
    rotation 3.1 ms (5%) of a 200-token request's 60.8 ms of operations. The attention row computed each score as one
    dependent chain a channel at a time and each context coordinate as one chain down every key row, a 512-byte
    stride a term; the rotation computed the same 16 angles' cosines and sines for each of 24 heads a token.
    `gqa_bench.c` (standalone, byte for byte against the committed loop over 3,000 random shapes) put blocking the
    context's coordinates at x0.54 and the scores' rows too at x0.41-0.45 at 205 positions.
  - DECODE-ROW (`f356f16`, the witness's flags `17871d0`): `bcir_decode.c`'s attention row runs eight key rows'
    score chains side by side, then four, the rest one at a time, and eight context coordinates' chains side by side
    down the rows, each chain the one it was -- from +0.0, in the same order, term for term -- the masked rows'
    terms still added. `bcir_rope`'s angle and rotated pair are one definition each, shared with `bcir_rope_table`
    (one position's cosines and sines) and `bcir_rope_apply` (a row turned by such a table); the decoder's workspace
    holds the table and its position (budgeted with the rest), `bcir_llama_op_rope` fills it when the position
    changes, for the monolithic runner and GEM alike, and GEM refills it at each run's first position.
  - Measured on the committed tree against the tree before (one host; clang 18 and GCC 13 at the qualification's
    -O2 -ffp-contract=off): a request's decode steps of the attention row x0.43 (clang and gcc) at the 860K
    decoder's shapes over 205 positions and x0.44-0.47 over 1,024, x0.34-0.35 at TinyLlama's (32 heads of 64, 4 kv
    heads) over 256 and x0.41-0.43 over 1,024; the rotation x0.06-0.10; the standalone C decoder's generated token
    over 200 tokens x0.75-0.83 (0.30-0.32 -> 0.24-0.25 ms) and the 200-token request x0.79-0.85, both lab models;
    over 200 tokens the operations 61.2 -> 51.6 ms (the attention 14.6 -> 7.7, the rotation 3.1 -> 0.35). Every token
    and final logit of 7 builds of the tree after is the tree before's bytes over 2 models x 3 prompts x 4 lengths;
    the qualification's logits keep one SHA-256 (0 from the oracle's, 23 of 23 checks), with 4.4-4.7% fewer
    simulated instructions on its two rails. At the qualification's 48 tokens a whole request reads x0.88-0.97, within its A/A
    control mostly: the attention is a small share of a 55-position request.
  - Witnesses: `test_decode.c`'s `order` mode holds `bcir_gqa_attention_row`, `bcir_gqa_attention`, `bcir_rope` and
    the table's rotation to the loops before DECODE-ROW, kept in the harness as the definition, byte for byte:
    2,482 comparisons over 21 head widths and 21 context lengths around the blocks, six head groupings, six value
    ranges and rotations at ten positions to INT_MAX - 300, built with -ffp-contract=off at -O0, -O2 and -O3
    (`test_the_attention_row_and_the_rotation_are_the_one_chain_loops_bit_for_bit`); the decoder's rotation is held
    to `bcir_rope`'s rows at positions repeated, backward and after a gap
    (`test_the_decoders_rotation_is_bcir_ropes_rows_bit_for_bit`), which the decoder's 1e-9 parity cannot see,
    attention reading only the positions' differences. `decode-row.json` DR1-DR15, each caught by its own check, 8
    of them by the new witnesses alone (the older tests' shapes never reach a block). CI's aarch64 job found the
    first form of the order witness wrong: its plain -O2 build let clang contract (aarch64 has FMA), and each code
    shape's chains fused as clang chose -- the one-chain loops' differently from the blocks'. The witness now builds
    as every decoder build here does, with -ffp-contract=off; the CMake library build of `bcir_decode.c`, which
    passes no contraction option, is left as its own task.
  - Not claimed: libm's exponential, now most of the attention row's time (one call a probability, kept for the
    oracle's values); the same bytes where a compiler may fuse a product into its sum; a gain at the qualification's
    48 tokens beyond its noise; any other host's figures; counter-based figures (no PMU on this VM).
  SP-ONCE (2026-10-25): a StreamPack validated once a call, GEM's execution and load crossing the CRC once where they
    crossed it three and four times, every bound and verdict as it was
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §33,
    [measurements/2026-10-25/sp-once/](research/measurements/2026-10-25/sp-once/README.json)).
  - Placed by NEXT-8 after DECODE-ROW, reading CRC-SLICE's trust-boundary record: `bcir_sp_execute` (1.43 ms clang,
    1.35 gcc) was a validation, `bcir_sp_verify_semantic` (which validated again) and the segment walk (a third
    time), three CRC passes over one pack in one call, about 0.95 / 0.75 ms of it; `bcir_dgem_load`'s walks were four.
  - SP-ONCE (`0f29c50`): `bcir_runtime.c`'s walks are bodies over the header `bcir_sp_validate` reads, each plain walk
    validating and running its body; `bcir_sp_check` records a validated pack in a `bcir_sp_checked` token and the
    checked semantic walk, generation check, segment and block walks run the same bodies over it without recomputing
    the CRC, refusing a refused check's token and one `bcir_sp_check` did not fill. `bcir_sp_execute` and its R11
    wrapper, `bcir_dgem_load` and the generation-vector walks check once; the qualification's trust-boundary row
    times what a run now pays. Every run still validates the pack it executes. Counted under callgrind (each
    tree's units at -O0, a profile part a call), a pack's CRC passes a call: `bcir_sp_execute` and `bcir_dgem_run`
    3 -> 1, `bcir_dgem_load` 4 -> 1, the program rail's lowering 13 -> 9.
  - Measured on the committed tree against the tree before (one host, one process, an A/A control):
    `bcir_sp_execute` with no kernel 1.39 -> 0.74 ms (x0.530, clang) and 1.29 -> 0.76 ms (x0.591, gcc); GEM's load
    1.97 -> 0.98 ms (x0.500) and 1.84 -> 1.07 ms (x0.584); a whole 48-token GEM request x0.94-0.96, within its
    control. In the qualification, over four runs of each tree: GEM's load 2.16-2.26 -> 1.24-1.31 ms, the trust
    boundary 0.98-1.00 -> 0.67-0.69 ms, the first token of a GEM run 2.68-2.89 -> 2.00-2.08 ms, GEM/monolithic
    1.06-1.25 -> 1.03-1.07; the logits one SHA-256 on both trees, 23 of 23 checks, GEM 214.9 M -> 211.1 M simulated
    instructions and the monolithic runner's the same 201,016,566.
  - Witnesses: the executor harness's `--checked` differential
    (`test_the_checked_walks_are_the_plain_walks_on_every_pack`: 1,100 packs -- the hydrated packs and their 21
    disorders, truncations, a flipped CRC, header fields moved by one and resealed, body octets changed and resealed
    -- each checked walk's status, header and every callback's view the plain walk's, forged and refused tokens
    refused); `fuzz_streampack.c`'s same differential on every input, raw and sealed; `sp-once.json` SO1-SO11, each
    caught by its own check; `qual.json`'s QL2 and QG3 re-anchored on the moved lines and the whole table (28 faults)
    swept again on the committed tree, each still caught.
  - Not claimed: validating once a load and trusting the bytes across runs (TRUST-ONCE, a change to the per-run
    contract); the other callers that validate and walk, which keep their plain walks -- the program rail's
    lowering (`bcir_decoder_lower_pack`: 9 CRC passes over its pack a lowering where it made 13, its two executions
    now one pass each; its time unmeasured), the verifier CLI, the control plane, the encoder; any other host's
    figures; counter-based figures.
  CONTRACT-OFF (2026-10-26): the decode and train kernels compiled without floating-point contraction in every
    build the repository writes, the manifest checker's M17 holding every build to it and a witness building them
    under a hostile default.
  - Found after DECODE-ROW's order witness failed on CI's aarch64 job, where clang contracts by default: the
    CMake and BCIR Make builds compiled `bcir_decode.c` and `bcir_train.c` (library `bcir_streampack`) with no
    contraction option, though every other decoder build here passes `-ffp-contract=off`. On this host's
    x86-64 FMA shows it (`-mfma`): clang 18's default contraction moves 6 of the training twin's 12 epoch losses
    and 1 of its 5 trained weights in their last bits (by at most 5.6e-17), and a default of fast contraction
    (GCC's own outside ISO mode) breaks `test_decode`'s `order` witness under clang and GCC, the fusion in either
    the kernels or the one-chain loops enough. The oracle differentials' 1e-12 tolerance hid both.
  - CONTRACT-OFF (`a46c3e1`): `runtime/manifest.json` gives `bcir_streampack` and the `test_decode` harness (the
    one-chain loops the kernels are held to) `-ffp-contract=off`, which CMake and BCIR Make apply to each unit's
    own sources and the variants and fuzzers take through their closures; the six Python builds that compiled the
    kernels without it (the decode and training oracle differentials, the preflight builds, the cfront allocation
    sweep's harness) and `tools/c/fuzz_streampack.sh` pass it. Built so, the training twin's losses equal the
    Python oracle's floats exactly on this host.
  - M17 (`tools/build/manifest.py`): the unit that owns `bcir_decode.c`, `bcir_train.c` or `test_decode.c` names
    the option, unscoped; no unit or variant turns contraction back on; a Python function under bcir/ or tools/,
    or a gate script, that names one names the option beside it (in the function, at the module's top level, or
    anywhere in the module for a name at the top level). A build assembling its flags elsewhere is a finding until
    it names the option where it names the source -- the declared scope.
  - Witness (`test_the_manifests_options_build_the_kernels_without_contraction`): test_decode, test_train and
    test_train_stream with their closures, compiled source by source with each unit's manifest options after a
    hostile default (`-O2 -ffp-contract=fast`, with `-mfma` on an x86-64 CPU that has FMA), print the plain
    build's bytes (`-O0 -ffp-contract=off`): the order mode's 2,482 comparisons for two seeds and both training
    runs' losses and weights to 17 digits. Its control requires the hostile default to move something without
    the units' options: under clang 18 all four runs move, under GCC 13 the order mode, under clang 23 the
    training runs. `fp-contract.json` FC1-FC15, each caught by its own check.
  - Not claimed: the other floating-point kernels' builds beyond what they already pass (M17 reads these three
    sources); compiles whose object is discarded (the freestanding section); MSVC, which takes no manifest option;
    a bitwise comparison with the Python oracle across libms.
  NEXT-9 (2026-10-26): the C BCIRQ8 decoder's next two kernel levers after DECODE-ROW, each prototyped with every
    value the same and measured; neither taken; the hoist's decoder gain this host's clock, its unchecked ceiling the
    kernel's own; records and docs only ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §34,
    [measurements/2026-10-26/next9/](research/measurements/2026-10-26/next9/README.json)).
  - Q8-HOIST, measured and not taken: the vector matvecs' x times a group's scale formed once a row and group,
    `code * (x * s)` for `x * (code * s)`, the same bits wherever `x * s` is exact. The prototype (`q8_hoist.patch`)
    checks every such product and computes a call that formed an inexact one again with the committed kernel, its
    harness's path probe holding every vector matvec to one of the two paths. Without the check the hoist runs the
    AVX-512 block matvec at x0.852-0.960 of its time; with it the prototype is x0.995-1.144 at the 860K decoder's
    shapes (x1.042-1.109 over its 28 matvecs in order, codes from L2) and x0.967-1.060 at TinyLlama's.
  - In one binary with the matvec's path chosen at run time the prototype makes a decoded token x0.937-0.957 of its
    time under clang and x0.967-0.978 under GCC (3 runs each), the committed kernel's copy at addresses of its own
    x0.997-1.012 and x0.971-1.006: not placement, and not the kernel. It is the core's clock: a dependent add chain
    run between the matvecs at a decode's duty cycle (`clock_probe.c`, the median of 3 runs) reads the committed
    kernel's AVX-512 matvecs lengthening this core's cycle x1.202 (clang) and x1.192 (GCC) against none, and the
    prototype's leaving it x0.946 and x0.958 as long as the committed kernel's, in x1.027 and x1.039 of its cycles.
    The clock answers the prototype's per-group checks, not its multiplies: the ceiling -- its block kernel with the
    checks taken out, the same 512-bit multiplies -- leaves clang's cycle x1.002. Not taken: a gain that is this
    host's clock answering the checks, through a kernel that is itself slower.
  - The ceiling (exact here because every byte is compared, never a kernel) takes a token to x0.917-0.941 under clang,
    its matvecs in x0.884 of the committed kernel's cycles with the clock as it was -- the kernel's own gain -- and to
    x0.925-0.965 under GCC, whose build of it moves the clock too (x0.960, 1 of 3 runs resolved): what a hoist whose
    exactness is proved once a call can reach (Q8-BOUND, next).
  - Q8-ROWS2, measured and not taken: the head's row dots sixteen rows a step in two interleaved blocks, two
    independent chains a coordinate, every row's chain the same terms in order: x1.050 (clang) and x1.043 (GCC) at the
    860K decoder's head, x0.992 and x1.061 at TinyLlama's.
  - Not claimed: any other host's clock, or a frequency in hertz; the matvec over rows as well as columns at
    TinyLlama's sizes, still open; counter-based figures (no PMU on this VM).
  Q8-BOUND (2026-10-26): the C decoder's vector matvecs form x times a group's scale once a row and group where the
    call's ranges prove every such product exact, one multiply a term where there were two, every output the portable
    loops' bit for bit ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §35,
    [measurements/2026-10-26/q8-bound/](research/measurements/2026-10-26/q8-bound/README.json)).
  - Q8-BOUND (`7224194`): `q8_matvec_lanes` reads a call's ranges once (`q8_call_exact`, AVX2): every exponent of its
    groups in [-300, 300], the range `bcir_ai_q8_values_validate` admits -- each biased by 300 and compared unsigned,
    sixteen a vector -- and every nonzero |x| in [2^-722, 2^724). Each `x * s` is then x scaled by a power of two
    from 2^-300 to 2^300 with neither overflow nor a subnormal result, exact, so `code * (x * s)` rounds to the
    loop's `x * (code * s)` (`code * s` is exact), and the call takes the hoisted kernels (`q8_matvec_block512_xs`;
    `q8_matvec_runs_xs` over `q8_run_avx512_xs` and `q8_run_avx2_xs`), no check inside their loops. A call outside
    the ranges -- the `_instance` and prevalidated entries never check exponents -- takes the committed kernels, as
    before. NEXT-9 placed it: checked a row and group, the hoist was slower.
  - Witnesses: the kernel harness's lanes test holds every vector matvec to the path its call's ranges choose, and
    `q8_bound_edges` runs the ranges' edges (inputs at 2^-722 and a step below, a step below 2^724 and 2^724 itself
    over zero codes, exponents at +-300 and +-301, inputs whose `x * 2^-300` is subnormal, each range's vectors and
    its tail), every instance against the portable loops, the status, every byte and the path the bound probe
    reports; `test_native_ai_c_harness_c11_and_c23` holds each build to 47 hoisted and 52 computed as before a vector
    instance. `q8-bound.json` QB1-QB18, each caught by its own check; the witnesses before Q8-BOUND catch 5 of the 18
    -- the ranges ignored, the block's and the runs' scale, the fused term and the summed vector (QB12, QB14-QB17) --
    and none of the other 13: with each of them in place every output the old harness compares stayed the portable
    loops' bytes. `q8-lanes.json`'s QL3, QL4, QL5, QL7 and QL22 re-anchored past the hoisted copies of their lines,
    all 22 caught.
  - Measured (`7224194` against `7e5572a`, a reading the median of 3 runs): against the tree before's unit the
    AVX-512 matvec takes x0.870-0.892 of its time at the 860K decoder's shapes under clang and x0.943-0.962 under GCC
    (x0.872-0.936 and x0.917-1.000 at TinyLlama-1.1B's), the head's row dots, which Q8-BOUND leaves as they were,
    x1.000-1.006 and x1.001-1.008. In one binary with the choice switched at run time a decoded token is x0.926 of
    its time under clang (x0.908-0.946, 2 of 3 runs resolved past the A/A control) and x0.972 under GCC
    (x0.942-0.982, none of 3); NEXT-9's unchecked ceiling read x0.917-0.941 and x0.925-0.965. The clock does not
    carry clang's: between the matvecs at a decode's duty cycle its cycle reads x0.976-1.002 of the committed
    kernel's (none of 3 past the A/A control) while the matvecs take x0.894 of its time; under GCC the cycle reads
    x0.951-0.973 (1 of 3) and the matvecs take x0.924. Under callgrind (the AVX2 instance) the qualification's GEM
    request takes 211.1 M -> 197.8 M instructions and the monolithic runner 201.0 M -> 187.7 M; its logits keep their
    SHA-256, 23 of 23 checks on both trees.
  - Not claimed: the head's row dots (a block's lanes carry eight rows' scales, so no product is shared); the AVX-512
    runs, which a ragged shape takes and which read x1.017 (clang) and x1.026 (GCC) with the choice on, the scan's
    cost past the multiply there (no decoder shape here is ragged); any other host's figures or clock;
    TinyLlama-1.1B's decoder end to end (no artifact of that size here: its shapes are timed at the kernel);
    counter-based figures (no PMU on this VM).
  Q8-PAIR (2026-10-26): the C decoder's vector matvecs take two input rows a pass where eight columns' terms lie in
    one group, every output the portable loops' bit for bit
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §36,
    [measurements/2026-10-26/q8-pair/](research/measurements/2026-10-26/q8-pair/README.json)).
  - Q8-PAIR (`d144970`): where a matvec's columns are a multiple of 8 and its group a power of two of at least 8,
    both vector instances run `Q8_PAIR_KERNEL` (`q8_matvec_pair512[_xs]`, and `q8_matvec_pair256[_xs]` on `q8_x8`,
    eight columns as two AVX2 vectors): eight outputs are loaded, row i's terms and then row i + 1's are added, and
    they are stored; a row's scale is read again at the step where its group changes, and an odd last row runs alone.
    Every output's chain keeps its rows in increasing order and every term is Q8-BOUND's, so every output is the
    portable loops' bytes. It replaces the AVX-512 block (`q8_matvec_block512[_xs]`: 64 columns' sums in registers, a
    strip walked down every row) and, at AVX2, the runs a row at a time; other shapes keep the runs. A survey before
    the change placed it (23 shapes from L2, from L3 and from memory, clang and GCC, 3 runs, on draft tools not
    kept): the block lost wherever a matrix passed L2 or its rows were wide; a first form, in spans between the rows'
    group ends, lost to it at 128 x 344 under GCC (x1.06), and four rows a pass had worse cells (x1.15).
  - Witnesses: the pair's probe (`bcir_ai_q8_pair_probe`) at every vector matvec of the lanes test; `q8_pair_edges`,
    17 shapes in four exponent ranges with every buffer at its exact size; 92 matvecs on the pair and 64 on the runs
    a vector instance (`test_native_ai.py`); `test_native_ai_c_harness_under_address_sanitizer`, the harness under
    AddressSanitizer and UBSan. `q8-pair.json` PA1-PA15, each caught by its own check. Under the witnesses before
    Q8-PAIR 13 of the 15 are caught, all but the two that move only which kernel runs (PA7, PA8); PA3, a pass one
    step past its rows' end, only through the crash its write past a small array happens to cause. QB14, QB17 and
    QL4-QL8 re-anchored on the pair, all caught.
  - Measured (`d144970` against `5677fc0`, a reading the median of 3 runs): in one binary with the choice switched at
    run time, the AVX-512 matvec takes x0.793-0.875 of the block's time at the 860K decoder's four shapes in L2 under
    clang and x0.808-0.882 under GCC. At TinyLlama-1.1B's widest three it takes x0.439-0.513 from L3 and x0.326-0.484
    from memory (x0.461-0.502 and x0.391-0.485 under GCC), reading 5.0 GB/s of codes at 2,048 x 5,632 against the
    block's 1.6 (5.1 against 1.9 under GCC). The AVX2 matvec takes x0.644-0.780 (clang) and x0.533-0.671 (GCC) of the
    AVX2 runs' time in L2. Each shape behind these figures resolved past its A/A control in at least two of its three
    runs but clang's AVX-512 pair at 128 x 64 in L2, in one. A token of the 860K decoder takes x0.917 under clang (3
    of 3 runs resolved past the A/A control) and x0.932 under GCC (1 of 3), x0.865 and x0.740 capped at AVX2; a token
    of a synthetic decoder at TinyLlama's widths, six layers whose 334 MiB pass this host's L3, x0.467 (201.2 -> 95.0
    ms) and x0.512, x0.881 and x0.821 capped at AVX2. Under callgrind (the AVX2 instance) the qualification's GEM
    request takes 197.8 M -> 147.2 M instructions and the monolithic runner 187.7 M -> 137.1 M; its logits keep their
    SHA-256, 23 of 23 checks on both trees.
  - Not claimed: the head's row dots, untouched (x0.979-1.039 and x0.990-1.003 of the tree before's, the control);
    clang's AVX2 pair at 128 x 64 read from memory, x1.034 (none of 3 resolved): 8 KiB matrices, which a decoder of
    that width holds in L2; a real TinyLlama-1.1B (no artifact of that size here: the synthetic decoder has its
    widths and six of its 22 layers, with pseudo-random codes); any other host's figures; counter-based figures (no
    PMU on this VM).
  HEAD-PACK (2026-10-26): the C decoder's head reads its rows from a copy packed once at load, 32 rows a pass, every
    logit the rows' own bit for bit ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §37,
    [measurements/2026-10-26/head-pack/](research/measurements/2026-10-26/head-pack/README.json)).
  - HEAD-PACK (`27091ae`): `bcir_ai_q8_rows_pack` packs a matrix's rows once -- blocks of eight rows, a block's
    groups in order, each group's segment its eight rows' exponents and then its coordinates' codes, eight bytes a
    coordinate -- and `bcir_ai_q8_rows_dot_f64_packed` reads it: four blocks a pass on both vector instances
    (`q8_rows_pack512x4`, `q8_rows_pack256x4`, every accumulator and scale a named register), then a block at a time,
    a group's eight scales one load and (e + 1023) << 52, no division. The loader packs the head (the embedding when
    tied, else the LM head) where `d_model` is a whole number of groups (`head_pack`), and `bcir_llama_op_head`,
    which both dispatchers call, reads it. A survey before the change (draft tools, not kept) placed the layout: the
    codes alone packed, a block a pass, read x1.09-1.15 of the committed transpose at TinyLlama-1.1B's head under
    clang, the same kernel with the exponents in the pack x0.62, and four blocks a pass x0.24-0.50 at AVX-512 and
    x0.30-0.59 at AVX2.
  - Witnesses: `q8_pack_shapes` (16 shapes in two exponent ranges, every buffer at its exact size, the probe's passes
    and blocks: 32 calls, 152 passes and 40 blocks a vector instance) and `q8_pack_test` (a 64-byte pack pinned, a
    pack serving fewer rows, non-finite totals, the refusals); the Python oracle and layout test; the head witness
    (the loader under an allocator refusing each of its allocations in turn, the head against the unpacked row dots
    byte for byte, 12 of 12 calls on the pack); the standalone decoder's parity and the memory-discipline gate on a
    toy whose head is packed. `head-pack.json` HP1-HP18, each caught by its own check; under the witnesses before
    HEAD-PACK 0 of the 18 are caught, since none of them calls a packed entry or reads a packed head (the rotation
    witness loads a toy whose head the loader packs and reads only its rotation). QL12 re-anchored on the committed
    row block's store.
  - Measured (`27091ae` against `06e67ce`, a reading the median of 3 runs): in one binary the packed row dots take
    x0.516 of the unpacked time at the 860K decoder's head (in L2) under clang and x0.401 under GCC, and x0.392
    (x0.330) at TinyLlama-1.1B's head from memory, reading 7.2 GB/s of the pack against 2.6 of the rows as they are;
    at AVX2 x0.599 (x0.448) and x0.586 (x0.352). A token of the 860K decoder takes x0.920 under clang (3 of 3 runs
    resolved past the A/A control) and x0.877 under GCC (3 of 3), x0.940 and x0.855 capped at AVX2; a token of a
    synthetic decoder at TinyLlama's widths, six layers whose 334 MiB pass this host's L3, x0.791 (93.9 -> 73.4 ms)
    and x0.742, x0.899 and x0.749 capped at AVX2. Under callgrind (the AVX2 instance) the qualification's GEM request
    takes 147.2 M -> 135.6 M instructions and the monolithic runner 137.1 M -> 125.5 M; its logits keep their
    SHA-256, 23 of 23 checks on both trees.
  - Not claimed: a real TinyLlama-1.1B (the synthetic decoder has its widths and six of its 22 layers, with
    pseudo-random codes); a head whose rows are not a whole number of groups (no pack: the rows as they are); any
    other host's figures; counter-based figures (no PMU on this VM).
  DECODE-THREADS (2026-10-26): the C decoder's large matvecs by columns and its head by rows in parts on a runner the
    caller lends, every token and logit the same at every thread count
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §38,
    [measurements/2026-10-26/decode-threads/](research/measurements/2026-10-26/decode-threads/README.json)).
  - DECODE-THREADS (`ef6717a`): every matvec kernel runs over a range of columns [k0, k1), the whole call [0, out), an
    output's chain its column's terms over every input row in order whatever its range
    (`bcir_ai_q8_matvec_f64_columns`). `bcir_llama_runner` (`bcir_llama.h`) lent to the workspace computes a matvec of
    at least 64 Ki weights in parts of its columns and the head over its pack of at least 1 Mi in parts of its rows,
    one part a worker, each output by one part; `bcir_llama_generate_greedy_with_runner`, `bcir_dgem_set_runner` and
    `bcir-llama --threads T`, which lends the trainer's deterministic pool. A survey before the change (draft tools,
    not kept) placed it: at TinyLlama-1.1B's widths from memory four threads took the matvecs to x0.41 and the head to
    x0.27 of their time; the thresholds were then calibrated end to end on the 860K decoder and synthetic decoders
    256, 512 and 1,024 wide.
  - The whole calls (`0c24748`, `ea6fc46`): one kernel serving the whole call and a range, its range a variable in
    every call, `ef6717a`'s whole calls ran the range's code -- under clang its pair kernels carried 7 to 12 more
    stack accesses in their vector loops -- and its first records found them slower. Each vector kernel now has two
    instances, the whole call's (its range the constant [0, out)) and a range's, and a call takes the whole call's
    where its range is the whole. `0c24748` inlined one body into both, which the compilers made other code of (its
    pair kernels' listings 84 to 286 lines from the tree before's under GCC, 134 and 322 for clang's AVX-512 two); in
    `ea6fc46` Q8-PAIR's kernels are one text in two, so the whole call's is the kernel before the ranges instruction
    for instruction under clang 18 and GCC 13. A probe (`bcir_ai_q8_columns_probe`) witnesses the instance each vector
    matvec ran.
  - Witnesses: `q8_columns_test` (ranges and splits on vectors' edges and off them, every output the whole call's,
    2,128 parts a named vector instance; every vector range on its kernel's whole-call instance exactly where it is
    [0, out); Q8-BOUND within a range; the refusals) and the Python column oracle test; the decoder's and GEM's runner
    witnesses (a runner running the parts last first, and the pool at two to four threads, every token and logit the
    same; the exact partition; a part left unrun failing the decode); the CLI's `--threads` parity.
    `decode-threads.json` DT1-DT32, each caught by its own check; under the witnesses before DECODE-THREADS 3 of the
    32 are caught -- DT27, DT28 and DT29, the whole-call instances a column or a vector short, on the path the lanes
    test always ran. HP16, QB8, QB10, PA3, PA6, PA11 and QL3 re-anchored.
  - Measured (`ea6fc46`, on the session's second host -- a Xeon at 2.80 GHz, 4 cores, 1 MiB of L2 a core and 33 MiB of
    L3 -- a reading the median of 3 runs, in one binary lent no runner and the pool at two to four threads): a token
    of a synthetic decoder at TinyLlama-1.1B's widths takes x0.473 of its one-thread time at four threads under clang
    (107.1 -> 51.5 ms) and x0.580 under GCC (x0.458 and x0.498 capped at AVX2), one 1,024 wide x0.585, x0.640, x0.731,
    x0.641 (clang AVX-512, clang AVX2, GCC AVX-512, GCC AVX2), 512 wide x0.775, x0.736, x1.007, x0.776 and 256 wide
    x0.962, x0.845, x1.169, x1.005: the one 256 wide slower under GCC's AVX-512 build, its operations past the default
    thresholds but too small for their parts here. Every generation's tokens and final logits are the one-thread
    decode's bytes. The 860K decoder splits none of its operations, its token x0.971-1.026 at two to four threads. On
    the session's first host, a Xeon at 2.30 GHz with 2 MiB of L2 a core and 260 MiB of L3 where the thresholds were
    calibrated, the same records of `0c24748` (`decode-0c24748.json`) read x0.370 at TinyLlama's widths and x0.454 at
    1,024 wide under clang AVX-512 (x0.351 and x0.432 under GCC), the one 256 wide x0.793. The whole calls, in one
    binary holding the three trees' kernel units with every function on a 64-byte boundary (`kernels.json`, 101
    rounds): `ef6717a`'s x1.102-1.250 of the tree before's time under clang AVX-512 and x0.996-1.046 on AVX2 (seven
    and two of seven shapes resolved slower), the tree after's x0.961-1.002 and x0.981-1.005, and under GCC
    x0.993-1.032 and x0.963-1.002, none resolved slower. At the compilers' own placements the tree after's read
    x0.880-1.223 under clang AVX-512 (six resolved slower), though its instructions are the tree before's: on this
    host where a kernel lies moves its time by up to a fifth. Over six qualification runs a tree its C matvecs took
    6.74 ms (6.30-7.70) before, 8.40 ms (8.27-11.45) at `ef6717a` and 6.61 ms (6.40-6.92) after; under callgrind its
    GEM request takes 135,617,235 -> 135,438,771 instructions, its logits keep their SHA-256, 23 of 23 checks on both
    trees.
  - Not claimed: a real TinyLlama-1.1B (the synthetic decoders have its widths, not its weights); thresholds right for
    every host (the defaults were calibrated on the session's first host, and on its second they split the 256-wide
    decoder at a loss under GCC); counter-based figures (no PMU on either VM).
  KERNEL-ALIGN (2026-10-27): the decoder's units built with the JCC-erratum mitigation on x86-64, so that where the
    linker places the vector kernels no longer moves their time, every value the same
    ([review](research/BCIR_GEOMETRIC_HEAD_REVIEW_2026-10-19.md) §39,
    [measurements/2026-10-27/kernel-align/](research/measurements/2026-10-27/kernel-align/README.json)).
  - KERNEL-ALIGN (`4329b80`): on the session's second host, a core of Cascade Lake's family with Intel's JCC erratum
    (a jump, or a compare fused with one, that crosses or ends on a 32-byte boundary leaves its loop uncached as
    decoded micro-operations), the same instructions of the vector kernels read up to a fifth apart at different
    placements of their 16-byte-aligned text (DECODE-THREADS' `kernels.json`). `bcir_models` now takes
    `x86_64:clang:-mbranches-within-32B-boundaries` and `x86_64:gcc:-Wa,-mbranches-within-32B-boundaries`
    (`runtime/manifest.json`), the assembler padding so that no such jump meets a boundary; the sources do not change.
    M6's grammar takes an `x86_64:` target scope before the compiler scope, each at most once, read by one predicate
    (`bcir.toolchain.manifest_options`) for BCIR Make and the Python builds and by its CMake twin
    (`bcir_manifest_resolve`, held to it under `cmake -P`, a joined list's empty element read as no option);
    `bcir.toolchain.unit_options` gives a build the options of the units that own its sources, and the native AI
    library, the model microbenchmark and the qualification's `bcir-qualify` take theirs there. M18 holds the two
    options to `bcir_models` alone, the three speed builds to `unit_options` and every other module and gate to never
    spelling the mitigation.
  - Witnesses: the grammar's spellings and the resolver's truth table, the machine read from `-dumpmachine`, a
    compiler's identity asked once, `unit_options` over the manifest (test_toolchain); M6's, M14's and M18's injected
    violations, M18's readers over the speed builds' texts and its tree over a scratch root, CMake's scopes against
    the resolver's under `cmake -P` (test_build_manifest); BCIR Make's compile lines on x86-64 and aarch64
    (test_make_rails); the native AI library's command and stamp, the kernels' object -- its text aligned to 32 bytes
    with the unit's options and to 16 without -- and the microbenchmark's command (test_native_ai); the
    qualification's command, plain and callgrind (test_qualification). `kernel-align.json` KA1-KA35, each caught by
    its own check; under the witnesses before KERNEL-ALIGN 10 of the 35, through M18 in the manifest's reconciliation
    test and the native library's stamp tests. FC4 re-anchored on M17's pattern as it now reads.
  - Measured (`4329b80` against `dbe444c` on that host, clang 18 and GCC 13; resolved where a 95% interval excludes
    1 and does not meet the A/A control's): in one binary holding copies of one build of the kernel unit, each copy
    at every placement its build admits (101 rounds, the 860K decoder's matvec shapes, 512 and 2,048 square and the
    packed head, on AVX-512, AVX2 and the portable loops), 18 of the tree before's 162 arms at 16, 32 or 48 bytes
    past a 64-byte boundary resolved slower than its copy at 0 -- clang's AVX-512 pair kernels up to x1.228, GCC's up
    to x1.328, GCC's portable head x1.527-1.619 -- and none of the mitigation's 108 at 0 and 32, nor against each
    shape's fastest placement before; GCC's portable pair kernels, slowest at 0 before, ran x0.764-0.831 of that.
    Beside it loop alignment left 1 of 162 arms slow, 64-byte function alignment 4 of 54, and loops aligned with the
    mitigation 1 of 108. The kernel unit's text grows 2.1% (clang) and 2.4% (GCC), `bcir_models`' 2.7% and 3.0%; in
    the copies timed, the hot loops' jumps on a 32-byte boundary go from 17-19 (clang) and 20-34 (GCC) to none.
    In the decoder (`decode_place.c`, copies of the four sources a decode links, 101 rounds, three runs a decoder) the
    tree before's slow placement took a token x1.069 (clang) and x1.093 (GCC) of its copy's at 0 on the 860K decoder
    and x1.108 and x1.143 on a 256-wide one, each run's interval above 1 and 9 of 12 resolved; the tree after's
    copies x0.989-1.023, none of 24 resolved. The trainer's GEMM (`bcir_tensor.c`, left as it was) read x0.948-1.032 at its placements and
    with the mitigation, none of 40 resolved. Under callgrind (the AVX2 instance) the qualification's GEM request
    takes 135,438,771 -> 138,792,520 instructions (x1.025, the padding's NOPs executed), its logits keep their
    SHA-256, 23 of 23 checks on both trees.
  - Not claimed: the mechanism (no PMU on this VM, so no count of the micro-operation cache's misses: the boundary
    jumps are a correlate the mitigation removes, and clang's slow placement at 32 holds as many as its fast one at
    0); any other host's figures, or a gain on a core without the erratum, where the padding costs its 2-3% of code;
    aarch64 and Windows timing (the option is x86-64's).
  SILICON-NULL (2026-10-27): `bcir.silicon`'s PMU flags, and every value it cannot read recorded as None -- never 0,
    never a nearby number.
  - The perf flags: `_perf_open` set bit 20 of perf_event_attr's flag word for exclude_kernel; bit 20 is
    exclude_guest, exclude_kernel bit 5 (include/uapi/linux/perf_event.h). The counter counted the kernel, and at the
    default perf_event_paranoid=2 an unprivileged open was refused, so a bare-metal host with a PMU read as having
    none -- and `MeasuredPlan`, whose `hardware_pmu` is the host's attestation, recorded "silicon" with 0 cycles. The
    attr is now built as bytes (`_perf_attr`, PERF_ATTR_SIZE_VER1): disabled, exclude_kernel and exclude_hv (bits 0, 5
    and 6, named as constants), the read format asking for the enabled and running times; a big-endian host, whose
    bitfields the kernel allocates from the top, opens nothing. The other two openers were right and stay as they
    are: `runtime/c/bcir_asn1_bench.c` sets the struct's named fields, `tools/perf/ot_contract.py` bits 5 and 6.
  - Null, never zero: a counter read that failed, came back short, was never enabled, never ran, or ran for part of
    its window (multiplexed) is None (`_perf_count`), and `HwSample.ipc_milli` None unless both counts were read and
    cycles moved; `pmu_counts` is the one test every record makes before it carries PMU counts, so the telemetry ring
    and `silicon_dna` carry the PMU's cycles and misses or the OS's pair, never one of each, and claim the `pmu`
    provenance only for the first. `RaplSampler.lap_uj` gains the range back on a wrap and is None when the range is
    unreadable or a reading lies past it (it was 0 J), and a lap after a failed read has no start (its energy was the
    next lap's). `cpufreq_info` has no fallback: `/proc/cpuinfo`'s "cpu MHz" -- the current clock on bare metal, the
    hypervisor's reported one in a guest (2800.198 MHz here, the TSC's rate) -- was taken for the nominal frequency;
    without a cpufreq driver the frequency is now None. The kernel's text is read in its own grammar (`_decimal`:
    ASCII digits, a '-' only for a thermal zone), never `int()`/`float()`, which take '+5', '1_000', 'inf', non-ASCII
    digits; a scaling table with one malformed entry is no table.
  - The evidence records (`MeasuredPlan`, `ProviderMeasurement`, `MeasuredScheduleCandidate`) take their counts
    through `pmu_counts` and keep their schemas' documented zeros when the PMU gave none; giving them a null of their
    own is a schema change, proposed rather than made here.
  - Witnesses, host-independent (a PMU-less guest, an unprivileged CI runner, a root workstation alike), through
    patchable hooks (`_perf_syscall`, `_perf_ioctl`, `_perf_read`, `_perf_close`, `_read`): the attr held as bytes to
    the uapi layout; every unreadable-count case, the work run once and every fd closed; the IPC; the ring's and the
    DNA's fallback; the grammar; the frequency with and without cpufreq, a "cpu MHz" offered and never taken; the
    RAPL laps; ot_contract's probe held to the same flag bits; the three records. DVFS quantization is now tested
    against a stated table, since a guest without cpufreq has no frequency of its own to anchor the host-read test.
    `silicon-null.json` SN1-SN25, each caught by its own check; under the tests before SILICON-NULL 1 of
    the 25 (SN12, incidentally: on a host without a PMU its close of a refused counter raises inside the measured-plan
    tests).
  NDT-BF16 (2026-10-17): the GEMMs on bfloat16 operands, accumulated as the AMX unit accumulates them.
  - The instance the container moved to at 10:15 has AMX and AVX512-BF16. The tile unit's
    `TDPBF16PS` was pinned first, bit for bit, against twelve candidate models
    ([measurements/2026-10-17/bf16/](research/measurements/2026-10-17/bf16/), the report's §31): per
    instruction two lanes from +0.0, the even terms' fused products in one and the odd terms' in the
    other, every step flushed when tiny by IEEE tininess after rounding with the exponent unbounded,
    then `c = daz(c) + (E + O)`; MXCSR ignored, DAZ on both operands and C. The conversion is
    nearest-even after DAZ on every one of the 2^32 inputs; `VDPBF16PS` chains the odd term first,
    another arithmetic.
  - That is the bf16 contract (`bcir_tensor.h`), with portable, AVX-512 (FMA under MXCSR's FTZ and
    DAZ) and AMX (admitted by host checks and a differential, once a process) instances over
    operands packed into workspace the caller lends; a shorter workspace runs in chunks to the same
    bits. An independent `__float128` reference finds 0 differences in 963 cases under GCC 13, Clang
    18 and Clang 23 (718 capped below AMX, 473 portable); the GEM+ row `train.mm.bf16.mismatch` (0
    of 28) holds it to the oracle's exact rationals; `tools/testing/faults/ndt-bf16.json` holds 17
    injected faults. The last two are clang's analyzer's (18 and 23, before the push): the grouped
    entries give every call they take its partition, and a call with no rows or no columns divided
    its region count by zero there; the region width is counted in whole 32-column blocks since, the
    same width on all 126,000,000 non-empty shapes compared (`bf16/cols_check.c`), and the C harness
    holds every entry to the empty shapes.
  - The trainer's `BCIR_DT_CONTRACT_BF16` (state ABI 4: a lent workspace of
    `bcir_decoder_workspace_bytes`, 11.5 MB for the 9.84M decoder; Python `bf16`, `bf16+exp`, and
    each `+reduce`; the workspace in the budget; no CBLAS provider admitted). Only the GEMMs change.
  - Measured at four threads: the step's GEMM shapes 16.5 -> 4.4 ms (x3.79), the 9.84M decoder's step
    188.1 -> 79.8 ms (5,445 -> 12,832 tokens a second, x2.36; x2.43 at one thread). The AVX-512
    instance alone runs at 0.56 of the FP32 kernel: without AMX the contract buys no speed. Six
    steps end in one state through the tile, AVX-512 and portable instances at one and four threads.
  LM-BF16 (2026-10-18): BCIR's bf16 against PyTorch's, five seeds paired.
  - The comparison harness gained `torch-eager-bf16` and `torch-compiled-bf16` -- PyTorch's CPU mixed
    precision, `torch.autocast` to bfloat16 around the forward and the loss, the weights, gradients,
    moments and fused AdamW in FP32 -- and BCIR's packing workspace in its memory split. LM-PAIRED's
    five seeds ran four arms each at four threads in one session, the arms' order rotated, with the
    analysis fixed before the runs ([bf16/lm/](research/measurements/2026-10-17/bf16/lm/), the
    report's §32).
  - BCIR under `bf16+exp+reduce` trains at 2.08 times the tokens a second of compiled PyTorch under
    autocast (95% interval 1.98-2.19, faster on five of five: 13,445-14,666 tokens a second against
    6,607-7,235) and 2.91 times compiled PyTorch's FP32. Its test cross-entropy is 0.0009 nats above
    PyTorch bf16's and 0.0012 above its own FP32 run's, both intervals inside the ±0.01-nat margin
    fixed beforehand: the quality parity the roadmap's NDT-BF16 row asked for. Autocast buys PyTorch
    1.40 times its FP32; BCIR FP32 against compiled FP32 is 1.23 times again (1.20-1.27). Eager
    PyTorch under autocast ran slower than compiled on the two seeds of the check. Peak RSS 540 MB
    against 1,141-1,153.
  - The two roadmap rows that still counted `ndt-bf16.json` as NB1-NB15 read NB1-NB17.
  LM-PAIRED (2026-10-17): BCIR against `torch.compile` with fused AdamW, five seeds paired.
  - An external review's bar for a public claim (the fastest matched PyTorch configuration, a paired
    interval, at least three seeds, a quality margin fixed beforehand): the comparison harness gained
    `prepare --seed` and a memory split (aeb7d15), and five seeds ran one pair each in one session,
    the order alternating ([measurements/2026-10-17/paired/](research/measurements/2026-10-17/paired/),
    the report's §30).
  - BCIR trains 1.262 times PyTorch's tokens a second (95% interval 1.240-1.284; 176.5-185.2 against
    218.4-237.4 ms a step), faster on all five seeds; the test cross-entropy difference -0.00001 nats
    (interval -0.00006 to +0.00003, inside the +-0.01 margin); 528-529 MB of peak RSS against
    1,055-1,121, the same 157,390,848 bytes of persistent state on both.
  - A check added after the runs: eager PyTorch on two seeds, 0.8% faster and 12.6% slower than their
    compiled runs; against the faster arm wherever both ran the ratio is 1.260 (1.241-1.279).
  - On the instance the container moved to at 10:15 (a 2.10 GHz Xeon, a 260 MB L3), where both sides
    ran faster than in r9's session; the margin there was 1.37 in one round.
  NDT-PACK (2026-10-17): the packed tiles kept between a thread's tasks, measured and not kept.
  - An external review's second item (the 2026-10-13 training-architecture blueprint: reuse packed
    panels where legal, from a workspace the caller lends, keyed by the value's version), as three
    arms over d68e3d3's tensor unit that keep the partition, so every product is the same floats
    ([measurements/2026-10-17/pack/](research/measurements/2026-10-17/pack/), the report's §29):
    a job's tasks numbered column block by column block; that order with a thread's copy of one
    column block's tiles, keyed by a per-job sequence number and never B's address; and groups of
    four column blocks row set by row set.
  - The GEMM shapes alone at four threads, warm: the column order 151.55 -> 141.34 ms a step in one
    run (10 rounds of 12) and 134.59 -> 131.56 in the next (6 of 12), the cache 138.66. Cold in L2:
    the column order 0.6% and the cache 2.2% faster, the groups 3.4% slower. In place the GEMMs
    143.7 -> 142.2 (column order, 4 rounds of 8) and 141.5 (cache, 6 of 8), inside base's own
    rounds (132.9-165.4). On the instance the container moved to at 10:15 (a 2.10 GHz Xeon, 2 MB
    of L2 a core, a 260 MB L3, AVX512-BF16 and AMX).
  - The order changes only how often a pack finds its panel in the packer's own L2, and the cache
    saves only the pack's copy, which the inner loop did not wait on in §18's probe. Not kept.
  LM-4T (2026-10-17): BCIR against PyTorch at four threads on 2afa3e3.
  - The 1,000-step language-model run with NDT-UPDATE, one round in its session
    ([measurements/2026-10-17/lm/](research/measurements/2026-10-17/lm/) r9-*, the report's §28):
    - BCIR 203.7 / 203.4 ms a step (5,026 / 5,035 tokens a second);
    - eager PyTorch 321.4 ms (3,187);
    - `torch.compile` 278.2 ms (3,681).
  - BCIR is 58% ahead of eager and 37% ahead of compiled, at the same held-out cross-entropy and
    529-530 MB against 1,088-1,123. Both BCIR states are `5137e013…`, and these are the fastest BCIR
    runs recorded.
  - Most of the wider gap is PyTorch's: against §26's session, eager ran 9% slower and compiled 4%.
  NDT-UPDATE (2026-10-17): the AdamW update's commit and first pass, the same values.
  - A probe of the trainer's own functions at the 9.84M decoder's size on four threads
    ([update/update_probe.c](research/measurements/2026-10-17/update/update_probe.c)) found neither
    pass memory-bound, as §11 of the report had read them. The commit was bound by its divider:
    8.40 ms a call against 4.17 for its streams with trivial arithmetic, and streaming stores made it
    slower. The first pass was bound by the reduce contract's norm chain, 16,384 dependent double
    additions a chunk after the chunk's vector loop: 6.84 ms against 3.72 for its flag and maxima alone.
  - The commit's two divisions an element by the bias corrections 1 - beta1^t and 1 - beta2^t, which
    every element of a call shares, go through `div_shared`: y = RN(1/d) once a call, q0 = x*y, and
    two FMA corrections, the second Markstein's theorem (Handbook of Floating-Point Arithmetic, 2nd
    ed., Theorem 4.10: RN(x/d) for a faithful q and y within half an ulp of 1/d). `copysign` keeps
    a zero's sign. It is used in the AVX-512 commit and a new AVX2+FMA commit (variant bit 8, the
    fused GEMM's AVX2 bit, which implies FMA). The portable commit, the FMA-less AVX2 commit and the
    preflight keep the true divisions; each element's own division and root stay.
  - The first pass under the reduce contract on a vector variant takes two chunks a task
    (`pair_avx512`, `pair_avx2`): both chunks' flag and maxima a vector at a time, and each chunk's
    squares added to its own chain in index order between the loads. Each chain then runs under
    the streams' reads and under the other's latency. Short pairs, an odd last chunk, the chain
    contract and the portable build take the per-chunk loop.
  - Evidence:
    - the C harness's `update_pairs`: two decoders of 19,616 and 16,032 parameters, whole and short
      pairs, NaN past the parameters. The state, verdict and norm match `ref_update` bit for bit
      under both norm contracts, no pool and three, two scales, clipping on and off. 512 planted
      refusals at the ends of sixteen chunks leave the state untouched, 64 -0.0 second moments are
      admitted, and a -0.0 weight's zero sign is kept;
    - under `BCIR_DT_DIVISION_PROBE` (the harness's builds only), the dispatched shared division
      against the division over 268,444,464 float dividends for sixteen divisors;
    - the Python test's new build capped at AVX2;
    - the clipping-norm test's new one-chunk bench (728 parameters). The chain found `ndt-reduce.json`
      NR11 escaping its own check once the threads bench's whole pairs took the paired loop;
    - `ndt-update.json` UP1-UP14, 14 of 14 caught;
    - [division_check.c](research/measurements/2026-10-17/update/division_check.c): 24,245
      divisors (14,837 distinct) times 16,782,765 dividends, 0 mismatches;
    - the statedump differential: the same 376,368,795 bytes and 9.84M digest;
    - the 20-step state `4bf4e057…`.
  - Measured in place at four threads, eight rounds rotated with aaa2e6c's sources
    ([measurements/2026-10-17/update/](research/measurements/2026-10-17/update/), the report's §27):
    the update dropped from 15.33 to 10.06 ms a step (commit 8.73 -> 6.84, first pass 6.62 -> 3.15)
    and everything but the GEMMs from 50.78 to 46.22, both in every round. The first pass now sits
    at its streams' bound. The commit's remaining 2.0 ms over its streams is mostly each element's
    own division and root.
  - The pre-push chain's runtime gate found the portable build (`BCIR_TENSOR_NO_DISPATCH`) refusing
    `adamw_commit_fma` as an unused function under Clang's `-Werror`: GCC does not warn on an unused
    static inline, and none of the slice's harness builds was portable. Every no-dispatch target
    (aarch64, Windows, the portable build) would have failed the same way. The three helpers are now
    defined among their only callers, inside the dispatch block, with the same text: `ndt-update` is
    14/14 and the 20-step state is `4bf4e057…`.
  LM-4T (2026-10-17): BCIR against PyTorch at four threads on cf435d4.
  - The 1,000-step language-model run with NDT-SCAN-W, one round in its session
    ([measurements/2026-10-17/lm/](research/measurements/2026-10-17/lm/) r8-*, the report's §26): BCIR 204.7 / 207.4 ms
    a step (5,002 / 4,936 tokens a second), eager PyTorch 293.6 (3,487), `torch.compile` 267.0 (3,835) --
    BCIR 42-43% ahead of eager and 29-30% of compiled, the same held-out cross-entropy, 529 MB against
    1,116-1,124, both BCIR states `5137e013…`; the first BCIR run past 5,000 tokens a second.
  NDT-SCAN-W (2026-10-17): the activations' finiteness taken by their writers.
  - The forward's verdict on its activations had been one scan of the whole arena after the forward
    -- 171 MB for the 9.84M decoder, read back from memory, 6.0 ms of the four-thread step; scanning
    each layer when it was done (NDT-GEMM-CAND) found the same 6.6 ms: the shared L3 does not keep a
    layer's 26 MB between their writes and the scan.
  - `bcir_tensor_rms_scan_pool`, `bcir_tensor_silu_mul_scan_pool` (and `_v2`) and
    `bcir_tensor_attention_scan_pool` (and `_v2`) compute their plain entries' values, bit for bit,
    and set a verdict when one of their regions holds an element that is not finite, read back by the
    tasks that wrote them into one byte each of the caller's stack (a job past the bytes' bound
    scanned after it on the pool). A region answers through another its value reaches -- the
    RMSNorm's input rows through its output, the SwiGLU product's gate and up through the product,
    the attention's queries, values and probabilities through its context -- and the attention reads
    its keys itself, the one region the softmax can hide (a key whose scores are all -inf weighs 0).
    The trainer's forward calls them for every region of the arena and scans only the logits after
    the head; a provider's SiLU has its three regions scanned after it.
  - Evidence: the C harness's `writer_verdicts` (each entry its plain entry's values and its regions'
    verdict over 560 cases: every task chunk's ends in every input region, overflowing outputs, a set
    verdict kept, the hidden key; through the trainer the key behind RoPE, logits that overflow alone
    and a provider's SiLU that answers 0 for an infinite gate, refused with every other activation
    finite) and its Python test's build with the bytes' bound at two tasks; `ndt-scanw.json` SW1-SW17
    (17 of 17 caught; the edit had left three earlier tables' entries without their anchors --
    `ndt-exp.json` NE5, `ndt-fuse.json` NU7, `ndt-threads.json` NT5 -- re-anchored on its calls in
    aaa2e6c); an old-against-new differential of the verdict over three specs, the twelve
    contracts, pools of one, two and four and ten crafted states, every observable byte hashed -- the
    same table and digest; the statedump differential's same 376,368,768 bytes and 9.84M digest; the
    20-step state `4bf4e057…`.
  - Measured in place at four threads, eight rounds rotated with 4fcaa5e's sources
    ([measurements/2026-10-17/scanw/](research/measurements/2026-10-17/scanw/), the report's §25):
    the activations' scan 5.95 -> 0.05 ms a step and everything but the GEMMs 56.0 -> 50.3, every
    round; the writers' reads 0.5 ms among them once the attention reads each region in one call (a
    first form, one call a strided row, spent 0.6 ms in the attention alone).
  LM-4T (2026-10-16): BCIR against PyTorch at four threads on 14b5941.
  - The 1,000-step language-model run with NDT-ATTN-7, one round in its session
    ([measurements/2026-10-16/lm/](research/measurements/2026-10-16/lm/) r7-*, the report's §24): BCIR 211.0 / 225.6 ms a
    step (4,853 / 4,538 tokens a second), eager PyTorch 304.5 (3,362), `torch.compile` 310.5 (3,298) --
    BCIR 35-44% ahead of eager and 38-47% of compiled, the same held-out cross-entropy, 529 MB against
    1,137, both BCIR states `5137e013…`; against §22's session eager's median step was 14% shorter
    and compiled's 5% longer, the slice's own share being §23's in-place reading.
  NDT-ATTN-7 and NDT-PREPACK (2026-10-16): the attention's diagonal terms in registers, and the
  GEMM's tiles packed once, measured and not kept.
  - Cycle counters around the attention's phases put 62% of its task time in four products -- the
    forward's context and the backward's dv, dq and dk -- whose kernels (`att512_context`,
    `att512_transposed`) took a block's diagonal terms in a loop bounded by the term, indexing the
    block's register arrays by a run-time row; at -O3 GCC kept the arrays in memory (21 of the
    context's 24 accumulators stored to the stack every step of its main loop) and Clang spilled
    the context's too.
  - NDT-ATTN-7: the diagonal terms run over all eight rows, a row that does not take a term
    merge-masked out of its sum (its accumulator the one it held); no stack store or load is left
    in either kernel at -O3 under GCC 13.3, Clang 18.1 or Clang 23.1.
  - Evidence: the C harness's attention reference (finite and not) under GCC and Clang, C11 and
    C2x, -O2 and -O3; `ndt-attn.json` NA5 and NA8 re-anchored onto the masks, NA15-NA16 new (16 of
    16 caught); the old-against-new differential of every observable byte: the same 376,368,768
    bytes and 9.84M digest; a differential over twelve shapes with two NaN payloads among its
    operands, every value the same under both compilers, only which NaN survives where two meet
    moving -- as it already moved between GCC's and Clang's builds of the old kernels and between
    every build and the scalar reference; the 20-step state `4bf4e057…`.
  - Measured in place at four threads, eight rounds rotated with 64bfd62's sources
    ([measurements/2026-10-16/attn7/](research/measurements/2026-10-16/attn7/), the report's §23):
    the attention forward 7.63 -> 6.65 ms a step and the backward 12.03 -> 10.06, everything but the
    GEMMs 62.2 -> 57.8, every round; alone 7-10% at one thread and inside the host's noise at four.
  - NDT-PREPACK, measured: a pack phase packing each column block's tiles once into a workspace
    the pool's tasks read, in place of every 128-row task packing its own (5.6-12.6% of a
    one-thread task, §18) -- the GEMM shapes alone at four threads 10.0% slower, in place the GEMMs
    173.0 -> 178.4 ms a step: the tiles come from L2 or L3 either way, so sharing them saves only
    the copy and costs a pass, a barrier and the cross-core reads of lines another core just
    wrote. Not kept ([measurements/2026-10-16/prepack/](research/measurements/2026-10-16/prepack/)).
  LM-4T (2026-10-16): BCIR against PyTorch at four threads on 64bfd62.
  - The 1,000-step language-model run with NDT-FUSE and NDT-ATTN-6, one round in their session
    ([measurements/2026-10-16/lm/](research/measurements/2026-10-16/lm/) r6-*, the report's §22):
    BCIR 227.3 / 226.0 ms a step (4,504 / 4,531 tokens a second), eager PyTorch 353.2 (2,900),
    `torch.compile` 296.9 (3,449) -- BCIR 55-56% ahead of eager and 31% of compiled, the same held-out
    cross-entropy, 529 MB against 1,119-1,123, both BCIR states `5137e013…`. The runs started from
    8b4f23d with the slices uncommitted; the sources they built are 64bfd62's blobs, hashed beside the
    records.
  NDT-FUSE and NDT-ATTN-6 (2026-10-16): the step's passes merged, and the attention's fills.
  - After NDT-SCHED-1 a third of the four-thread step was outside the GEMMs (68.2 of 228.9 ms a step
    in place), and five of its passes were there only because a neighbouring pass stopped short: the
    SwiGLU forward's SiLU and product as two passes over each layer's hidden activations, the
    residual's adds after the RMSNorm VJPs had stored their dx, the gradients' 9.84M-float zero fill,
    the backward's closing gradient scan that the update's first pass repeats, and the attention's
    fills under kernels that then add their first terms to the zeros.
  - NDT-FUSE: `bcir_tensor_silu_mul_pool` and its v2 twin (the two roundings the passes made);
    `bcir_tensor_rms_backward_into_pool` with `BCIR_TENSOR_RMSB_ADD` (the VJP's element added to dx in
    place) and `BCIR_TENSOR_RMSB_FIRST` (the gamma adjoint's sums from +0.0); under a fresh backward on
    BCIR's GEMM each gradient's first writer starts it -- the weight adjoints at beta 0, which
    NDT-BETA's first K block adds to +0.0 without reading C, a sum begun at +0.0 never being -0.0 --
    with only an untied embedding filled for its scatter and a provider's GEMM keeping the fill and
    beta 1; inside a run the last micro-batch's closing scan is the update's first pass, a
    non-finite gradient still reported as `BCIR_DT_NUMERIC` ahead of the update's own
    `BCIR_DT_INVALID`, and later micro-batches skip the opening scans already answered.
  - NDT-ATTN-6: the AVX-512 attention's scores, context and dq blocks start from +0.0 in registers on
    a head's first write, dk and dv on a KV head's first query head; the fills cover only the
    probabilities' future columns past a row's blocks and the v1 kernel's rows; the scores' last
    channel block stores its future lanes as +0.0; the key panel is packed by 16 x 16 register
    transposes.
  - Evidence: the C harness's `fused_passes` (the product over special operands and in place, the
    VJP's four modes, a NaN-filled gradient under accumulate 0 against the zero-filled one on a tied
    and an untied fixture under four contracts, the scans' refusals on a crafted state whose
    activations are finite and whose gradient overflows); `ndt-fuse.json` NU1-NU14 and NZ1-NZ7,
    `ndt-exp.json` NE5, `ndt-lanes.json` NL4 and `ndt-scan.json` NS6-NS7 (onto the fills that remain)
    re-anchored; an old-against-new differential of every
    byte the API can observe (three small specs under the twelve contracts at one, two and four
    threads, and the 9.84M spec): the same bytes from both trees; the 20-step state at one and four
    threads, widest, AVX2 and portable builds, `4bf4e057…`.
  - Measured in place at four threads, eight rounds of the three arms rotated
    ([measurements/2026-10-16/fuse/](research/measurements/2026-10-16/fuse/), the report's §21):
    everything but the GEMMs 68.2 -> 64.5 ms a step with NDT-FUSE (lower in seven rounds of eight) and
    61.3 with both (every round), the step 228.9 -> 221.8 with both.
  LM-4T (2026-10-16): BCIR against PyTorch at four threads on 78cbd29, again.
  - The 1,000-step language-model run with NDT-BETA and NDT-SCHED-1, two rounds in one session
    ([measurements/2026-10-16/lm/](research/measurements/2026-10-16/lm/) r4-*, r5-*, the report's
    §20): BCIR 241.8 / 236.7 / 256.6 ms a step (3,991-4,327 tokens a second), eager PyTorch 362.6 /
    369.7 (2,824 / 2,770), `torch.compile` 321.6 / 310.3 (3,184 / 3,300) -- BCIR 44-53% ahead of
    eager and 21-36% of compiled round by round, the same held-out cross-entropy, 529 MB against
    1,060-1,142, every BCIR state `5137e013…`. That session's host ran PyTorch a quarter slower than
    the first session's, so the gap is wider than the slices alone would make it.
  NDT-SCHED-1 (2026-10-16): the linears that read one input as one pool job.
  - Five of a layer's seven linears read one input -- q, k and v the first norm's output, gate and
    up the second's -- and each was its own job behind its own barrier; their input adjoints add
    into one dx, so the backward ran their pair jobs (NDT-SCHED-0) one after another, the narrow k
    and v leaving threads idle at each job's tail.
  - `bcir_tensor_mm_group_pool` runs a chain of up to four calls into one C (the same c, m and n:
    a tile's task runs each chained call's task for that tile in turn, the order the calls would
    run whole) beside up to four independent calls, each its own partition. The forward's q, k, v
    and gate, up are independent calls; the backward's input adjoints the chain, the weight
    adjoints beside it -- six barriers fewer a layer. A provider, one thread, more calls or a
    chain of other shapes runs the calls one after another; every element receives the same
    operations in the same order.
  - Evidence: the C harness's group tests (a chain of three with two independent products, each
    alone, both serial fallbacks, under the three GEMM contracts at every thread count);
    `ndt-threads.json` NT13-NT18; the 20-step state at one and four threads `4bf4e057…`.
  - Measured in place at four threads, four rounds alternated with NDT-BETA's tree
    ([measurements/2026-10-16/sched1/](research/measurements/2026-10-16/sched1/), the report's
    §19): the GEMMs 179.5 -> 171.3 ms a step, the step 254.2 -> 241.2, every round.
  NDT-BETA (2026-10-16): the blocked GEMM's beta scaling in its first K block.
  - NDT-GEMM-L1's probe of the AVX-512 kernel over the pool's partition found its inner loop alone
    at the FMA probe's rate, 162-171 GFLOP/s, at every depth from 64 to 448 -- the contract's depth
    and the tile's L2 residence cost nothing, against NDT-GEMM-4T's guess -- and the rest of a task
    its tile's pack (5.6-12.6% of its cycles) and the pass scaling its elements of C by beta
    (1.6-8.9%); fourteen variants that keep the arithmetic (depth slices, prefetch distances,
    row-task heights) moved nothing beyond one run's noise.
  - Under the blocked contract the pass is gone: the kernels scale C in their first K block's
    epilogue, under beta 0 adding the terms to +0.0 without reading C, under any other beta scaling
    the loaded C by the same rounded product, a later block adding to C as stored -- the same
    operations in the same order. Beta and the lane masks are formed after the depth loop: with
    beta live across it the block's 28 of 32 vector registers spilled every depth (2-4x slower). A
    depth of zero and the v1 and fused contracts keep the pass.
  - Evidence: the C harness's blocked-contract test (a depth of zero at beta 0, 0.5, 1 and -2, and
    a zero product at alpha -1 under beta 0 storing +0.0 at depths 1, 449 and 897, on the serial
    entry, both pool partitions and every instance); `ndt-kblock.json` NK3b and NK11-NK17, NK10 and
    `ndt-threads.json` NT3 re-anchored; `train.mm.kblock.mismatch`'s 897-deep case at alpha 1.1
    (was 0.5, a product the binary32 multiply scales exactly), so a fused rounding in a later K
    block's epilogue -- now its own code -- stays in the row's sight, and the row's witness test
    given the fold's wrong copies; the 20-step state at one and four threads, widest, AVX2 and
    portable builds, `4bf4e057…`.
  - Measured ([measurements/2026-10-16/beta/](research/measurements/2026-10-16/beta/), the report's
    §18): the step's GEMM shapes alone at four threads 198.6 -> 175.3 ms a step over twelve
    alternated rounds (10 faster), neutral at one thread (551.3 -> 546.7; the gate and up input
    adjoint 4% slower there); in place at four threads the GEMMs 199.6 -> 179.5 ms a step in every
    round of four, the step 267.8 -> 254.2.
  NDT-GEMM-4T (2026-10-16): the four-thread GEMMs measured, and BCIR against PyTorch at four threads.
  - On this instance an FMA probe gives one thread 162-165 GFLOP/s alone and 144-163 each with four
    running; the step's GEMMs, timed call by call in place, are 162.8 ms of a 235 ms step at 371
    GFLOP/s, 57-65% of it, with 2% of the pool idle inside the calls -- merging jobs cannot buy
    more. Cold operands cost 10%; the narrow weight gradients' 16-24 tasks run 50-60% slower at four
    threads than alone; the transposed-A penalty of the previous instance is gone (NDT-TA: no
    slice), and L1-sized depth slices gained nothing (NDT-KSUB: dropped).
  - The 1,000-step language-model run at four threads: BCIR 234.9 / 235.1 ms a step (4,359 / 4,356
    tokens a second; 247.2 / 249.4 before NDT-LANES, NDT-ROPE and NDT-ATTN-5, the same state
    `5137e013…`), eager PyTorch 283.8 (3,608), `torch.compile` 259.6 (3,945), the same held-out
    cross-entropy, 529 MB against 1,084-1,128. `train_comparison.py --threads` now sets PyTorch's
    intra-op threads too and records them ([measurements/2026-10-16/gemm4t/](research/measurements/2026-10-16/gemm4t/),
    [lm/](research/measurements/2026-10-16/lm/), the report's §17).
  NDT-ATTN-5 (2026-10-16): the backward's probability-adjoint tiles from a transposed panel.
  - Each row block's 32 x 64 tiles of `dp = dY V^T` went through the general GEMM kernel, which
    packed the transposed values again for every tile -- a third of the backward at one thread. A
    backward task now transposes its KV head's values once into its stack (`d x t` floats, up to
    64 KB; a longer context takes the GEMM kernel as before), and every query head's tiles are
    AVX-512 register blocks of eight rows by 32 columns reading that panel, only the blocks that
    hold some `j <= i` (the others are never read), with the v1 kernel's arithmetic for those
    operands: from the tile's zero, a rounded product and a rounded sum per channel in order.
  - Evidence: the C harness's attention reference (its 16 shapes, finite and not); `ndt-attn.json`
    NA10-NA14; the 20-step state at one and four threads `4bf4e057…`.
  - Measured in place at four threads, four rounds alternated with NDT-ROPE's tree
    ([measurements/2026-10-16/dp/](research/measurements/2026-10-16/dp/), the report's §16): the
    attention backward 13.64 -> 12.37 ms a step, every round.
  NDT-ROPE (2026-10-16): RoPE's frequencies once a range of pairs.
  - The kernel computed `angle = pos * pow(base, -2j/dim)` in double and its float cosine and sine
    per pair and position -- 3,072 `pow`s a call where 24 differ -- and then turned every head's
    pair at stride `dim`. Each position-range task now computes the frequencies once per range of
    64 pairs, each position's angles once, and turns a head's pairs contiguously: the same double
    angle, the same cosine and sine, the same rotation per element.
  - Evidence: a per-pair reference (the kernel before it) in the C harness at 2 x 40 x 3 x 8 and
    1 x 33 x 2 x 260 (three ranges of pairs), forward and inverse, serial and pooled -- the old pool
    test compared the kernel with itself; `ndt-rope.json` NO1-NO5; the 20-step state at one and four
    threads `4bf4e057…`.
  - Measured in place at four threads, four rounds alternated with NDT-LANES's tree
    ([measurements/2026-10-16/rope/](research/measurements/2026-10-16/rope/), the report's §15):
    RoPE forward 2.05 -> 0.96 ms a step, backward 2.53 -> 1.13, every round.
  NDT-LANES (2026-10-16): the step's per-row chains as the lanes of a vector.
  - The step's in-place attribution on this host (the container restarted on a faster instance of
    the same CPU model; the four-thread step about 236-254 ms) put RMSNorm's two kernels at 8.9 ms
    and the attention at 25. Each RMSNorm row and each softmax row was one dependent chain -- a
    rounded add per column, four cycles each -- run one row at a time, the softmax's exponential
    ran each row's remainder one element at a time, and the gamma adjoint's column tasks loaded
    and stored their sums once per row.
  - On AVX-512 a whole chunk of 16 rows now runs as the lanes of one vector: 16 columns of the 16
    rows transposed in registers (the GEMM pack's transpose), lane r adding row r's terms in column
    order, so every sum, dot and maximum is its scalar loop's bit for bit (the scalar
    `x > max ? x : max` is VMAXPS's operand rule). The softmax's exponentials under the exp contract
    run in those transposed blocks whole, the diagonal block's upper lanes stored as the zeros they
    were; the gamma adjoint's column sums stay in registers and reach dg once.
  - Evidence: the C harness's references (RMSNorm at 40 x 37 and 48 x 389 -- whole chunks, a part
    chunk, a ragged column tail -- and the attention reference's shapes, finite and not, at every
    variant build); `ndt-lanes.json` NL1-NL11, `ndt-reduce.json` NR5 re-anchored; the 20-step state
    at one and four threads `4bf4e057…`, the one it was.
  - Measured in place at four threads, five rounds alternated with the tree before
    ([measurements/2026-10-16/lanes/](research/measurements/2026-10-16/lanes/), the report's §14):
    the attention forward 11.10 -> 8.56 ms, the RMSNorm VJP 6.34 -> 4.48, RMSNorm 2.62 -> 1.45,
    every round; the step 248.1 -> 240.6 ms, its GEMMs unchanged within the host's few percent.
  NDT-PAGES (2026-10-14): the trainer's state on 2 MB transparent huge pages.
  - The six state buffers (335 MiB on the 9.84M decoder) were ctypes arrays: glibc serves them from
    anonymous mappings it never advises, and under the host's `madvise` THP mode not one of their
    pages was huge; a GEMM walking a transposed operand's rows touched a new 4 KB page every few
    rows, past the second-level TLB's reach.
  - Where Python exposes `MADV_HUGEPAGE` (Linux), `NativeDecoder` places them on one private
    anonymous mapping advised before any page is touched, the first buffer on a 2 MB boundary and
    each next 64 bytes past the last one's end (`_state_buffers`); elsewhere, or with
    `huge_pages=False`, ctypes arrays as before, and `resume` takes the placement. Private, because
    shared anonymous memory is shmem, whose huge pages follow `shmem_enabled` instead.
  - Evidence: the placement witness (the layout, the private mapping, the advice in its VmFlags,
    three steps' state equal to the arrays' at one and four threads, `resume`'s placement, the
    refusals); `ndt-pages.json` NG1-NG8, NG8 a kernel whose arithmetic follows its operand's
    address; the 20-step state at one and four threads 4bf4e057…, the one it was.
  - Measured ([measurements/2026-10-14/pages/](research/measurements/2026-10-14/pages/), the
    report's §13): the step alternated in one process 403.9 / 405.5 -> 349.2 / 341.4 ms at four
    threads (19 of 20 pairs each), 668.8 -> 627.6 at two, 1101.8 -> 1009.9 at one; the step's GEMM
    time 16% lower at one and at four threads, the transposed-A weight gradients most (384 x 384 x
    1024: 59.1 / 56.1 -> 89.0 / 88.3 GFLOP/s at one thread); in place at four threads 407.8 / 381.0
    / 388.0 -> 348.0 / 340.0 / 330.6 ms. The GEMMs at four threads reach 37-52% of the four cores'
    peak on the step's large shapes (26-45% before): the TLB was part of the gap, not all of it.
  NDT-POOL (2026-10-13): the pool's tasks claimed from one counter, a hot path without the lock.
  - The four vCPUs are four cores with their own FMA units (twelve FMA chains in registers:
    141-148 GFLOP/s pinned to one, 579-584 to all four), yet the GEMMs gained about 2x from four
    threads; the pool's latency was the first suspect and was part of it:
    an empty four-task job cost about 80 us (a futex wake costs tens of microseconds on this VM),
    sixteen 25 us tasks 191-195 us for 100 us of work a thread.
  - Claims from one counter (GCC's and Clang's atomics; MSVC keeps task t on thread t mod threads),
    the job published by the generation's release store, a report as one atomic decrement (the
    last reporter alone takes the lock, to signal the caller), a wait of 64 pauses, 512 yields,
    then sleep. The budget came from three measured stages: a long pure spin wins on a quiet host
    and loses badly on a busy one (8,192 pauses: 281-298 us an empty job with two CPUs busy).
  - Evidence: the harness's `pool_claims` (600 jobs of 1 to 1,024 tasks at two to four threads,
    every task exactly once: back to back, after the workers sleep, at the stop); the runtime
    gate's ThreadSanitizer section race-free, its injected partition fault still reported (the
    split of a conflicting pair is now the run's timing; the first multi-threaded backward reports
    it); `ndt-threads.json` NT2 re-anchored on the claim, NT11-NT12.
  - Measured ([measurements/2026-10-13/pool/](research/measurements/2026-10-13/pool/), the
    report's §12.4): an empty job 77.4 / 85.3 -> 3.6 / 3.3 us; the GEMM shapes' work-weighted
    median 179.6 / 174.2 / 196.1 -> 199.7 / 210.1 / 222.0 GFLOP/s; the step in place 535.8 / 482.1 /
    554.4 -> 431.3 / 422.1 / 412.6 ms (486.6 / 545.6 / 556.8 before NDT-PASS); with three of four
    CPUs busy the largest job 12% slower than the sleeping pool. The 20-step state with all four
    slices at one and four threads is the one it was (4bf4e057…).
  NDT-ATTN-2/3 (2026-10-13): the attention's masked products as AVX-512 register blocks.
  - The five masked products (the forward's scores and context, the backward's dv, dq and dk) ran
    through the GEMM kernel's v1 blocks: 64-wide column blocks computed whole above the causal
    diagonal, 48 channels padded to 64, a mask test per term where the diagonal crosses a block.
  - On the AVX-512 instance: blocks of eight rows with mm_block's v1 arithmetic exactly (an element
    from the value it holds, one rounded product and one rounded sum per term in the term order),
    the scores 32 keys at a time from their diagonal on, the channels 48 at a time; a ragged tail
    of rows through the kernel; the AVX2 and generic instances keep it.
  - Evidence: the attention reference gains the blocks' edges (t = 7, 8, 9, 31, 32, 33, 40; d = 1,
    3, 15, 16, 17, 47, 49), bit for bit, finite and not; `ndt-attn.json` NA4-NA9 (9 of 9 caught
    with NA1-NA3).
  - Measured ([measurements/2026-10-13/slices/](research/measurements/2026-10-13/slices/)): at one
    thread, best of 21, the forward kernel 11.12 / 11.22 / 11.31 -> 9.09 / 9.28 / 9.42 ms and the
    backward 14.71 / 14.55 / 15.36 -> 8.36 / 8.95 / 8.96 (with NDT-ATTN-1; 11.01 / 10.88 / 11.11 the
    blocks alone); in place at four threads the attention backward 46.8 / 39.8 / 44.5 -> 23.1 /
    26.3 / 27.2 and the forward 26.6 / 32.1 / 27.6 -> 24.1 / 21.6 / 22.8.
  NDT-ATTN-1 (2026-10-13): the attention backward's probability adjoint evaluated once.
  - The in-place backward evaluated each row block's `dp = dY V^T` tiles in both of its passes, as
    the reference does (the row sums first, the score adjoints second) -- the same product over
    operands neither pass writes (0.98M multiply-adds a head at T=128, half of them repeated).
  - A row block keeps its first 256 columns' tiles between the passes (32 KB on the task's
    stack); a longer context evaluates its further tiles in each pass through one spare slot.
  - Evidence: the attention reference gains a 330-token context (the tiles at 256 and 320 through
    the spare slot), bit for bit, finite and not; `ndt-attn.json` NA1-NA3.
  - Measured ([measurements/2026-10-13/slices/](research/measurements/2026-10-13/slices/)): the
    backward kernel at one thread 14.71 / 14.55 / 15.36 -> 12.07 / 12.28 / 12.48 ms (best of 21).
  NDT-PASS (2026-10-13): the update's first pass as vector loops.
  - NDT-SCAN's in-place record put the first pass at about 20 ms of the four-thread step: four
    39 MB streams read by a loop GCC had compiled to scalar code ("control flow in loop": the
    refusal flag's short-circuit and the maxima written as floating selects).
  - One refusal flag without a branch, the maxima as the largest bit patterns read as int32 (a
    non-negative float orders as its bits; the -0.0 second moment the pass admits reads as
    INT32_MIN and never wins, as the select from +0.0 never took it), AVX2 and AVX-512 instances
    dispatched as the preflight's and the commit's are; GCC and Clang vectorize all three. The
    norm's chunk partial stays the index-order chain.
  - Evidence: the harness puts a first moment only the preflight refuses (FLT_MAX/4 under a zero
    gradient) at either end of either chunk and at the instances' block edges -- refused, nothing
    written; `ndt-pass.json` NV1-NV2; `ndt-perf.json` NP8 re-anchored on the one flag.
  - Measured ([measurements/2026-10-13/slices/](research/measurements/2026-10-13/slices/), the
    report's §12): in place at four threads, 23.2 / 20.3 / 24.6 -> 12.0 / 14.6 / 13.2 ms.
  NDT-SCAN (2026-10-13): the step's whole-buffer passes on the pool.
  - Attribution first: in place (`measurements/2026-10-13/scan/attribution.py`: a clock read after every statement of the step), the serial passes were 11% of the four-thread step (the activations' scan 27.3 ms, the gradients' fill 8.5, the attention's fills 8.0 + 4.2, the weights' and the gradients' scans 6.7 + 6.2, of 571.4).
  - The scans and the gradients' fill as one task per chunk (a scan's chunks report into the
    caller's stack, bounded by construction and guarded where the array is), the attention's
    fills in the tasks that own the slices (a KV head's dk/dv only in the task holding its first
    query head: zeroed in every task, the runtime gate's injected race -- one query head a task --
    erased the earlier heads' sums and failed serially before it could race, so the gate caught
    that first form); every verdict and value the serial pass's. The BCIR
    exponential's clamp is total over NaN (UBSan found a NaN reaching an integer conversion), as a
    maximum then a minimum -- the first form, a third select, cost the SiLU and the SwiGLU VJP their
    vector code (22-30 ms a step in place), which only the in-place rows showed.
  - Evidence: `scan_references` (19,104 parameters over two chunks with a ragged tail), NaN-filled
    attention outputs on every entry, the harness under UBSan in its Python test; `ndt-scan.json`
    NS1-NS9 (9 injected, 9 caught), `ndt-exp.json` NE6.
  - The thorough tier on Python 3.15 found a blind spot in the GEM+ threads row: the sweep's NT2
    fault (the runner drops each thread's last task) now drops a scan's task too, whose report is
    never written, so the mutant refuses on a pool instead of training a different state, and the
    row's helper raised on the refusal -- a traceback where the row owed a verdict (L1). A refused
    run is now a state of its own, a refused one-thread run counts once (no baseline is a finding),
    and the test holds both, with a library that refuses at every count.
  - Measured ([measurements/2026-10-13/scan/](research/measurements/2026-10-13/scan/)): 3 in-place pairs: the serial passes 60.9 / 68.4 / 68.3 -> 17.4 / 16.2 / 18.6 ms (-43.5 / -52.2 / -49.7); 6 breakdown pairs at four threads under `kblock+exp+reduce`: the step 575.4 / 634.9 / 601.2 / 653.9 / 682.9 / 621.1 -> 550.9 / 588.2 / 615.7 / 616.6 / 642.6 / 563.4 ms (-24.4 / -46.6 / +14.6 / -37.2 / -40.3 / -57.7, below in 5 of 6), the forward stage 213.3 / 223.2 / 219.4 / 233.0 / 263.7 / 201.5 -> 199.9 / 201.4 / 194.0 / 213.9 / 230.6 / 199.4 (-13.4 / -21.9 / -25.4 / -19.0 / -33.1 / -2.1); the 20-step state at one and four threads the same on both trees (4bf4e057…).
  TRAIN-POLICY (2026-10-13): the rail trains under the fast contract at the host's threads.
  - The lab-model recipe's `native` block: `contract` (one the trainer admits; the shipped recipes
    name `kblock+exp+reduce`) and `threads` (`"host"`: the CPUs the process may run on, or
    `BCIR_TRAIN_THREADS`; else an integer). The contract is the model's -- it changes the bits, so
    the recipe names it -- and the threads the host's, because the state is the serial one at
    every count. `build_lab_model.py` checks the block before any work and the card records the
    contract, the threads, the policy and the widest kernel instance the host ran. A CTest entry's
    trainer takes the processors the entry declares (`train-corpus` two, the BCIR Make gate's
    train task one).
  - Evidence: the refusals (a contract the trainer does not admit, a count neither `"host"` nor
    positive, a missing field); the tiny model the same checkpoint at another thread count and
    another one under `v1`; `BCIR_TRAIN_THREADS` read only as a positive integer; the rail's
    claim table names the policy. The 40-step smoke on this host: 2706 -> 6690
    tokens/s (30.3 -> 12.2 s of training on 4 threads, avx512f;
    3705 under the fast contract on one thread), the checkpoint at one thread and at
    4 the same; the rail's run 314 -> 286 s, the
    rest the corpus, the tokenizer, the validation and the export
    ([measurements/2026-10-13/policy/](research/measurements/2026-10-13/policy/)).
  NDT-SCHED-0 (2026-10-13): the two VJPs of a linear as one pool job.
  - NDT-SCHED's first item. `dx = dy W` and `dW += dy^T x` read one adjoint and write apart, so
    `bcir_tensor_mm_pair_pool` runs the first GEMM's tasks and then the second's in one pool run,
    each with the partition its own entry would have used, so every output element keeps its depth
    order and the results are the two entries' bit for bit; one barrier in place of two for each of
    the step's 43 linears, the narrow K/V projections' idle threads on the other product's tasks.
    The trainer dispatches the pair under each GEMM contract; a provider's GEMM keeps its two calls;
    the breakdown harness times the VJPs as the trainer runs them.
  - Evidence: the C harness holds every shape pair of its GEMM section under the three contracts,
    in both orders, to the two entries, and the twelve contracts' states at one to four threads to
    the serial run through the trainer's pairs; `ndt-threads.json` NT9 (a pair split at the first
    GEMM's column count) and NT10 (the pair's serial path skips the first GEMM); NT1, NF3, NK4 and
    NK5 re-anchored to the partition's, the one routing predicate's (`gemm_contract`, read by the
    forward's linears and the pair alike) and the serial branch's new homes.
  - Measured ([measurements/2026-10-13/sched0/](research/measurements/2026-10-13/sched0/), the
    report's §10) in 3 same-session pairs at four threads under `kblock+exp+reduce`: the
    backward pass 631.9 / 633.8 / 622.4 -> 601.0 / 586.3 / 594.6 ms (-30.9 / -47.5 / -27.8), the
    GEMM-backward row timed alone 292.7 / 285.9 / 266.6 -> 238.0 / 254.8 / 282.6 (-54.6 / -31.0 / +16.0),
    the step 684.2 / 679.4 / 659.3 -> 645.1 / 681.7 / 668.1 (-39.2 / +2.4 / +8.8); the backward pass (the `loss_backward` stage, the forward recomputed inside it, as the trainer runs it -- the one stage the pair job changes) is below the two-call stage in 3 of 3 pairs, by -30.9 / -47.5 / -27.8 ms (631.9 / 633.8 / 622.4 against 601.0 / 586.3 / 594.6), and the GEMM-backward row timed alone by -54.6 / -31.0 / +16.0; the step's own cell (-39.2 / +2.4 / +8.8) is within this host's run-to-run noise, as NDT-REDUCE's was -- the one barrier fewer per linear and the narrow projections' tasks shared are worth about 5% of the backward at four threads here.
  NDT-REDUCE (2026-10-13): the reductions the pool left serial, on the pool; the clipping norm's
  declared contract.
  - NDT-THREADS admitted only partitions in which each output element, and each reduction, is
    produced by one task in the serial order, and left four index-order reductions serial: the
    loss's rows, the gamma VJP's row sum, the embedding scatter and the update's first pass
    (41 of the four-thread update's 53 ms, measured for NDT-NORM). Three of them admit such a
    partition: the loss's rows are tasks whose terms (in double, as the serial loop computed
    them) the caller adds in row order; the gamma VJP runs rows for `dx` and each row's `1/rms`
    (a row of scratch the trainer lends) and then column ranges, each column's sum over the rows
    in row order (`bcir_tensor_rms_backward_pool`); the scatter is column ranges, every row's
    columns added to its token's row in row order. Every loss and state is the serial one at
    every thread count under every contract, bit for bit.
  - The update's first pass and preflight are one task per chunk of `bcir_decoder_reduce_chunk`
    elements (1,024 doubled until at most 1,024 chunks; the results on the caller's stack), the
    verdicts and maxima exact whatever the partition. The norm is the contract's: the one
    index-order chain in double where the contract was, or under the new `reduce` bit
    (`BCIR_DT_CONTRACT_REDUCE`, `NativeDecoder(contract="<any>+reduce")`) each chunk's squares
    summed in index order from zero and the chunk sums added in chunk order from zero -- a
    declared reduction whose shape is a function of the parameter count alone, emulated by the
    oracle operation for operation (`grad_norm_squared_reference`, `reduce_chunk`), with
    `bcir_decoder_gradient_norm` reporting the square the update takes its root of. The bit
    changes the norm's last bits and nothing else; the float rounding of every update absorbs
    that for a while (the harness's reduce states coincide with their chain's after three steps
    while the norms are apart at 15 of 18), and on the 9.84M decoder a rounding boundary is
    crossed within twenty steps, after which the state is the contract's own.
  - Evidence: `reduction_references` in the C harness holds the pooled loss and scatter to
    serial references written apart from the pool (untied fixtures of 32 and 40 rows, tokens the
    batch repeats without repeating a row, no pool and a pool of two); `pool_identity` holds
    twelve contracts' states and the losses and norms the steps report at one to four threads to
    the serial run, and the gamma VJP to the serial kernel over three row and three column
    chunks; `adamw_reference` holds the update under both norm contracts to a reference of each
    on a three-chunk fixture, serially and on a pool of three, the square against the
    reference's sum, with the chunk rule on both rails; `refusals` holds a non-finite element in
    the first pass's last chunk refused under both; `train.norm.reduce.mismatch` (0 of 6 norms
    and 8 counts) and `train.state.threads.mismatch` (now under v1 and `reduce`) are the rows;
    `ndt-reduce.json` NR1-NR11, 11 of 11 caught. Two faults were first reported uncaught and
    both were the witnesses' findings: a periodic token stream made every repeat of a token the
    same row, so a reversed accumulation summed the same values, and the norm's root hides a
    one-ulp change of the chunked sum half the time -- hence the streams, the three-chunk
    fixture and the square's entry.
  - Measured ([measurements/2026-10-13/reduce/](research/measurements/2026-10-13/reduce/), the
    report's §9): in three same-session pairs at four threads (each contract's breakdown alternated with the other's, minutes apart, so both see one host) the update reads 40.7 / 49.7 / 47.5 ms under `kblock+exp` -- the first pass pooled, the chain serial, NDT-NORM's content -- against 31.8 / 47.2 / 39.6 under `kblock+exp+reduce`, below it in every pair (-9.0 / -2.6 / -7.9), with §8's 53.1 the day before on a faster host; the gamma VJP row 11.0 / 14.3 / 12.4 against 6.7 / 8.8 / 11.8 (-4.3 / -5.5 / -0.6; 9.2 in §8); the step itself is not resolved on this day's host, whose unchanged GEMM rows drifted 20% within the session, more than the slice's effect on the step; the 1,000-step `kblock+exp+reduce` run at 625.8 ms a step on
    four threads (1419.6 on one), ending in `5137e013…`, its own, not §7's `d7fc618c…` (the norm's last bits crossed a float rounding boundary within the first steps, as the 20-step identity rows already show, and the state is the reduce contract's from there), and the one-thread run ends in the same state, the held-out
    model 1.5003 / 1.6319. NDT-NORM landed with it.
  PR817-P0 (2026-10-13): the two portability findings of PR #817's same-host rerun.
  - The provider admission is host-specific and the tests said otherwise. Two assertions held
    that every importable OpenBLAS wheel is admitted on the comparison decoder's shapes; on
    PR #817's AMD EPYC host the same wheel (0.3.34, DYNAMIC_ARCH) blocks differently, 14 of 30
    cases differ, the trainer refuses it -- the guard working -- and the tests failed. The
    claim is now the guard's: `train.mm.provider.unsound` (`provider_unsound`) is 0 wherever
    the wheel is importable -- the verdict `use_cblas` gives agrees with the differential it is
    defined by, admitted exactly when no case differs, refused with the count otherwise -- and
    `train.mm.provider.differences` is the host's count (0 of 30 here, 14 of 30 there),
    reported everywhere and graded only on the baseline host (`host_dependent`).
  - The rank-ordered BPE encoder kept a duplicate pair's LAST merge; the pass encoder, the
    definition, applies the first, whose pass consumes every occurrence, so an artifact naming
    one pair twice (training never writes one; the schema and digest admit it) encoded `ab` to
    the later token by rank. `_ranks` keeps the first merge now; the witness holds the
    constructor's and the imported tokenizer to the pass encoder on both duplicate shapes
    (a plain repeat, and a pair an earlier merge created) and a last-wins table disagrees;
    `tools/testing/faults/bpe.json` (BP1 the last merge kept, BP2 a merge dropped).
  NDT-BLOCK (2026-10-12): the v3 GEMM kernel's register block and packing, under the same contract.
  - The gap NDT-KBLOCK left to the admitted provider (BCIR's own v3 kernel at 61 GFLOP/s
    work-weighted against OpenBLAS's 89 at one thread on the step's shapes) was the schedule,
    not the arithmetic: the 4 x 64 block read its 112 KB tile from L2 for every four rows
    (272 bytes per 16 fused multiply-adds, at the L2 limit), the autovectorized A gathers went
    through the stack, the tile was packed scalar (16 us a transposed tile) and the transposed-A
    operand's lines came one per depth without prefetch. Measured first in a scratch prototype
    against the tree's kernel bit for bit on 42 cases: 12 x 32 (92 GFLOP/s), 8 x 48 (108), 6 x
    64, 4 x 96, and the row-panel-outer nest that repacks the tile per row panel (worse at every
    panel height, 60-77, because the packing is the cost).
  - The AVX-512 and AVX2 instances of v3 are now intrinsic micro-kernels (`kb512_block`,
    `kb256_block`): 8 x 48 and 6 x 16 accumulators, every A term one broadcast from memory,
    the tile three (two) vector loads per depth -- eleven loads per 24 fused multiply-adds --
    and the block's partial added as a rounded product and a rounded sum; rows that do not fill
    a block run 4-row (2-row) and single-row blocks of the same chain. The tile is packed by
    vector (`kb512_pack`: masked row loads, 16 x 16 register transposes of B^T, ragged edges
    scalar); a transposed A's next line is prefetched eight depths ahead; the pool's tasks are
    128 rows by one column block of the instance's width (`mm_nr` per contract: 48, 16, 32),
    so each tile is packed half as often. The generic instance keeps the plain-C block.
  - Evidence: the C harness's `gemm_kblock_contract` over 19 shapes (three new ones with 4-row
    and single-row tails and 16..48-wide ragged blocks) on every instance, serially and on a
    pool; `train.mm.kblock.mismatch` now holds every instance the host runs to the exact
    emulation (`mm_variants`, `mm_native(variant=)`), so the generic block's injected faults
    fire where the AVX-512 instance runs and the instance has its own: `ndt-kblock.json`
    NK7-NK10 (its last single row dropped, a task's column block narrower than its register
    block, a transposed tile's ragged columns stale, a ragged block's fused epilogue), 10 of 10
    caught; the kernel mutants test carries the same three AVX-512 copies with an honest skip
    where the host lacks the instance. gcc and clang builds bit-identical on every case.
  - Measured ([measurements/2026-10-12/block/](research/measurements/2026-10-12/block/)): the
    kernel-rate record over the step's 15 shapes, best of 7 -- work-weighted 61.2 -> 99.4
    GFLOP/s at one thread against OpenBLAS's 88.8 in the same process, 135.9 -> 218.9 on the
    pool at four against OpenBLAS's 219.1 at four (its workers sleeping); every result at beta
    1 identical to the old serial kernel's. Still behind the provider: `dW` with 1,024 rows and
    depth 1,024 (73.6 against 89.6; its A lines alias one cache set, and the per-panel packing
    that would fix it needs a buffer the serial path has no allocation law for) and the
    depth-192 product (67.4 against 91.1). The step by kernel under `kblock+exp`: the GEMMs
    569 ms at one thread (952 before) and 290 at four (531); the 1,000-step
    run on BCIR's kernel ends in `d7fc618c…` at 516 ms a step on four threads (753
    before) and 974 on one (1,425); the identity rows on the widest, AVX2 and portable
    builds end in §7's kblock states.
  NDT-ATTN-0 (2026-10-12): the v2 softmax's row through the vector exponential.
  - NDT-MEASURE's finding, fixed the same day: `att_head`'s `exp` branch called the scalar
    exponential element by element with the row sum in the same loop, which GCC does not
    vectorize, so the attention forward read 79 ms under `exp` against 62 under libm at one
    thread. The row now has its maximum subtracted in place, goes through the exponential's
    vector instance (`exp_inplace_run`: one pointer, no alias check, the same `exp1` per
    element) and is summed in the same index order -- the same values, so the `exp` and
    `fused+exp` states are unchanged (`aaa9998c…`, `b18d89fa…`, at one and three threads).
    Re-measured with the report's stages ([attn0/](research/measurements/2026-10-12/attn0/)):
    the attention forward 79.0 to 55.2 ms under `exp` and 84.7 to 58.1 under `fused+exp` at one
    thread (41.9 to 31.7 on three), v1's 62.7 in the same run; the C harness's `exp_contract`
    holds the v2 softmax to its definition through the new path, and `ndt-exp.json`'s NE4 (the
    v2 softmax on libm) is re-anchored to it.
  NDT-MEASURE (2026-10-12): the comparison re-run after the pool and the two v2 contracts
  ([BCIR_NDT_MEASURE_2026-10-12.md](research/BCIR_NDT_MEASURE_2026-10-12.md)).
  - The `identity` stage per contract and thread count (`--contract`, `--variants`): v1's 16 rows
    one state, the 10-11 serial digest; each v2 contract one state of its own at 1..4 threads on
    the widest and AVX2 builds (the portable fused instance, libm's `fmaf` per term, is a
    correctness path the C harness holds, not a run). The 1,000-step v1 run on three threads ends
    in the 10-11 one-thread run's state byte for byte (`aed5ffbf…`); the `fused+exp` runs on
    three and four threads end in one state of their own (`eb43c96d…`, the same at three and
    at four threads).
  - A `bcir-breakdown` stage: the step's kernels timed through the entry points the trainer calls
    under the run's pool and contracts (the pooled kernels, `bcir_tensor_mm_fused_pool`, the v2
    SiLU, SwiGLU VJP and softmax), so "the rest" is the step less its kernels under that pool --
    the serial entry points the first cut timed read a negative rest under the pool, which is how
    the stage found its own defect. At one thread the fused contract takes the step's GEMMs
    x1.24, not the 2x of its instruction count (the packing, loads and stores are
    the same under both contracts); three threads take the `fused+exp` step x1.71
    and four x2.03 (the 64-row task partition behind one barrier per GEMM, and the
    reductions the pool leaves serial by design). One finding against NDT-EXP itself: the v2
    softmax calls the scalar exponential element by element inside its row loop, so the
    attention forward reads 79 ms under `exp` against 62 under libm at one thread -- the row
    through the vector instance (the same values, the same digest) is the next attention
    slice's first item.
  - The 1,000-step language-model run: v1 on three threads 989 ms a step, `fused+exp` 819 (v1 on
    one thread here 1,952; the 10-11 one-thread v1 run 1,726, so that host was about 13%
    faster; eager PyTorch there 774 on one thread), the same held-out model to four
    decimals (validation 1.5003, test 1.6319 nats per byte; the two trajectories within 0.0002
    nats at every hundredth step). The training roadmap's §2 budget gets its measured table and
    the ladder two candidate slices the residual names, NDT-BLOCK (the fused kernel's register
    block and packing) and NDT-REDUCE (the serial reductions under a declared tree contract);
    PyTorch is not installable on this host, so the equal-thread comparison is the next run.
  NDT-EXP (2026-10-12): a BCIR-owned exponential under the v2 contracts of SiLU, the SwiGLU VJP
  and the softmax.
  - `bcir_tensor_expf`: x = k ln2 + r by a Cody-Waite reduction (LN2_HI with 16 significant bits,
    so k*LN2_HI is exact), a degree-7 polynomial in r by Horner, 2^k as two exact power-of-two
    scalings (one rounding into the subnormals), the argument clamped and the three special cases
    (+inf above ln FLT_MAX, 0 below the last subnormal, NaN) selected at the end -- branch-free, so
    the scalar, AVX2 and AVX-512 instances run one operation sequence under -ffp-contract=off and
    agree bit for bit (clang vectorizes both instances; GCC the AVX-512 one). SiLU v2, the SwiGLU
    VJP v2 (the sigmoid 1/(1+e) for x >= 0 and e*(1/(1+e)) for x < 0, e = exp(-|x|), a declared
    form) and the softmax v2 use it; the `exp` contract bit (BCIR_DT_CONTRACT_EXP) selects them, a
    provider's SiLU keeps its own; `NativeDecoder(contract="exp" | "fused+exp")`.
  - Evidence. `bcir_tensor_exp_sweep`: every binary32 in [-104, 89] at stride 2^8 (8,388,592
    inputs) through every instance the host runs: 0 inputs where two instances differ, 1 ulp at
    most from exp in double rounded to binary32 (99.6% exact) -- `train.exp.mismatch` 0 and
    `train.exp.ulp.max` 1, the stated row, which the harness holds exactly (an exact row has no
    band); identical sweeps from the gcc and clang builds. The C harness's `exp_contract`: the
    instances against the scalar, the special values, SiLU/SwiGLU VJP/softmax v2 on a pool against
    their definitions and apart from v1; `pool_identity` runs its three steps under all four
    contracts, each state its own, each identical at 1..4 threads. `ndt-exp.json`: 5 defects, 5
    caught -- NE5, the trainer's SiLU ignoring its contract, escaped the first witness (the four
    states still differed, through the softmax), so `exp_contract` now holds a forward pass's
    layer-0 hidden activations to each contract's own SiLU, the check that sees that kernel
    alone (L11: the witness hits the law it exists to test). The threads witness holds the `exp`
    and `fused+exp` states identical across thread counts and apart from the other contracts'.
  NDT-FMA (2026-10-12): the fused GEMM contract, v2, beside v1.
  - `bcir_tensor_mm_fused` (and `_fused_pool`, `_fused_variant`, `_fused_variants`): the same
    register-blocked nest as v1 with one `fmaf` per term in place of a rounded multiply and a
    rounded add -- the product `alpha*A` still rounded as in v1 -- so each element receives its
    terms k ascending, rounded once each. Three instances: portable (the libm `fmaf`, correctly
    rounded where the C library keeps Annex F), AVX2+FMA and AVX-512F, the last two fusing in
    registers; `-ffp-contract=off` stays, since the fusion is spelled, not left to the compiler.
    ABI 3: `bcir_decoder_state` ends in `contract` (0 v1, `BCIR_DT_CONTRACT_FUSED` v2, other bits
    refused); the trainer's `gemm` routes by it, a provider's GEMM keeps its own;
    `NativeDecoder(contract="fused")`; the checkpoint schema is `bcir.native_decoder.v2` and
    names the contract, which `resume` restores -- a state trained under one is not the other's.
  - Evidence. The C harness's `gemm_fused_contract`: every fused instance this host runs, serially
    and on a pool of three, bit for bit against the `fmaf` ascending-k loop over the 44 (shape,
    transpose) cases, and apart from v1 on them; `pool_identity` now runs its three training steps
    under both contracts, each identical at 1..4 threads and the two states different; an
    undefined contract bit is refused. `train.mm.fused.mismatch` (GEM+ `train`, exact, bound 0):
    0 of 20 against `mm_reference_fused`, an exact rational emulation rounded once per term through
    `round_f32` -- never through a double, whose own rounding resolves a tie wrongly (the fixture
    shows 1 + 2^-24 + 2^-60); `round_f32` agreed with glibc's `fmaf` on 20,000 random operand
    triples before it was trusted. `ndt-fma.json`: 5 defects injected, 5 caught. The threads
    witness holds the fused state identical at every thread count and apart from v1's, and a
    fused checkpoint resumes as fused.
  - Measured: the fused kernel against v1 at one thread on the 9.84M decoder's 13 GEMM shapes, best of 7 calls each:
    work-weighted 46.9 -> 66.9 GFLOP/s, x1.43, every shape between x1.33 and x1.51
    (`docs/research/measurements/2026-10-12/gemm-fused-rates.json`; GCC 13.3, the AVX-512 instance, a
    4-vCPU container, indicative). Short of the arithmetic's x2: the kernel is not only its
    multiply-adds -- the packing, the loads and the block's stores stay -- which is what the wider
    register block and the per-panel packing of the transposed operand, still open, are for.
  NDT-THREADS (2026-10-12): a deterministic thread pool under the native trainer, bit-identical
  at every thread count.
  - `bcir_tensor_pool` (`bcir_tensor.c`): pthreads on POSIX, Win32 threads on Windows, one thread
    elsewhere, in memory the caller provides (`bcir_tensor_pool_size`, `_init`, `_stop`: the unit
    allocates nothing, as the trainer's arenas allocate nothing -- the memory-discipline check
    refused the first draft's `calloc`, L5's class rule doing its job); thread 0 is the caller, task t runs on thread t mod threads, each thread its tasks
    in order, and every partition is a function of the shapes alone -- the GEMM over (column block
    of the variant's width, 64-row range) with its beta scaling inside the task, so each element
    of C keeps its depth order; attention over (batch, query head) forward and (batch, KV head)
    backward with the `group` = heads/kvheads query heads of one KV head in order, the order the
    serial loop adds the shared dk/dv; SiLU, the SwiGLU VJP, the residual add and the gate
    multiply in 8192-element chunks; RMSNorm rows and RoPE positions in chunks of 16; the AdamW
    commit in 65536-element chunks with the instruction-set variant chosen once per call. The
    loss, the embedding scatter, the gamma VJP's row sum and the update's norm pass stay serial:
    each is an index-order reduction (the training roadmap's 3.1). The old entry points are the
    NULL-pool kernels, unchanged in every bit; ABI 2 ends `bcir_decoder_state` in its pool and the
    step routes BCIR's kernels through it (a provider's GEMM or SiLU is threaded as the provider
    is). `NativeDecoder(threads=n)`, `close()`, `resume(threads=n)`; the checkpoint does not name
    the count, since the state does not depend on it.
  - Evidence, three kinds. `test_decoder_train.c`'s `pool_identity`: every pool kernel at one to
    four threads against the serial kernels and the scalar references (GEMM shapes crossing the
    64-row tasks and both block widths under every transpose pair and a beta that scales, a depth
    of zero that still scales; attention forward and in-place backward on the reference test's
    shapes; the elementwise, row and position kernels over three chunks; the task runner itself;
    three training steps through `bcir_decoder_run`), in the C23, C11, portable and AVX2-capped
    builds. `tools/c/sections/decoder_train_tsan.sh` runs that harness under ThreadSanitizer (no
    report, exit 0) and holds the same build of `test_decoder_train_race` -- the manifest's
    mutation `a.group=nh/nk` -> `a.group=1`, one query head per task, so one KV head's two heads
    land on two threads -- to a race report on its dk/dv; the sanitizer finds it from the
    happens-before order, so on every run, and the run halts on the first report (TSan exits 66)
    so the harness's own assertions, which the racing sums fail, are not what decides.
    `train.state.threads.mismatch` (GEM+ group `train`, exact, bound 0): the state after three
    steps of the threads bench (vocab 250, width 72, 4 heads over 2 KV heads, 2 layers, batch 2 x
    40 tokens: an 80-row GEMM with a 16-row tail, T past the backward's 32-row chunks, every width
    off the block edges) at 2, 3 and 4 threads against one -- 0 of 3; its fault table
    `ndt-threads.json`: NT1 (the GEMM's row tasks stop before the ragged tail) and NT2 (the runner
    drops each thread's last task) fire the row -- they reach the state only on a pool, which is
    why the row compares thread counts and not kernels -- and NT3-NT8 (a task adding before
    scaling by beta; a dropped last chunk of the elementwise, RMSNorm, RoPE and commit partitions;
    a backward task one query head short) fire the C harness.
  - The report's `identity` stage, with `--threads 1,2,3,4` (the 2026-10-11 harness): the 9.84M
    decoder's 20-step state SHA-256 `6c0419e85d345904...` on GCC 13.3 at every thread count and
    every instruction-set variant, 16 rows with a repeat, one state
    (`docs/research/measurements/2026-10-12/identity-threads.json`) -- the digest the report
    measured on the serial trainer, so the slice changed no arithmetic. The same rows' per-step
    medians on this four-vCPU container, indicative only: the widest build 1,786 / 1,229 / 976 /
    1,103 ms at 1, 2, 3 and 4 threads (three threads the best on a host that is not the pool's
    alone), AVX2 2,512 / 1,655 / 1,254 / 1,409, portable 4,155 / 2,508 / 1,825 / 2,125. The hosted-model gate (`tools/models/test_native_decoder.py`)
    trains five steps at 1, 2 and 4 threads on both of its hosts, the Windows cell on Win32
    threads. Speed is not this slice's claim: NDT-MEASURE re-runs the comparison on the pool.
  TRAIN-RAIL (2026-10-12): the training corpus is the IR's data, one way.
  - The law TRAIN-0 stated now has its rail. `tools/models/configs/bcir-corpus-860k.json` is the
    docs recipe with `training/*.md` and `training/**/*.md` beside `docs/`'s patterns (407 tracked
    files, 5.3 MB; the docs recipe reads 79), the same 860K-parameter decoder, tokenizer and
    licence, its own split salt. `tools/models/train_rail.sh` runs it through `build_lab_model.py`
    with the trainer compiled by CC, then reads the corpus witness `tools/models/train_rail.py`:
    the tracked Markdown files of `docs/` and `training/` the model did not see,
    `train.corpus.unselected`, 328 before the rail (every file of `training/`), 0 under it. The
    witness reads the row from the recipe's own selection (`corpus_documents`, the builder's one
    predicate) without a training run, so the GEM+ harness measures it beside `train.step.ms`.
  - One script in every place the rail lives (L14): CTest's `train-corpus` (label `train`,
    `ctest --preset train`, `BCIR_TRAIN_STEPS` 40 by default, refused at 0); BCIR Make's task
    `train.corpus` in the rails' BCIRfile, the same script and count through `run_section.sh`,
    reading the recipe, the trainer's two C units, `bcir/`, `docs/` and `training/` -- a corpus
    edit retrains, a runtime edit elsewhere does not, and the make gate's scratch copy carries the
    two trees so its generation tags can say so; the qualification job, which trains the corpus
    recipe for 200 steps and qualifies that model where it trained and qualified the docs model;
    the gcc cell of `cmake-build` runs the entry. `test_train_rail.py` holds the recipe's selection
    to the tree, fires the witness on the docs recipe (exit 1) and on a manifest it cannot read
    (exit 2), refuses the script's four ways of not running (exit 2, before any work), and reads
    the script, recipe, roots and step count out of each place.
  - Found by the task: the make gate's change step copies the tracked tree, so a target reading a
    file git does not track got a traceback from the scratch copy minutes in. `untracked_reads`
    now refuses before any run (UNUSABLE, exit 2) naming the files, with its witness in
    `test_make_rails.py`. The smoke's own cost, 568 s for 40 steps, was the encoder's: BPE-FAST.
  S5-B (2026-09-25) landed G10: escape analysis and indirect-call target narrowing, and with them
  a sound effect footprint behind `CompileResult.commute`.
  - RED, measured on the parent (`8d3aab84`) and judged by this slice's fixtures:
    - the footprint recorded every store as a read of its base (`c.store rd=(base, [index,]
      value) wr=()`), so writes through a pointer, to a static and to the heap were invisible;
    - the witness shows 119 pairs reported as commuting that diverge;
    - the rails disagreed on 24 of 200 units, and the twin had no escape report;
    - no local was proved private and no indirect site was narrowed.
  - What landed (`bcir/frontends/cfront/escape.py`, the twin's `esc_*` in
    `runtime/c/bcir_cfront.c`, `bcir-cc --emit-effects` / `--emit-escape`):
    - one points-to analysis over the unit: Andersen's, open world, monotone and so independent of
      the order the claims are visited in;
    - three answers from it: escape verdicts for named locals, target sets for indirect sites
      (added to the call graph), and the footprint;
    - the footprint's rules: every store is a write, a static is `function.name`, the heap is one
      object per allocating function, a pointer made from an integer points anywhere, and an
      ops-table callback has callers the unit cannot see;
    - a device access is decided by the access's base resource, not by how the claim was spelled;
    - a call with more operands than a twin claim holds is refused on both rails.
  - The judges share no code with the analysis:
    - generated units whose verdicts and targets are known by construction;
    - a dynamic witness that runs every pair of functions in both orders, in separate processes;
    - 386 forms of every declaration kind against every access form, in every storage place.
  - Outcomes:
    - `escape.unproved` 39 → 0: every candidate is proved nonescaping;
    - `icall.unknown` 22 → 15 and `icall.unresolved` 22 → 18, of 22 sites. 7 sites are narrowed and 4
      resolved. The 15 unknown are the open world's floor: each calls a pointer an exported
      function takes as a parameter;
    - `escape.verdict.mismatch` 73 → 0 and `icall.target.mismatch` 20 → 0;
    - `effects.commute.unsound` 119 → 0, over 1,867 decided pairs;
    - `effects.parity.mismatch` 24 → 0 and `escape.parity.mismatch` 200 → 0;
    - `tools/testing/faults/escape.json` injects 26 defects, each caught by its own row;
    - the analysis costs about 5% of `compile_unit`'s time over the corpus (0.09 s of 1.7 s), and 0.09 s on the 7,630-claim scale unit;
    - `native.*` is untouched: its rows import no cfront module.
  - Found and fixed in the slice:
    - the two rails visited sibling expressions in different orders, and the first indirect-call
      rule decided once from a partial target set, so the rails' answers differed. Every rule
      now only grows its sets;
    - the twin named compound literals per unit where the oracle names them per function;
    - the witness saw `_BitInt` padding bits as a divergence, until it hashed integers by value;
    - the first RED leaned on the lowering it judged (`init_refs`), until the witness read the
      AST;
    - the first fault sweep caught 18 of 19: dropping static locals from the footprint fired only
      the parity row, because no two generated functions shared a static (L11);
    - the parity rows skipped units the twin refused, so a twin that reported on nothing passed
      both (L2);
    - the harness admits a zero baseline only on a guard row, and the corpus counts rose from
      zero. They now count what is not yet proved or narrowed, falling toward proved floors, and
      the gate holds each floor from below as well: a count under its floor is an unsound claim;
    - the forms sweep found two defects of the twin's port that the corpus, the generated units
      and 300 fuzz programs could not show: a file-scope pointer was touched in place, and an
      index load through a volatile pointer was read as an ordinary load (L14, L22).
  - Not claimed: flow, context and field sensitivity; the contents of global initializers;
    type punning; a volatile access that neither frontend carries the qualifier to (a volatile
    member, a volatile global, a global or member pointer to volatile), queued as a frontend
    follow-up (closed by CF-VOL, below); and one heap object per allocating function.
  CF-VOL (2026-09-25) carried `volatile` through both cfront rails, to every place a C program
  puts it.
  - RED, measured on the parent (`c72d9e29`) over 432 forms -- every place `volatile` can sit,
    against every access form it admits, at eight element widths -- and judged by Clang:
    - 356 refusals (the oracle 170, the twin 186). The oracle typed the value a volatile load
      yields as volatile, so `x |= 1` on it broke R3, and neither rail made a local, global or
      member pointer to volatile a device region;
    - 369 forms whose emitted C did not perform the original's volatile accesses. The oracle's
      declarations dropped the qualifier, and the twin wrote every register access as 32 bits;
    - 69 forms that returned or wrote different bytes: a byte store through a `volatile uint8_t *`
      landed as a word at `p + 4*i`;
    - 282 functions whose effect report missed the device, so two readers of a volatile global,
      member or register block were reported to commute.
  - What landed (`bcir/frontends/cfront/{ctype_model,lower,emit}.py`, `runtime/c/bcir_cfront.c`):
    - one model on both rails. A resource is a device region when its type holds volatile storage
      or points at it: a parameter, local, static, global or temp alike. An access is volatile when
      its lvalue is. R3's pass then makes every claim touching a device region device-domain and
      ordered;
    - a loaded value is an ordinary value: lvalue conversion drops the qualifier;
    - the emit performs each volatile access through a volatile lvalue of exactly its type.
      Declarations keep the qualifier, and a member or dereference goes through
      `*(volatile T *)(base + off)`, or `T volatile *` when the slot is itself a pointer.
  - The judges share no code with either rail: Clang's volatile loads and stores, in order and at
    their LLVM types, for each original function against its emitted twin; a seeded harness that
    compares every return value, buffer and global; the structural digest; and each rail's effect
    report.
  - Outcomes: `volatile.refused` 356 → 0, `volatile.emit.mismatch` 369 → 0,
    `volatile.behaviour.mismatch` 69 → 0, `volatile.device.missed` 282 → 0, and the parity guard
    held at 0 (group `volatile`); the G10 forms went from 386 to 476 on a second sweep;
    `tools/testing/faults/volatile.json` injects 23 defects, each caught by its own row.
  - Found and fixed on the way. All were pre-existing and none was volatile's own:
    - a pointer cast `(T *)x` gave an integer temp on both rails. The oracle's was 32 bits wide,
      truncating the address, and the spelling dropped the pointee's sign and qualifier. No corpus
      unit held a pointer-to-pointer cast, so the emitted C had never been compiled (L22). The
      register idiom `*(volatile uint32_t *)ADDR` now lowers on both rails;
    - the twin typed every file-scope variable as `uint32_t`. `(g >> 1) < 0` was false for a
      negative `int32_t g`, and `-g` truncated a float, under equal digests
      (`cfront_globaltype.c`);
    - the twin declared a local array of pointers as the pointer-wide integer, and a `(float *)`
      cast as a float;
    - the twin refused `*a` on an array and `*g` through a file-scope pointer, both of which the
      oracle lowers. Its emitter spelled a base's address at six sites, and each addressed a
      file-scope pointer's slot instead of reading it (L14, `holds_pointer`);
    - the gate's first run judged the twin's output without the header its driver includes for a
      masked access. It graded 362 forms as mismatches that were compile errors in the judge (L11).
  - Not claimed: `++`/`--` on a volatile lvalue (both rails refuse); on the twin, a member of a
    file-scope struct and `**` through a file-scope pointer; and a `_BitInt` pointee's temp.
  - Found after the series, by the escape table's sweep over its head (2026-09-26): 25 of 26
    caught. The G10 device rule read a load's base resource as well as its domain, because the twin
    once lowered `p[i]` through a `volatile T *` as an ordinary load. R3's pass now makes every claim
    touching a device region MMIO-domain on both rails before any analysis runs, so no input reached
    the base reading, and its fault passed the whole corpus. The reading was removed from both rails
    (`escape._device`, `esc_device` read the domain alone), and the table's fault moved to R3's pass,
    one per rail, each caught by the effect parity row: 27 of 27 (L22).
  CF-IDX (2026-09-25) made a subscript chain through a pointer element index what the element
  holds, on both cfront rails.
  - The defect: a base took every subscript that followed it. `q[j][i]` on `T *q[N]` and
    `pp[j][i]` on `T **pp` were Horner-flattened as if the base were a two-dimensional array,
    into `q[j + i]`: a load read a pointer out of the table and returned it as a number, and a
    store wrote a value over one. The oracle refused `*q[j]`; the twin read `*q` and subscripted
    what it loaded. The twin also declared a one-element array (`T *a[1]`) as a scalar and left
    its subscript unguarded, and decayed a parameter `T *rows[]` to `T *` instead of `T **`.
  - RED, measured on the parent (CF-VOL, `b386191b`) by the volatile gate over the 138 forms that
    reach a register block through a table of pointers (two elements, one element, and a
    parameter `volatile T *rows[]`), per rail: 126 refused, 120 emit mismatches, 134 behaviour
    mismatches, 22 parity mismatches and 40 device accesses missed. `cfront_ptrindex.c` is
    refused by the parent's oracle; with its dereference taken out, both rails emit C that is
    not behaviour-equivalent, under equal digests.
  - What landed: one rule on both rails. A base takes one subscript per declared dimension (a
    multi-dimensional VLA's too), else one; while subscripts remain, its element must be a
    pointer, which is loaded, and the rest index what it holds (`lower._lvalue(Index)`; the
    twin's `index_chain`, `subscript_dims` and `step_to_elem_ptr`). `*q[j]` and `*(q[j] + i)`
    take the same step, in an expression and in a store. A one-element array is an array
    (`decl_array`), an array of pointers is one predicate (`ptr_array`), and `T *rows[]` is
    `T **`.
  - Outcomes: every `volatile.*` row 0 over 548 forms; `cfront_ptrindex.c` equivalent on both
    rails with equal digests; `tools/testing/faults/volatile.json` gains eight faults, each caught
    by its own row. The pinned G10 forms are unchanged (476); the parameter forms they
    leave out now lower on the twin as well.
  - Found on the way:
    - the G10 forms held `X[0][i]` on an array of pointers and a pointer to pointers in every
      storage place, and both rails' reports agreed on them. The forms are compared, never run,
      and two rails that share a misreading agree (L11);
    - the rule's first cut counted a multi-dimensional VLA as one dimension, and the corpus
      refused `a[i][j]` on one at once;
    - the fixture's first cut lent a local table to a call. The escape analysis reports that
      array lent, rightly, and `escape.unproved` counts it, so the table moved to file scope;
    - both rails declare a `static` local array as a scalar, so the emitted C does not compile
      (CF-STATIC, below).
  CF-MEMCONV (2026-09-25) made every store the emit spells as a byte copy convert the value to the
  slot's declared type, on both cfront rails.
  - The defect: both rails chose the stored bytes' type from the VALUE. A float member, member-array
    element, array-of-structs field or `*p` received an integer's bits (`s->f = v`), an integer slot
    received a float's (`s->si = x`), and a real value landed in a `_Complex` member at the wrong
    width. A float stored into a bitfield did not compile, and `(s->si += x)` yielded the
    unconverted float sum.
  - RED, measured on the parent (CF-IDX, `609b3413`) with `cfront_memberconv.c`: the rails' claim
    graphs differ, and both rails' emitted C fails to build (a float inserted into a bitfield).
    Without those stores, both rails' emitted C returns a different value from the first input on.
  - What landed: C's assignment conversion, in the claim graph and on both rails. A byte-copy store
    whose value is of another arithmetic class (integer, real floating, complex) converts first,
    through the `c.cast` an explicit `(T)v` lowers to: `lower._store_conversion` over `_cast_value`,
    and the twin's `store_conv` over `emit_cast`. So the digest carries the conversion. A width or
    sign change within a class stays the emit's, and a `_Bool` slot normalizes by its flag. Every twin
    store path asks the one predicate: the store helpers convert and return the value they stored,
    and on both rails a compound assignment's value is that stored value.
  - Found and fixed on the way:
    - the twin spelled the member store three times, and one copy lacked parts of it. A store through
      a pointer member (`s->p->f = v`) dropped a `_Bool` member's flag and wrote a bitfield as a
      plain member. Both copies are now the helper (L14);
    - on the twin, an initializer reached through a nested designator (`.in.b = x`) took its `_Bool`
      flag and bitfield unit from the top member, not the leaf;
    - the oracle's emit converted a complex value of another width through a real float, and wrote
      that value's bytes into the slot.
  - Outcomes: `cfront_memberconv.c` is equivalent on both rails with equal digests, and a focused
    test pins the cast in the claim graph and the complex width on the oracle. Six injected defects,
    one per part of the fix, are each caught. The differential fuzzer ran 450 programs over three new
    seeds with no divergence.
  - Found, not fixed here: the twin aligns a `double _Complex` member to 16 bytes where the ABI and
    the oracle use 8. For a struct that holds one after a smaller member, the rails' digests differ
    (CF-CALIGN, below).
  CF-STATIC (2026-09-25) made a `static` local array or aggregate keep its shape in the emitted C,
  on both cfront rails.
  - The defect: both rails lowered a static array or struct as the object it is, its subscripts
    guarded against its extent, and then declared it as a scalar (`static uint32_t hist = 0u;`,
    `static struct Q s = 0u;`). The unit was reported clean, but its emitted C did not compile. An
    initializer on one is refused on both rails, and still is.
  - RED, measured on the parent (CF-MEMCONV, `46c87786`) with `cfront_staticarr.c` (a scalar,
    two-dimensional, pointer, one-element and volatile array, a struct and an array of structs): the
    digests are equal, since the digest carries no declarations, and both rails' emitted C fails to
    build.
  - What landed: a static array takes the array declaration and a static aggregate the aggregate one,
    with `static` and a zero initializer (`emit._static_decl`; the twin's declaration chain, whose
    local branches now take the storage class). A static scalar or pointer keeps its baked-in value.
  - Outcomes: `cfront_staticarr.c` is equivalent on both rails with equal digests, guards and
    storage extents. Three injected defects, one per rail and one per shape on the oracle, are each
    caught.
  CF-CALIGN (2026-09-25) made a scalar's layout the target ABI's on the twin, and gave both rails the
  ABI's atomic promotion.
  - The defect: the twin aligned every scalar member to its size. A `double _Complex` after a smaller
    member sat at offset 16 where the ABI and the oracle put it at 8, and every member after it moved
    too. `sizeof` and `_Alignof` gave the same wrong answers, and on i386 a `long double` aligned to 12,
    not 4. The oracle places each by `CType.align`, so the rails' digests and emitted offsets disagreed.
  - RED, measured on the parent (CF-STATIC, `c18f4deb`):
    - with `cfront_complexalign.c`, the rails' claim graphs differ, and both rails' emitted C returns a
      different value from the first input on. The oracle's is wrong through the `_Atomic` member
      (below);
    - over the five targets, the parent twin refuses the cross-target source (`sizeof(_Atomic cf)`),
      and the parent oracle folds three of its thirteen constants wrong on each LP64 target, four on
      Windows and two on i386, all of them `_Atomic` layouts (besides the two i386 `double` entries
      it still gets wrong, below).
  - What landed: one twin predicate, `scalar_align`, mirrors `CType.align`:
    - a complex type takes its element's alignment, and a `long double _Complex` the long double's;
    - a `long double` takes the ABI's alignment;
    - any other scalar takes its size.
    A member's placement asks it, and `sizeof` and `_Alignof` ask `type_layout`, which is built on it.
  - Found and fixed on the way:
    - the ABI's atomic promotion, on both rails. The twin placed an `_Atomic double _Complex` member
      where Clang does only because it aligned everything to its size, so aligning a complex to its
      element would have moved it. Neither rail modeled the promotion, and the oracle dropped `_Atomic`
      on members. The rule: an `_Atomic` type no wider than the target's promotion width (16 bytes on
      the 64-bit targets, 8 on i386) rounds its size up to a power of two and aligns to it. Both ABI
      tables now carry that width (`atomic_promote_size`). Each rail asks one predicate for a local's,
      a parameter's, a member's and a global's type (`with_atomic`, `atomic_layout`), and `packed`
      still wins over it. An `_Atomic` struct or union is refused on both rails;
    - the twin sized a typedef'd complex twice. `typedef float _Complex cf;` was 16 bytes, not 8, and
      a `long double _Complex` typedef was 16, not 32. A complex type reached through `typeof` or
      `_Atomic(T)` was doubled the same way. A typedef of it also dropped `_Atomic` on the twin;
    - the twin spelled "does a type-name start here" at five sites, and refused `sizeof(_Atomic T)`,
      which the oracle folds. Two predicates now answer, as on the oracle. One is for a declaration,
      `sizeof` and `typeof`, and it takes `_Atomic`. The other is for a cast and a compound literal,
      and it does not, so both rails refuse `(_Atomic T)x`.
  - Outcomes: `cfront_complexalign.c` is equivalent on both rails with equal digests. A cross-target
    test pins thirteen layout constants per target. The rails agree on every target, and they agree with
    Clang on every target except two i386 entries. A refusal test holds both rails to one refusal of an
    `_Atomic` aggregate and to the same folds of `sizeof`, `_Alignof` and `typeof` of an `_Atomic`
    type. Eighteen injected defects, one per part of the fix on either rail, are each caught.
  - Found, not fixed here (each queued as its own task):
    - on i386, Clang aligns a `double`, a `long long` and a `double _Complex` to 4, where both rails
      use 8;
    - the twin lays out an array-of-pointers member by its pointee and truncates the pointer on a
      store;
    - the twin ignores an `__attribute__((packed))` written after the closing brace;
    - both rails access an `_Atomic` object through a pointer or a member with a plain byte copy, not
      an atomic operation.


  S5-C (2026-09-26) landed G8: data movement as a first-class transformation, chosen jointly with
  compute, and closed Stage 5.
  - RED, measured on the parent (`2221db5e`), judged by this slice's fixtures and, where the parent
    had them, by its own verifier, codec, ASN.1 projection and C twin:
    - four example programs held nine claims that read RAM and HBM at once, with no move (D-R2);
    - the plan's movement family, declared by G11, was always empty;
    - the only transfer-aware rule, `channels.orchestrate`'s site choice with movement priced after
      it, sat 597,840 ticks off the joint optimum across eleven fixtures (16.2%);
    - the parent's R9 refused every plan that moves data, the nine legal ones included, because
      its lifetime cover compared phase ids with topological positions. Its refusals of the thirty
      law variants were therefore all by R9 or R13, never by the law each breaks;
    - neither rail read a plan that moves data, and both read a v1 plan whose "move" stays in its
      bank and one whose writeback is lossy; its ASN.1 decoders read all 18 malformed documents its
      projection could spell.
  - What landed (`bcir/kbcir/movement.py`; the reference is
    `docs/kernel/BCIR_DATA_MOVEMENT.md`):
    - a module transform M → M′. The optimizer chooses each claim's site and makes every crossing
      an explicit `mem.move.{far,near}` claim over a per-(resource, bank, episode) copy. Moves go in
      inbound, outbound and final phases, because the scheduler orders a phase by claim id. M′ is
      realized, placed, laid out and minted by the machinery every rail already shares;
    - Semantic Swap as the spec's classes: immutable resources drop and reload; recomputable ones
      rematerialize only from a `replay_safe` producer, with a replay certificate; mutable ones are
      written back home at their final version; a lossy codec needs every reader's R17 tolerance and
      an accuracy certificate; Belady eviction with an anti-thrash law; a deadline;
    - the joint planner over the transformed module's scheduled makespan: exact within a budget
      (TMSAO-1), greedy search otherwise (TMSAO-4); the three pre-G8 rules priced as baselines;
    - MV1–MV11 over (M, spec, M′, plan), trusting none of the planner's bookkeeping, and total;
    - ExecutionPlan v3, append-only: the move tail and the source/spec binding, on the codec, the C
      twin, BCAB, the control plane, the ASN.1 projection (version 3), the decoder fuzzer and the
      decoder campaign.
  - Outcomes (group `movement`; gate `tools/perf/check_movement.py --require-cc`):
    - `movement.implicit_cross_tier` 9 → 0 and `movement.excess` 597,840 → 0: every fixture at the
      optimum of the joint objective. The optimum comes from an independent enumeration, never from
      the planner's own answer;
    - `movement.laws.misattributed` 30 → 0 and `movement.laws.accepted` 0 → 0: each variant is
      refused by exactly its own law;
    - `movement.plans.unlawful` 9 → 0, `movement.roundtrip.mismatch` 9 → 0,
      `movement.parity.mismatch` 11 → 0 and `movement.asn1.accepted` 18 → 0 (of 87);
    - `movement.identity.drift` 0 → 0: a plan that moves nothing is the parent's, byte for byte;
    - `tools/testing/faults/movement.json` injects 51 defects into the planner, the transform, the
      producer, every movement law, R9, the codec, the ASN.1 projection and the C twin, and each is
      caught by its own row.
  - Found and fixed on the way:
    - R9's lifetime cover read phase ids as positions (above);
    - the C plan/pack binding located the generation vector at the body's end, past the v3 trailer,
      and called a matching v3 pack stale. One locator predicate now serves both readers (L14);
    - the Python codec applied the v3 laws only when a decoded plan "needed" v3, so it read a v3
      buffer with an all-zero binding that the C twin refused. The wire version now decides (L11);
    - the ASN.1 plan decoders applied no wire law. They now hold the native predicate in both
      directions, and a newer projection version is refused;
    - `verify_movement` raised on four malformed inputs and accepted a copy re-declared in the
      MMIO domain (L1);
    - the decoder fuzzer's plan passes had never reached a plan law: its corpus held only a
      StreamPack, and every plan is CRC-sealed (L2);
    - three sweep findings on the gate itself. The law rows depended on the planner's corpus, and
      the tolerance variant was also caught by the accuracy certificate. And `movement.excess` first
      measured the planner against its own answer, so a planner that returned the worst candidate
      read 0 (L1, L11);
    - CI found what the pre-PR mapping missed: `training/` was untouched, but a training chapter's
      `axes` table counts the non-test `bcir/` files that pass a cost axis by keyword, and
      `movement.py` prices fabric and sync. The table was regenerated, and the plan baseline
      bound to the corpus fingerprint re-recorded, with all 29 plans unchanged. S4-A met the same
      trap, so CONTRIBUTING now names the trigger.
  - Found, not fixed here: `realize` prices a far move like a near one, since the base cost is
    shared by four rails. `device_manifest`'s docstring claims distance-aware pricing that
    `realize` does not do.

  Five cfront fixes (2026-09-29) closed the four CF-CALIGN found-not-fixed items and a preprocessor
  defect found while testing the last. RED for each was measured on the parent (S5-C, `590b5130`).
  CF-I386 laid out an 8-byte scalar as i386 does, on both rails.
  - The defect: both rails aligned `double`, `long long`, `int64_t`, a `_BitInt(33..64)` and a
    `double _Complex` to 8 on every target. i386 aligns them to 4 in a struct and in `_Alignof`, while
    their size stays 8, so every member after one moved and `sizeof` and `_Alignof` folded wrong there:
    `struct { uint8_t c; double d; }` was 16 bytes, not 12. The rails agreed with each other, so the
    parity digest could not see it.
  - RED: a unit folds 22 layout constants (8-byte members, a union, an array, an `_Atomic long long`,
    a `_BitInt(40)` and eight bitfield structs), each checked against Clang with a `_Static_assert`.
    Both rails fold 18 of the 22 wrong on i386, and none on the other four targets.
  - What landed:
    - both target tables carry the ABI's `eight_byte_align` (4 on i386, 8 elsewhere), and one predicate
      on each rail asks it (`ctype_model.scalar_align`, the twin's `scalar_align`);
    - a bitfield follows Clang's placement rule (`bitfield_start` on both rails): a field moves to its
      type's next alignment boundary only when it would overflow a storage unit of its type's size, and
      a zero-width one aligns to its type's alignment, packed or not. Where alignment equals size, which
      is every other target and type, this is the old rule, so no other layout moves;
    - a bitfield of an under-aligned type is accessed over only the bytes it spans
      (`narrow_bitfield`), as a packed one is: an 8-byte unit at the start of a 4-byte struct would read
      past its end.
  - Outcomes: the constants match Clang on every target, all 22 where the rails implement the target's
    bitfield rules (x86-64 and RISC-V Linux, i386) and the 13 without a bitfield on AArch64 and
    Windows. Each bitfield's bit offset matches Clang's record layout on i386 and x86-64, and code
    reading and writing them lowers digest-equal on both rails. Two injected defects, one per rail, are
    each caught.
  CF-PTRARR laid out an array-of-pointers member as pointers, on the twin.
  - The defect: the twin gave the elements of a `T *arr[N]` member its pointee's size, so a later
    member sat at the wrong offset, `sizeof` was wrong, a store truncated the pointer to 4 bytes, and a
    load read the element into an integer, which Clang rejects. `*t->arr[i]` dereferenced `t` and then
    failed on the `->`, and so did `*s->p` through a plain pointer member.
  - RED: `sizeof` and `_Alignof` of `struct { uint8_t c; uint32_t *arr[2]; uint8_t d; }` fold to 16
    and 4 on x86-64, not 32 and 8. Of 29 access forms, 15 lower to a different claim graph from the
    oracle's and 12 are refused by one rail only. The twin refuses `cfront_ptrmember.c`.
  - What landed: each element takes the ABI's pointer size and is loaded and stored whole, typed `T *`;
    `*t->arr[i]` dereferences the postfix expression, as its precedence says. `t->arr[i][j]` and
    `t->arr[i]++` stay refused on both rails.
  - Outcomes: every form lowers or is refused alike on both rails, digest for digest on x86-64 and
    i386; the layout matches Clang on every target; `cfront_ptrmember.c` runs equivalent to the
    original on both emits. One injected defect is caught.
  CF-TRAILPACK read `__attribute__((packed))` written after a struct's closing brace before laying out
  its members, on the twin.
  - The defect: the twin honoured the attribute before the body but read the trailing spelling, the
    common one, only for the aggregate's alignment, after every member had been placed at its natural
    offset. `struct { uint8_t c; uint64_t n; } __attribute__((packed))` put `n` at 8 and was 16 bytes,
    where Clang and the oracle say 1 and 9, so every access to it read and wrote the wrong bytes.
  - RED: four of ten folded constants are wrong on every target (a nested packed struct, a straddling
    bitfield, an array member and a typedef'd struct), and `cfront_trailpacked.c` digests differently
    on the two rails.
  - What landed: the attributes after the `}` are read before the members are laid out
    (`trailing_attrs`); a trailing `aligned(N)` still raises the alignment.
  - Outcomes: the constants match between the rails on every target and Clang wherever the rails
    implement its bitfield rules; `cfront_trailpacked.c` is equivalent on both emits with equal
    digests. One injected defect is caught.
  CF-ATOMIC made every access to an `_Atomic` object one atomic operation, on both rails.
  - The defect: both rails lowered an `_Atomic` object reached through a pointer, a member, a member
    array, an array of structs or a subscript as a plain byte copy, and a compound assignment or an
    increment of one as a load, an operation and a store. A concurrent update between them was lost,
    though C makes each one read-modify-write (C11 6.5.16.2p3, 6.5.2.4p2). The digest could not tell them apart:
    it did not read an access's order.
  - RED:
    - the four new tests fail on the parent;
    - the parent twin refuses `cfront_atomicaccess.c`, and the parent oracle's emit diverges from the
      original at the first float generic, whose value it converted through a `uint32_t`;
    - `acc_bump`, three counter updates run from four threads 100,000 times each, lost 52 to 65% of
      the updates in three runs of the parent oracle's emit (162,719 of 400,000 in the first).
  - What landed:
    - a load or store of an `_Atomic` object keeps its claim, on lane A with the atomic hazard
      (`lower._access_order`, the twin's `mark_atomic`). A compound assignment or an increment is one
      `c.c11atom.rmw:<op>` claim, addressed as the store is: `ATOMIC_ADD`, `ATOMIC_SUB` or
      `ATOMIC_XOR` for those operators, `CMPXCHG` for the rest. Both emits spell every one through an
      `_Atomic` lvalue (`(*(_Atomic T *)addr) op= v`), never a byte copy;
    - the value of an assignment to one is the value stored, never a second atomic read;
    - `unqualified` drops `_Atomic` and restores the layout the ABI's atomic promotion widened
      (`CType.natural`); a value read out of an `_Atomic float _Complex` aligns to 4, not 8;
    - the digest canon names the order (`c.load!atomic`, `c.store!atomic`), so a source with `_Atomic`
      and one without digest differently. No existing claim carries the hazard, so no record moved;
    - the generics type their value by the pointee: a 64-bit `atomic_load` or `atomic_fetch_add` had
      been a `uint32_t`;
    - a bit-field of `_Atomic` type is refused on both rails, as GCC and Clang refuse it.
  - Found and fixed on the way, on the twin: a typed store through an `_Atomic _Bool *` stored the raw
    byte (2, not 1), because a pointer parameter or local did not record a `_Bool` pointee; a store
    through any `_Bool *` did not normalize the value; and a compound assignment to a subscripted
    pointer typed its value as the pointer.
  - Outcomes: every function of `cfront_atomicaccess.c` has its exact count of atomic accesses and
    read-modify-writes on both rails, digest-equal, and runs equivalent to the original on both emits
    under Clang and GCC; `acc_bump` loses no update on either emit. An 84-form probe over five targets
    agrees wherever both rails lower a form. Ten injected defects, on both rails, are each caught. The
    escape and effect reports stay byte-identical across the rails over the corpus, the new unit
    included.
  CF-PASTE kept apart, in both preprocessors, tokens that would lex as others.
  - The defect: both preprocessors re-spell every source line from its tokens, and kept a space only
    between two words, so tokens whose spellings run together into others came out as those (maximal
    munch, C 6.4p4). `a + ++g` came out `a+++g`, which is `(a++) + g`; `-NEG(a)` with
    `#define NEG(x) -x` came out `--a`, a decrement. Each lowered clean and computed another value, on
    both rails alike, so the parity digest agreed. `y / *p` came out as a comment opener, and
    `a + +b` as a parse error.
  - RED: 186 of 2,000 generated token sequences re-lex as other tokens. On a unit of the four
    silently wrong forms both rails lower clean with equal digests, and all 2,400 calls return
    another value than the original's. Both preprocessors fail the Clang and GCC differential over
    `cfront_pp_avoidpaste.c`, where 16 of 24 lines lex as other tokens.
  - What landed: one predicate on each rail (`cpp._pastes`, `bcir_cpp.c`'s `pastes`) puts a space
    between two words, before a comment opener, where maximal munch would extend a punctuator, between
    two dots (an ellipsis spans three tokens), and where a pp-number would run on. The twin's tokenizer
    now reads every multi-character punctuator and a pp-number that begins with `.`. It decides exactly
    where the next pass cannot respace: the last pass and a gathered macro argument, which a `#`
    keeps.
  - Outcomes: every one of the 357 tracked C sources preprocesses byte-identically to the parent on
    both rails; the space goes only where the text lexed as other tokens. The generated sequences,
    stringized and rescanned through macros too, re-lex exactly and match byte for byte between the
    rails. `cfront_paste.c` runs equivalent to the original on both emits under Clang and GCC. Thirteen
    injected defects, on both rails, are each caught.
  Found, not fixed here (each a suggested follow-up):
  - `sizeof` folds a wrong constant, silently, for most operands that are not a plain scalar or struct
    name. The twin folds an array variable to its element's size (`sizeof buf` of a `uint32_t buf[10]`
    is 4) and a global the function has not yet read to 4. The oracle folds a subscripted row or
    element, a member and a dereference to 4, and a string-initialized array to 0 (`sizeof s.a` of a
    `uint32_t a[4]` member is 4, `sizeof *sp` of a 32-byte struct 4, `sizeof str` of `"hello"` 0).
    Of 30 probed forms, 24 fold a constant other than Clang's on one rail or both, and every such unit
    lowers clean;
  - the twin reads before its struct table on a file-scope struct's member (`gs.n`), whose entry
    carries no struct index, and crashes; the oracle lowers it;
  - the oracle refuses a statement whose first identifier is also a struct tag
    (`struct st *st; st->n += 1u;` is "expected a type"), though C keeps tags in their own name space;
  - a stringized argument drops its whitespace on both rails: `S(a + b)` is `"a+b"`, where C 6.10.4.2
    and Clang make it `"a + b"`;
  - `_Atomic(T *)`, an atomic pointer object, is modeled on both rails as a pointer to `_Atomic T`;
  - neither rail models AArch64's alignment for an unnamed bitfield or the MSVC bitfield layout.

  Four cfront fixes (2026-09-29) closed the first two found-not-fixed items above and two defects found while
  measuring them. RED for each was measured on the parent (the five fixes, `b958d890`).
  CF-SIZEOF folded `sizeof` to its operand's own size and typed it `size_t`, on both rails.
  - The defect: both rails folded a wrong constant for most operands that are not a plain scalar or struct
    name, and every such unit lowered clean. The twin measured a local array by its element (`sizeof buf`
    of a `uint32_t buf[10]` was 4) and every global as 4, and refused a type-name array and most operators
    (a comparison, a shift, `&`, a call, a conditional, a comma, an assignment). Its `sizeof` bound only a
    primary expression, so `sizeof -x` lowered `4 - x`. The oracle measured a row,
    a struct element, a member and a dereference as 4, a string-initialized array as 0, and an operator's
    result without its operands' conversions (`sizeof(x8 = x64u)` was 4, not 1). Both typed the result a
    32-bit unsigned, so `sizeof(uint32_t) - 8u` wrapped at 2^32 where `size_t` wraps at the pointer
    width.
  - RED: two probes hold 159 operand forms against Clang on each of the five targets, 795 cases. The
    parent oracle folds a wrong constant in 368 of them and refuses 30; the parent twin folds a wrong
    constant in 138 and refuses 510. The three sizeof tests fail on the parent.
  - What landed:
    - the oracle types the operand with neither the lvalue nor the array-to-pointer conversion
      (`_sizeof_type`): an array stays its array, a row a row, a member array its member. `&`, a call, a
      conditional, an assignment, an increment, a comma and a nested `sizeof` are typed as C types them, an
      operator's bit-field operand by its promotion (`_sizeof_operand`), and the parser reads a type-name's
      `*`s and `[N]`s (`_abstract_type_name`);
    - the twin has no AST, so a static walk reads a designator (a name, `[...]`, `.m`, `->m`, `*`, `&`,
      parentheses, a string literal) off the tokens without lowering it (`sz_unary`, `sz_postfix`,
      `sz_name`). It lowers any other operand speculatively for its value's type and rolls the lowering
      back (`spec_mark`), as `typeof` does. A global keeps every dimension, and an unsized one takes its
      extent from its initializer (`brace_init_extent`);
    - the result is a `size_t` temp on both rails. A bit-field operand, a function designator, an incomplete
      type and a type larger than the target's `PTRDIFF_MAX` are refused, each for its own reason;
    - the oracle types a `long double` literal and a floating operation by the target's ABI, not the host's:
      `sizeof 1.0L` is 12 on i386 and 8 on Windows.
  - Outcomes: every case matches Clang on both rails except two Windows ABI facts neither rail models
    (`wchar_t` is 2 bytes there, and MSVC lays out bit-fields its own way) and the forms a rail refuses:
    the oracle one (a string-initialized local array), the twin three (that and two typedef'd array forms).
    `cfront_sizeof_forms.c`, 38 functions, runs equivalent to the original on both emits under Clang and
    GCC.
  CF-GSTRUCT lowered a file-scope struct's members, on the twin.
  - The defect: the twin bound a struct global with struct index -1, so every member access read
    `c->s[-1]` and the frontend crashed: a read of `gs.n`, a store through it, a nested `gt.in.a[1]`, a
    pointer global's `gp->n`. It computed a global's device domain without the struct, so a global holding
    volatile members was placed in ordinary memory. The oracle lowered all of these.
  - RED: of 26 access forms on file-scope structs, the parent twin crashes on 19, refuses 3 and lowers 4.
  - What landed: `use_global` finds the struct index once and gives it to both the device domain and the
    environment; a member access on an object that is not a struct or union is refused, never indexed at
    -1.
  - Outcomes: 25 of the 26 forms lower digest-equal on both rails and run equivalent to the original from
    the same seeded globals, the result and every global's bytes compared. Both rails refuse the 26th, a
    member read through a subscripted array of structs. `cfront_globalstruct.c` runs equivalent on both
    emits.
  CF-GINIT kept an initialized global's declared type, on the oracle.
  - The defect: the oracle typed an initialized scalar or struct global as a one-element array of it, a
    relic of the read-only lookup-table model. `gw + 1` of a `uint64_t gw = 0x100000000u` was computed in
    32 bits and returned 1, and `gi.n` of an initialized struct was refused.
  - RED: 5 of 9 forms on initialized globals fail on the parent: two oracle miscompiles, one oracle refusal
    and two `sizeof`s.
  - What landed: an initialized scalar or struct keeps its declared type; only an unsized array takes its
    extent from its initializer.
  - Outcomes: all 9 forms lower digest-equal and run equivalent to the original on both emits.
  CF-DECAY made an array used as a value the address of its first element, on both rails. With it,
  CF-PTRDIFF typed a pointer difference `ptrdiff_t`, and CF-DEREFIDX lowered the index of `*(la + j++)`
  once, on the twin.
  - The defect: both rails read a member array used as a value (`gs.a`, `l.a`, `sp->a`) as a load of its
    first element, so `return gs.a;` did not compile and `(uintptr_t)gs.a` returned the element. Both
    typed `la + 1` and a conditional over pointer arms as a 32-bit integer (`int32_t t = la + 1;`, which
    does not compile), and `p - q` as one too; the twin's was unsigned, so a difference of -2 returned
    4294967294. The twin lowered the index of `*(la + j++)` twice, reading `la[j + 1]` and stepping `j` by
    two.
  - RED: 25 of 28 forms fail on the parent, and 20 of those digest equal on both rails: the digest does
    not read a temp's type, and both rails shared the misreading.
  - What landed: a member array's value is the member's address typed as a pointer to its element
    (`_array_value`, the twin's `member_array_value`); an array operand of `+`, `-` or a conditional
    decays to a pointer to its element (`_arith_decay`, `rid_addr`); the difference of two pointers is a
    `ptrdiff_t`. A row of a multi-dimensional array used as a value had been its flat element, silently;
    it is refused on both rails now. The twin's deref-assignment lookahead rolls back what it lowered.
  - Outcomes: 23 of the 28 forms lower digest-equal and run equivalent to the original; both rails refuse
    the other 5, each a row of a multi-dimensional array. `cfront_decay.c` runs equivalent on both emits.
  Found and fixed on the way:
  - the twin sized a function pointer 8 bytes on every target, where i386's is 4;
  - the oracle raised a bare `KeyError` on a pointer to a struct it had not laid out (an opaque
    `struct fwd *`, a member pointer to a later struct or to the struct being defined): a traceback where
    the pipeline needs a refusal it can route. It is a `CLowerError` on every path that names a type
    (`_aggregate`);
  - the oracle's check for a function designator under `sizeof` was unreachable: its VLA path looked the
    name up first and refused `sizeof g` as an undeclared identifier. The refusal test passed on that and
    on a parse error, so each refusal witness now asserts its own reason on each rail;
  - the G10 escape gate caught the first cut of `cfront_decay.c`. One function read past a local array
    for an index of 10 or more, which the commute witness passes, so nine pairs diverged on stack
    garbage (`effects.commute.unsound` 9). Two local arrays had their address subtracted or converted
    to an integer, and the escape analysis reports them escaping, rightly, which `escape.unproved`
    counts (2). The index is bounded, those arrays moved to file scope, and the test runs the local
    forms inline; every G10 row is back at the parent's value.
  The ten new tests fail on the parent, and 35 injected defects, on both rails, are each caught.
  Found, not fixed here (each a suggested follow-up):
  - brace elision: a braced list for a struct whose first member is an array initializes that array's
    elements first (C11 6.7.9p20). Both rails give the first value to the whole member and the next to
    the next member, so `struct { uint8_t a[2]; uint8_t c; } l = {1, 2}` computes another value on both
    emits, silently; the oracle's emit for a 16-byte array member does not compile;
  - a string-initialized local array (`char str[] = "hello"`) is not sized by its literal, so `sizeof str`
    is refused on both rails;
  - both rails make `wchar_t` 4 bytes on every target; Windows makes it 2 (`sizeof(L"ab")` is 6 there);
  - the twin refuses a typedef'd array type (`typedef uint32_t row_t[3];`);
  - both rails refuse a subscript of a two- or three-dimensional global (`gb[i][j]`) and a member read
    through a subscripted array of structs (`gs_arr[i].n`);
  - the oracle refuses a pointer to a struct it has not laid out, a self-referential one included
    (`struct node { ...; struct node *next; }`), which the twin lays out;
  - both rails refuse `sizeof &g`, the size of a function's address.

  CF-GAPS (2026-09-29) closed those seven items on both rails, and three defect families found while
  measuring them. RED for each was measured on the parent (`05e182c5`), whose probe groups fail 39 of
  43 forms.
  CF-BRACE walked an initializer as C walks the current object (C11 6.7.9p17-21).
  - The defect: neither rail elided braces; each gave a list's values to whole members in its own way.
    `struct { uint8_t a[2]; uint8_t c; } l = {1, 2}` returned another value on both emits, a union took a
    value for every member, and the entries after a designator did not continue past it. The oracle
    crashed on a nested struct's elided list (`tuple index out of range`), where the twin lowered it to
    another value. An excess or overriding initializer, a string too long for its array, a non-character
    array given a string, a designator past the array (a store one past a 4-element local) and a 65-deep
    nesting were lowered, not refused: the oracle lowered 15 and the twin 14 of 18 such units.
  - RED: 14 of 16 brace forms fail on the parent -- 11 return another value on at least one emit (both,
    on 2), 4 oracle emits do not compile, and the oracle crashes on 4.
  - What landed: one walk per rail over a stack of frames (`lower._init_list`, `_init_next`,
    `_init_designate`; the twin's `init_list` over an `iwalk`). A positional entry fills the next scalar
    subobject, descending through unbraced sub-aggregates; a designator resets the walk to the list's
    object, and the entries after it continue past the designated subobject at the innermost level. An
    anonymous member is one subobject, a union takes one value, a string literal fills the character array
    it meets, a struct value fills its subobject whole, and `{e}`/`{}` fill a scalar. The walk records the
    bits each store covers, so an override of an initialized subobject and a union's change of member are
    refused, as is an excess entry; 64 nested subobjects and an `int` designator bound it on both rails. A
    store is an indexed `base[i]` store only when it writes a whole element of the declared array its own
    list initializes, else a byte-offset store.
  - Outcomes: the 16 forms lower digest-equal and run equivalent on both emits, and `cfront_braceelide.c`
    runs equivalent function by function under Clang and GCC.
  CF-STRLOCAL sized a character array by its string literal.
  - The defect: both rails declared `char s[] = "abc"` one element long, its emit and a sized
    `char s[8] = "hi"`'s did not compile, and `sizeof s` was refused as an incomplete type.
  - RED: all 8 string-initialized forms fail on the parent.
  - What landed: a literal's code units after concatenation (an escape is one unit) size an unsized
    array with its NUL and fill it unit by unit, the rest zero; a wide literal fills an array of its unit
    (`u` 2, `U` 4, `L` the target's `wchar_t`); a literal exactly as long as a sized array drops its NUL,
    a longer one is refused; rows of strings take their count from the list (`clex.str_units`, the twin's
    `str_units`). A non-ASCII character and a universal character name in a string initializer are refused
    on both rails.
  - Outcomes: the 8 forms and `cfront_strlocal.c` run equivalent on both emits.
  CF-SELFREF laid out self-referential and forward-referenced structs, on the oracle.
  - The defect: the oracle refused a pointer to any struct it had not laid out -- the struct being
    defined, one defined later, one only declared (`struct t;`), an opaque handle. The twin laid out the
    self-reference and refused the forward and opaque ones at their declaration.
  - RED: all 6 forms fail on the parent, 3 on both rails.
  - What landed: an incomplete struct type (C11 6.2.5p22, `ctype_model.incomplete_aggregate`), completed
    by its tag when, and if, it is defined (`lower._complete`; the twin's `declare_struct` and
    `complete_struct_refs`); `struct t;` declares the tag; a pointer-returning call is a pointer on both
    rails, dereferenced in place (`f()->v`). A member access through, `sizeof`, a local or a parameter of
    a struct still incomplete is refused on both rails.
  - Outcomes: the 6 forms and `cfront_selfref.c` run equivalent on both emits.
  CF-SMALL closed the four smaller gaps.
  - `wchar_t` and an `L"..."` literal's unit are the target's own integer type (Clang's `__WCHAR_TYPE__`):
    4-byte signed on x86-64, RISC-V and i386 Linux, 4-byte unsigned on AArch64 Linux, 2-byte unsigned on
    Windows (`TargetABI.wchar_size`, `wchar_signed`). Both rails had made it a 4-byte `int` on every
    target. `sizeof &f` is a pointer's size.
  - The twin keeps a typedef'd array's dimensions inner to a declarator's own (`row_t rw[2]` is two rows)
    in a local, a member, a parameter, a global, `sizeof`, a type-name and a typedef of it; the oracle's
    parser had put them outer. A pointer to such a type is refused on both rails.
  - A 2-D/3-D global indexed in full is one row-major element of the whole object at its byte offset, for
    a read, a store, `OP=`, `++`/`--` and `&` (`lower`'s `whole` index, the twin's `global_md_field`).
    `a[i].m[j]` of an array of structs folds the element index with the member's -- `i*K + j`, K the struct
    size in `m`'s elements -- in every access form (`aos_member_array`, and the value and increment paths
    now share `member_elem_assign_value` and `member_elem_incdec` with `s.m[j]`). `&m[i][j]` of a local, a
    VLA or a `T m[][N]` parameter flattens every subscript on the twin, as the oracle did.
  - RED: 11 of 13 forms fail on the parent. Beside them, all 11 `a[i].m[j]` access forms and all 8
    multi-dimensional global value and increment forms were refused on both rails, and the twin refused 10
    of 12 multi-dimensional address forms -- and took a row's address as an element's.
  - Outcomes: every accepted form lowers digest-equal and runs equivalent on both emits;
    `cfront_typedefarr.c` and `cfront_mdglobal.c` run equivalent function by function, and the sizeof
    table holds `wchar_t`, `L"ab"`, `u"ab"`, `U"ab"` and `&hfun` against Clang on all five targets.
  CF-STORAGE honored a storage class wherever a declaration spells it (C11 6.7p1).
  - The defect: both rails kept only a leading `static`. `volatile static uint32_t n` and `unsigned static
    n` were uninitialized locals on both rails, digest-equal, so a second call returned another value;
    `uint32_t static n` was one on the oracle. The twin refused a qualifier or storage class after a
    typedef name or a struct tag. A block-scope `extern T g;` bound a new uninitialized local in place of
    the global on the oracle (and `const extern T g;` on both).
  - RED: of 14 storage and qualifier forms, the parent oracle returns another value on 7 and the twin on
    3, and the twin refuses 7 more.
  - What landed: a qualifier or storage class after the type specifier is the one before it
    (`cparse._TRAILING_SPEC`, the twin's trailing scan in `p_type_base`); a block-scope declaration reads
    its own storage classes; a block-scope `extern` and a local array with no size and nothing to count
    are refused on both rails.
  - Outcomes: 9 forms lower digest-equal and run equivalent on both emits over three rounds of calls, and
    both rails refuse the other 5: the three `extern`s and the two static arrays with a brace initializer
    (refused before on both, and now in either specifier order, until CF-STATICTAB below lowered them).
  Found and fixed on the way:
  - the twin's speculative `a[i].f` value and increment paths refused a member array after `a[i].`, before
    the read path could take it;
  - the round-trip gate classified "the emit names a struct only the original defines" by the parser's
    refusal of the undefined struct; once such a pointer lowered, three fixtures' emits re-parsed and their
    signatures drifted. The gate now names the undefined tags itself, and its included set is the parent's
    64.
  The ten new or extended tests fail on the parent, and 36 injected defects, on both rails, are each
  caught.
  The same measurement queued six follow-ups, each closed here with the defects found while closing it.
  CF-STRUCTVAL typed a struct or union value as the struct, on both rails.
  - The defect: a struct-valued load -- an element of an array of structs (`ps[i]`, a global's, `pp[i]`
    through a pointer parameter), a member (`g.b`), `*p`, a struct returned through a function pointer,
    `va_arg(ap, struct T)` -- was emitted into an integer temp on both rails (`uint32_t t = ps[i];`), so
    neither emit compiled, and the twin refused a member access on such a value. A select of two structs in
    a brace list gave the whole value to the first member on both rails, digest-equal
    (`{c ? ps[0] : g.p, 9u}` put `9u` into `.y`): a silent miscompile.
  - RED: every form's emit fails to build on both rails on the parent, and the twin refuses the fixture
    (`member access on an object that is not a struct or union`).
  - What landed: an aggregate temp spelled `struct T` on both rails (`emit._load_ctype`; the twin's
    `tempagg` on every value path, a function pointer's struct return carried as `bcir_ctype.fp_ret_agg`);
    a select of two arms of one struct type is that struct, and any other pairing with a struct arm is
    refused on both rails. Four twin-only defects on the way: an array parameter of structs was a struct by
    value (it decays to `struct T *`), `&ps[i]` was an `int32_t *`, a struct global's type went unrecorded
    so the brace walk missed it (a runtime mismatch), and a struct assignment used as a value lowered where
    the oracle refuses it.
  - Outcomes: `cfront_structvalue.c` lowers digest-equal on the four targets and runs equivalent function
    by function under Clang and GCC; `typeof(ps[0])` and `_Generic(ps[i], ...)` agree (the twin had picked
    `default:`).
  CF-STRUCTINIT refused a struct or union given a value of another type (C11 6.5.16.1p1, 6.7.9p13).
  - The defect: `struct s x = 5;`, `= "a"`, `h.inner = 5`, `*p = 5`, `ps[0] = 5` and 13 more such units
    lowered on both rails, digest-equal, where Clang rejects each.
  - What landed: a struct or union takes a value of its own type only, compared by kind and tag with
    qualifiers ignored, at every declaration and store site (`lower._struct_value`; the twin's
    `agg_value_ok`), refused with one reason per context on both rails; a cast to a struct type is refused.
  - Outcomes: the 18 units are refused, each reason asserted on each rail, and the valid same-type copies
    and stores run equivalent to the original.
  CF-NESTMEM lowered a member of a member of an array element (`a[i].m.k`, `a[i].m.arr[j]`).
  - The defect: both rails refused it (the oracle: `unsupported base expression Index`).
  - What landed: one load or store at the element's stride plus the member's flattened offset, each hop
    adding its offset and, through a volatile member, its qualifier -- which the twin's emit had dropped
    for `o.in.v` (`lower._aos_member`; the twin's `sdef_elem_field` through `member_descend`). The elements
    a pointer member points at (`h.next[i].v`) read and store on the twin as on the oracle, and a statement
    `a[i].f++;` lowers on the twin.
  - Outcomes: `cfront_aosnest.c` -- local, global and parameter arrays, member arrays of structs, unions,
    three levels, 1-D and 2-D member arrays, volatile and `_Atomic` leaves, in every access form -- runs
    equivalent on both emits, and its volatile function keeps its five volatile accesses on each rail.
  CF-PAREN let the twin take a postfix after parentheses (`(a)[1]`, `((T[N]){...})[i]`, `(*p).x`).
  - The defect: the twin refused each (`PARSE-ERR ;`). The oracle lowered them, two wrongly: `(*p).m` read
    the whole struct into a 4-byte temp -- a wrong value past its first word, and a lost store -- and
    `(*p)[i]` through a row pointer emitted a subscript of a scalar, which does not compile.
  - What landed: the twin drops an operand's redundant parentheses before a postfix, after the mutation
    scan has read the source spelling and before the body parses (`unparen_body`), never around a call, a
    condition, a cast's type name or a declarator. Both rails read `(*X).f` as `X->f` and `(*X)[i]` as
    `X[0][i]` (`cparse._postfix_tail`).
  - Outcomes: `cfront_parenpostfix.c` runs equivalent on both emits, and the 11 forms both rails still
    refuse (a parenthesized declarator among them) assert their own reason on each rail.
  CF-STATICTAB folded a static local's initializer into its declaration, on both rails.
  - The defect: a static local array, struct or union with a brace initializer -- a lookup or CRC table --
    was refused on both rails. Beside it, statics miscompiled with parity intact, because the canon read no
    static's value: the oracle declared `static int64_t n = -5;` as `-5u`; the twin folded `~0u` into a
    `uint64_t` as `18446744073709551615u`, gave a second scope's same-name static the first's value, and
    dropped `static` and the initializer of a static pointer (CF-STATICPTR: `static uint32_t *sp = 0;`
    became an uninitialized automatic pointer). The oracle declared two scopes' same-name statics twice. An
    integer literal's type ignored the target's `long`: `3000000000` was a 4-byte `long` on the oracle for
    LLP64 and ILP32, and `5L` 8 bytes on the twin.
  - RED: the four new tests fail on the parent.
  - What landed: a static's initializer runs through the initializer walk in constant mode
    (`lower._static_init`, `_kfold`; the twin's `init_image`, `kfold`): each entry is lowered speculatively
    and its claims folded as C evaluates an integer constant expression, each operation in its own type --
    promotions, the usual arithmetic conversions, unsigned wrap; signed overflow, an out-of-range shift and
    a division by zero refused (C11 6.6p4) -- and each value converted for its subobject. The image is
    rendered once, as the declaration's initializer, never as stores at each call; an inferred `[]` takes
    its count from the list. The image is no claim: the canon gains one sorted `static NAME = INIT` line
    per static with a nonzero image, identical on both rails, so two tables one value apart digest apart. A
    typedef'd function-pointer local is declared by its alias on the twin, and an integer literal's type
    follows the target's `long` on both rails.
  - Outcomes: `cfront_statictab.c` runs equivalent to the original over four rounds of calls on both emits,
    and 15 initializers that are no integer constant expression -- an address, a float constant, another
    static -- are refused with one reason on both rails. `cfront_storage.c`'s digest moves on both rails in
    lockstep: its nonzero statics gain canon lines.
  CF-RTVOL made the round trip idempotent for a volatile access at a byte offset.
  - The defect: the emit spells a volatile member access as one access at its byte offset,
    `*(volatile T *)((const volatile char *)p + K)`, and the oracle re-lowered that as a pointer
    computation and an access at offset 0, so each round added a cast and an add per access. A re-parsed
    emit names its locals `t<rid>`, the emitter's own spelling for a temporary, so the next emit redeclared
    them.
  - RED: the new round-trip tests fail on the parent
    (`cfront_rmw.c: e1 re-lowers to other accesses than the original's`; `unsigned int t103 = 3u;` inside
    `bcir_f(uint32_t t101, uint32_t t103)`).
  - What landed: both rails fold `*(volatile T *)((char *)p + K)` -- T a volatile, non-atomic integer or
    floating type, K an integer literal, p a declared pointer, an array or `&s` whose region holds volatile
    storage -- into the one load or store at offset K from p that the member access lowers to
    (`lower._byte_offset_access`; the twin's `byte_off_access`, over the tokens, rolled back on a miss). A
    temporary is `t<rid>` only while no declared name spells it, on both rails (`emit._Names`,
    `uniq_local`).
  - Outcomes: the three volatile register maps (`cfront_rmw.c`, `cfront_bitfield.c`,
    `cfront_bfcompound.c`), given their definitions, reach a fixed point over three rounds with the
    original's accesses, and `cfront_bitint.c` now round-trips, so the included set is re-pinned at 65. No
    fixture's digest or emitted C moves on either rail.
  CF-MEMDECAY stored an array's address, not its bytes, into a pointer slot.
  - The defect: an array stored into a pointer the emit writes by `memcpy` -- a member (`h.p = arr`,
    `{7u, arr}`, `{.p = arr}`, `hp->p = arr`, a compound literal's), a dereference (`*pp = arr`), an
    element of a member array of pointers (`r.slot[i] = arr`) -- lowered to one claim graph on both rails,
    which read the array. The oracle's emit copied the array's first bytes into the pointer
    (`memcpy(&h.p, &arr, 8)`), a wild pointer the next access dereferenced; the twin's did not compile
    (`uint64_t _v = arr`).
  - RED: 11 of 17 decay forms fail on the parent -- the oracle's emit crashes or reads another value on 10,
    and the twin's does not build on 9.
  - What landed: an array source is staged in a pointer object on both rails -- `emit._store_conv`'s array
    case, the twin's member and member-array element store emits (`const volatile void *`, which any array
    converts to). The claim graph is unchanged: no fixture's digest moves on the four targets, on either
    rail.
  - Outcomes: the forms, and `cfront_memdecay.c` function by function, run equivalent on both emits.
  CF-NULLPTR typed a null pointer constant as the pointer that takes it.
  - The defect: `T *p = 0;`, `p = 0;` (a local, a parameter or a global), `return 0;` from a function
    returning a pointer, and `ps[i] = 0` or `{&v, 0}` for an array of pointers lowered to the same claim
    graph on both rails, and each emit assigned an `int` temp to the pointer (`int t = 0u; p = t;`), which
    C forbids and Clang and GCC reject; a null function pointer failed the same way.
  - RED: 15 of 20 null-pointer forms fail to build on the parent's oracle emit and 14 on the twin's.
  - What landed: a `c.const 0` taken by a pointer -- a declaration's initializer (braced or not), an
    assignment to a named pointer, a typed element store, a `return` -- types its temp as that pointer
    (`lower._null_pointer` over the zero constants `_emit` records; the twin's `null_pointer`), so the emit
    declares `T *t = 0u;`, a null pointer, and a function-pointer temp by its type. Only a temp's C type
    moves, so no digest does.
  - Outcomes: the forms, and `cfront_nullptr.c`, run equivalent on both emits.
  CF-UAF: the twin's `tempptr` read its source resource through a pointer into the resource table after
  `add_res` had grown it -- a heap use-after-free in the compiler, which the sanitized twin found over a
  probe corpus. It now reads the source first, as `temp_like` already did. `cfront_resgrow.c` makes a
  pointer temporary at every count across the table's first growth, so `tools/c/sanitize_cfront.sh`, which
  runs every fixture through the sanitized twin, holds it; a scan of every resource or claim pointer held
  across a call that can grow its table found no other.
  CF-RAILSPLIT closed four splits the slices above found, each an accepted form no fixture held, where the
  rails lowered one function to two claim graphs.
  - The defect: the twin gave a logical not the ADD opcode where the oracle's `_UN` gives SUB; the twin's
    probe of a bare `*e;` as a store kept the operand's claims, and the statement then lowered it again;
    the twin counted `&p[i]` as taking `p`'s address, so it did not recover an allocated buffer's extent,
    which the oracle does; and the twin laid a VLA of volatile elements out as ordinary memory, failed R3
    on it and declared it without `volatile`.
  - RED: `cfront_railsplit.c` digests apart on the four targets on the parent, and the twin's lowering of
    it fails verification.
  - What landed: `!` takes SUB on the twin; the store probe rolls back (`spec_begin`, `spec_end`); `&`
    marks a name's address taken only when no subscript or member follows the name; a volatile VLA is a
    device object, declared `volatile`.
  - Outcomes: the fixture lowers to one claim graph on both rails, and each emit declares its VLAs
    `volatile` and runs equivalent to the original.
  The twin's test driver (`runtime/c/test_cfront.c`) now builds under `gcc -Wall -Werror`, where one
  misleadingly indented line had failed it.
  The G10 escape gate caught three of these fixtures' first cut: `cfront_memdecay.c` stored eight local
  arrays into pointer slots, and `cfront_nullptr.c` and `cfront_parenpostfix.c` each lent one to a call,
  which the analysis reports escaping, rightly, and `escape.unproved` counts (10). The fixtures hold those
  forms on file-scope arrays, and the local forms run inline in their tests; every G10 row is back at the
  parent's value.
  The follow-ups' 16 new tests fail on the parent, and with CF-GAPS's, 86 injected defects, on both rails,
  are each caught. The campaign is committed as four fault tables (`tools/testing/faults/cfront-*.json`),
  run by `red_sweep` through a gate that names each failing test as a finding
  (`tools/testing/check_tests.py`, itself driven into its refusals by `test_red_sweep.py`). One fault
  needed a new witness first: the pointee size a struct's completion back-fills into a pointer member
  (`complete_struct_refs`) changed no digest and no run until a function took `typeof` of such a member and
  subscripted through it (`sr_typeof_next` in `cfront_selfref.c`).
  CF-TERNARY evaluated the operands C may leave unevaluated only when C does.
  - The defect: both rails computed both arms of `?:`, and the right operand of `&&` and `||`, and chose
    after (C11 6.5.15p4, 6.5.13p4, 6.5.14p4) -- `d ? n / d : 0u` divided by zero, `p ? *p : s` and
    `p && *p` read through a null `p`, and a call in the operand C skips ran. A conditional whose arms are
    void (`c ? f() : g();`, an `assert`'s `c ? (void)0 : fail()`) assigned a void call's non-value and did
    not compile, `(void)e` was a cast of `e` to `uint32_t`, and the twin returned a void call's placeholder
    from a void function (`return t;`) and refused a void function whose only effect is the calls it
    makes, which the oracle verifies clean.
  - RED: on the base below this change both rails lower `cfront_condeval.c` with no branch where an arm
    divides, and neither verifies it: the oracle's void conditional copies the void sentinel (R2, an
    undeclared RID), and the twin's R12 refuses `ce_twice`, which only calls. Without the void forms, the
    fixture's first cut built on both rails, and both emits crashed on the driver's first zero divisor.
    The void-return unit splits the rails.
  - What landed: an operand is still computed eagerly -- a `c.select`, `c.bin.land` or `c.bin.lor`, so
    a pure conditional's digest does not move -- when it can neither trap nor change state: every claim it
    made is on a short list of arithmetic, comparison, address and constant ops, none volatile, none
    writing a declared variable (`lower._operand_pure`, the twin's `operand_pure`; `div` and `mod` are
    off the list). Any other lowers as a branch that assigns one named local of the select's type in each
    arm; the arm the left operand of `&&` or `||` decides stores 0 or 1. The twin decides by lowering the
    operand speculatively, memoized by its first token so a chain of conditionals stays quadratic. A void
    value is the oracle's `_VOID_RID` and a resource marked `is_void` on the twin: two void arms branch
    with no local (both pure, they run in place), one void arm is refused, `(void)e` is `e` for its
    effects, a void function returns a void value as `return;`, and the twin's R12 counts a call as an
    effect.
  - Outcomes: `cfront_condeval.c` runs function by function on both emits with every guard false as well
    as true -- a zero divisor, `INT32_MIN / -1`, a null pointer, a bounds guard at the array's end, a
    device register at an address nothing maps -- counting the calls each makes; a unit of typed branch
    values round-trips as a fixed point. Five corpus fixtures with a guarded arm and two with
    `(void)b;` move their digests in lockstep on the four targets; the `(void)cb;` of
    `cfront_sec_sigoverflow.c` moves the oracle's, and the twin, whose driver reports that fixture's emit
    over its 32 KiB capacity before and after this change, prints no digest to compare.
  CF-NULLARG typed a null pointer constant passed to a pointer parameter as that parameter's pointer.
  - The defect: `g(0, s)` for a callee taking a pointer -- to a scalar, to `const`, to `void`, to a pointer
    or to a struct, a function pointer, an array parameter `T a[]`, `T a[N]` or `T m[][N]`, a VLA -- lowered
    to one claim graph on both rails, and each emit passed an `int` temp where the callee takes a pointer
    (`int t = 0u; bcir_g(t, s);`), which Clang and GCC reject. Neither rail carried a callee's parameter
    types to its calls, and the twin, which parses in one pass, cannot see a callee defined after its
    caller. The oracle also declared a prototype's array parameter by its element type, which conflicted
    with the prototype.
  - RED: on the base below this change both rails pass each fixture's `0` arguments as `int` temps, and both
    fixtures' tests fail; on the parent the fixtures do not lower at all, since they hold self-referential
    structs (CF-SELFREF).
  - What landed: an argument converts to its parameter's type as if by assignment (C11 6.5.2.2p7). The
    oracle reads every function's and prototype's parameter types before it lowers (`_param_types`) and
    types each direct or prototyped call's constant-0 arguments by them (`_null_pointer_args`); the twin
    does the same once the unit is parsed (`null_pointer_args`), from the callee's definition or its
    prototype. A variadic callee's extra `0` keeps the `int` it is read as. Only temps' C types move, so
    no digest does.
  - Outcomes: `cfront_nullarg.c`, and `cfront_nullarg_link.c` (callees defined late, only prototyped, or
    variadic), run function by function on both emits, built with `-Werror=int-conversion`. The G10 escape
    gate caught the fixtures' first cut: three local arrays lent to calls (`escape.unproved` 3) and three
    calls through a function-pointer parameter of a callee another unit may call (`icall.unknown` 18, over
    its open-world floor of 15). The arrays are at file scope, the two callees that call through a pointer
    are `static` -- so both calls narrow to the one function the unit passes -- and `nl_later_node`, the
    external callee defined late, tests its pointer rather than calling it.
  The five new tests of these two slices fail on the base below them, and 31 injected defects, on both rails,
  are each caught (`tools/testing/faults/cfront-operands.json`). Of the 34 tests this whole change set adds or
  extends, 33 fail on the parent; the 34th, an existing refusal test it extends, already held there.
  Found beside them, and each closed by a later slice below: the round trip's continue labels and `memcpy`
  (CF-RTWIDE); the undeclared calls, the functions no emit declared, the prototypes' `const` and unnamed
  parameters, and the anonymous structs (CF-DECLS, CF-ANON); the integer constants, file-scope arrays, nested
  braces and thread storage (CF-INTCONST, CF-GARRAY, CF-GBRACE, CF-TLS); and the void callbacks, the
  function-designator arms of `?:`, the null pointer constants and the struct arithmetic (CF-VOIDCB, CF-FNSEL,
  CF-NULLCALL, CF-STRUCTARITH).

  The ring drain race, the G2 residual, transport replay and measured floors (2026-09-30) closed the
  GEM+ items that needed no hardware, and reconciled the roadmap with the harness.
  - The ring drain race. AArch64 CI failed once in `test_ring.c --procs`, reporting
    `unaccounted=1`: a consumer read `stop` only after an EMPTY verdict. Descheduled between the two
    reads, it let the last producer publish and exit and the supervisor set `stop`, and then quit
    with records still in the ring (`stop` was also a plain volatile read). The consumer now loads
    `stop` with acquire before it consumes and finishes on EMPTY only if `stop` was already set. The
    supervisor stores it with release after every producer has exited.
    - The race needed a preemption, so a test could not count on it. A `--procs` run takes a drain
      window: the producer finishing the stream pauses halfway, and the consumer sleeps through the
      end of the stream on its first EMPTY after that. A window run must report its window and at
      least one nap, or it is itself a violation.
    - With the window, the late read fails 6 runs of 6 with CI's signature, and the fix passes 12 of
      12. `ring.json` gains three faults (25 of 25 caught).
  - The G2 residual. `optimize_scheduled`'s dispatch read every RAW/WAR/WAW conflict of the §6.3
    chain: n(n − 1)/2 = 130,816 edges, walked on every trial placement.
    `concurrency.hazard_frontier` keeps each resource's last writer and the readers since that write.
    That is the same closure in O(touches) edges, and every placement is the full DAG's, slot for
    slot, because a dropped predecessor is an ancestor of a kept one.
    - `optimize_scheduled.edges.512` 130,816 → 511, its proved floor. The slowdown row went
      10.9× → 2.8× A/B on one host, under the 4× bound.
    - The token plan and the exact solver keep the full DAG. `frontier.json`: 6 of 6.
  - Transport replay (the 2026-09-04 review's finding 16, now closed). `DurableLog` replays values,
    ordered by static claim ids. `bcir/telemetry_replay.py`'s `EnvelopeLog` keeps what the transport
    delivered instead: every envelope's bytes in arrival order and every live-generation change,
    sealed by a trailer holding the intake's report, the ring's overwrite count and a SHA-256.
    `replay_envelope_log` re-decides through a fresh intake and refuses any log whose report differs.
    - A ring capture that overwrote records replays with its gaps.
    - 18 tamper cases are each refused by their own law, and the C twin's intake decides a replayed
      log identically. `replay.json`: 7 of 7.
    - Two first-cut defects were caught by the sweep: two faults were masked by other laws until each
      tamper case isolated one law (a resealed log, and a forged record only the digest can see).
  - Measured floors. Thirteen harness rows carried no lower bound, so no optimality statement was
    available for them. Each now measures a floor in the same run as its value (`Metric.floor_key`,
    §0.3 of the roadmap), and a value past its floor blocks the table:
    - the digest's own FNV-1a chain over the pre-rendered stream, which returns the same digest:
      40.4 of 50.7 ms;
    - the emission floor of the call-count rows: the call plus one constructor per fresh record.
      The one-claim delta's floor is 13 calls of its 472, the same at every scale;
    - writing the output's bytes once, for the byte-producing ratios and the native planner:
      24 µs of 3.73 ms, through `test_kplan --bench-floor`;
    - each audit case's own fixture, which it builds inside its timed interval. The refactor that
      exposed the fixtures left the audit's definition and correctness digests unchanged;
    - re-deriving each step's cost once through R9's predicate, which must sum to the plan's score.

    `test_gemplus_floors.py` holds each floor to the work it stands for, and `floors.json`
    catches 20 of 20 injected defects.
  - The roadmap caught up with the harness. §0.3 had listed five unbounded rows while the harness
    listed thirteen. It now lists the measured-floor rows, and a test pins that list to the harness.
    §8 had stopped at Stage 3 and called the `verify.*` rows unmeasured; it now carries every stage's
    rows. The earlier plan to bound the audit rows with G4's lower-bound stack was wrong, and §0.3
    says why: that stack bounds a schedule's makespan, not the time to compute it.

  CF-BUF (2026-09-30) lifted the C twin's fixed name and emit buffers, two items the list above had recorded --
  the claim op of 31 characters and the emit of 32 KiB -- with the truncations an audit of every fixed field found
  beside them. RED was measured on the parent (`916d3e3a`).
  - The defect: the twin kept every name in 32 bytes -- a function, a parameter, a local, a global, a struct tag, a
    member, an alias -- and a claim's op in the same 32, and cut what did not fit, silently. A callee past 24
    characters became another `c.call:` op (19 in `c.call.void:`, 21 in `c.call.tu:`), so the digest differed from
    the oracle's and the emit called the cut name: one nothing defines, or another function. A floating constant's
    spelling rides in its op, so one past 22 characters lost its tail, digest apart and with another value:
    `1.0000000000000000000000000e-300` was emitted as `1.00000000000000000000`. `va_arg`'s type was cut to 16
    characters (`unsigned long long` emitted as `unsigned long lo`, which does not compile). A `struct <tag>`
    spelling, a pointer cast's op, a shadowing local's emitted name (`<name>_2`) and the emitter's type spellings
    were cut the same way; GCC's format-truncation analysis finds 32 such writes in the parent. The emitted C was a
    fixed 32 KiB, so corpus fixtures were trimmed to fit it, and each synthesized `typedef RET (*__bcir_fpN)(...)`
    line was rendered into 512 bytes, past which the whole emit was marked impossible: `cfront_sec_sigoverflow.c`,
    whose emit is 1.8 KB, printed `EMIT-ERR`, so its digest was never compared with the oracle's.
  - RED: on the parent the twin refuses `cfront_longnames.c` (`undefined identifier`) and cuts the variadic unit's
    `va_arg` type. The oracle lowers each of the twelve 64-character units; the twin lowers five of them with the
    name or the constant cut (a function, a label, a callee, both constants) and refuses the other seven for another
    reason (`undefined identifier`, `unknown field`). A unit of 140 functions and `cfront_sec_sigoverflow.c` print
    `EMIT-ERR`; `bcir-cc --emit-c` refuses a 700-statement function (`emitted C exceeds 32768-byte result
    capacity`); GCC's analysis reports the 32 writes. The six new tests and the three this change extends (the
    capacity-edge test, and the corpus parity and GCC/Clang groups that now hold the fixture) fail. The extended
    allocation-failure sweep does not build against the parent's result type; a variant written for that type fails
    at the sweep's fault-free run, whose emit does not fit.
  - What landed: a limit on both rails, and fields sized for it on the twin. An identifier or a floating constant is
    at most 63 characters -- C11 5.2.4.1's significant initial characters of an internal identifier -- and a longer
    one is refused where it is lexed, with one reason per kind (`clex._too_long`; the twin's `lex`). The twin's name
    holds 63 and a NUL (`BCIR_CIR_NAME`), a `struct <tag>` spelling 71 (`BCIR_CIR_AGG`), an op a prefix and a name,
    a floating constant or a pointer cast's type (`BCIR_CIR_OP`), and the emitter's scratch a suffixed name, a type
    and an expression of several (`BCIR_EMIT_*`); every write into a field goes through `fits` or `idcpy`, which
    fail the compile where they would cut. `va_arg`'s type is its tokens, one space apart, or, when they do not fit
    the op, the type's own spelling. The emitted C (`bcir_cfront_result.emitted`, with `emitted_len`) is owned by the
    result and grows through its allocator: each piece is measured, the block grown two-phase, then the piece
    rendered, and a function rendered into too little room is rendered again once the block holds it.
    `bcir_cfront_free` releases it; an allocation failure fails the compile (`oom`), so the text is whole or absent.
    The two preludes grow the same way (`ctext_putf`), and `bcir-cc --linkable` sizes its rename scratch from the
    emit. The claim op is also the data-plane hand-off's segment label, and the freeze refuses a label that does
    not end inside the op (G16), so the oracle's bound (`LABEL_MAX` in `bcir/gem/handoff.py`) follows the op from
    31 to 127 characters; the law graphs sit at 127 and 128 characters, and
    `docs/kernel/BCIR_DATA_PLANE_HANDOFF.md` states the new bound.
  - Outcomes: `cfront_longnames.c` -- a 63-character name in each place the graph keeps one, and a 63-character
    constant -- and a unit with a 63-character variadic callee lower digest-equal on the four targets and run
    equivalent function by function under Clang and GCC. At 64 characters, ten name positions and both constant
    forms are refused on both rails for the same reason; at 63 they lower digest-equal. The 140-function unit (103 KB
    of emitted C on the twin, 190 KB on the oracle) and `cfront_sec_sigoverflow.c` lower to the oracle's digest on
    the four targets and run as the original. No other fixture's summary, emit or canon moves on the twin, on the
    four targets; the G10 and volatile rows are unchanged. GCC's analysis finds no write that could cut. The
    allocation-failure sweep fails each allocation of a unit whose emit grows four times, and each failure is the
    compile's, with every block released. The sanitizer script now requires `cfront_sec_envrealloc.c` and
    `cfront_sec_sigoverflow.c` to compile clean, as its comment said. The G16 rows stay zero on the oracle, the C
    twin and the C++ seam.
  14 injected defects, on both rails, are each caught (`tools/testing/faults/cfront-buf.json`), and a hand-off
  oracle that keeps the old 31-character label bound is caught by the G16 rows (`tools/testing/faults/handoff.json`).
  Three faults of the earlier tables anchored into lines this change rewrote; they were re-anchored, and each is
  still caught. The three items it found but did not fix -- a macro name past 63 characters, the canon cut at
  its caller's buffer, the driver's 64 KiB source -- are closed by CF-LIMITS (below).

  CF-SPLIT2 (2026-09-30) closed the rail splits the CF-NULLARG list recorded and those found beside them: forms the
  cfront rails lowered to two claim graphs, or that one rail lowered and the other refused. RED was measured on the
  parent (`59740254`).
  - The defect: a written-out `p = p + n` of a pointer was a `c.ptradd` on the oracle, which folded any `x = x ± e`
    into the compound's in-place step, and a sum and a copy on the twin, as `q = p + n` and `p = n + p` were on both
    rails. The twin refused an increment, and an assignment used as a value, through a pointer the lvalue itself
    loads (`h.next->v++`, `x = (h.next->next->v ^= s)`, `s->p[i]++`), through a pointer variable (`p[i]++`,
    `x = (p[i] = v)`) and through a dereference (`(*p)++`, `++*p`); it refused the address of such an object
    (`&h.next->v`) and a call through a parenthesized callee (`(fp)(x)`, `(o.fn)(x)`), all of which the oracle
    lowered. The oracle refused a dereference of a pointer value other than a name, a member or a call (`*&a`,
    `*(c ? &a : &b)`, `*p++`, `*(q - 1)`), which the twin lowered, and both rails took `++h` of a statement
    `++h.next->v;` and failed on the `.`. The twin also lowered `*q->a` of a member array as the array's address and
    an access through it, where the oracle accesses the first element at the member's offset.
  - RED: `cfront_splits.c` splits in each of its ten functions on the parent. The twin refuses nine, the oracle four
    of those nine, and `sp_ptr_sum` lowers on both rails to different claim graphs (36 claims against 39). Its test
    fails at the oracle's parse.
  - What landed on the oracle: only the parser's desugaring of a compound assignment, which reuses the target node
    as the left operand, steps a pointer in place. The statement shortcut takes `++name` only for a bare name, so
    `++h.next->v;`, `++p[i];` and `++*p;` parse as expressions. A dereference of any other pointer value is an access
    at offset 0, as a cast's already was.
  - What landed on the twin:
    - One resolver handles an object reached through a pointer the lvalue loads (`plv_chain`). The object is read,
      written, stepped, assigned and addressed as the oracle's `_lvalue` is. A device object is refused, as the
      oracle's `_mmio` gate refuses it: `plv_volatile` reads the loaded pointer's domain.
    - An element of a pointer variable subscripted once takes the scalar array paths (`ptr_elem_lv`).
    - `deref_incdec` steps `*p`, `*(p + i)` (as `p[i]`), `*q->a` and any other pointer value.
    - `*q->a` of a one-dimensional member array of scalars is its first element at every site
      (`deref_member_array`).
    - The parenthesis pass drops a callee's parentheses before its call. It keeps a subscripted callee's, which
      the oracle refuses.
    - The statement shortcut `++name` takes only a bare name.
  - Outcomes: the fixture lowers to one claim graph on the four targets. Each emit runs as the original does under
    Clang and GCC, and the globals it touches are compared after every call. A device object stays refused on both
    rails (`test_split_forms_on_a_device_object_are_refused_on_both_rails`, a guard that already held on the parent).
  - Stack (CF-SPLIT2.1): the twin's new lvalue paths first held their locals in the recursive descent's own frames.
    The address sanitizer gives every address-taken local its own stack slot, so under GCC's ASan a level of
    `p_unary_inner` grew from 7024 to 8016 bytes and one of `p_assign` from 6624 to 7552, and
    `cfront_sec_deepnest.c`, whose nesting runs past the depth guard's cap, overflowed the stack before the guard
    refused it; `tools/c/sanitize_cfront.sh` caught it. Five functions marked `BCIR_NOINLINE` now hold those
    locals out of the descent, and the two frames are 6544 and 2832 bytes, smaller than on the parent. Three
    fault anchors moved with the code (T6 and T11 of `cfront-splits.json`, T41 of `cfront-roundtrip.json`), and
    T7 of `cfront-gaps.json` spans two lines: `plv_incdec` had repeated its one-line anchor.
  17 injected defects, 4 on the oracle and 13 on the twin, are each caught (`tools/testing/faults/cfront-splits.json`).
  Found, not fixed here (each a suggested follow-up; both are closed by CF-FPTAB, below):
  - both rails refuse a call through a dereferenced function pointer, `(*fp)(x)`: the oracle's parser takes a call
    only after a name or a member, and the twin's `*` of a function pointer is "dereference of a non-pointer";
  - both rails refuse a call through an element of an array of function pointers (`(ops[i])(x)`), and the twin
    refuses the declaration `int32_t (*t[2])(int32_t);` ("expected declarator name").

  CF-RTWIDE (2026-09-30) widened the cfront emit -> re-parse round trip: 117 of the corpus's 167 fixtures reach its
  fixed point, where 65 of 166 did. RED was measured on the parent (`5c49f334`).
  - The defect: four idioms of the emit were not fixed points of their own re-lowering, and the gate excluded every
    fixture that held one.
    - The emit spells each plain memory access as a `memcpy`. Both rails refused a call of `memcpy`, `memmove` or
      `memset` as a call to an undefined function (R18), so no emit with a memory access re-lowered.
    - A structured loop emits as `while (1) { ...; if (!cond) break; ... }`, which a re-parse reads as a loop whose
      condition is the constant 1 and whose test is a branch of its body. The emitter tested that constant again,
      and the test re-lowered as one more branch each round. Its continue label `__cont_<id>` also collided with a
      label the function defines: a re-parsed emit keeps the labels the emit before it placed, and a source label
      may spell one. Such an emit defines one label twice and does not compile -- on the twin too, whose labels
      are `__cont_<n>`.
    - An aggregate local's zero baseline emitted as `= {0}`, which a re-parse reads as the baseline and a store of 0
      to the first scalar. The next emit spelled that store out, one more each round.
    - The classifier re-parsed an emit without the original's struct, union and enum definitions, which the emit
      names by tag. The definition regex the volatile register maps used missed a nested member and an attribute,
      and read a comment inside a definition (`struct align 1`) as a tag.
  - Found beside them and fixed: the twin lowered a call of `malloc`, `calloc`, `realloc`, `aligned_alloc` or `free`
    as the libc edge even when the unit defined that function, and a printf-family function defined after its call
    as the external variadic; the oracle lowers both as calls of the unit's own function. The linkable emit
    included `<math.h>` for any libc edge outside malloc/calloc/realloc/free, and `<math.h>` declares neither
    `aligned_alloc` nor the string routines.
  - RED: 9 of the 11 new or changed tests fail on the parent. The two that pass test the harness alone: the
    definition reader, and the CF-RTVOL fixed point read through it.
  - What landed:
    - `memcpy`, `memmove` and `memset` lower on both rails as `<string.h>` libc edges (`c.call.libm:<name>`, a
      `void *` result, opaque to R18, no link flag), unless the unit defines the function.
    - The twin asks whether the unit defines a library name through one predicate over the whole unit
      (`unit_defines`, a cached scan of the file-scope definitions), for the allocators, `free`, the string routines
      and the printf family, as the oracle's `func_rets` holds every definition.
    - Both emitters spell a loop whose condition is the constant 1 with no test, and name each loop's continue
      label clear of every label the function defines (`__cont_<id>_<k>`).
    - Both emitters spell the zero baseline as `= {}`, the C23 empty initializer, which GCC and Clang take in every
      mode the harnesses use. `test_local_aggregate_initializers_oracle` still asserted `= {0}`, and failed on this
      tree: only the slice's focused tests had run. CF-RTWIDE.1 asserts `= {}`. The aggregate check of
      `tools/c/check_runtime.sh` looked for `= {0}` as well, and only the final serialized gates ran it;
      CF-RTWIDE.2 looks for `= {}`.
    - The linkable emit includes `<string.h>` for a string routine and `<stdlib.h>` for `aligned_alloc`.
    - The classifier supplies the original's definitions, read whole from the preprocessed source, and the
      control-flow-not-idempotent exclusion is retired.
  - Outcomes: 52 fixtures joined the round trip, the new `cfront_strmem.c` among them. The 50 still excluded are 42
    with a masked-access guard and 8 whose emit names what only the original declares, or holds a form the re-parse
    refuses. `cfront_strmem.c` lowers to one claim graph on the four targets, and each emit runs as the original
    does. 15 injected defects -- 8 on the oracle and the classifier, 7 on the twin -- are each caught
    (`tools/testing/faults/cfront-roundtrip.json`, now 22 faults).
  Found, not fixed here (each a suggested follow-up):
  - the re-parse refuses three forms the emit writes: a cast to a function-pointer type (`cfront_fnptrmember.c`,
    `cfront_signedfnptr.c`), a `_Complex` type in a cast (`cfront_complexalign.c`), and `_BitInt` arithmetic beside
    a bit-field's standard type (`cfront_bitint_bitfield.c`) (closed by CF-RTFP, below);
  - the oracle refuses a unit that declares a `<stdint.h>` or `<stddef.h>` name itself (`typedef unsigned long
    size_t;`, common in freestanding code), which the twin accepts (closed by CF-RTFP, below);
  - the twin refuses a braced initializer of a function-pointer local (`uint32_t (*fp)(uint32_t) = {g1};`), which
    the oracle lowers (closed by CF-RTFP, below).

  CF-DECLS and CF-ANON (2026-09-30) made the emits declare functions and name types as C does, on both rails.
  RED was measured on the parent (`a8fa97fd`).
  - The defects (CF-DECLS):
    - Neither emit declared a function before its definition, so a call to a callee defined after its caller did
      not compile as emitted, even with a prototype before the call.
    - Both rails accepted a call made before any declaration of its callee, which C99 does not (C11 6.5.1p2), and
      lowered it as if prototyped. The twin typed the result by its `uint32_t` default (`uint32_t t = bcir_g(s);`
      for a `uint64_t g`) under the same claim graph as the oracle, so no digest showed the truncation. A
      prototype after the call split the rails: the oracle typed the call by it, the twin lowered an undefined
      callee.
    - A prototyped callee's `extern` declaration dropped `const` from a pointer or array parameter on both rails,
      which conflicts with the original prototype in one translation unit, and the oracle spelled a
      function-pointer parameter by its name (`extern uint32_t g(fn, uint32_t);`). Both rails refused a prototype
      that leaves its parameters unnamed.
    - Found beside them: the oracle lowered a designator and a `sizeof` operand naming a function no declaration
      precedes, which the twin refused; the twin refused a `sizeof` of a call through a prototype and a designator
      of a function its prototype declares before its definition, which the oracle lowered.
  - The defects (CF-ANON): both emits spelled a struct or union declared without a tag and named by a typedef as a
    tag no compiler knows -- the oracle `struct  a` (the empty tag), the twin `struct $anon0 a` -- so no function
    using one compiled; the digests agreed, so no gate noticed. The oracle registered each such aggregate under the
    empty tag, so a unit with two of them refused the first's members ("no member named 'x' in struct ''"), where
    the twin lowered it.
  - RED: on the parent the oracle refuses `cfront_decls.c` and `cfront_decls_link.c` (an unnamed parameter) and
    `cfront_anonstruct.c`; the twin's emit of `cfront_nullarg_link.c`, with the hand-supplied declarations gone,
    does not build; the oracle lowers the undeclared calls; the twin refuses a `sizeof` through a prototype. The
    six new or changed tests fail there.
  - What landed, on both rails:
    - One notion of a function declared here. The oracle's parser records the functions each definition can see
      (`Func.declared`), and `_require_declared` refuses a call, a designator or a `sizeof` operand naming a unit
      function outside them (`call to undeclared function 'g'`, `use of undeclared identifier 'g'`). The twin
      records each call no earlier definition, prototype or the function itself declares (`undecl`), and refuses
      it once the unit is parsed if the unit defines or prototypes the callee after it; `declared_ret` types a
      `sizeof` of a call and a designator by an earlier prototype. A callee the unit never declares stays R18's
      undefined edge.
    - Each emit declares the unit's functions a function calls ahead of it, with the definition's own signature
      (`emit_function(lf, unit)` and `_signature`; `emit_callee_decls` and `emit_sig`).
    - A prototype's `extern` declaration keeps a pointer or array parameter's `const` (`proto_consts`;
      `bcir_ctype.is_const`) and spells a function-pointer parameter `RET (*)(PARAMS)` (the twin keeps its
      `__bcir_fpN` alias). A prototype may leave a parameter unnamed; a definition that does is refused.
    - An anonymous aggregate keeps a synthesized tag inside each rail, and the emit spells it as C names it -- the
      typedef's name, `__typeof__(*(PP)0)` when only a pointer typedef names it, or `__typeof__(((P *)0)->m[0])`
      for the type of a nested member `m` -- by one pass over the finished emit that leaves literals and comments
      alone (`respell_anon` on both rails; the chain is walked without recursion).
    - The twin's diagnostic for an identifier nothing declares is the oracle's (`fail_undeclared`, out of the
      recursive descent's frames).
  - Outcomes: `cfront_decls.c` and `cfront_anonstruct.c` join the corpus; they and `cfront_decls_link.c` lower to
    one claim graph on the four targets, and every emit runs as the original with no declaration supplied by hand.
    The fault `B12` of `cfront-buf.json` anchored on a line `ctext_putn` repeats; it takes one more line of context.
  32 injected defects, 16 on each rail, are each caught (`tools/testing/faults/cfront-decls.json`).
  Found, not fixed here (each a suggested follow-up):
  - a qualifier between the stars of a prototype's pointer parameter (`const char *const *argv`) and a qualifier in
    a function-pointer parameter's own parameters are dropped from the `extern` declaration on both rails, which
    then conflicts with the original (closed by CF-QUALS, below);
  - a function designator of an external function, declared by its prototype and defined by another unit, is
    refused on both rails (`use of undeclared identifier`), which C allows (closed by CF-EXTDESIG, below).

  File-scope constants, arrays and initializers (2026-09-30), on both rails, each slice closing an item of the
  CF-NULLARG list above. RED was measured on the parent (`59740254`), with the new fixtures and tests copied in.
  CF-INTCONST made an integer constant its exact value in its C11 6.4.4.1 type, and folded a file-scope
  initializer in C's own types.
  - The defect: the twin read a constant with `strtoll`, which saturates, so one past LLONG_MAX was LLONG_MAX
    (`0xFFFFFFFFFFFFFFFFu` emitted as `9223372036854775807u`), and read an octal constant in base 10 (`017` was
    17, and 10 as an array dimension or a `case` label written `010`); it typed a constant from a 63-character
    copy of its spelling, which cuts 64 binary digits. The oracle folded a file-scope initializer, for its
    linkable emit, with unbounded integers: `-7 / 2` rendered -4, `~0u` -1 and `-0x80000000` -2147483648, and
    `7 % -2` raised a bare `ValueError` (the fold computed every operator's result, a shift by -2 among them).
    Neither rail refused a constant no type can hold (the oracle lowered `18446744073709551616u`), and the oracle
    read `0o17` through Python's `int`, as 15.
  - RED: on the parent the fixture's functions alone digest apart on the four targets; the whole fixture raises
    `ValueError: negative shift count` in the oracle and is refused by the twin (`non-constant enum
    initializer`: it read a global's initializer with the enum evaluator, which takes no cast and no `sizeof`).
    Both new tests fail.
  - What landed: one reading of a constant per rail (`clex.int_literal_parts`, the twin's `int_literal`): the
    separators dropped, the suffix split off, the base from the prefix, each digit checked against it, the value
    exact. The twin keeps a constant past LLONG_MAX as its 64 bits, and its canon spells a constant of an
    unsigned type unsigned (`vn_imm`), as the oracle's does. A constant past `unsigned long long`, or decimal
    without `u` past `long long` (GCC types that one `__int128`, Clang `unsigned long long`), is refused on both
    rails (`an integer constant too large for every type its base and suffix allow`), a malformed one for its
    digits. The oracle's `_fold_const` is gone: a global's initializer is lowered on a lowerer whose scope is the
    globals declared so far and folded by the fold a static's initializer takes (`_const_value`, CF-STATICTAB),
    then spelled exactly (`_const_spelling`: `Nu` past LLONG_MAX, LLONG_MIN as an expression). The twin skips a
    global's initializer instead of reading it with the enum evaluator, and refuses a call in it, as the oracle
    does.
  - Outcomes: `cfront_intconst.c` -- every base and suffix at the edges of its type, each constant's type read
    through a comparison with -1 and through `sizeof`, octal and binary `case` labels and array dimensions --
    lowers to one claim graph on the four targets and runs as the original on both emits under Clang and GCC, and
    the oracle's linkable emit defines each of its globals with the value the original's holds. No other
    fixture's digest moves on the four targets.
  CF-GARRAY passed file-scope arrays and character tables through the rails as local ones.
  - The defect: a multi-dimensional global passed for a `T (*p)[N]` or `T m[][N]` parameter was passed by name in
    both emits, where its type is still `T[A][N]` and the emitted parameter the flat `T *`, so Clang and GCC
    refused the call (`-Werror=incompatible-pointer-types`, as the harness builds); a local one works, being
    declared flat. The twin refused `char buf[] = "bcir";` at file scope (`non-constant enum initializer`), and
    bounded each access to an unsized global array by one element (`T g[] = {...}` set its dimension but not its
    count), so an in-bounds `g[2]` reached the quarantine. Both rails refused a declaration's second declarator --
    `uint32_t a[3], b[2];`, `uint32_t gx, gy;`, `struct pt ga, gb;` -- and a declaration defining the struct of the
    objects it declares (`static struct t { ... } a, b;`), each rail for its own reason.
  - RED: on the parent the fixture is refused by both rails -- the oracle at its first list, the twin at its first
    character table -- and the test fails; a unit with only the row-pointer call builds on neither rail's emit.
  - What landed: a call's argument that is a file-scope multi-dimensional array is its first element's address,
    `&m[0][0]`, in both emits (`emit._args`; the twin's `emit_arg` over a new `bcir_resource.ndims`) -- the emit
    only, so no digest moves. Each declarator of a file-scope declaration is a global of its own type
    (`cparse._globals`; the twin's `p_global` over `p_global_declarator`), and a declaration may define its
    struct first (`_aggregate_definition`; the twin's `try_top_decl`), an untagged one refused on both rails. The
    twin sizes a character array by its literal, as the oracle does, and counts an unsized global's elements.
  - Outcomes: `cfront_garray.c` -- a 2-D and a 3-D global passed to row-pointer parameters, character tables
    sized by their literals (concatenated, with escapes, and in a larger array), and lists of arrays, scalars, a
    pointer, initialized objects, `static` objects, structs and the objects of a struct the declaration defines --
    lowers to one claim graph on the four targets, and each function of each emit returns, and leaves the
    globals, as the original does.
  CF-GBRACE walked a file-scope initializer as C walks the current object.
  - The defect: the oracle parsed a global's initializer as a flat list of expressions, so it refused a nested
    brace (`struct pt g[2] = {{1u, 2u}, {3u, 4u}};`: `unexpected PUNCT '{'`), which the twin lowered, and the twin
    read a designator chain at the top of the list (`[5].y = 9u`) as a parse error. Both rails sized an unsized
    global by its top-level entries, not by the walk: `struct pt g[] = {1u, 2u, 3u, 4u, 5u};` was five elements on
    both, so `sizeof g / sizeof g[0]` returned 5 in both emits where the original returns 3, and of
    `uint32_t m[][3] = {1u, 2u, 3u, 4u, 5u, 6u};` the oracle counted six rows and the twin none (`sizeof of an
    incomplete type`). Neither rail refused an initializer C refuses: `uint32_t g[2] = {1u, 2u, 3u};` lowered on
    both.
  - RED: on the parent the fixture is refused by both rails, the oracle at its first nested brace and the twin at
    `[5].y =`, and both new tests fail; the oracle lowers `uint32_t g[2] = {1u, 2u, 3u};`.
  - What landed: a global's initializer is a local's -- an expression, or a brace list whose entries are
    expressions or lists, positional or designated (`cparse._global` over `_init_value`) -- walked for its shape:
    the walk a local's or a static's initializer takes, in a mode that neither lowers nor folds an entry and
    records only each store's range (`_InitWalk.shape`, `_file_scope_shape`; the twin's `global_init_shape`, an
    `iwalk` with `skip`). The walk's constraints hold, so an excess entry, an override of an initialized
    subobject, a string too long, a designator outside its object and two entries for a scalar are refused on both
    rails for one reason, and an unsized array takes the extent the walk reaches, a nested one its rows. An
    initialized array of more than three dimensions is refused on both, the twin's walk holding three. The twin
    scans the initializer for a call before the walk, as the oracle's parser refuses one before lowering; the scan
    CF-INTCONST gave it now also finds a call through a parenthesized callee (`(k)()`). The oracle's linkable
    emit renders the initializer as the source spells it, braces and designators kept, each entry folded or
    spelled as before and a string literal that initializes a character array by its own spelling
    (`_file_scope_rendering`, `_spell_init`), and declares a nested global with every dimension
    (`emit._object_declarator`).
  - Outcomes: `cfront_gbrace.c` -- arrays of structs, rows, a 3-D array, character tables, a nested struct, a
    union and a braced scalar at file scope, with nested braces, brace elision and designators -- lowers to one
    claim graph on the four targets, each function of each emit returns as the original does, and the oracle's
    linkable emit, given the unit's struct definitions, holds each global's bytes as the original does. No other
    fixture's digest moves on the four targets.
  CF-TLS kept `_Thread_local` on a static local.
  - The defect: both rails read `_Thread_local` as a storage class and dropped it, so `static _Thread_local
    uint32_t n = 3u;` was emitted as `static uint32_t n = 3u;`, one object every thread shares, and digested as the
    plain static. The twin read a declaration that begins with it (`_Thread_local static uint32_t t[3];`) as an
    expression (`undefined identifier`), and the oracle lowered a block-scope `_Thread_local` object that is not
    `static`, which C11 6.7.1p3 forbids, as an automatic local. The oracle's linkable emit dropped it from a
    global.
  - RED: on the parent both rails emit the fixture's first thread-local static as `static uint32_t n = 3u;`, at the
    plain static's digest.
  - What landed: the storage duration rides on the static (`Decl.thread_storage`, `LoweredFunc.thread_statics`;
    the twin's `bcir_static.thread_storage`, from `saw_thread`), both emits declare it `static _Thread_local`, and
    the canon keeps its line whatever its image (`static _Thread_local n = <image, or 0>`), so a static differing
    only in its storage duration digests apart. A declaration may begin with `_Thread_local` on the twin, and a
    block-scope one without `static` is refused on both rails. The oracle's linkable emit declares a thread-local
    global `_Thread_local` (`LoweredUnit.thread_globals`).
  - Outcomes: the fixture's thread-local global and its four thread-local statics -- a scalar, an array, a struct
    and a zero one, the storage class before `static`, after it and after the type -- run as the original on both
    emits, and again in a second thread, where each starts at its initial value; the linkable emit's thread-local
    global starts there too.
  The new tests of these four slices fail on the parent, and 37 injected defects, on both rails, are each caught
  (`tools/testing/faults/cfront-globals.json`). One fault of an earlier table, ST3 of `cfront-statics.json`,
  anchored into a line CF-TLS rewrote; it was re-anchored, and is still caught. The G10 and volatile rows are
  unchanged.
  Found, not fixed here (each a suggested follow-up; one more, an enumerator and a `case` label folding `/`, `%`
  and a comparison apart from C, is closed by CF-ENUMFOLD, below):
  - the twin types a plain `char` element of an array `int8_t` in its emit, where the oracle and the source say
    `char`: on a target whose `char` is unsigned (AArch64 Linux), an element past 0x7F reads back negative (closed
    by CF-CHARELEM, below);
  - a member of an element of a 2-D array of structs (`gm[i][j].x`) is refused on both rails (`a subscript of an
    element that is not a pointer`; closed by CF-AOS2D, below);
  - the oracle's linkable emit names the unit's structs and unions without defining them, and drops `const` from
    a global (`const uint32_t g = 5u;` renders `uint32_t g = 5;`); it still refuses a pointer table's string
    (`const char *tab[] = {"a"};`) by name (closed by CF-LINKEMIT, below);
  - a parenthesized string literal initializing a character array (`char s[] = ("abc");`, at file scope, at
    block scope or `static`) lowers on the oracle, whose parser drops the parentheses, and is refused by the twin
    (`an array is initialized by a brace list or a string literal`; closed by CF-PARENSTR, below);
  - both rails accept a malformed integer suffix (`1lL`), and the twin names a designator's unknown member
    without its name (`no member of that name to designate`, the oracle's `no member named 'z' to designate`;
    closed by CF-SUFFIX, below).

  CF-VOIDCB, CF-FNSEL, CF-NULLCALL and CF-STRUCTARITH (2026-09-30) typed what goes through a function pointer --
  its call, the arms of `?:` that point to functions, the null pointer constants passed or compared -- and refused
  a struct or union where C takes a scalar, on both rails. The twin's function-pointer type now records a `void`
  return and the whole function type: `bcir_ctype.fp_sig` indexes a table of return and parameter types, filled
  wherever a declarator, a typedef, a parameter or a struct member is parsed (`sig_add`, `fp_param_list`).
  CF-VOIDCB made a call through a pointer to a void function a call with no result.
  - The defect: a call through a function-pointer local, parameter or global, or a struct member by `.`, `->` or
    a loaded pointer chain, whose function returns `void`, lowered on both rails as a claim writing a `uint32_t`
    temp, and each emit declared `uint32_t t = cb();`, which does not compile. The twin read a zero return width
    as both `void` and a return it had not captured.
  - RED: on the parent both rails refuse `cfront_voidcallback.c` ("one arm of `?:` is void and the other is
    not"), since `c ? cb() : (void)0` had a non-void arm; a unit of `cb(); o.step();` lowers digest-equal on both,
    and both emits declare the call's result.
  - What landed: the call writes no result, and its value is the void value a direct void call has
    (`_VOID_RID`, the twin's `void_temp`), so a void conditional and `return cb();` in a void function take it;
    each emit spells a bare call. The twin sets `fp_ret_void` where a return type is captured -- a declarator, a
    typedef, a parameter, a struct member and the member's `field` copy (`fp_capture_ret`). No digest moves: a
    claim's result is not in its record.
  - Outcomes: `cfront_voidcallback.c`, and a unit of structs of several callbacks, `p->ops->step` chains and a
    file-scope ops table, run function by function on both emits.
  CF-FNSEL typed the arms of `?:` that point to functions.
  - The defect: `s > 3u ? tw : th`, whose arms are function designators (C11 6.3.2.1p4) or function-pointer
    objects, lowered on both rails to a `c.select` typed as an integer (`uint32_t` on the oracle, `uint64_t` on
    the twin on a 64-bit target), and the emit's `uint32_t t = (c ? tw : th); fp = t;` does not compile. Arms of
    two function types were not refused.
  - RED: on the parent the rails lower `cfront_fnselect.c` digest-equal and neither emit compiles; both lower
    every unit of two function types.
  - What landed: a designator has its definition's function type and a function-pointer object or a select of
    them its declaration's; the select is that pointer, and a null pointer arm is typed as it (6.5.15p6). Arms of
    two types -- a return, a parameter count or a parameter's type, compared as `_Generic` compares types -- are
    refused with one reason on both rails (6.5.15p3). The twin synthesizes a typedef for a designator's type
    (`designator_sig`); `res_sig` and `sig_same` are the oracle's `_fn_type` and `_fn_key`. Only temps' C types
    move.
  - Found and fixed on the way: the change's first cut split the rails on three arm pairs its own probes found.
    The oracle had left a designator untyped when a `typeof` or a `va_list` types one of its parameters, which
    its parameter pre-scan cannot type (it now reads the lowered definition's); it had typed a function pointer
    read from a member, which the twin loads as an integer (now neither rail types one); and the twin had
    compared a `va_list` parameter as the integer of its size.
  - Outcomes: `cfront_fnselect.c` returns its selects to the driver, which compares them with the original's and
    calls them; a unit calling through selects in place -- designators of functions with `typeof` and `va_list`
    parameters among them -- runs on both emits.
  CF-NULLCALL typed the null pointer constants CF-NULLPTR and CF-NULLARG had left an `int`.
  - The defect: the constant 0 compared with a pointer by `==` or `!=` (C11 6.5.9p5), passed through a function
    pointer or a function-pointer member to a pointer parameter (6.5.2.2p7), given to `free` or as `realloc`'s
    pointer, or the arm of a select beside a pointer, was an `int` temp in both emits: a pointer compared with an
    integer, which Clang and GCC only warn about, or an integer passed for a pointer, which they reject.
  - RED: on the parent the twin refuses `cfront_nullconst.c` (`unknown field`, the `p -= 1` below); without
    `nc_step` the rails agree, and neither emit builds under Clang or GCC with the pointer/integer mixes made
    errors.
  - What landed: each such constant is typed as the pointer it converts to -- the oracle's `_null_pointer` at
    each site; the twin's `null_compared`, `null_pointer_sig` over the pointer's recorded parameters,
    `null_arms_as`, and the `free` / `realloc` branch of `p_call`. Only temps' C types move.
  - Outcomes: `cfront_nullconst.c` runs function by function on both emits, built under Clang with
    `-Werror=int-conversion -Werror=pointer-integer-compare` and under GCC with `-Werror`.
  CF-STRUCTARITH refused a struct or union where C takes a scalar.
  - The defect: both rails lowered `a += 5`, `a++`, `a + 1`, `-a`, `a && v`, `uint32_t k = a;`, `(uint32_t)a`,
    `return a;` from a function returning a scalar and `g(a)` for a scalar parameter, and Clang rejects each emit.
  - RED: the parent's oracle lowers every unit of the refusal test, and its twin every unit but `a -= 1u`, which
    it refused only by reading `-=` as `->` (below).
  - What landed: a struct or union operand of an arithmetic, bitwise, shift, relational, equality, logical or
    unary operator, a compound assignment or an increment is refused with one reason on both rails
    (`_scalar_operands`, the twin's `agg_operand`), and one converted to a scalar -- an initializer, braced or
    not, an assignment, a list element, a `return`, a cast, an argument -- with another (`_scalar_value`,
    `scalar_value_ok`). A struct copied whole, its members in arithmetic, and a pointer to structs stepped and
    compared still lower digest-equal. The twin refuses the operand of `-`, `~` and `!` once the unit is parsed
    (`agg_unary_operands`), and the new ctype fields and the member's void flag sit in padding. On the CF-BUF
    snapshot this change was first built on, the same check in the unary path made GCC stop inlining
    `p_primary` into `p_unary_inner`; with the ctype grown by the new fields, one nesting level of `(` took
    16,512 bytes of stack under GCC ASan, not 13,776, and `cfront_sec_deepnest.c` overflowed the native stack
    before the depth guard (`tools/c/sanitize_cfront.sh`). On CF-SPLIT2.1's frames the change leaves the
    recursive frames as they are: the fixture runs clean down to a 5,420 KiB stack, as it does without it.
  - Found and fixed on the way: the twin's statement parser read any two-character token after a name that
    starts with `-` as `->`, so `p -= 1` of a pointer to a struct was refused (`unknown field`), which the oracle
    lowers; `nc_step` in `cfront_nullconst.c` holds it.
  The G10 escape rows stay at the parent's values: each fixture's indirect calls go through a pointer that holds
  one function, and calls through a select, several callbacks or an ops table are the tests' own units. The new
  tests fail on the parent, and 46 injected defects, on both rails, are each caught
  (`tools/testing/faults/cfront-calls.json`).
  Found, not fixed here (each a suggested follow-up):
  - a void value used as a value -- `uint32_t k = f();`, `f() + 1u`, `!f()`, `(uint32_t)f()` of a void function,
    called directly or, now, through a pointer -- lowers digest-equal on both rails, and neither emit compiles;
    the oracle's verifier fails the unit and the twin's passes it; `if (f())` and `return f();` from a non-void
    function lower on both (closed by CF-VOIDVAL, below);
  - a struct or union as a controlling expression (`if (a)`, `while (a)`, `for (; a;)`, `a ? x : y`,
    `switch (a)`), as the operand of unary `+` (`struct s b = +a;`) or of `__real__` lowers on both rails, which
    Clang rejects (closed by CF-STRUCTCOND and CF-UNARY, below);
  - an increment of a struct in memory (`(*p)++`, `p[0]++`, `p->in++`, `h.in++`, `g_a[1]++`) is refused on both
    rails with different reasons (the oracle: "inc/dec of this lvalue form is a follow-on"; the twin: a parse
    error) (closed by CF-STRUCTCOND, below);
  - a function pointer read from a member as a value (`op_t g = o.fn;`) is a `uint32_t` temp in the oracle's
    emit and a `uint64_t` in the twin's, neither of which compiles, and as an arm of `?:` it is compared on
    neither rail, so a member arm beside a function of another type is not refused (closed by CF-FPTAB, below);
  - a call to a function returning a function pointer types its result `uint32_t` on both rails
    (`uint32_t t = bcir_pick(s);`), which Clang rejects; the oracle refuses a `typedef T *(*pf)(T *)`, and the
    twin types a call through one as an integer (closed by CF-FPRET, below);
  - both rails drop `const` from a function-pointer declarator's parameter (`uint32_t (*fn)(const uint32_t *)`
    is emitted taking `uint32_t *`), so assigning it a function that takes `const uint32_t *` does not compile
    (closed by CF-QUALS, below);
  - the twin refuses a postfix on the struct a call through a function pointer returns (`m(s).a`), which the
    oracle lowers; its reason is now the conversion refusal, since `return m(s)` is checked before `.a` (closed
    by CF-FPRET, below);
  - a variadic function's designator as an arm of `?:` is still an integer select on both rails, and both
    refuse a variadic function-pointer declarator (`uint32_t (*g)(uint32_t, ...)`) (closed by CF-FPRET, below);
  - an array compared with the constant 0 (`garr == 0`, `arr != 0`) compares it with an `int` temp in both
    emits (closed by CF-STRUCTCOND, below);
  - `void *vp = malloc(4u);` lowers to two claim graphs: the twin records an allocation extent for the
    `void *` (two claims), the oracle does not (closed by CF-STRUCTCOND, below).

  The final serialized gates, run once over the three change sets above together (2026-09-30), found two defects
  that no slice's own gates had reached. One is CF-RTWIDE.2 (above). The other: the Clang analyzer, over the
  combined twin -- each change set alone analyzes clean -- found a path on which `p_struct_body` rounds a struct's
  size to an alignment of 0 (`core.DivideZero`). The alignment is read back from the struct table after calls
  that may move it, and nothing bounded it below, where every other reader of an alignment takes at least 1. It is
  bounded there now; the twin's output for every fixture, in each of its driver's four modes, is unchanged.
  The fault sweeps of the final head then ran every cfront table and the G10 and volatile tables: 335 of 336
  injected defects were caught. The one missed, the volatile table's fault making the oracle refuse `*q[j]`, had
  become an equivalent mutation: it disabled the branch that lowers a dereference of a subscripted element, and
  CF-SPLIT2's general dereference of a pointer value now lowers `*q[j]` the same way, so no check could fire. The
  fault now injects the refusal it names, in that branch, and is caught.

  CF-LIMITS (2026-10-01) closed the three items CF-BUF's list recorded: the bound CF-BUF set where the lexers read a
  name now holds where the text enters before them, in the preprocessors and the twin's driver. RED was measured on
  the parent (`6d6ed7cc`).
  - The defect: the twin's preprocessor kept a macro name in 64 bytes and refused a longer one inconsistently -- at
    `#define` and `-D` as `macro name is too long`, at `#undef`, `#ifdef`, `#ifndef`, `#elifdef`, `#elifndef` and in
    a `#if` for its 64-byte token buffer (`preprocessor token too long`), and as a `defined` operand only past 255
    characters; it also read `#ifdef` and `#elifdef` operands in groups C skips. The oracle's took a name of any
    length, so a unit naming a 64-character macro lowered on the oracle alone. The oracle also defined a macro
    named `9x`, ignored a bare `#define` or `#undef`, raised `IndexError` on a bare `#ifdef` and evaluated a skipped
    group's `#if`. `bcir_cfront_canon` cut the canon at its caller's capacity without saying so: the twin driver's
    `--canon` held 128 KiB, and the 140-function unit's 239 291-byte canon came back as its first 131 071 bytes,
    exit 0. The same driver read the first 64 KiB of a source and lowered that prefix (a 70 067-byte unit of two
    functions lowered one), and handed a source holding a NUL on as a string that stopped there.
  - RED: the five new tests fail on the parent's rails (5 run, 5 findings).
  - What landed: each rail reads every macro name a directive reads through one predicate (the twin's `macro_name`,
    the oracle's `_macro_name`) -- the name `#define` or `-D` defines and `#undef` removes, the operand of `#ifdef`,
    `#ifndef`, `#elifdef`, `#elifndef` and `defined`, and each name an evaluated `#if` or `#elif` looks up, before
    and after expansion. Past 63 characters (C11 5.2.4.1) it is refused as `macro name is too long`, a parameter as
    `macro parameter is too long`, and a directive with no name where it reads one for the twin's reasons. A
    directive in a skipped group, or an `#elif...` after a taken one, is read only through its name (6.10.1p6) on
    both rails. `bcir_cfront_canon` and `bcir_cfront_canon_with_allocator` return the whole canon's length with
    snprintf semantics (a NULL buffer measures), and `SIZE_MAX` with an empty buffer when an allocation failed; the
    driver measures, allocates and prints all of it, and `test_memory_discipline.c` holds the whole, measured and
    cut canon under every injected allocation failure. The driver reads the whole source, growing two-phase through
    the host allocator to `bcir-cc`'s 64 MiB bound, and refuses a file it cannot hand on whole as `READ-ERR`.
  - Outcomes: the twin driver's output over the corpus (227 fixtures, four targets, `--canon`, `--emit-cpp`; 1 589
    runs) is byte-identical to the parent's, and the oracle's preprocessed output and lowering unchanged but for
    `cfront_sec_cppmacro.c`, an overlong macro parameter both preprocessors now refuse, so the escape tests' pin of it
    as the twin's limit alone (`TWIN_PREPROCESSOR_LIMITS`) is empty. The canon's FNV-1a still equals the digest.
  16 injected defects, one per defect per rail, are each caught (`tools/testing/faults/cfront-buf.json`, now 30).
  The six items it found and did not fix -- the 64 KiB loop driver and preprocessed text, the 256-byte token, a
  malformed `defined` and a NUL, `-D`/`-U` of one name, the out-of-memory digest, a name past ASCII -- are closed by
  CF-PPLIMITS (below).

  CF-ENUMFOLD (2026-10-01) folded every integer constant expression the parsers fold -- an enumerator, a `case`
  label, an array dimension, a designator -- in C's own types on both rails, closing the item the file-scope
  slices recorded. RED was measured on the parent (`66dce62d`).
  - The defect: the oracle folded with unbounded Python integers (`-7 / 2` was -4, `-7 % 2` 1, `~0u > 5` 0) and,
    computing every operator's result for each node, raised a bare `ValueError` on any negative right operand
    (`enum { N = 7 / -2 };`); the twin folded in `long long` (`~0u > 5` was 0), itself undefined on `LLONG_MIN /
    -1` and wide shifts. Both picked a value where C requires a diagnostic (`1 / 0` folded to 0, `enum { N =
    0x100000000 }` became 2^32 in an int). The twin spelled a negative enumerator as its 64-bit two's complement
    (`int32_t t = 18446744073709551613;`), the oracle as `-3u`, and the oracle a case label past LLONG_MAX without
    `u`. An enumerator as an array dimension split the rails: the twin made `uint32_t a[N]` a VLA and refused it as
    a global, member or parameter dimension, the oracle made `a[2 + 1]` a VLA. Both refused a cast in an
    enumerator, and the oracle typed `0xFFFFFFFFL` with a 64-bit `long` on every target.
  - RED: the four new tests fail on the parent (4 run, 4 findings).
  - What landed: both rails fold through one parse-time driver over the predicates a static's initializer already
    folds with (the oracle's `_kbin`/`_kun`/`_ksel`/`_kconvert`, the twin's `kbin`/`kun`/`ksel`/`kconvert`): the
    integer promotions and the usual arithmetic conversions, `/` and `%` truncating toward zero, shifts, a cast to an
    integer type, `?:` in its arms' common type, and an operand C does not evaluate typed but not folded (`0 && 1 /
    0` is 0). The oracle's parser takes the target's ABI, so a constant's type follows the target's `long`. A
    dimension that is an integer constant expression makes a fixed array on every declarator (6.7.6.2p4). Each rail
    refuses, for one shared reason, an enumerator no int holds (6.7.2.2p2), a division or remainder by zero, a
    signed overflow, a shift C leaves undefined and an operand no integer constant expression has (6.6), and a
    constant dimension outside 0..INT_MAX. Both emits spell a negative constant signed and a case label past
    LLONG_MAX with `u`. Clang folds some of what both rails refuse (`1 << 31`, `-1 << 1`, `INT_MIN / -1`, `0 && g`,
    `INT_MAX + 1` with a warning) and C23 gives a non-int enumerator a wider type; these divergences are recorded in
    `docs/languages/CFRONT_GUIDE.md`.
  - Outcomes: `runtime/c/cfront_enumfold.c` (51 enumerators, case labels of a signed, an unsigned and a 64-bit
    switch, constant dimensions of a local, a member, a typedef and a global) lowers to one claim graph on the four
    targets, each emit runs as the original, and every enumerator equals Clang's value on each target. Over the
    corpus (176 fixtures, four targets) only the new fixture changed. The new `c.const` spelling moved T45 of
    `cfront-values.json`, and the evaluator's new signature broke the replacement of IC6 of `cfront-globals.json`;
    both were re-pointed and are each caught.
  24 injected defects, on both rails, are each caught (`tools/testing/faults/cfront-consts.json`).
  Found, not fixed here (each a suggested follow-up):
  - a duplicate case value after conversion (`case -1:` beside `case 4294967295u:` in a `uint32_t` switch) is
    accepted by both rails, where C requires a diagnostic (6.8.4.2p3) and the emit does not compile (closed by
    CF-CONSTEXPR2, below);
  - a character constant `'\xff'` is -1 on every target on both rails; where plain `char` is unsigned (AArch64
    Linux) C gives 255 (closed by CF-CONSTEXPR2);
  - an `enum` defined at block scope is refused by both rails, for different reasons, and so are an enumerator in
    a compound literal's dimension `(T[N]){...}` and a row pointer's `(*p)[N]`, where each rail reads a literal
    only (closed by CF-CONSTEXPR2);
  - `sizeof`, `_Alignof` and `(int)1.5` in an enumerator are refused on both rails (6.6p6 allows them) (closed by
    CF-CONSTEXPR2);
  - a 3000-term `1 + 1 + ...` chain raises `RecursionError` in the oracle's parser, which the twin lowers (closed by
    CF-CONSTEXPR2);
  - the twin refuses `typedef T row[0];`, which the oracle takes (closed by CF-CONSTEXPR2).

  CF-ENUMSCOPE (2026-10-01) made a block-scope name hide an enumerator on both rails, a silent miscompile the
  CF-ENUMFOLD triage found. RED was measured on the parent (`04d575dc`).
  - The defect: both rails read a name as an enumerator before anything else, wherever it stood, so a local, a
    parameter or a loop's own declaration of the same name never hid it (C11 6.2.1p4). Beside `enum { N = 4 }`,
    `uint32_t N = (s & 1u) + 1u; uint32_t a[N]; ... return a[0] + sizeof a + N;` lowered on both rails as `a[4]` and
    `+ 4`, digest-equal: `f(1)` returned 21 where C gives 11. A parameter or a loop variable named for an enumerator
    was refused on both rails, for different reasons (the oracle read `K++` as `3++`), and the twin's `env` kept a
    function's parameters bound at file scope after its body.
  - RED: on the parent the twin refuses `runtime/c/cfront_enumscope.c` (`PARSE-ERR ;`), the oracle raises
    `unsupported base expression IntLit`, and the new test fails at the oracle's case label (`case N:` naming the
    local folded to `case 4:`).
  - What landed: the oracle's parser keeps the names the function declares, one set per block scope (the
    parameters around the body, a block, a `for`'s own declaration), each in scope from the end of its declarator,
    its own initializer included (6.2.1p7), and `_enumerator` reads a name as an enumerator only where none of them
    is in scope. The twin reads every enumerator through one predicate, `visible_enum`, which hides one a name in
    `env` binds -- an expression's primary, an integer constant expression (so `case N:` naming the local is refused
    as no constant on both rails), `sizeof` of a name, a fence's memory order and a volatile access's byte offset --
    and its unit loop clears `env` at file scope.
  - Outcomes: the fixture (a local as a VLA's extent and under `sizeof`, a parameter, an inner block, a loop's
    declaration, `uint64_t N = sizeof N;`, a local array under `sizeof`, and file scope after a function whose
    parameter hid the name) lowers to one claim graph on the four targets and each emit runs as the original; the
    fence and byte-offset readers hold parity on the four targets. Over the corpus the twin's output in its four
    modes (916 runs) and the oracle's summaries and emits (229 fixtures) are unchanged but for the new fixture.
  13 injected defects, 6 on the oracle and 7 on the twin, are each caught (`tools/testing/faults/cfront-enumscope.json`).
  Found, not fixed here (a suggested follow-up): a local that hides a typedef name is refused on both rails where a
  statement begins with it (`typedef uint32_t T; ... uint32_t T = s; T = T * 3u;` -- the twin: `expected declarator
  name`, the oracle: `expected 'IDENT', got OP '='`), where C reads the statement as an expression (closed by
  CF-TYPEDEFSCOPE, below).

  CF-FPTAB (2026-10-01) closed the calls through `(*fp)` and through tables CF-SPLIT2 recorded, and the member reads
  CF-CALLS recorded (the first item of its list), and made both rails refuse, for one reason each, the operands C
  refuses to the operators those calls are built from. RED was measured on the parent (`c0680c87`).
  - The defects: a call took only a name or a member as its callee on the oracle's parser, so `(*fp)(x)`, `(**fp)(x)`,
    `ops[i](x)` and `(ops[i])(x)` were refused there, and on the twin `*` of a function pointer was `dereference of a
    non-pointer` and a call after a subscript a missing `;`. The twin refused every inline declarator of a table or a
    pointer to one (`uint32_t (*t[2])(uint32_t)`: `expected declarator name`), and declared a typedef'd table parameter
    (`op_t t[2]`) as one function pointer -- an emit no compiler takes -- where the oracle refused it as `unknown type
    'op_t'`. Both rails typed a function pointer read from a member as an integer, so `op_t g = o.fn;` emitted an
    integer temp that does not compile, and a member arm beside a function of another type was compared on neither.
    Beside them, the operand kinds of `*`, `[]`, `.` and `->` (C11 6.5.3.2p2, 6.5.2.1p1, 6.5.2.3p1-2): the oracle read
    `*s` of an integer through memory at `s` as a `uint32_t` (`*s`, `*g`, `(*s)++`, `++*s` and `*s += 1u` lowered on the
    oracle alone); both rails lowered `*s = 1u`, `*(s + 1u)`, `&*s`, `s[1]`, `s[1] = 2u`, `s[1]++`, `&s[1]`, `fp[0]` (a
    subscript of a function pointer) and `*fp = 1u`; the twin lowered `s->x` of a struct as `s.x`, `p.x` of a pointer
    as `p->x`, `o[1].x` of a struct, `sizeof s[1]` and `sizeof *fp`; and the forms both refused, each rail refused for
    a reason of its own.
  - RED: on the parent all five tests fail. Both rails refuse `runtime/c/cfront_fptab.c` (the oracle at its first
    inline declarator, the twin at a call through an element) and `_FPTAB_CALLS`; of the 41 forms `_FPTAB_REFUSED`
    holds, the oracle gives 41 another verdict and lowers 15 of them, and the twin gives 31 another verdict and lowers
    16; of the nine `_FPTAB_LOWERED` holds, the oracle refuses eight and the twin six; the oracle lowers CF-FNSEL's
    member arm beside a function of another type, and refuses `sizeof` of a function, and a member access on a
    non-struct, for reasons of its own.
  - What landed: on the oracle, the parser takes a call after any postfix expression (`cast.CallPtr`) and an inline
    declarator of a table of up to three dimensions or a pointer to one (`_funcptr_declarator`, `_declarator` keeping
    the shape of a typedef'd one); lowering calls through `_call_ptr`, which names the function a `*` of a function
    designates (`_fn_valued`, which sees through a conditional and a generic selection) and calls any other value that
    is a function pointer through `_call_through`, refusing the rest as `NOT_CALLABLE`; a member and an element read
    are typed (`fn_loaded` is gone); `_type_of` takes a subscript chain as `_lvalue` does, one subscript per dimension
    of a multi-dimensional array's shape; and a 2-D file-scope table is indexed whole as a 2-D scalar global is. One
    predicate per operator decides the operand: `_indexable` for `*` and `[]` (at the lvalue, the value, `&*`, the
    type and `sizeof`), `_member_agg` for `.` and `->`, and a function designator is no lvalue (`FN_NOT_LVALUE`). On
    the twin, `fp_inline_decl` parses the inline declarators at every site (a local, a parameter, a member, a global)
    with up to three dimensions; `call_value` calls through any value that is a function pointer -- a parenthesized
    expression, a postfix chain, a call's result, a literal, a generic selection -- and refuses the rest for the
    oracle's reason; `deref_named_callee` takes `(*NAME)(x)` of a function or a function-pointer object, never of a
    table, whose `*t` is its first element; a table's element, a member read and a pointer to a table's pointee are
    function-pointer values typed by their alias (`fp_value_temp`, `ptee_fp`, `bcir_ctype.ptr_to_fp` and
    `bcir_resource.ptee_funcptr`, both in padding: `bcir_ctype` stays 120 bytes); a typedef'd table parameter decays to
    a pointer to its element; `use_global` gives a file-scope table its element's alias; and `global_md_field` takes a
    2-D table of function pointers. Its predicates mirror the oracle's: `names_object_ptr` at every fast path that
    reads or writes through a named operand and in `array_index_n`, where every named subscript is parsed;
    `member_base_ok` at the member read and store; `deref_lvalue_refused` at a store and a step through `*`;
    `star_names_fn` in `sizeof`. Each reason is one string on both rails (`DEREF_NOT`, `SUBSCRIPT_NOT`, `DOT_NOT`,
    `ARROW_NOT`, `FN_NOT_LVALUE`, `SIZEOF_FN`, `NOT_CALLABLE`).
  - Outcomes: `runtime/c/cfront_fptab.c` -- tables typedef'd and inline, local and file-scope, 2-D, members, pointers to
    them and parameters of them, their elements picked and compared, calls through `(*fp)`, `(**fp)` and `(*f)`, and
    `*fp` as a value, held, selected and returned -- lowers to one claim graph on the four targets and both emits run as
    the original; every call it makes in place goes through a pointer that holds one function, so the G10 rows are
    unchanged, and a call through a table in place is `_FPTAB_CALLS` (local, file-scope and 2-D tables by a runtime
    index, a pointer, a parameter of every spelling, a member, under `*` and `**`, a generic selection and a select),
    which lowers alike and runs as the original on both emits. The 41 refusals of `_FPTAB_REFUSED` hold on both rails
    for the one reason each names, the nine valid forms of `_FPTAB_LOWERED` lower alike on the four targets, and
    CF-FNSEL's member arm beside a function of another type is refused on both. CF-SIZEOF's `sizeof` of a function has
    one reason on both rails.
  - Faults: `tools/testing/faults/cfront-fptab.json` holds 34 injected defects, 16 on the oracle and 18 on the twin,
    and each is caught by its own test. The first sweep caught 31: `*fp` read as a value was in no test, so the oracle
    reading it through memory survived; the twin reading `p.x` of a pointer as `p->x` was caught only by the refusal
    table, not by the member test; and the oracle typing a member read as no function pointer only by the fixture, not
    by CF-FNSEL's test. The fixture now returns `*fp` as a value (`ft_pick_deref`), the member test refuses `.` of a
    pointer, and CF-FNSEL's lowered set holds a member read through `*`. Four faults of earlier tables whose anchors
    this change rewrote were re-anchored -- `cfront-splits.json` S3, `cfront-calls.json` CL18, `cfront-decls.json`
    DT5 and `cfront-globals.json` GA7 -- and each is caught again; `cfront-calls.json` CL46, which injected the oracle
    typing a member read as a function pointer, describes what both rails now do and is retired.
  Found, not fixed here (each a suggested follow-up):
  - arithmetic on a function pointer -- `fp++`, `fp + 1`, `fp += 1` -- lowers on both rails, digest-equal (C11 6.5.6p2
    requires a pointer to an object type; GNU's extension), and an increment of a table's element (`(*t)++`, `t[0]++`)
    is refused on both rails for different reasons (the increment closed by CF-STRUCTCOND, below: one reason);
  - a static local table of function pointers (`static op_t t[2] = {f, g};`) is refused on both rails as no integer
    constant expression, where a function designator is an address constant (C11 6.6p9);
  - a call to a function returning a function pointer is still typed `uint32_t` on both rails (`uint32_t t =
    bcir_pick(s);`, an emit no compiler takes) -- an item of CF-CALLS's list -- so the fixture's table parameters
    compare the element they pick rather than return it (closed by CF-FPRET, below);
  - `i[p]` -- the pointer as the index (C11 6.5.2.1p2) -- is refused on both rails, an integer constant base for one
    reason (`a subscript of an integer constant is not supported`) and any other integer as a subscript of no array.

  CF-FPRET (2026-10-01) typed a function pointer a call returns, and what a call through a function pointer returns, on
  both rails, and declared pointers to variadic functions: three items of CF-CALLS's list and the one CF-FPTAB recorded
  beside them. RED was measured on the parent (`b831ba78`).
  - The defects: a call to a function returning a function pointer -- directly, through its prototype or through a
    pointer to such a function (`op_t g = pick(s);`, `mk_t m = pick; op_t g = m(s);`) -- typed its result as an integer
    on both rails (`uint32_t`, or on the twin a `uint64_t` through a pointer), and Clang rejects the emit. Built by
    GCC, which only warns, both rails' `op_t g = pick(s); return g(s);` called a pointer cut to 32 bits and crashed, and
    `pick(s) == inc` returned 0 where the original returns 1; a call through the result (`pick(s)(s)`) was refused on
    both as calling no function pointer. A function pointer whose function returns a pointer (`typedef uint32_t
    *(*pf)(uint32_t *);`, and a local, member or parameter so declared) was a parse error on the oracle, and the twin
    typed a call through one as an integer, refusing `*h(&v)` as a dereference of a non-pointer and `h(s)->a` as a
    missing `;`. The twin refused the member of a struct a call through a function pointer returns (`m(s).a`, as a
    struct converted to a scalar), and the oracle refused `(*m)(s).b` (`unsupported base expression CallPtr`). Both
    rails refused every declarator of a pointer to a variadic function (`uint32_t (*g)(uint32_t, ...)`: `expected a
    type`), and lowered a variadic function's designator as an arm of `?:` to an integer select Clang rejects -- one
    beside a function of another type included. A function declared to return a function pointer by a nested
    declarator (`uint32_t (*pk(uint32_t s))(uint32_t)`) was a parse error of each rail's own.
  - RED: on the parent both new tests fail. Both rails refuse `runtime/c/cfront_fpret.c` (the oracle at its
    pointer-returning typedef, the twin at its variadic one) and `_FPRET_CALLS`; each rail gives each of the five forms
    `_FPRET_REFUSED` holds another verdict, lowering the two variadic arms, and refuses six of the seven
    `_FPRET_LOWERED` holds -- the seventh, a function-pointer return declared by a prototype, lowers to an integer.
  - What landed: on the oracle, a call's function-pointer return is that function pointer (`_call_result_ct`), and
    `_call_ret` reads what a call returns off the declarations alone -- a function-pointer object's function's, a
    definition's or a prototype's, and through any callee whose value is a function pointer, its function's -- so
    `_fn_valued` and `_fn_value_type` see a call that returns a function pointer and `_call_ptr` calls through it; the
    result of a call through a function pointer or a member is addressable as a direct call's (`_addr`: `(*m)(x).f`,
    `o.mk(x).f`); the parser takes the `*`s before a function-pointer declarator into its function's return type
    (`_fp_return_stars`, at every declarator, a typedef and a member) and a trailing `...` into its parameter list
    (`TypeRef.func_variadic`, `CType.variadic`); and a variadic function's designator has its named parameters and
    `...` for its function type (`_fn_type`, `_fn_key`), which the emit (`_funcptr_decl`) and a claim's callee
    signature spell. On the twin, the function-type table records `...` (`fsig.variadic`, `sig_addv`), which
    `fp_param_list` parses at every capture site, `sig_alias` and `designator_sig` spell and `sig_same` compares; a call
    returning a function pointer -- direct, through a prototype, through a function pointer or a member -- yields a
    function-pointer value of its type (`fp_ret_temp`), and one through a function pointer returning a pointer a
    pointer (`fp_result_temp`, and `field_call_temp` at both member-call sites); one predicate, `call_result`, takes the
    postfix on any call's value -- `.` of a struct, `->` and `[` of a pointer, a call through a function pointer -- at a
    direct call, through a function-pointer object, through any value (`call_value`) and through a member
    (`member_call_result`); and the indirect and member call emits declare a pointer result by its type. A function
    declared to return a function pointer by a nested declarator is refused on both rails for one reason (`FN_RET_FP`,
    the twin's `CC_FN_RET_FP`); with a typedef for its return type it lowers.
  - Outcomes: `runtime/c/cfront_fpret.c` -- function pointers returned by calls, held, compared, selected and returned,
    through a definition, a prototype and a pointer to a function that returns one; a pointer and a struct returned
    through a function pointer, the struct's members read; pointers to variadic functions declared inline and by a
    typedef, selected and called -- lowers to one claim graph on the four targets, and both emits run as the original,
    the driver calling every pointer the fixture returns and comparing it with the original's. Each call it makes in
    place goes through a pointer that holds one function, so the G10 rows are unchanged; the calls through a returned
    pointer in place are `_FPRET_CALLS` -- a call's result called directly, under `*` and through a pointer to a
    function that returns one, a member's, a struct's member and a pointer returned through a member or a table, a
    select of variadic functions, a variadic member -- which lowers alike and runs as the original on both emits. The
    five refusals of `_FPRET_REFUSED` hold on both rails for the one reason each names, the seven forms of
    `_FPRET_LOWERED` lower alike on the four targets, and a call through a pointer to a variadic function carries
    `uint32_t(uint32_t, ...)` as its callee signature. The corpus harness includes `<stdarg.h>`, which a variadic
    fixture's preprocessed source needs.
  - Faults: `tools/testing/faults/cfront-fpret.json` holds 28 injected defects, 13 on the oracle and 15 on the twin,
    and each is caught by its own test. Seven faults of earlier tables whose anchors this change rewrote were
    re-anchored -- `cfront-calls.json` CL8, CL13, CL16, CL25, CL26 and CL27 and `cfront-fptab.json` FT17 -- and each
    is caught again.
  Found, not fixed here (each a suggested follow-up):
  - a function pointer initialized or assigned with a function of another type (`op_t g = two;` of a two-parameter
    function, `op_t g = va;` of a variadic one) lowers on both rails, which C forbids (6.5.16.1p1) and Clang rejects;
  - a cast to a function-pointer typedef (`(op_t)inc`, `(op_t)0`) lowers on the twin and is `unknown type 'op_t'` on
    the oracle -- beside CF-RTWIDE's cast to an inline function-pointer type, which the round trip refuses (closed
    by CF-RTFP, below);
  - a function declared to return a function pointer by a nested declarator (`T (*f(P))(Q)`) is refused on both rails,
    now for one reason; a typedef for its return type is the spelling both take.

  CF-EXTDESIG (2026-10-01) let a function the unit only prototypes -- another unit defines it -- be named as a value
  on both rails, the second item of CF-DECLS's list, and made both rails take the address of any function and refuse
  storing to one alike. RED was measured on the parent (`bcf225d4`).
  - The defects: a designator of a function declared by a prototype and defined by another unit (`apply(ext, s)`,
    `op_t g = ext;`, `s ? ext : inc`, `{ext}`) was refused on both rails (`use of undeclared identifier 'ext'`), though
    a call to it lowered as an external edge (CF-DECLS). `&f` of any function -- defined or prototyped -- was refused
    on both rails for reasons of their own (the oracle: `use of undeclared identifier`, the twin: `unsupported
    address-of`); `sizeof &f` of a prototyped function lowered on the oracle alone, and `sizeof f` was a function on
    the oracle and an undeclared name on the twin. The oracle's `extern` declaration of a variadic prototype dropped
    its `...` (`extern uint32_t vext(uint32_t);`), so a call passing more arguments, `vext(s, 2u, 3u)`, did not
    compile; the twin spelled it. Storing to a function or stepping one -- `f = g`, `f++`, `++f`, `f += 1` -- was
    refused for reasons of each rail's own, the oracle's naming the function an undeclared identifier. And a `0`
    compared with a designator (`f != 0`) was an `int` temp in the twin's emit, a pointer to a function of no
    parameters in the oracle's, each a comparison Clang or GCC warns about.
  - RED: on the parent both new tests fail. Both rails refuse `runtime/c/cfront_extdesig_link.c` and `_EXTDESIG_CALLS`
    (`use of undeclared identifier 'ed_ext'`); of the 13 forms `_EXTDESIG_REFUSED` holds, the oracle gives 10 another
    verdict and the twin 12; of the 8 `_EXTDESIG_LOWERED` holds, the oracle refuses 7 and the twin all 8.
  - What landed: on the oracle, one predicate names the function a node designates -- a name no object hides, of a
    function the unit defines or prototypes, under `&` too (`_designator`) -- and `_fn_valued`, `_fn_value_type`,
    `_call_ptr` (`(&f)(x)` is the direct call `f(x)`), the address-of and `_lvalue` (`FN_NOT_LVALUE`) read it; a
    prototyped function's value has its prototype's type (`_func_ptr_value`, `_fn_type`) and is declared `extern` as
    a called one is (`_tu_declare`, which the call path shares); the parser keeps a prototype's `...`
    (`Unit.variadic_protos`), which `func_variadic`, the function type and the `extern` declaration carry; and `f !=
    0` types the 0 by the designator's whole function type. On the twin, one function makes the value of a function
    declared at that point -- by an earlier definition or a prototype -- for a designator and for `&f` alike
    (`fn_value`, read where `declared_ret` answers); `designator_sig` reads a prototype's type when the unit has not
    defined the function, with the `...` the prototype table now records; `sizeof f` and `sizeof &f` read
    `declared_ret`, and `star_names_fn` reads `sizeof *f` and `sizeof *&f` as the function; `deref_named_callee`
    takes `(&f)(x)` and `(*&f)(x)` as `f(x)`; `f = g`, `f++` and `++f` are refused with
    `CC_FN_NOT_LVALUE`; and `null_compared` types a 0 compared with a designator by the designator's function type.
  - G10 and R18: a function the unit does not define is code no analysis sees. An indirect call whose pointer may
    hold one is an external edge -- its targets are not narrowed, it counts as `icall.unknown`, its actuals escape and
    it returns unknown pointers -- as a direct call to it is (`c.call.tu`); both rails' analyses already said so
    (`targets`, the twin's `esc_targets`: a function of another unit). R18 sees no edge to it, as for a call. The
    fixture makes no indirect call, so the G10 rows stay at their bounds (15 and 18); the 8 indirect calls of
    `_EXTDESIG_CALLS` are each an external edge, and both rails' effects and escape reports for it are byte-identical.
  - Outcomes: `runtime/c/cfront_extdesig_link.c` -- prototyped functions passed to a function of this unit and of
    another, held, selected, stored in a member and a table, compared with a pointer, a designator and 0, their
    addresses taken, called directly, through `*f` and `(&f)`, and a variadic one called with more arguments than it
    names -- lowers to one claim graph on the four targets, each emit declares every function it names as the
    prototype does, and both emits run as the original with the driver defining the functions, built with a pointer
    compared with or converted to an integer made an error under Clang and GCC. `_EXTDESIG_CALLS` lowers alike and
    runs as the original; `_EXTDESIG_REFUSED` is refused on both rails for the one reason each names, and
    `_EXTDESIG_LOWERED` lowers alike on the four targets.
  - Faults: `tools/testing/faults/cfront-extdesig.json` holds 22 injected defects, 11 on the oracle and 11 on the twin,
    and each is caught by its own test. The first sweep caught 20 of 21: XO8, the oracle's `_fn_valued` blind to
    `&f`, reached no test, because `*&f` was in none -- and probing it found the twin calling `(*&f)(x)` through a
    pointer (another digest) and refusing `sizeof(*&f)` for a reason of its own. Both are fixed (`deref_named_callee`,
    `star_names_fn`), `_EXTDESIG_LOWERED` and `_EXTDESIG_REFUSED` hold the forms, XT11 injects the second, and the
    re-sweep caught 22 of 22. Six faults of earlier tables whose anchors this change rewrote were re-anchored --
    `cfront-calls.json` CL13, `cfront-decls.json` DO2 and DT11, `cfront-fpret.json` FT8 and FT11, `cfront-fptab.json`
    FT2 -- and each is caught again.
  Found, not fixed here (each a suggested follow-up):
  - a function declared at block scope (`uint32_t f(uint32_t s) { uint32_t g(uint32_t); ... }`) is refused on both
    rails with parse errors of their own;
  - the oracle's linkable emit refuses a file-scope initializer that names a function (`static op_t t[2] = {f, g};`:
    `non-renderable constant initializer`), where each function's emit takes it (closed by CF-LINKEMIT, below);
  - a function-pointer type with a `const` parameter (`uint32_t (*g)(const uint32_t *)`) is spelled without it on both
    rails, so a prototyped function taking `const uint32_t *` does not convert to it in the emit (CF-DECLS's first
    item; closed by CF-QUALS, below).

  CF-QUALS (2026-10-01) kept every qualifier level of a function type on both rails -- CF-DECLS's first item and the
  `const` parameter item of the CF-FNSEL and CF-EXTDESIG lists -- and closed, beside it, the twin's reading of an
  element that is a pointer (CF-IDXARROW). RED was measured on the parent (`b27c37e1`).
  - The defects: what a pointer points to and each `*` under the outermost (`const char *const *v`, `char *restrict
    *`), and the qualifiers of a function pointer's own parameters and return, are part of a function type (C11
    6.7.6.1p2, 6.7.6.3p15). Both rails dropped every qualifier past the first level: an `extern` declaration of a
    prototype taking `const char *const *` read `const char **` (the oracle, `const char * *`), which conflicts with
    the prototype it repeats, and a function pointer of a function taking `const uint32_t *` -- a typedef, a
    declarator, a member, a table -- took `uint32_t *`, which the function does not convert to. The oracle refused a
    qualifier after a `*` in a function pointer's parameter list outright. A pointer object that is itself volatile
    (`T *volatile p`, `volatile str_t p`) lowered on both rails with its qualifier dropped. And `_Generic` matched an
    association of a qualified type as the unqualified one on both rails -- `_Generic(p, const char *: 1, char *: 2)`
    of a `char *p` gave 1 -- a silent miscompile.
  - The twin's typedef of a struct pointer (`typedef struct S *SP;`) kept no pointer depth, so `SP *pp` was one level
    shallow; making it the level the qualifiers count by exposed CF-IDXARROW: the twin read a member straight through
    an element that is a pointer -- `arr[i]->m` of an array of pointers, `pp[i]->m` through a `T **`, read, stored,
    stepped, compounded, addressed -- as a member of the pointer's own slot, a silent miscompile the oracle refuses
    (`unsupported base expression Index`); flattened `pp[i][j].f`, assigned, compounded or stepped as a value, to the
    index `i*1+j` of `pp`, another; typed `pp + 1` and `arr + 1` of a `T **` as a `T *` and `&arr[i]` of an array of
    pointers as a `T *` four bytes apart, so their emits did not compile or compared unequal pointers; and read a
    file-scope array of pointers as integers.
  - RED: on the parent all three new tests fail. Both rails refuse `runtime/c/cfront_quals_link.c` (the oracle:
    `expected ')', got OP '*'`; the twin: `expected declarator name`); the oracle refuses `_QUALS_CALLS` and the twin's
    emit of it conflicts with the prototypes. Of the 17 forms `_QUALS_REFUSED` holds, the parent oracle lowers 13 and
    the twin 15; of the 15 `_QUALS_LOWERED` holds, the oracle refuses 7 and the twin 2; of the 17 `_IDXARROW_REFUSED`
    holds, the parent twin lowers 16. The twin refuses `runtime/c/cfront_idxarrow.c` (`a struct or union is
    converted to a scalar type`), and its claim graph of `_IDXARROW_LOCALS` differs from the oracle's, its emit not
    compiling.
  - What landed: the parsers keep each `*`'s qualifiers -- `TypeRef.ptr_quals` on the oracle, a byte per kind
    (`ptr_const`, `ptr_restrict`) in two padding bytes of `bcir_ctype` on the twin, which stays 120 bytes -- through
    declarators, casts, `sizeof`, `typeof`, `va_arg`, function-pointer declarators and a qualifier on a pointer
    typedef, which qualifies the pointer (6.7.8p3). A function type carries them: the oracle's `fquals` (`_fn_quals`,
    `_qual_sig`), the twin's parameter and return ctypes; the type's identity compares them (`_fn_key`, `sig_same`'s
    `qual_key`), so arms of `?:` of two function types that differ in a qualifier are refused as CF-FNSEL refuses
    any; and every function type the emit spells keeps them (`_extern_decl`, `_funcptr_decl`, `_qual_type`; the
    twin's `ctype_qstr`). The emit spells its own objects without qualifiers, so where C converts none -- an
    argument to a parameter qualified more than one level down, a qualified result -- a call casts: the oracle's
    `_qual_args` and `_qual_result`, the twin's cast table (`bcir_func.qcasts`, which `bcir_claim.qcast` names, in the
    claim's tail padding so a rolled-back lowering takes its casts with it). A volatile pointer object
    (`VOLATILE_PTR`), a qualified `_Generic` association (`GENERIC_QUALIFIED`) and a qualified `*` past the eighth
    (`QUAL_DEEP`, the masks' width) are refused on both rails for one reason each. For CF-IDXARROW, one predicate on
    the twin, `index_member_ok`, refuses a member access through an element that is not the struct itself, read by
    `aos_elem_field`, `aos_member_array` and `elem_field`; the assignment-as-value and step paths walk subscripts with
    `index_chain`, as the read and store paths do; `tempptr`, `&a[i]` and a file-scope array of pointers keep the
    pointer's depth and its pointee's spelling.
  - G10: both fixtures keep their objects at file scope and their indirect calls on functions of the unit, so the
    rows stay at their bounds (`escape.unproved=0 icall.unknown=15 icall.unresolved=18`); a local lent to an
    external function, a call through a member and the local forms of CF-IDXARROW are the tests' own units.
  - Outcomes: `runtime/c/cfront_quals_link.c` -- prototypes taking and returning pointers qualified below the top,
    a `const` pointer typedef and a pointer to `const` function pointers among them, function pointers whose
    parameters keep theirs, typedef'd, inline, in a table and `const` -- lowers to one claim graph on the four
    targets, each emit declares every callee as its prototype does, and both emits run as the original under Clang
    and GCC with a qualifier mismatch an error, the driver defining the callees. `_QUALS_CALLS` -- locals passed to
    those parameters, calls through a member by `.`, `->` and a chain, through a call's result and through a pointer
    to a variadic function -- lowers alike, runs as the original, and both rails' effects and escape reports are
    byte-identical. `runtime/c/cfront_idxarrow.c` joins the corpus and `_IDXARROW_LOCALS` runs beside it; both run as
    the original function by function. `_QUALS_REFUSED` (17) and `_IDXARROW_REFUSED` (17) are refused on both rails
    for the one reason each names and `_QUALS_LOWERED` (15) lowers alike on the four targets. The corpus is unmoved:
    every other fixture's summary and digest is the parent's on both rails.
  - Faults: `tools/testing/faults/cfront-quals.json` holds 38 injected defects -- 11 on the oracle, 17 on the twin
    for CF-QUALS, 10 on the twin for CF-IDXARROW -- each caught by its own test. Fourteen faults of earlier tables
    whose anchors this change rewrote were re-anchored -- `cfront-calls.json` CL2, CL7, CL8 and CL45,
    `cfront-decls.json` DO6, DO7, DO8 and DT7, `cfront-fpret.json` FO6, FO10, FT13 and FT14, `cfront-fptab.json`
    FO3, `cfront-gaps.json` O15 -- and each is caught again.
  Found, not fixed here (each a suggested follow-up):
  - a member access through a subscripted pointer element, `arr[i]->m`, is refused on both rails (the oracle has no
    subscript base); C allows it, and driver code spells it often (`devs[i]->ops`);
  - a member access through a parenthesized dereference, `(*pp)->m`, is refused on the twin (`;` expected) and
    lowered by the oracle, and `(pp + 1)[0][0]` is refused on both rails for reasons of their own;
  - `x[i]->m` of an element that is a struct, and `p[i]->m` through a pointer to one, are invalid C that both rails
    lower as `x[i].m`.

  CF-RTFP (2026-10-01) closed CF-RTWIDE's three follow-ups -- the round trip's re-parse of a cast to a function-pointer
  type, of an `_Atomic` member access beside a typedef only the original declares, and of `_BitInt` beside a
  bit-field's type; a unit that declares `size_t` itself; a braced function-pointer initializer on the twin -- with
  CF-FPRET's cast to a function-pointer typedef, and the defects found beside them. RED was measured on the parent
  (`e4424bdf`).
  - The defects: the emit stores a function-pointer member through a generic slot at its byte offset, `*(void
    (**)(void))((char *)p + K) = (void (*)(void))f;`, and reaches an `_Atomic` member as `(*(_Atomic T *)((char *)p +
    K))`. Both rails refused every cast to a function-pointer type (`(R (*)(P))f`, `(R (**)(P))p`) and to a pointer to
    an `_Atomic` type; the oracle refused `(op_t)f` of a function-pointer typedef as an unknown type (its cast path
    rebuilt the type from its base, dropping the function type) where the twin lowered it as an integer temp. The emit
    names a function it takes as a value by its source name (`op_add`) and a typedef the original declares (`typedef
    float _Complex cf;`), and the classifier supplied neither; it stored a `_BitInt` into a bit-field as `(v & 31u)`,
    `_BitInt` arithmetic beside a standard type, which the oracle refuses. The oracle read `size_t` in `typedef unsigned
    long size_t;` as a keyword of the specifier before it -- its builtin scalar names ran into the keyword run -- and
    refused every unit declaring a standard type itself. The twin refused `uint32_t (*fp)(uint32_t) = {g1};` (6.7.9p11
    allows the braces) and declared `uint32_t (*fp)(uint32_t) = 0;` through an `int` temp, which GCC rejects.
  - Found beside them, and closed: both rails cast the null pointer constant through an `int` temp, `int t = 0; T *q =
    (T *)t;`, which GCC rejects (a pointer from an integer of another size), for any pointer type; a cast to an array
    type (C11 6.5.4p2) lowered on the oracle as the pointer the array decays to (`(uint32_t[2])s`), was refused on the
    twin as an undeclared name, and lowered on both to two claim graphs for an array of function pointers; `sizeof` of
    an array compound literal was the size of a pointer on the twin (`sizeof((uint32_t[3]){1u, 2u, 3u})` 8 where C
    says 12, a silent miscompile), and `sizeof (T[N]){...}` without parentheses was refused on both; `sizeof` of an
    array of function pointers counted one pointer on the twin; the twin refused a typedef of a table of function
    pointers (`typedef uint32_t (*tab_t[2])(uint32_t);`) and of a pointer to one, which the oracle lowers, typed
    `__typeof__(t[0])` of a table's element and `__typeof__(o.fn)` as integers (its emit failing its verification),
    and refused `( E ) = v;` and `( E ) OP= v;` where a statement begins -- the emit's own spelling of an atomic member
    store -- which the oracle lowers. Reading `_Atomic` as a cast's type name opened a compound literal of `_Atomic`
    type, an `_Atomic` object, on both rails, which CF-ATOMIC had refused as a parse error; and the casts opened a
    compound literal of function pointers on the oracle (`(op_t[2]){f, g}[i](x)`), which the twin refused.
  - Found by running every probe's emits against the original under both compilers, and closed -- each on the parent
    too: each emit stored a function pointer into a member through a generic slot, `*(void (**)(void))((char *)p + K)
    = (void (*)(void))f;`, and read it back as the member's own type, which C leaves undefined (6.5p7); GCC at -O2
    dropped the store, so a unit that fills an array of structs of function pointers and calls through one called a
    null pointer -- under both rails' emits, digest-equal, a silent miscompile. The twin stored a function into an
    element of a member table through `uint64_t _v = f`, which does not compile. A compound literal of a pointer type
    was a `uint32_t` on the twin, the pointer cut into it -- `*(T *){p}` refused as a dereference of a non-pointer,
    `&(T *){p}` a `T *`, `(op_t){f}(x)` refused -- and its `{0}` an `int` on both rails. And Clang (18, 22 and 23)
    rejects a cast to an `_Atomic` scalar type wherever its value is taken -- `uint32_t y = (_Atomic uint32_t)x;` --
    typing it `_Atomic` where C17 6.5.4p5 and GCC drop the qualifier, so the lowering of it this slice had added could
    not be held to the original.
  - RED: on the parent the three new tests fail, and so do the three this changes -- the round trip's pin and its
    definitions test, against the parent's classifier, and CF-ATOMIC's refusal test, whose `_Atomic` casts and
    literals the parent refuses as parse errors (`expected ')', got IDENT 'uint32_t'`). Both rails refuse
    `runtime/c/cfront_rtfp.c` and `_RTFP_CALLS` (the oracle: `expected ')', got PUNCT '('`; the twin: `)` expected).
    Of the 12 forms `_RTFP_REFUSED` holds, the parent oracle gives 8 another verdict and the twin 12; of the 33
    `_RTFP_LOWERED` and `_RTFP_UNITS` hold, the oracle refuses 16 and the twin 31, and of the ones both lower there,
    the emits of an array of structs filled with functions and called through crash under GCC at -O2 on both rails,
    and the twin's emit of a store to a member table's element does not compile.
    `cfront_fnptrmember.c` is excluded from the round trip, whose pinned count is 119 on the parent with the new
    classifier -- which alone brings `cfront_fnptrlocal.c` and `cfront_fpret.c` in.
  - What landed: one reading of a type name per rail -- the oracle's `_fp_type_name`, the twin's `p_cast_type` -- read
    by casts, `sizeof`, `_Alignof`, the byte-offset fold's cast and the slot store, takes an abstract function-pointer
    declarator and a typedef'd function pointer whole and refuses a named one (`TYPE_NAME_NAMED`); `_Atomic` starts a
    type name on both. A cast to a function pointer is `c.cast:fnptr`, a pointer to one `fnptr *`, a pointer to an
    `_Atomic` object `_Atomic T *`; the emit spells the cast by the temp's own type, the alias. The byte-offset fold
    (the oracle's `_byte_offset_access`, the twin's `bo_match`) takes a function-pointer slot, stored as the member
    store it was emitted from with its cast dropped (`_assign`, `fp_slot_value`), and an `_Atomic` integer or floating
    access -- a volatile one only of a device region, as before -- as one atomic operation, a compound and a step
    included (`emit_rmw`; the twin's `deref_incdec` form 5). A cast of an array type is refused as `CAST_ARRAY`, a
    compound literal of an `_Atomic` type as `ATOMIC_LITERAL`, and a cast to an `_Atomic` type as `CAST_ATOMIC` --
    spelled, `_Atomic(T)`, a typedef of one, under `sizeof` and `typeof` -- which CF-CALIGN had refused as a parse
    error; a cast to a pointer to an `_Atomic` object, the emit's own, lowers. A null pointer constant cast to a pointer
    is a null pointer of that type (`_null_pointer`, `null_pointer`). The oracle's keyword run ends at a builtin
    scalar name after a type, which is then the declarator. On the twin: the braced function-pointer initializer
    (`{f}`, `{f,}`, `{}`; two expressions refused as the oracle refuses them) and a null one; a typedef through
    `fp_inline_decl`, its element spelled by a synthesized alias; a compound literal of function pointers, called
    through (`call_value`); `sizeof` of a compound literal from its type name (`sz_literal`, `literal_close`; the
    oracle's `_literal_follows`), one whose initializer sizes it refused as the oracle refuses it; `typeof` of a
    function-pointer value by its function type (`res_sig`); and `( E ) = v;` read as `E = v;` when E is an lvalue of
    unary `*`s and postfix operators (`up_paren_lvalue` in `unparen_body`). The emit converts a `_BitInt` stored to a
    bit-field to the unit's type, and the classifier supplies the original's typedefs and each function it defines or
    prototypes, as an external prototype. Each emit stores a function pointer at a byte offset -- a member, an element
    of a member table -- through a pointer to its own type, `*(R (**)(P))((char *)p + K) = f;`, which a read of the
    member's type may alias: the oracle's `_fp_store`, a function's name typed whole by `LoweredFunc.fn_types` (its rid
    type carries the return alone), and the twin's `fp_slot_ty`, a value by its alias and a function's name by
    `__typeof__(&*f)`. A compound literal of a pointer type is a pointer local of its type on the twin, its `{0}` a null
    pointer on both, `&` of one a pointer one level deeper (`addr_temp`, the one predicate `&v` shares now), and one
    of a function-pointer type is called through.
  - Two earlier tests pinned what CF-RTFP changes: CF-RTVOL's near miss `*(volatile _Atomic uint32_t *)((char *)d +
    4)` now folds as one atomic access (it moved to the folded forms, each emit run against the original), and
    CF-ATOMIC's refusal of `(_Atomic uint32_t)x`, a parse error on both rails, names its reason now (`CAST_ATOMIC`)
    and holds the other spellings of the cast.
  - G10: the shared fixture keeps each indirect call on one function of the unit; calls through a table or a select of
    two functions and through a function pointer read from memory are `_RTFP_CALLS`, the test's own unit, so the rows
    stay at their bounds.
  - Outcomes: the round trip reaches its fixed point for 123 fixtures (117 before): `cfront_fnptrmember.c`,
    `cfront_signedfnptr.c`, `cfront_complexalign.c`, `cfront_bitint_bitfield.c`, `cfront_fnptrlocal.c` and
    `cfront_fpret.c` joined, each emit compiling with the original's headers over three rounds; the new
    `runtime/c/cfront_rtfp.c` is excluded for its masked guards. It and `_RTFP_CALLS` lower to one claim graph on the
    four targets and both emits run as the original under Clang and GCC with the mixes `_QUALS_WERROR` names made
    errors. `_RTFP_REFUSED` (12) is refused on both rails for the one reason each names, and `_RTFP_LOWERED` (31) and
    `_RTFP_UNITS` (2) lower alike, each emit the original where `f` takes a `uint32_t` -- at -O2, where GCC exploits
    what the generic slot left undefined. Beyond the tests, 207 probe forms were compared on the four targets: 151
    lower to one claim graph, 43 are refused on both rails for one reason, and the 13 that split are the follow-ups
    below; every emit of the ones that lower built and ran against the original under both compilers (604 builds,
    436 runs). Every other fixture's summary and digest is the parent's on both rails.
  - Faults: `tools/testing/faults/cfront-roundtrip.json` takes 54 more -- 20 on the oracle, 2 on the classifier, 32 on
    the twin -- each caught by its own test (the table now holds 76). The first sweep caught 42 of the first 43: an
    `_Atomic` cast that kept its qualifier (O54) left the claim graph as it was and declared the emit's temp `_Atomic`,
    so the witness, comparing claim graphs, could not see it. Holding the emit to the original instead -- the cast's
    and every probe's, under both compilers -- is what found Clang rejecting the cast, the slot store GCC dropped and
    the pointer compound literal; O54 now injects the cast's lowering, and the 11 faults of those three are caught with
    it. Three faults of `volatile.json` whose anchors this change rewrote were re-anchored and are caught again.
  Found, not fixed here (each a suggested follow-up):
  - a step of a volatile access at a byte offset, `(*(volatile uint32_t *)((char *)p + 4))++`, is refused on both
    rails for reasons of their own;
  - a call through a function-pointer member of an indexed struct element, `a[i].fn(s)`, is refused on both rails
    for reasons of their own (the oracle has no subscript base), and reading that member as a value, `op_t g =
    a[i].fn;`, is refused by the oracle (`array-of-structs non-scalar element field`) and lowered by the twin;
  - `typeof` of a call, of `?:` and of a function designator is refused by the oracle as not yet supported and lowered
    by the twin, whose `typeof(f)` fails its own verification;
  - `sizeof` of a compound literal its initializer sizes, `sizeof((uint32_t[]){1u, 2u})`, is refused on both rails,
    where C sizes it;
  - an assignment to an expression that is no lvalue -- `(a, b) = s`, `(s ? a : b) = 1u`, `(x + 1u) = s`, `(x++) = s`
    -- is refused on both rails for reasons of their own.

  CF-VOIDVAL, CF-STRUCTCOND and CF-UNARY (2026-10-01) closed CF-CALLS's void value and struct items and CF-FPTAB's
  step of a table's element -- the value of a void expression, a struct as a controlling expression or as the operand
  of unary `+` or `__real__`, an increment of a struct in memory, an array compared with 0, `void *` of `malloc` --
  and the defects found closing them, CF-GCOND among them. RED was measured on the parent (`601ca6f0`).
  - The defects: the value of a void expression (C11 6.3.2.2) -- `uint32_t k = f();`, `f() + 1u`, `!f()`,
    `(uint32_t)f()`, `if (f())`, `switch (f())`, `garr[f()]`, `s && f()`, `k = ({ f(); })`, `return f();` from a
    function that returns a value, of a void function called directly, through a pointer or prototyped -- lowered on
    both rails, digest-equal, and neither emit compiled. A struct or union as a controlling expression (`if (a)`,
    `while (a)`, `for (; a;)`, `do ... while (a)`, `a ? x : y`, `switch (a)`; 6.8.4.1p1, 6.8.5p2, 6.5.15p2) or as the
    operand of `+`, `__real__` or `__imag__` lowered on both rails, as did `+p`, `-p` and `~p` of a pointer, `-f` of a
    function, `~d` of a real floating value and `__real__ p` (6.5.3.3p1), and `switch` of a pointer, a floating value,
    an array or a function (6.8.4.2p1) -- emits Clang rejects; under `sizeof`, `typeof` and `_Generic` the oracle
    refused some for reasons of its own (``sizeof of `-` applied to a struct``) and lowered the rest. An increment of a
    struct in memory (`(*p)++`, `p[0]++`, `p->in++`, `++*p`, `g_a[1]++`) was refused for two different reasons (the
    oracle's follow-on reason, the twin's parse error), as was a step of a table's function pointer (`(*t)++`).
  - Found beside them, and closed -- silent miscompiles, digest-equal on both rails unless named: both parsers dropped a
    unary `+`, which promotes its operand (6.5.3.3p2), so `sizeof(+c)` of a `char` was 1, `_Generic(+h, int: ...)` of a
    `uint16_t` chose by `h`'s type and `__typeof__(+h) k = -1;` declared a `uint16_t` -- and the twin's byte-offset
    reader skipped a unary `+` because the oracle's parser dropped it; `+x = 1u` lowered on the oracle as `x = 1u`. The
    twin gave `-z` and `~z` of a complex a real temp, losing the imaginary part, promoted a `_BitInt` under `-` and `~`
    (C23 6.3.1.1p2 promotes none) and typed `__real__ c` of an integer a `double`; the oracle typed `__real__ z` under
    `sizeof` and `_Generic` as the complex itself, a bit-field operand by its declared type (`_Generic(x.a - 1, int: 1,
    unsigned: 2)` of an `unsigned a : 3` chose 2, where its value is an `int`) and `&g` as `g` (`_Generic(&g, uint32_t
    *: ...)` chose `default`, and `__typeof__(&g) p = &g;` made `*p` a dereference of a non-pointer). An array compared
    with 0 (`garr == 0`, `arr != 0`, a VLA) compared with an `int` temp in both emits, which neither compiler takes;
    `void *vp = malloc(4u);` lowered to two claim graphs (the twin bound a bounds extent to the `void *`, two claims
    nothing read); `++(x)` and `++(a[1])` were parse errors on the twin.
  - Found by running the emits against the original, and closed (CF-GCOND): the oracle spelled a file-scope object
    that only a control node reads -- `if (g)`, `switch (g)`, `while (g)`, `for (; g;)`, `do ... while (g)`, an early
    `return g;` -- as its raw rid temp (`if (t900000)`), which nothing declares: no claim touched it, so the function
    neither took its resource nor named it, and the emit did not compile. The digest hashes claims, not names, so
    parity never saw it; the twin names a global wherever it reads one.
  - RED: on the parent the two new tests fail, as do the port-I/O red team's two `outb` tests this
    changes and the two shared-corpus tests that hold the new fixture
    (`test_python_c_parity_and_equivalence_across_fixtures_g0`,
    `test_emitted_c_is_equivalent_under_both_gcc_and_clang_g0`). The twin refuses `runtime/c/cfront_unaryops.c`
    (`++(x)`) and the oracle two of its functions (`sizeof(+x.a)`: `sizeof of a bit-field`; `__typeof__(&uo_g) p =
    &uo_g;`: a dereference of a non-pointer); of the rest, each alone, both emits of `uo_plus` and `uo_parts` return
    what the original does not (`uo_parts` on two claim graphs), neither emit of `uo_null` compiles and the oracle's
    `uo_gconds` does not -- while `uo_conds` and `uo_voids`, which hold only forms C allows, run as the original there
    too. Of the 72 forms `_CARD5_REFUSED` holds, the parent oracle gives 70 another verdict (it lowers 60) and the twin
    72; of the 38 `_CARD5_LOWERED` and `_CARD5_BITINT` hold, each rail refuses 2, and of the ones both lower 21 fail:
    6 lower to two claim graphs, in 7 an emit returns what the original does not, and in 8 an emit does not compile
    (6 the oracle's, 2 the twin's).
  - What landed: the value of a void expression is refused on both rails (`VOID_VALUE`, `CC_VOID_VALUE`): the oracle's
    `_rvalue` refuses what `_rvalue_void` returns, read only where C discards the value -- an expression statement,
    `(void)e`, the operands of `,`, an arm of `?:`, a `_Generic` association, a statement expression's last statement,
    `return f();` in a void function; the twin, which parses and lowers in one pass, refuses a statement whose claims
    read a void value at the statement's end (`void_read`, over the `n_void` void values the unit made) and `return
    f();` from a function that returns a value where it is read. One predicate per rail answers whether an operand
    takes a unary operator (`_unary_operand`, `unary_operand_ok`), read as each operator lowers and by `sizeof`,
    `typeof` and `_Generic`: a struct or union for CF-STRUCTARITH's reason, and a pointer, a function, an array or `~`
    of a real floating value as `invalid argument type to unary expression` (`UNARY_NOT`) -- the twin's pass over the
    unit's final claims (`agg_unary_operands`), which never saw a speculative lowering's, is gone. One predicate per
    rail for every controlling expression (`_condition`, `cond_value_ok`): a scalar, and for `switch` an integer
    (`SWITCH_NOT`). An increment of a struct is refused for that reason on both (`_incdec_value`; the twin's
    `incdec_rest`, the prefix or postfix step `incdec_value` did not take), a table's function pointer for the
    follow-on reason on both. Both parsers keep `+`: its operand integer-promoted, by a `c.cast` to `int` where that
    changes its type (the twin's `unary_plus`), its promoted type under `sizeof`, `typeof` and `_Generic`, a fence's
    order under it its constant (`_fence_order_kind`, `fence_order_op`). `-z` and `~z` of a complex are complex
    (`tempc`), `-b` and `~b` of a `_BitInt` keep its type (`tempbi`), `__real__` and `__imag__` take a complex's
    element type and a real operand's own (`_part_type`, `part_temp`); a bit-field operand has its value's type
    (`_operand_type`, over the `_bitfield_value_type` `sizeof` shares); `&x` is a pointer to x's type. An array
    compared with 0 compares with a `void *` null pointer (the oracle's `==`/`!=`; `null_compared`); a `void *` takes
    no bounds extent (`bind_extent`: R21 reads the lifetime events alone, which both rails record); the twin reads
    `++(P)` as `(++P)` (`unparen_body`), the expression the oracle reads. The oracle takes and names every global a
    control node reads (`_control_rids`: a condition, a `switch` discriminant, a computed `goto` target, a returned
    value).
  - One earlier test pinned what CF-VOIDVAL changes: the port-I/O red team's `unsigned y = outb(v, 0x60);` lowered,
    failing verification, where C refuses the value of the void `outb`; it and `outb(v, 0x60) + 1` are refused now.
  - Outcomes: `runtime/c/cfront_unaryops.c` (`+`, the parts, bit-field and `&x` operands, arrays compared with 0,
    controlling expressions, a global only control nodes read, `++(x)`, void values where C reads none) lowers to one
    claim graph on the four targets and both emits run as the original under Clang and GCC with the mixes
    `_QUALS_WERROR` names made errors; the round trip excludes it (its emit names its file-scope objects, which the
    standalone re-parse never declares), so the round trip's count stays 123. `_CARD5_REFUSED` (72) is refused on
    both rails for the one reason each names, `_CARD5_LOWERED` (36) lowers alike, each emit the original, and
    `_CARD5_BITINT` (2) the same under Clang, the one compiler here that builds a `_BitInt`. Beyond the tests, 62
    probe forms lower or are refused alike on both rails; every other unit of the corpus (235) keeps the parent's
    summary and digest on both rails, on the four targets. G10: the fixture compares only file-scope arrays with 0 --
    a local array compared with one is an escape candidate the analysis does not prove (its first cut raised
    `escape.unproved` to 1) -- and the local and VLA forms are `_CARD5_LOWERED`'s own units, so every row stays at
    its bound.
  - Faults: `tools/testing/faults/cfront-unary.json`, 62 -- 33 on the oracle, 29 on the twin -- each caught by its
    own test. The first sweep caught 61: the oracle's `typeof` and `_Generic` typing of `!`, `__real__` and
    `__imag__` checking no operand (UO31) reached no test, as `_CARD5_REFUSED` held those operators only under
    `sizeof`; three such forms joined it, and the re-sweep caught UO31. Seven units joined `_CARD5_LOWERED` before the
    sweep so that each of CF-GCOND's control reads, a fence's order under `+` and the byte-offset reader's `+` reach a
    test of their own. Two older faults anchored in code this card replaced, which the quick tier's anchor test
    caught: `cfront-calls.json` CL36 (the post-parse struct-operand walk, now `unary_operand_ok`) and
    `cfront-operands.json` TV3 (the void `return`, now refusing a void value from a non-void function) were
    re-anchored to the new code, and each re-swept caught by its own test.
  Found, not fixed here (each a suggested follow-up):
  - an assignment to, a step of or the address of an expression that is no lvalue -- `+x = 1u`, `(+x)++`, `&+x`,
    `x++++`, `++x++`, `(s, x)++`, `((uint32_t)x)++` -- is refused on both rails for reasons of their own (the oracle:
    `not an lvalue`; the twin: a parse error or the follow-on reason), with CF-RTFP's `(a, b) = s` family;
  - a `_Generic` association naming a pointer to an array or to a function (`uint32_t (*)[2]: ...`,
    `uint32_t (*)(uint32_t): ...`) is refused on both rails with parse errors of their own;
  - on the Windows target, where `long double` has `double`'s representation, `_Generic(x, double: 1, long double:
    2)` of a `long double` chooses 1 on both rails, digest-equal, where Clang chooses 2 (a silent miscompile; the two
    are distinct types whatever their representation, 6.2.5p14);
  - `_Generic` of a 40-bit bit-field declared `uint64_t` is typed differently by GCC and Clang, so neither rail can be
    held to one compiler there.

  CI-ARM (2026-10-01): on CF-UNARY's commit (`2a76a01e`) the native AArch64 job failed two tests that passed on every
  x86-64 job -- the fixture's emits run against the original
  (`test_unary_operators_controlling_expressions_and_void_values_run_as_the_original`) and the shared corpus's
  equivalence run (`test_python_c_parity_and_equivalence_across_fixtures_g0`). The witness was undefined, not the
  emit: `uo_parts` converted a negative `double` to `uint32_t` once `s` passed 500 (C11 6.3.1.4p1), a conversion
  x86-64 wraps and AArch64 saturates, so on AArch64 the original and the emit computed one undefined value two ways.
  - Reproduced on x86-64 under UBSan (`-fsanitize=undefined,float-cast-overflow -fno-sanitize-recover=all`): `-130070
    is outside the range of representable values of type 'unsigned int'`, in the original's `uo_parts`.
  - The sweep: each test that runs an emit against the original, its builds under Clang and GCC made that way, and
    then each unit of a test that stops at its first failing unit one at a time. Five more of this branch's witnesses
    ran undefined behaviour no platform here exposed: `cfront_idxarrow.c` summed what `ix_steps` and `ix_globals`
    read through its pointers as `int32_t` after a function stored `s` there (signed overflow, 6.5p5); of
    `_CARD5_LOWERED`, one unit shifted a promoted byte left by 24 (`255 << 24` is no `int`, 6.5.7p4), two converted
    a negative complex part to `uint32_t` as `uo_parts` did, and one converted `s * 2.5` past `UINT32_MAX`.
  - What landed: `uo_parts` and the two part units read a byte of `s`, which keeps each part they convert in range;
    `cfront_idxarrow.c` sums what it reads as `uint32_t`; the shift is by 23, beside a comparison of `+c - 256` with
    0 that still tells an `int` from an `unsigned` operand; the scale is 0.5. RED holds unchanged on each parent: on
    `601ca6f0` both emits of `uo_parts` still return what the original does not and `_CARD5_REFUSED` and
    `_CARD5_LOWERED` count as above; on `b27c37e1` the twin still refuses `cfront_idxarrow.c` for the same reason.
    Under the sweep every witness this branch added runs clean.

  CF-CHARELEM, CF-STRELEM, CF-AOS2D, CF-DEREFSUM, CF-PARENSTR, CF-SUFFIX and CF-LINKEMIT (2026-10-01) closed the five
  file-scope items CF-GLOBALS found and CF-EXTDESIG's linkable refusal of a function named in an initializer, and the
  defects found closing them. RED was measured on the parent (`4f55b1a5`).
  - The defects: the twin declared a plain `char` element's temp, and `(char)v`'s, `int8_t` -- right where `char` is
    signed, and reading an element past 0x7F back negative where it is not (AArch64 Linux) -- and read a string
    literal's element as an `unsigned char` (`*"\xf0"` was 240 where `char` reads -16, a wide literal's element one
    byte of it); the oracle typed an `L` literal's element an unsigned integer of its width, where `wchar_t` is `int`
    on x86-64, i386 and RISC-V Linux. Both rails typed a concatenated literal by its first piece's prefix (`"a" U"b"`
    was a `char[3]`, where C11 6.4.5p5 makes it a `char32_t[3]`) and took the pieces of two encodings (`u"a" U"b"`)
    that Clang refuses. A member of an element of a 2-D or 3-D file-scope array of structs or unions, `gm[i][j].x`
    -- read, stored, compounded, stepped, addressed -- and an element of a 2-D file-scope table of pointers were
    refused on both rails (`a subscript of an element that is not a pointer`). The twin read `*(p + i - j)` as
    `p[i - j]`, the index computed in the unsigned type of `i - j`, which wraps where `p + i - j` steps back -- a
    silent miscompile -- and the oracle refused `*("abc" + 1 + i)` (`unsupported base expression Binary`). The twin
    refused `char s[] = ("abc");` at file scope, at block scope and `static` (`an array is initialized by a brace list
    or a string literal`), which the oracle lowered. Both rails took any run of `u`s and `l`s as a suffix -- `1lL`,
    `1uu`, `1lul`; `1lll` raised a bare `KeyError` in the oracle -- in code and in a `#if`, where the twin read its
    constants with `strtol` (`0777` was 777), and the twin refused a designator naming no member without naming it.
    The oracle's linkable emit (`--linkable`) named the unit's structs, unions and typedefs without defining them,
    spelled a typedef'd anonymous struct `struct $anon0`, dropped `const` from a global (`const uint32_t k = 5u;`
    rendered `uint32_t k = 5;`, another type than the `extern const uint32_t k;` another unit declares, C11 6.2.7p2),
    refused a pointer's string literal (`const char *tab[] = {"a"};`), `&g[k]`, a decayed array and a function in an
    initializer, declared its static functions only after its globals, and called memcpy, `va_start`,
    `atomic_fetch_add` and `creal` undeclared: of the 236 fixtures' linkable emits, 94 built alone under Clang and
    GCC, 63 of them under `-Wall -Werror`.
  - Found by running the emits against the original, and closed: both default emits read a `const` global into a temp
    of their own, unqualified, type -- `char *t = names[i];` of `const char *names[2]`, `uint32_t *p = gk;` of `const
    uint32_t gk[3]` -- which discards a qualifier: a constraint violation (6.5.16.1p1) that Clang refuses under
    `-Werror=incompatible-pointer-types` and GCC warns about.
  - RED: on the parent the four new tests fail, as does
    `test_linkable_static_forward_declaration_and_pointer_string_table`, which pinned the string table's refusal.
    Both rails refuse `runtime/c/cfront_filescope.c`: the oracle at `*("abc" + 1 + (s & 1u))` (`unsupported base
    expression Binary`), the twin at its first parenthesized string and, past it, at `gm[i][j]`. Of the 16 units
    `_CARD4_REFUSED` holds, the parent oracle lowers 12 and raises a bare `KeyError` on 2, and the twin lowers 14 and
    refuses 2 without naming the member; of the 13 `_CARD4_LOWERED` holds, 9 fail -- 4 (a member of a 2-D array of
    structs) refused on both rails, `*("abc" + 1 + i)` by the oracle and `char t[] = ("xyz")` by the twin, `#if 0777 ==
    511` and `*(q[i] + 1u - 1u)` on two claim graphs, and `("a" L"b")[i]` with a twin emit that returns what the
    original does not. The
    other 4 -- every suffix C spells, in code and in a `#if`, designators out of order, and a `const` global under
    `_Generic` -- lower alike there too: they hold what the new refusals and records must leave alone.
  - What landed: the twin's temp of a plain `char` element, and of `(char)v`, is a `char` (`elem_temp`,
    `emit_cast`). One reading of a literal's prefix on each rail (`clex.lit_prefix`, the twin's `str_prefix`): the
    one prefix its pieces carry, a piece without one taking the others', two different ones refused
    (`string literals with different encoding prefixes are concatenated`); the element's type follows it
    (`lower._LIT_ELEM`, `str_elem_type`: `char`, `char16_t`, `char32_t`, the target's `wchar_t`), read by `"ab"[i]`
    and `*("ab" + i)` alike (`lit_venv`, `deref_lit_index`). The oracle's `_addr` takes the subscripts of a 2-D or
    3-D file-scope array of structs, unions or pointers as the element `i*B + j` of a flat array, as a local's
    (`nest_shape`, `nest_elem`); the twin binds such a global with its dimensions (`global_md_elems`); each emit
    indexes it from its first element, `(&g[0][0])[i]`, bounded by the whole array (`emit._elem_base`, `elem_base`).
    The twin reads `*(E + i)` as `E[i]` only when `i` is the whole right operand of that `+` (`deref_sum_index`, the
    operand `+` takes at its precedence), and past it a pointer value, dereferenced (`deref_paren_fast`,
    `deref_rvalue_assign_value`, `deref_incdec`); the oracle bases no subscript on a sum's inner sum (`_lvalue`). The
    twin drops a string literal's redundant parentheses where an initializer or an argument begins (`str_unparen`).
    One reader of an integer constant per rail, shared by its lexer and its `#if` -- `clex.int_literal_parts`, which
    `cpp._int_lit` now reads through, and the twin's header-only `bcir_intlit.h`, which `bcir_cpp.c`'s `lit` now
    reads through in place of `strtol` -- takes the suffix C spells (a `u` before or after an `l`, `L`, `ll` or `LL`)
    and refuses any other run in Clang's words (`invalid suffix 'lL' on integer constant`); the twin names the member a
    designator does not find (`no member named 'z' to designate`). The oracle's parser records the unit's file-scope
    type definitions as their tokens spell them (`Unit.type_defs`, `cparse._spelled`: a struct, union or enum
    definition, a tag's forward declaration, a typedef), which the linkable emit defines first, in source order --
    copied, never re-rendered from a layout, so a `packed` or `aligned` attribute stays; each global keeps its
    qualifiers level by level (`lower._object_quals`, `emit._object_declarator`); a typedef'd anonymous aggregate is
    spelled by its name and a global of an untagged one no typedef names refused by name; every function is declared
    before the globals (`_proto_param`); `_spell_init` renders a pointer's string literal, `&g[k]`, a decayed array and
    a function designator as the address constants they are (C11 6.6p9); the headers follow what the artifact's own
    text names outside its literals and comments (`_code_only`: `<stddef.h>`, `<string.h>` for its own memcpy,
    `<stdarg.h>`, `<stdatomic.h>`, `<complex.h>`). Both default emits name a global `const` at any level through an
    lvalue of its unqualified type, `(*(char * (*)[2])&names)`, as a call meets a qualified parameter
    (`emit._unqualified_global` over `LoweredFunc.global_quals`; the twin's `qglobal_add`, recorded per function in
    `bcir_func.qglobals` and rolled back with a speculative lowering); a volatile, `_Atomic` or function-pointer
    global keeps its name, and a bounds guard's label the plain one.
  - Two earlier tests pinned what this changes: `test_linkable_static_forward_declaration_and_pointer_string_table`
    asserted the string table's refusal and now asserts its rendering, built under `-Wall -Werror`; and
    `test_register_driver_composes_register_map_surface` looked for `QUANTA[` in the twin's emit, which reads the
    `const` table through its cast now (`&QUANTA)[`). The globals' bytes check
    (`_linkable_bytes_are_the_originals`) no longer builds the emit after the source's struct definitions.
  - Outcomes: `runtime/c/cfront_filescope.c` -- `char` elements and casts, string literal elements, `gm[i][j].x` of
    2-D and 3-D arrays of structs and unions read, stored, stepped, copied, addressed and passed, 2-D tables of
    pointers, `*(p + i - j)`, parenthesized string initializers and `const` globals of every level -- lowers to one
    claim graph on the four targets, and both emits run as the original under Clang and GCC with a signed and with an
    unsigned `char`, the mixes `_QUALS_WERROR` names made errors, and clean under UBSan; its linkable emit builds alone
    under `-Wall -Werror` (Clang's `-Wstring-plus-int` aside, which names the source's own `"ab" + 1`) and, linked
    with a driver, prints what the original prints. `_CARD4_REFUSED` (16) is refused on both rails for the reason each
    names; `_CARD4_LOWERED` (13) lowers alike, each emit the original. Each of `_LINKEMIT_UNITS`' 15 linkable emits
    builds alone under `-Wall -Werror`, builds with another unit's declarations of what it defines after it, and
    prints what the original prints, under Clang and GCC. Of the same 236 fixtures, the linkable emit now builds
    alone for 218, 148 under `-Wall -Werror`, and every emit that built on the parent still builds. Beyond the
    tests, 75 probe forms agree on both rails: 47 lower alike, each emit run as the original under Clang and GCC
    with either `char`; 5 more lower alike; 21 are refused in the same words and 2 for reasons of their own
    (below). Every other unit of the corpus (236) keeps the parent's summary and digest on both rails. The round
    trip's count stays 123 (the fixture's emit names its file-scope objects, which the standalone re-parse never
    declares); the fixture's local arrays are only indexed, so every G10 row stays at its bound.
  - Faults: `tools/testing/faults/cfront-filescope.json`, 26 -- 14 on the oracle, 12 on the twin -- each caught by
    its own test. The first sweep caught 25: FS19, the twin's `(char)v` typed `int8_t`, reached no test, as the
    fixture stored the cast only into a `char` local, which converts it back; two uses of the cast's own value
    joined `fs_chars`, and the re-sweep caught FS19, and FS18 again. Five older faults anchored in code this card
    moved, which the anchor test (`test_red_sweep`) caught before the gates ran: `cfront-gaps.json` T12 (an `L`
    literal's element size, now read through `str_prefix`), `cfront-globals.json` IC1 and IC4 (the twin's constant
    reader, now `bcir_intlit.h`, which its `#if` shares), `cfront-roundtrip.json` O38 (the linkable emit's
    `<string.h>`, now included for its own memcpy too) and `volatile.json`'s `*q[j]` fault (the name read under `*`,
    now after `deref_paren_fast`) were re-anchored to the new code, and each re-swept caught by its own check.
  Found, not fixed here (each a suggested follow-up):
  - an object of an enumerated type none of whose enumerators is negative is `unsigned int` under GCC and Clang (C11
    6.7.2.2p4 leaves the type to the implementation) and `int` on both rails: `c - 5 < 0` and `_Generic(c - 5, ...)`
    of `enum col { RED, GREEN = 4 } c` differ from the original, digest-equal on both rails (a silent miscompile),
    and the linkable emit defines an `enum col` global as `int`, which another unit's `extern enum col g;` conflicts
    with (closed by CF-ENUMOBJ, below);
  - `#if` evaluates outside C's arithmetic on both rails: `#if -1 > 0u` and `#if 'a' == 97` take the `#else` branch
    on both, digest-equal (a silent miscompile; C reads -1 as `UINTMAX_MAX` there, and `'a'` as 97), and `#if
    0xFFFFFFFFFFFFFFFF == -1` raises a bare `ValueError` in the oracle where the twin refuses it as an overflow
    (closed by CF-PPARITH, below);
  - the linkable emit's definitions drop a parameter's qualifiers below its top level, so a function pointer of the
    source's type takes such a function only through a cast (`cfront_quals_link.c` under Clang: `incompatible function
    pointer types assigning to 'uint32_t (*)(const uint32_t *)'`); with the four `_BitInt` fixtures GCC 13 cannot
    build, it is the one fixture that lowers whose linkable emit does not build alone;
  - of the 219 linkable emits that build alone, 71 draw a `-Wall` warning: a loop's `continue` label its body never
    takes (46), a temp declared and never read (17), a bit-field's storage unit read before its first store (4), and
    four the source's own spelling carries;
  - the twin's `--linkable` emits the unit's functions alone, without its type definitions and globals;
  - a `u8` string literal's element is `char` on both rails, as Clang 18 reads it under C23, where GCC 13 reads
    `char8_t` (`unsigned char`) under `-std=c2x`, so the two compilers disagree about such an element's sign;
  - a subscript of a pointer member of an element of an array of structs (`gt[i].name[1]`), a pointer to an array of
    structs (`struct pt (*q)[3] = gm;`), a member through the address of an element (`(&gm[0][1])->y`), a bit-field of
    a 2-D array of structs and an element of a string literal plus an offset (`("ab" + 1)[i]`) are refused on both
    rails, each for reasons of its own.

  CF-ENUMOBJ (2026-10-02) types an object of an enumerated type by the integer type its enumeration is compatible
  with, closing card 4's first found-not-fixed item. RED was measured on the parent (`83e53595`).
  - The defect: C11 6.7.2.2p4 leaves an enumerated type's compatible integer type to the implementation. GCC and Clang
    make one with no negative enumerator `unsigned int` on the System V targets (x86-64, AArch64 and i386 Linux, and
    MinGW) and one with a negative enumerator `int`; the MSVC ABI makes every one `int` -- each measured by Clang's own
    fold for the target. Both rails typed every enum object `int` (the oracle mapped `enum [tag]` to `int`, the twin
    to a signed 4-byte scalar): `(c - 5) < 0` of `enum col { RED, GREEN = 4 } c` was 1 where Clang and GCC give 0,
    `_Generic(c - 5, ...)` chose `int`, `(int64_t)c` sign-extended, `c -= 1` of `RED` read back as -1 and a 3-bit
    bit-field of the type read 4 as -4 -- the same claim graph on both rails, so parity never saw it (a silent
    miscompile). The oracle's linkable emit defined such a global `int`, which another unit's `extern enum col g;`
    conflicts with (6.2.7p2). Both rails also read `enum nope` -- a tag no definition gave, or one used before its
    enumerator list -- as `int`, where C requires the list first (6.7.2.3p3).
  - RED: on the parent the four new tests fail. Of the fixture's ten driven functions, eight return what the original
    does not on both rails (`eo_switch` and `eo_signed`, which `int` and `unsigned int` answer alike, hold); each of
    the eight units of `_ENUMOBJ_LOWERED` returns another value; the six of `_ENUMOBJ_REFUSED` lower on both rails; of
    the eight units of `_ENUMOBJ_TARGET_UNITS`, six fold otherwise than Clang on each System V target on both rails (36
    of the 64 rail-target-unit triples; the enumeration with a negative enumerator and the enumeration constant hold,
    and every unit holds on the MSVC target); and the fixture's linkable emit redeclared `extern enum col gcol;` does
    not compile (`conflicting types`).
  - What landed: an ABI field per rail, `TargetABI.enum_unsigned` and `bcir_abi.enum_unsigned`, false on the MSVC
    target. The enumerator list decides the type (`cparse._enum_body`, the twin's `p_enum_body`: `unsigned int` where
    no enumerator is negative on a target that makes it so, else `int`) and records it for its tag (`enum_tags`; on
    the twin a tag entry in the enumerator table, so a speculative parse that rolls the enumerators back rolls the tag
    back with them, and the enumerator and tag lookups each skip the other's entries -- tags are a name space of their
    own, 6.2.3p1). One reader of an `enum` specifier per rail (`_enum_spec`, `enum_spec`) serves a declaration, a
    typedef, a cast, `sizeof`, a parameter and `_Generic`, and refuses a tag no definition gave (`an enumerated type
    with no definition`). The enumeration constants stay `int` (6.4.4.3).
  - Outcomes: `runtime/c/cfront_enumobj.c` -- enum objects as locals, a static and an external global, a parameter, a
    return, members, a 3-bit bit-field and typedef names of a tagged and an untagged enumeration, read by comparisons,
    widening conversions, `_Generic`, division, shifts, compound assignment and a switch, beside an object and an
    enumerator named as tags -- lowers to one claim graph on the four targets, and both emits run as the original under
    Clang and GCC. Each unit of `_ENUMOBJ_TARGET_UNITS`, lowered by both rails for each target, folds under Clang for
    that target to the original's constant, so the MSVC rule holds where nothing here can run it. `_ENUMOBJ_REFUSED`
    (6) is refused on both rails in the same words; `_ENUMOBJ_LOWERED` (8) lowers alike, each emit the original. The
    linkable emit declares an enum global by its compatible type: the fixture's, redeclared `extern enum col gcol;`
    and `extern enum sg gsg;`, builds under Clang and GCC. Every other unit of the corpus keeps the parent's summary
    and digest on both rails on the four targets: none reads an enum object where the two types part.
  - Faults: `tools/testing/faults/cfront-enumobj.json`, 18 -- 8 on the oracle, 10 on the twin -- each caught by its
    own test. EO8 and ET8 injected together -- both rails reading the MSVC enumeration as `unsigned int` -- pass every
    test but the cross-target fold, parity included: a shared misreading is invisible to parity, so the fold against
    Clang is the witness that hits the MSVC rule (L11). Two older faults anchored in lines this changed were
    re-anchored and caught again: `cfront-consts.json` EF17 (the enumerator range check, whose body now returns the
    type's signedness) and `cfront-gaps.json` T11 (the twin's AArch64 target row, which carries `enum_unsigned` now).
  Found, not fixed here: a pointer to an enumerated type before its definition (`enum later *p;`, which C allows while
  the type is incomplete) is refused on both rails with the rest.

  CF-PPARITH (2026-10-02) evaluates `#if` and `#elif` in C's integer arithmetic (C11 6.10.1p4) on both rails, closing
  card 4's second found-not-fixed item. RED was measured on the parent (`a8d3ef07`).
  - The defect: both preprocessors evaluated in a signed host integer -- Python's unbounded `int`, the twin's `long`
    -- so no operand was ever unsigned: `#if -1 > 0u` took `#else` on both rails, digest-equal (a silent miscompile).
    A character constant was 0 on both (`#if 'a' == 97` took `#else`). The oracle's `_apply` built every operator's
    result in one dictionary, so any operator whose right operand was negative computed `a << -1` and raised a bare
    `ValueError` (`0xFFFFFFFFFFFFFFFF == -1`, `-7 % 2 == -1`); the twin refused every constant past `LONG_MAX`. The
    twin took neither `?:` nor a unary `+` (`#if 1 ? 0 : 1` was 1); both ignored a token past the expression (`#if
    1 2`), an unclosed `(` and a `?` with no `:`; the oracle read a string or an `=` as 0, divided by zero to 0 and
    bounded neither a `#if`'s tokens nor its nesting (the twin held 512 tokens of 63 characters). Both tokenizers
    split `L'a'` into the name `L` and a plain constant, so with `L` a macro it became `7'a'`; the twin ended `1'000`
    at its digit separator and read a character constant from there.
  - Measured (Clang 18 and GCC 13, `-std=c2x`, `-E` per target): a character constant in `#if` is a `uintmax_t`
    where plain `char` is unsigned -- AArch64 and RISC-V Linux; `'a' - 98 < 0` is false there -- `u8`, `u` and `U`
    ones are unsigned, an `L` one is `wchar_t`'s signedness and width, and a multi-character one packs big-endian
    into 32 bits extended by `char`'s signedness. GCC reads that last one as a signed `int` whatever `char` is, so
    for it the two compilers part; both rails read Clang's.
  - RED: on the parent the four new tests fail (two at the target the preprocessor now takes as a parameter). Measured
    with the parent's own interface: of the 46 forms of `_PPARITH_FORMS`, 30 keep another branch than Clang's on some
    target -- 113 of the 184 form-target pairs, 101 on the oracle and 92 on the twin, 48 with the rails apart -- and
    the oracle raises a bare `ValueError` on 9; 36 of the 37 forms of `_PPARITH_REFUSED` are not refused alike (the
    oracle reads 140 of the 148 form-target pairs, the twin 76); the first unit of `_PPARITH_RUN_EXPRS` returns what
    the original does not, and `_PPARITH_TOKENS` preprocesses otherwise.
  - What landed: one evaluator per rail (`_ConstEval` and `_apply`; the twin's `ce_*` and `apply`): each operand its
    64 bits and whether it is a `uintmax_t`; an integer constant unsigned where its suffix has `u` or its value is
    past `INTMAX_MAX`, read by the lexer's reader (a constant no type holds refused with its words); the usual
    arithmetic conversions, a shift typed by its left operand, `!`, comparisons, `&&` and `||` an `int`, `?:`
    unsigned where either arm is; `true` 1 and any other identifier 0 (C23). A `live` flag carries what C evaluates,
    so the right of `&&` after 0, of `||` after a nonzero and the arm of `?:` not taken refuse for none of the
    operators' reasons: signed overflow, a negative value shifted left, `INTMAX_MIN / -1` and `% -1` (`integer
    overflow in #if expression`), a shift by a negative count or by 64 or more (`invalid shift ...`), a division by
    zero, an evaluated comma operator (6.6p3); a form the grammar does not admit is `malformed #if expression`
    wherever it stands. Both rails hold a `#if` to 512 tokens of 63 characters, nested 63 deep (C11 5.2.4.1), and
    unary operators apply as a run, so no form takes stack in proportion to its length. One reader of a character
    constant per rail (`clex.char_constant_units`; `char_literal` in `bcir_intlit.h`, which the lexer's integer
    reader already shared) takes ASCII source characters and simple, octal and hexadecimal escapes within the code
    unit, a plain one up to 4 characters' worth, a prefixed one one, and refuses the rest as `unsupported character
    constant`. The target reaches the preprocessor: `TargetABI.char_signed` and `bcir_abi.char_signed` (false on
    AArch64 and RISC-V), `preprocess(abi=...)` from `compile_unit` and the CLI's `-E` and `-M`;
    `bcir_cpp_context_set_chars` and `bcir_cpp_set_chars` on the twin, fed from `bcir_cfront_target_chars` by its
    two drivers. A prefixed character constant is one token on both rails, a word that may be a prefix is kept apart
    from a constant after it, and the twin's numbers keep C23's separators, as the oracle's did.
  - Outcomes: every form of `_PPARITH_FORMS` keeps Clang's branch on the four targets on both rails, and GCC's on the
    host for the host's own ABI with either `char` (but the multi-character form with an unsigned one); the 37 forms
    of `_PPARITH_REFUSED` are refused on both rails in the same words on every target, and a nesting one short of the
    bound and a token count at it are read; the 9 units of `_PPARITH_RUN_EXPRS` and `_PPARITH_RUN_ELIF`, lowered by
    both rails for x86-64 and for AArch64 Linux to one claim graph, each return what the original does, built with
    that target's `char` under Clang and GCC. Every unit of the corpus keeps the parent's summary and digest on both
    rails on the four targets: none reads `#if` where the two arithmetics part.
  - Faults: `tools/testing/faults/cfront-pparith.json`, 47 -- 24 on the oracle, 23 on the twin -- each caught by its
    own test. The first sweep caught 46: the twin dropping its check of a negative value shifted left (PT21) reached
    no test, for its range check refuses `-1 << 1` too -- only a count of 0 tells the two apart -- so `-1 << 0`, as
    undefined as any (6.5.7p4), joined `_PPARITH_REFUSED`, and the re-sweep caught PT21 and PO23 with it. Five
    older faults anchored in lines this changed were re-anchored and caught again: `cfront-buf.json` B18 (the names
    a `#if` looks up, now bounded with the evaluator's token checks), `cfront-filescope.json` FS2 and FS16 (the `#if`
    constant readers), `cfront-enumobj.json` ET8 and `cfront-gaps.json` T11 (the target rows, which carry
    `char_signed` now).
  Found, not fixed here: the twin's `#` stringize copies a string-literal or character-constant argument as it is,
  where C escapes its `"` and `\` (the oracle does): `S("q")` is `""q""` on the twin; and a prefixed string literal is
  still two tokens on both rails, so a macro named `L` expands the prefix of `L"ab"`.
  CF-PPARITH.1 (2026-10-02): on `de096434` the native AArch64 oracle job failed one test the x86-64 jobs passed,
  `test_if_expressions_take_the_branch_clang_takes_on_each_target`. The rails were right; the test held the host's
  GCC, which preprocesses for the machine it runs on, to the x86-64 ABI whenever it passed `-fsigned-char` -- and
  AArch64's `wchar_t` is `unsigned int`, so `L'a' - 98 < 0` is false there. The GCC comparison now takes the host's
  own ABI (`platform.machine()`) with the `char` each flag gives it, which also holds the wide forms under
  `-funsigned-char` on x86-64, where the test had skipped them.

  CF-UBGATE (2026-10-02) gates every cfront run-as-the-original witness against undefined behaviour in its original,
  closing CI-ARM's found-not-fixed item. RED was measured on the parent (`5475a6ad`).
  - The defect: a witness compiles the original C beside an emit into one program and compares them; where the
    original runs undefined behaviour, the platform supplies the verdict (CI-ARM's `uo_parts`). CI-ARM's one-off
    sweep, through `_build_run_c` and `host_link_args`, had found five older fixtures whose originals overflow `int`
    under `_equiv`'s full-width random arguments -- `signed_scale` (`x * y - x + y` of two full-width `int`s),
    `ps_unary` (`a + +b - -b + (a - -1)` in `int32_t`), `aostruct_init`, `structcall` and `assign_expr` (sums of a
    full-width `int` parameter) -- and no gate would have said so again, nor of the next one.
  - The gate: `tools/testing/ubsan_witnesses.py` runs the 17 cfront test modules with compiler shims first on `PATH`. A
    shim runs the real compiler as asked -- the witness's own `-O2` build, unchanged -- and, where the build links an
    executable from C sources outside the repository or among its `cfront_*.c` fixtures (the declared scope: the twin
    and the harness drivers belong to `sanitize_cfront.sh`), builds the same sources once more with
    `-fsanitize=undefined,float-cast-overflow -fno-sanitize-recover=all` (Clang's runtime at `-O2`, GCC's unoptimized
    where Clang cannot build the program, or the one `--engine` names; C23 by its draft name where the engine knows it
    so, as GCC 13 does) and puts a launcher at the output: the sanitized program runs first, a report fails the run (exit 125, the report on
    stderr and in the gate's log, naming the test that ran it -- `check_tests.py` exports it -- with a copy of the
    witness's source), else the unsanitized program runs and the test sees what it always saw. Every exit is a verdict:
    a report or a failing test is a finding; a run that sanitized nothing, a shard that could not run its tests or
    `-j 0` is INVALID; no UBSan runtime that reports here -- the probe runs signed overflow and must see the report --
    is an honest skip, and a failure under `--require-ubsan`, which CI's own `cfront-ubsan` job passes because it
    installs the runtime. The shim sits where every witness is built, so it reaches harnesses the sweep's two paths
    never did.
  - The instrument, measured under GCC 13 and Clang 18: Clang's check is a branch to a call of its handler, which no
    optimizer deletes while the branch can be taken -- an overflow folded into a comparison, stored where nothing reads
    it, or discarded as a statement (`x * x;`) reported at every level. GCC's check of signed arithmetic is a pure
    function of its operands, deleted with any value nothing reads at `-O1`, `-Og` and `-O2`: a witness comparing an
    original with an emit that overflows alike folds the comparison and both checks with it (`cfront_signed.c` with its
    overflow restored ran clean under GCC at `-O1` and `-O2`), and even at `-O0` GCC's front end drops a discarded
    statement unchecked. The gate first built every witness at `-O1`, GCC's runtime first, and its own fault sweep
    found six of its thirteen fixture faults (`UB1`-`UB3`, `UB7`, `UB9`, `UB10`) running clean so; built at `-O0`
    instead, one witness's loop of up to 2**31 trips, which `-O2` folds to its closed form, ran past 20 minutes before
    the run was stopped. So Clang's UBSan is the gate's instrument, at `-O2`; GCC's, unoptimized, builds only a
    program Clang cannot; and `--require-ubsan` with no engine named requires Clang's.
  - What the gate found beyond the five, on its first full run (Clang's UBSan): seven witnesses of the suite's own
    harnesses and units. `compile_unit(check_clang=True)` runs a unit against Clang over full-width arguments, and
    three units overflowed `int` there: `incdec`'s `x*100` of `int x=a--`, the signed VLA unit's `i*2-n` and its
    sum, the user-call unit's `g` returning `x*100000`. The integer-promotion test's driver passed `umix` a 35-bit
    `long`, so `a * b` overflowed `long`. `cfront_compoundwide.c`'s `l_array` and `l_ptr`, and
    `cfront_atomiclocal.c`'s `a_long`, shifted a negative `long` left under their tests' drivers (Clang's UBSan cannot
    even print the `_Atomic long` one: its runtime fails its own `CHECK`). And the differential's fairness
    fingerprint, `_original_xcc_verdict`, converted each complex part to `int64_t`, which for
    `cfront_complextrans.c`, `cfront_complexlong.c` and `cfront_complexmember.c` is infinite or past `INT64_MAX` --
    undefined in the harness itself, so the fixture was charged `xcc-divergent` for what its fingerprint did. Six of
    the seven passed on every platform CI runs; the seventh only excluded three fixtures from the differential.
  - RED: on the parent (`5475a6ad`), this gate's tool alone copied in, the whole cfront suite under Clang's UBSan
    sanitizes 2 672 witnesses and fails with 32 UBSan reports in 12 failing tests: the five fixtures and the seven
    witnesses above, each reported where its original overflows, shifts or converts.
  - What landed: each witness keeps what it tests. `signed_scale` still multiplies negative and positive signed
    operands, read from the low 16 bits of each argument and biased into `[-32768, 32767]`, so no product overflows;
    `ps_unary`, whose spelling (`a + +b - -b`) is the point, computes in `uint32_t`; `aostruct_init`, `structcall`
    and `assign_expr` keep their parameters to 16 bits at entry. The three units reduce their operand at entry (`a %=
    1000`, `n %= 100000`, `x % 20000`), negative values kept; the promotion driver passes a 30-bit `long`; the two
    fixtures shift right, an `OP=` on a wide lvalue still (implementation-defined for a negative value, arithmetic on
    both compilers, never undefined); the fingerprint saturates a complex part to `int64_t` and reads a NaN as 0.
    GREEN: the 17 modules under the gate build all 2 689 witnesses under Clang's UBSan -- none fell back to GCC's --
    and each runs clean: 0 reports, 0 unsanitized, 0 failing tests, in 810 s at two workers.
  - Faults: `tools/testing/faults/cfront-ubgate.json`, 5 -- each fixture made undefined again, each caught by the gate's
    report over the three equivalence groups that hold the five; `cfront-ubtests.json`, 8 -- each of the seven again
    (`cfront_compoundwide.c` twice), over the six tests and one differential group; and `ubsan-witnesses.json`, 22, the
    gate's own: a build from the runtime's sources or an object build taken for a witness, a sanitized twin that keeps
    `-O2` or loses `float-cast-overflow`, GCC's built at `-O1`, Clang's unoptimized, an engine the gate does not know
    built optimized, GCC's runtime tried first, a launcher that reads no report, never runs the sanitized program or
    runs a reported one on, an engine that failed taken as having built, a probe that takes a mute engine,
    `--require-ubsan` passing an absent runtime or GCC's alone, a run that sanitized nothing or a shard that could not
    run passing, the CI step dropping `--require-ubsan`, C23 not tried by its draft name, a report that keeps no source
    or names no test, an unsanitized witness's reason read from its diagnostic's caret line; each caught by its own
    test. Swept on the final tree: 5 of 5, 8 of 8 and 22 of 22 caught. The first sweep, of the `-O1` gate with GCC's
    runtime first, had caught 2 of the 5 and 5 of the 8 -- `UB1`-`UB3`, `UB7`, `UB9` and `UB10` ran clean, each an
    original and an emit overflowing alike, the comparison folded -- and 15 of its own 16, `UW14` never reached because
    the table's command named no C23 test; the table names every test of the module now. A later sweep stalled on one
    fault: with the Clang requirement removed, a test that named no test to run ran the whole cfront suite under GCC's
    unoptimized UBSan, for hours. Every driver call in the tests names one helper now, so a gate that no longer refuses
    runs that helper alone.

  CF-CONSTEXPR2 (2026-10-02) closes the six items CF-ENUMFOLD found and did not fix, on both rails. RED was measured
  on the parent (`b684841e`).
  - The defect: both rails took two case labels equal in the switch's promoted type (`case -1:` beside `case
    4294967295u:` on a `uint32_t`) and a second `default:`, an emit no compiler takes. Both read every character
    constant as an `int` of signed bytes: `'\xff'` was -1 on AArch64, where plain `char` is unsigned and C gives 255;
    `u8'a'`, `u'a'` and `U'a'` were `int`s, so `U'a' - 98 < 0` was 1 where C gives 0; `L'a'` was 4 bytes on Windows;
    and a constant C does not define (`'\q'`, `''`, `'\x100'`) was read as some value. Both refused an enumeration
    declared in a block, for different reasons, an enumerator as a compound literal's or a row pointer's dimension,
    and `sizeof`, `_Alignof` and a cast of a floating constant in an enumerator (6.6p6 allows each); a bit-field's
    width, `_BitInt(N)` and `aligned(N)` were read as a literal only -- the oracle refused `uint32_t a : W;` and the
    twin laid it out with a width of 0, a member as wide as its type -- and a width past its type (`uint8_t a : 9`)
    was laid out on both. The twin refused `typedef T row[0];` and laid a `uint32_t t[0]` member out as one element
    (`sizeof` 8 for Clang's 4). A 3000-term `1 + 1 + ...` raised `RecursionError` in the oracle -- folding it, scanning
    the body for mutations, lowering, typing and sizing it, and judging an allocation's count pure. And both comment
    strippers took the `'` of `u8'a'` (and `case'a'`) for a C23 digit separator, a `'` between two hex digits, so the
    constant's closing quote opened a literal that ran past the next comment, which survived as the tokens `/ *`.
  - Measured (Clang 18, `-std=c2x`, per target): `'\xff'` is 255 on AArch64 and -1 elsewhere; `L'\xff'` is 255
    everywhere; `L'\xffffffff'` is -1 on x86-64 and i386 Linux, 4294967295 on AArch64, and refused on Windows, whose
    16-bit `wchar_t` the escape is past; `sizeof(L'a')` is 2 on Windows; `u8` is an `unsigned char`, `u` a
    `char16_t`, `U` a `char32_t`; `(int)16777217.0f` is 16777216, the `float` rounding it first; an unsigned
    `_BitInt(12)` switch takes `case -1:` as its 4095 (C23 promotes no `_BitInt`).
  - RED: on the parent the seven new tests fail (7 run, 7 findings): the fixture is refused by both rails (`expected a
    type`), the oracle refuses the per-target unit's `sizeof` (`not an integer constant expression`) and reads
    `L'\xffffffff' < 0` as 1 on AArch64, lowers the duplicate labels, refuses `_BitInt(W)`, and raises
    `RecursionError` on the chain; the comment stripper keeps `/ *a comment's quote */` after `u8'a'`.
  - What landed: case labels converted to the switch's promoted type (6.8.4.2p5: an `int` for a narrower type or
    `_Bool`, a `_BitInt(N)` itself) and compared there, per switch, a nested one apart: `duplicate case value`, and
    `multiple default labels in one switch` (`lower._switch_promoted`; the twin's `switch_promoted`, `case_label` and
    `p_switch`). One reader of a character constant per rail, the one `#if` reads through (`clex.char_constant` over
    `char_constant_units`; the twin's `char_value` over `char_literal`): a plain one an `int` of its byte by the
    target's `char`, a multi-character one its last four bytes packed big-endian, a prefixed one of its prefix's type
    (`u8` `unsigned char`, `u` `char16_t`, `U` `char32_t`, `L` the target's `wchar_t`, signed or not; the twin's
    `lit_int_type`), one C does not define refused (`unsupported character constant`). An enumeration declared in a
    block: its constants and tag scoped to the block (6.2.1p4), the innermost declaration of a name the one it
    denotes -- the oracle's scopes map a name to its enumerator's value or to `None` for an object; the twin orders
    its locals and enumerators by one sequence (`venv.seq`, `econst.seq`) and restores its constants, tags included,
    at each block's and body's end -- and `enum [tag] { ... };` a declaration of its own. One integer constant
    expression reader where C takes one (`cparse._const_int`, `_const_dim`, `_bit_width`; the twin's `ce_int`,
    `ce_dim`, `bf_width`): a bit-field's width (negative, named and zero, or past its type: `invalid bit-field
    width`), `_BitInt(N)`, `aligned(N)`, `alignas(N)`, a compound literal's and a row pointer's dimension, the
    dimension's range checked by one predicate (`_dim_in_range`). `sizeof` and `_Alignof` of a type-name fold, the
    oracle's parser laying out the unit's aggregates as far as it has read (`lower.layout_aggregates`, shared with
    `lower_unit`; the twin's `sizeof_type_name`, shared by `p_sizeof` and `ce_sizeof`), as does `sizeof` of a
    constant (`sizeof(L'a')`); one of an incomplete type, or of an expression, is no constant. A floating constant
    cast to an integer type folds as C converts it (6.3.1.4p1): read by its own grammar and rounded to its type,
    ties to even (`_float_value`, an exact `Fraction`; `strtof`/`strtod` on the twin), truncated toward zero, refused
    where its type cannot hold it and C evaluates it; `_Bool` is whether it is nonzero; a `long double` one, whose
    format is the target's, is refused. A zero-length or flexible member is an array of no bytes on the twin (`farr`,
    at the 22 sites that read `arr_count` as the test), and `typedef T row[0];` is taken. The oracle walks a binary
    chain down its left operands with a loop where it folds, scans, lowers (`_binary_chain`, `_binary_step`), types,
    sizes and judges it pure. Both comment strippers copy an identifier and a preprocessing number whole (6.4.8), a
    digit separator and an exponent's sign inside the number.
  - Outcomes: `runtime/c/cfront_constexpr2.c` (switches whose labels differ only in the promoted type, prefixed and
    multi-character constants, a block enumeration hiding and hidden in turn, enumerators and `sizeof` as widths and
    dimensions, `sizeof`/`_Alignof` of type-names and floating casts in enumerators, zero-length and flexible
    members) lowers to one claim graph on the four targets and each function of each emit returns what the original
    does under Clang and GCC; the 38 enumerators of `_CONSTEXPR2_TARGET` are Clang's on each target and fold to one
    claim graph on both rails, as `L'\xffffffff'` does (refused alike on Windows); the 29 units of
    `_CONSTEXPR2_REFUSED` are refused on both rails for one reason each, the 7 of `_CONSTEXPR2_TAKEN` lower alike on
    the four targets, and nine 3000-term chains lower and run as the original on both. Over the corpus (239
    fixtures, four targets) only the new fixture moved: 887 of 956 fixture-target pairs keep equal summaries on both
    rails, the parent's 883 and the new fixture's four.
  - Faults: `tools/testing/faults/cfront-constexpr2.json`, 81 -- 40 on the oracle, 41 on the twin -- each caught by
    its own test. The first sweep caught 78: KO10 and KT8, each rail reading an `L` constant unsigned where `wchar_t`
    is signed, reached no test, for `_CONSTEXPR2_WIDE` read the constant only in an enumerator, which wraps it to its
    type whatever its sign -- the unit returns it as a value now (`cw_val`, `cw_half`), held to Clang's; and KO13, a
    block's enumeration tag outliving the block, reached none, for no unit named a tag after its block -- two do now,
    in `_CONSTEXPR2_REFUSED`. The re-sweep caught all three. Twenty-three older faults anchored in lines this changed
    were re-anchored: `cfront-consts.json` EF3-EF5, EF8, EF9, EF13 and EF16 (the fold's new `layouts`, the shared
    dimension check, the twin's `lit_int_type` taking the target), `cfront-enumscope.json` ES1-ES6 and ET1 (the scopes
    as maps, `visible_enum`'s order), `cfront-enumobj.json` ET6 (`find_enum`), `cfront-statics.json` SO3 and ST4 (the
    literal's type), `cfront-calls.json` CL17, `cfront-extdesig.json` XO10 and `cfront-unary.json` UO16, UO24 and
    UT21 (the binary operators, now `_binary_step`, and `p_switch`), and `cfront-roundtrip.json` T62
    (`sizeof_type_name`). All but two were caught again. ET6, the twin reading a tag as an enumerator, passed: its
    witness, an object named as the tag, now meets the object first (`visible_enum`'s order), so a tag read as an
    identifier joins CF-ENUMOBJ's refusal test (`_ENUMOBJ_TAG_ALONE`) and the re-sweep caught it. SO3, the lowering
    not re-typing a `long` constant past a 32-bit `long`, passed for a reason of its own: the parser has typed
    every constant by the target's `long` since CF-ENUMFOLD (`int_literal_type`; `cfront-consts.json` EF2 holds it),
    so the branch it disabled could not run. The branch is gone, one typing per rail as the twin's `lit_int_type`,
    and SO3 is retired.
  Found, not fixed here (each a suggested follow-up):
  - `sizeof` of an expression in an integer constant expression (`enum { N = sizeof g_x };`, an integer constant
    expression by 6.6p6 when the operand is no variable-length array) is refused on both rails;
  - a floating constant with a C23 digit separator (`1e+5'0`) is read by neither rail's lexer, which refuse it in
    different words (`unterminated character constant`, `unsupported character constant`);
  - `sizeof(struct nope)` of an undefined struct outside a constant expression is refused in different words (`the
    incomplete struct or union 'nope' has no layout here`, `unknown struct`);
  - a 5000-term `(10 / x) && ...` the oracle lowers is refused by the twin's driver (`input too large`): its
    preprocessed-text bound (64 KiB) and its 16384 tokens a unit.

  CF-TYPEDEFSCOPE (2026-10-02) makes a block-scope name hide a typedef name on both rails, closing the item CF-ENUMSCOPE
  recorded. RED was measured on the parent (CF-CONSTEXPR2's commit).
  - The defect: both rails read a typedef name as a type wherever it stood, so a local, a parameter, a loop's own
    declaration or a block's enumerator of the same name never hid it (C11 6.2.1p4). A statement starting with the
    name was refused on both (`uint32_t T = s; T = T * 3u;`). Where the name was read in an expression the rails took
    it for the type, digest-equal: `(T) - s` was a cast of `-s` where C subtracts, and `sizeof(B)` of a `uint64_t`
    local beside `typedef uint8_t B` was 1 where C gives 8 -- silent miscompiles. `T * x;` was read as a declaration
    of a pointer `x`, an emit that does not compile, and a declaration through the hidden name (`T y = 3u;`, which C
    refuses) lowered on the oracle.
  - Found while fixing it: both emits declare every local at the top of the function, so a block's local named as a
    file-scope name the function spells after the block captured it there. The oracle kept such a local from a
    parameter's or a used global's name (`g_2`), the twin only from an earlier local's: a block's `uint32_t g` beside
    a global `g` the function read after the block made the twin's emit read the local -- a silent miscompile,
    digest-equal with the oracle. A local named as a typedef (`{ uint32_t S = s; } S v;`) or as a function the
    function calls emitted C that does not compile, on both rails. And a review of the slice before its gates found
    the oracle asking one scope too late: a parameter entered its scopes with the body, so `f(uint32_t T, T x)` -- and
    a prototype's `g(uint32_t T, T x);` -- read `T x` through the typedef there, where the twin, which binds each
    parameter as it reads it, refused it as Clang does. The slice's gates found one more of the kind in a witness: the
    shared corpus's harnesses set the original and its emits beside a generator state named `S`, which the fixture's
    `typedef struct { uint32_t a; } S;` redefined, so the harness did not build; they name it `bcir_h_seed` now
    (`_HARNESS_RNG`).
  - RED: on the parent the three new tests fail (3 run, 3 findings): both rails refuse `cfront_typedefscope.c`, the
    oracle lowers `T y = 3u;` through the hidden name, and neither rail holds a list of the names its emit spells for
    itself.
  - What landed: one visibility predicate per rail -- the oracle's `cparse._typedef_visible`, the twin's
    `visible_typedef` -- a typedef name being a type only where no local, parameter, loop declaration or block
    enumerator of its name is in scope, read where a declaration starts (`_is_decl_start`; `decl_start_tok`), where a
    cast's or `sizeof`'s type-name starts (`_is_cast`; `type_name_tok`) and where a type specifier expands the alias
    (`_type_spec`; `p_type`). The twin's token pre-pass, which runs before a body's locals are bound, keeps the
    typedef names a block's declarations bind to the end of that block (`up_hide`, `uph`), so it rewrites `(T)[1]` of
    a local array as the parser will read it; a for-init's declaration stays a type to it, the direction in which it
    only declines a rewrite. One naming rule per emit: no parameter or hoisted local takes a name the emit spells for
    anything else -- a name it spells for itself (the libc routines it calls or copies through, the C11 atomics, the
    twin's store helper `_v`, the standard type names: `_EMIT_SPELLED`, the twin's `emit_spelled`, one list, which a
    test reads out of both sources), a global or a function it reads, each function it calls as the emit calls it (`X`
    and `bcir_X`), a typedef name -- nor another object's: a second local of a name takes the first of `_2`, `_3`, ...
    that is none of those (the oracle's `_uniq` over `_spelled_names`; the twin's `uniq_local` over `emit_name_taken`,
    `source_named` and `bcir_resource.typedef_named`, which holds the name and its `_2` to `_8`). A parameter is
    renamed alike, and the signature spells it so. The twin reads a function's claims for the callees they name only
    where it calls one by name (`bcir_func.named_calls`), and a candidate's checks only where a name could be it
    (`spelled_scan`): naming 700 locals of one name takes it 0.66 s, the parent's 0.48 s (the rule's first cut,
    checking every candidate in full, took 92 s). The oracle's parameter list is a scope of its own (`_func_body`),
    each parameter in it from the end of its declarator, as the twin's `env` holds it.
  - Outcomes: `runtime/c/cfront_typedefscope.c` (a local, a parameter, a loop's declaration and a block's enumerator
    hiding a typedef in statements, casts' places, `sizeof` and `typeof` operands; a parameter and a local declared
    with the typedef they name, `T T`; a member named as it; the typedef declaring and casting again after; locals
    named as a global read, a global written, a callee and typedefs, and as the emit spells `bcir_td_twice`, `memcpy`,
    `memset`, `fabs` and `size_t`; parameters named `_v` and `memcpy`, each stored whole into a member; a second local
    whose `_2` is a local's, a typedef's or a global's name) lowers to one claim graph on the four targets and each
    function of each emit returns what the original does under Clang and GCC; the 11 units of `_TYPEDEFSCOPE_REFUSED`,
    the hidden name used as a type -- three of them a parameter's, in a definition's list and a prototype's -- are
    refused by Clang and by both parsers; and the two rails' lists of the names their emits spell for themselves are
    one. Over the corpus (240 fixtures, four targets) only the new fixture moved: 891 of 960 fixture-target pairs keep
    equal summaries on both rails, the parent's 887 and the new fixture's four.
  - Faults: `tools/testing/faults/cfront-typedefscope.json`, 30 -- 12 on the oracle, 18 on the twin -- each caught by
    its own test. The first sweep, of the 17 then, caught 15: HO6 and HT10, each dropping a callee from the names a
    hoisted local may not take, passed, for both emits call a unit's function `bcir_f`, which no local of its source
    name meets. Holding the emits to what they spell, not to what the source does, found the rule partial on both
    rails, each a defect of valid C older than the slice: a local or a parameter named `memcpy` -- which spells every
    member store -- or a library routine a block's local outlived (`memset`, `memmove`, `malloc`, `fabs`), or `bcir_g`
    beside a call to `g`, made an emit no compiler takes on both rails, as a block's `size_t` did the oracle's (it
    declares `size_t` temporaries); a parameter or a local `_v` stored whole into a member made the twin's emit store
    its own helper, `{ uint32_t _v = _v; ... }` -- a silent miscompile, digest-equal; and the twin's `x_2` for a
    second `x` redefined a source's own `x_2`, or named a typedef. The fixture's `td_spelled`, `td_helper` and
    `td_suffix` and `td_shadow`'s `td_g_2` hold each, HO5, HO6, HT8, HT9 and HT10 were re-pointed at the new rule and
    thirteen faults added; the re-sweep caught 30 of 30.
  Found, not fixed here: a comma expression as a statement (`a, u = 2u;`) is refused on both rails, in different words
  (`expected ';', got PUNCT ','`, `;`).

  CF-PPLIMITS (2026-10-02) closes the six items CF-LIMITS found and did not fix, on both rails, and the splits fixing
  them found. RED was measured on the parent (CF-TYPEDEFSCOPE's commit).
  - The defects: the loop driver (`test_cfront_loop.c`) lowered the first 64 KiB of a source, silently -- a 70 KB unit
    ran its first function as the entry -- and every twin driver held the preprocessed text in 64 KiB, so a unit of
    6 000 globals was refused there that the oracle lowered. Past that, the twin held 16 384 tokens in a fixed array
    (`input too large`) and the oracle none, whatever the comments on both said. The twin's preprocessor held a token
    in 256 bytes, a macro argument in 1 KiB and a substitution in 2 KiB, and refused a longer one (`preprocessor token
    too long`) where the oracle took it. A malformed `defined` (`#if defined(X`, `#if defined +`) lowered on the
    oracle, whose two regular expressions matched a well-formed one only. The oracle's tokenizer dropped every
    character no alternative matched -- a NUL, `@`, a character past ASCII -- and lowered what surrounded it (`x +
    caf€` as `x + caf`). `bcir-cc -DXQ=1 -UXQ` failed every compile (`macro name must be an identifier`): the `-U`
    turned the `-D` into an empty definition. `bcir_cfront_digest_with_allocator` returned the digest of a canon that
    ran out of memory as an ordinary value. And the twin read the ASCII start of a name where the oracle read a whole
    Unicode word: beside `#define caf 5`, `#ifdef café` kept its group on the twin and skipped it on the oracle,
    digest-different, and the oracle's lexer took `int café;` (`str.isalpha`), which the twin's refused.
  - Found while fixing them: the spacing token the twin's substitution judged a run-on by kept the first 255 bytes of
    an argument, so after a longer one it read the wrong last byte -- `F(...+b)` with the body `t y` wrote `...+by`,
    another identifier, which a parameter `by` made a program C refuses lower on the twin alone. The oracle split a
    macro's parameter list at its commas and took whatever lay between two as a parameter, so `F(a b)`, `F(1)`, `F(a,
    a)`, an unclosed list and seventeen parameters each defined a macro there that the twin refused. It read
    `#line`'s number with a Unicode `\d`, numbering the line after `#line ٣` 3 where the twin, which reads no number
    there, left it 2. It read a directive line with `str.strip`, so ` #define K 3u` defined `K` there where the
    twin refused the no-break space, and ended a directive's name only at a space, so `#define\tK 3u`, `#if\t1` and
    `#if(1)` were unknown directives there; the twin read `#café x` as `#caf`, an unknown directive it ignores.
  - Found measuring the work budget the bound sets: the structural canon -- the digest every summary line carries, and
    `bcir-cc` prints one for each file it compiles -- spelled a value number as its value's whole dataflow tree and
    kept the string, so a chain of n dependent values cost it O(n^2) bytes and a value read twice at each of n steps
    2^n, on both rails: 2 000 chained locals took the twin 551 MB and 11 s, a probe of 8 000 more than 4 GiB, and 18
    doublings -- 500 bytes of source -- 32 MB of canon. The twin sorted a function's records by insertion, the cube of
    a chain. It read a value number's memo before its depth where the oracle reads the depth first, so a value reached
    again one step past the cap of 96 was its string on the twin and `cyc` on the oracle (`*p = t + 1u + ... + 1u` of
    97 terms beside `return t`, digest-different). And it numbered a function's claims from 1000 + 1000 * its index,
    so a function of more than 1000 claims ran into the next one's ids, and R1.1 refused a valid unit
    (`duplicate claim id 2000`, ok=0) that the oracle -- one sequence over the unit -- lowered clean. Each predates
    the slice; the bound it raises took each four times further.
  - RED: on CF-TYPEDEFSCOPE's commit, 11 of the slice's 12 tests fail, each on its defect -- the twin refuses a unit
    of 6 000 globals the oracle lowers, a token past 255 bytes, a source past 64 KiB and a preprocessed text past it
    (`preprocessed output too large`); the oracle lowers a malformed `defined`, `F(a b)` and `#define caf 5`'s
    neighbour `#ifdef café` otherwise than the twin; `bcir-cc -DXQ=1 -UXQ` fails (`macro name must be an identifier`);
    the oracle's canon spells `(a - b) - a`'s value numbers as trees; the twin's digest of `*p = t + 1u + ...` of 97
    terms is not the oracle's; and the twin refuses two functions of 1 800 claims (`duplicate claim id 2000`). The
    twelfth, `test_the_emit_grows_whole_under_a_failing_allocator`, drives the memory-discipline harness of the tree
    it runs in, and the checks the slice adds there -- the digest's status -- hold an interface the parent does not
    have.
  - What landed: the three twin drivers read a source through one reader (`bcir_cpp_read_source`: whole, to 64 MiB,
    refused as `READ-ERR` past it, unreadable or holding a NUL) and preprocess into a block grown as the text needs
    (`bcir_cpp_run_alloc` and its `_ex` forms, to 64 MiB). The twin's token array grows to the bound both lexers now
    hold, 65 536 (`MAXTOK`, `clex.MAX_TOKENS`): `MAX_TOKENS - 1` tokens lower on both rails, one more is refused on
    both as `input too large`. CF-CONSTEXPR2's 5 000-term `(10 / x) && ...` chain, which the twin's drivers refused
    for its 80 KB of text, lowers on both rails to one claim graph, digest-equal. The bound is also the compile's work
    budget -- see the found-not-fixed item on the twin's linear lookups, before which it does not grow. Each
    preprocessor buffer that holds a token, an argument or a substitution holds a logical line (`BCIR_CPP_LINE`), so
    no token is refused for its own length, and the spacing token keeps an argument's last bytes. Both rails read a
    `#if` token by token and refuse a malformed `defined` for the twin's reason; the oracle's tokenizer keeps an
    unmatched character as a token for the lexer to refuse, and both refuse a NUL in a source or a header before
    reading it, in the twin's words. `bcir-cc`'s `-U` drops every `-D` of its name, as the oracle's CLI does.
    `bcir_cfront_digest` and `bcir_cfront_digest_with_allocator` return a status and store the digest through a
    pointer: a failed allocation is a nonzero status, never a digest, and the summary line says `out of memory` in its
    place; `test_memory_discipline.c` checks both under every injected failure. An identifier is ASCII on both rails
    -- a name a directive reads, a directive's own name, a macro parameter, an identifier or a number the lexer reads,
    a `#line` number -- and a character past ASCII outside a literal or a comment is refused for one reason,
    `non-ASCII character outside a literal` (the oracle reads with `re.ASCII` and its own ASCII predicates: a host
    predicate is no grammar, docs/security/laws.md L4). The oracle reads a parameter list by the twin's grammar
    (`cpp._params`) and a directive line too: its `#` and name across spaces and tabs, the name an ASCII identifier,
    its operand stripped of ASCII white space only.
  - What landed for the canon: a value number is the FNV-1a (64-bit) of its one-level spelling, its reads' own numbers
    in it, in 16 hex digits, on both rails (`_vn_number`, `vn_claim`) -- `(a - b) - a` numbers its inner value
    `fnv("c.bin.sub(in:p0,in:p1)")` and returns `fnv("c.bin.sub(<that>,in:p0)")` -- so a record is its claim's op, imm
    and reads' numbers and the canon is linear in the function, 31 to 40 bytes a claim. The walk, the memo, `cyc` and
    the depth cap are the spelled canon's, and a commutative op's reads are sorted by their numbers as they were by
    their spellings, so two values number alike exactly when their trees spell alike, but for a 64-bit collision: over
    the corpus, the canon with its value numbers spelled level by level is the parent's on all 960 fixture-target
    pairs, and its 5 504 value numbers, 1 259 distinct function canons and 286 distinct unit canons map one to one
    onto the spelled ones. A budget on the canon was the slice's first answer -- each value number, record and store
    pair charged before it was built, 64 MiB over a unit, `canon too large` past it -- and the slice's gates refused
    it: CF-CONSTEXPR2's 3 000-term chains, which the parent digested on both rails, charge it 153 to 380 MB. The twin
    merges a function's records in sorting them. It reads a value number's depth before its memo, as the oracle does,
    and starts each function's claim ids past the last one taken: a function of 1000 claims or fewer keeps the ids it
    had, and every fixture of the corpus emits the same bytes on the twin as before.
  - Outcomes: every `runtime/c/cfront_*.c` on the four targets through both rails, against the parent's: every summary
    but its digest and every preprocessed text the parent's but `cfront_sec_lextail.c`'s, whose comment this slice
    rewrites and which the twin refused at the old 16 384-token cap (`input too large`) and now refuses where it ends,
    in the unterminated chain the oracle refuses too (891 of 960 pairs equal on both rails, as on the parent); every
    twin emit, read past its summary line, the parent's on all 960 pairs; every digest and canon moved with the
    numbering, the two rails' alike. CF-CONSTEXPR2's 5 000-term chain lowers digest-equal on both rails: the oracle in
    14 s and 231 MB, where it had spent 156 s building a digest, the twin in 54 s and 32 MB. Its 3 000-term test,
    133 s and 0.9 GB on the parent, takes 39 s and 141 MB.
  - Faults: `tools/testing/faults/cfront-pplimits.json`, 43 -- 20 on the oracle, 23 on the twin -- each caught by its
    own test; `cfront-buf.json`'s B15, B16, B29 and B30, anchored in lines this changed, were re-anchored and caught
    again. The first sweep of the whole table, then 48 faults with the budget's 14, caught 46. PO13 removed one of two
    checks of a parameter list's separator that the other held alike: one is gone, the fault holds the other, and
    `F(a b c)` -- `F(a, c)` without it -- is its witness. PT3 cut the loop driver's preprocessed text at 64 KiB where
    the defect was the read of its source (the test's 70 KB comment is gone before the text is cut); it cuts the
    source now. The re-sweep caught both. The budget's faults left with it; the numbering's eight (a value number
    spelled whole, numbered by its op alone, its reads run together, its hash cut to 32 bits or seeded with 0, a
    commutative op's reads unsorted) and the oracle reading its memo before its depth came in, and a sweep of those,
    of PT11, of the twin's precedence and claim-id faults and of `cfront-buf.json`'s B27, back to the parent's, caught
    13 of 13.
  Found, not fixed here (each a suggested follow-up):
  - the twin's lookups are linear in a function's size -- a resource by its id (`res_of`, in the parser, the verifier
    and the emitter), a local's emitted name against every earlier one (`uniq_local`), R1.1's pairwise claim-id scan,
    the canon's writer of a rid (`writer_of`) -- so a function costs it its size squared: near the 65 536-token bound,
    8 000 locals (56 000 tokens) take it 31 s and 6 400 statements updating one local 35 s, in some 25 MB (the oracle
    7.5 s and 9 s);
  - a call with more than six arguments: the twin keeps six operands per claim, so it lowers the call with the rest
    dropped -- its emit passes six arguments to a function of more, which C refuses, its summary still `ok=1` and
    its digest the oracle's no longer;
  - white space C reads as such (6.4p3): a form feed or a vertical tab lowers on the oracle, in code and before a
    directive's `#`, and is refused by the twin (`expected expression`, `expected a type`);
  - an unknown directive (`#foo`) is refused by the oracle (`unknown directive #foo`) and ignored by the twin;
  - the twin's preprocessor holds what the oracle's does not: a macro body of 1 023 bytes (`macro replacement is too
    large`), a logical line of 8 190 (`preprocessor line too long`), 16 arguments in an invocation (`too many macro
    arguments`, where the oracle also takes 17 arguments to a macro of 16 parameters, which C refuses), and a
    `#line` number followed by more (`#line 12abc`: the oracle reads 12, the twin refuses it as out of range);
  - `@` and `$` outside a literal are refused on both rails for different reasons (the oracle's `unexpected
    character '@'`, the twin's `;`).

  The final serialized sweep (2026-10-02) ran every fault table that injects into cfront code -- the 25
  `cfront-*.json`, `ubsan-witnesses.json`, `escape.json` and `volatile.json`, 905 faults -- each whole, on
  CF-PPLIMITS's commit (`b233b7d5`), two workers in clean worktrees, and caught 902. Each fault it missed had gone
  equivalent under a later slice's change to another line, which that slice's own sweep -- its table and the faults
  whose anchors it moved -- could not see. T45 of `cfront-values.json` and CL15 of `cfront-calls.json` cut the alias
  branches CF-NULLPTR and CF-FNSEL had given the twin's `c.const` and `c.select` emits; CF-FPTAB then taught `tty` to
  spell every function-pointer value by its alias, and the branches chose what `decl_ty` already chose. They are gone
  -- `tty` is the one place an emit learns a function pointer's type, and the twin's output, summaries and digests are
  its parent's on every fixture of the corpus on the four targets -- T45 and CL15 now drop the alias where the null
  constant's and the select's temps take it, T22 follows the `c.const` line it mutates, and a new fault, FT19
  (`cfront-fptab.json`), cuts `tty`'s. GB8 of `cfront-globals.json` dropped a character array's literal from
  `strings`, the oracle's record of the literals that initialize a character array, which the linkable emit read to
  refuse every other literal; card 4's CF-LINKEMIT rendered a pointer's literal too, and since then nothing reads the
  record but FS10, which injects the old refusal of a literal outside it. GB8 is FS10's mirror now: it refuses a
  literal the record holds. Swept again on the change: `cfront-values.json` 32 of 32, `cfront-calls.json` 45 of 45,
  `cfront-globals.json` 37 of 37, `cfront-fptab.json` 35 of 35.

  CF-NAMECACHE / CF-CANONFAST (2026-10-03) closed the first found-not-fixed item above -- the twin's lookups linear in
  a function's size -- and the two regressions an A/B measurement of #786 against #797 found, both from the slices
  above: the twin's compile of a run of statements on one local (chain@500 17 ms to 31 ms, chain@1800 96 ms to
  290 ms; CPU, Clang `-O2`) and the oracle's digest of the fixture corpus (66 ms to 79 ms).
  - What landed, twin: each function's emitted names are computed once before the unit is rendered
    (`bcir_emit_names`; `names_build` in `emit_unit`): a name per resource index, the declared names in a hash set
    (`declared_name`), and the ranking walks of `uniq_local_compute` tabulated (`names_rank`: each object's rank
    among the objects of its source name counted by name in the walks' order, and `spelled_scan`'s two answers from a
    table of the names `spelled_each` enumerates, with an entry for each prefix of a name that ends before an `_`).
    One enumeration and one spelling decision (`uniq_local_rank`) serve the walks and the table, so the emit is the
    walks' byte for byte. A resource is found by halving the rid-ordered `res` array (`res_index`: `add_res` gives
    rids in order, a rollback drops them with their resources, nothing reorders them) in the lowering, the canon and
    the emit; the verifier halves too, for a function whose rids increase (`rids_increasing`, read once per
    function), and walks any other, since it reads caller-owned memory it did not lay out. The canon's writer of a
    rid is a hash (`wslot`), and R1.1 sorts the unit's claim ids (`claim_ids_unique`: a heap sort, a duplicate found
    by halving) where it compared every pair, and R21's lifetime walk reads back from a use to the last event on
    its rid -- the `free` that reads it, the write that re-validates it, the write last when one claim does both --
    where it rescanned every prior claim for every read (4 800 updates of one local: 0.3 s of the driver's 0.35 s;
    the first push's scaling witness missed its bound on a CI runner by this). A failed allocation of the name
    table is the compile's `oom` failure,
    never a slower success: the first draft fell back to the walks, and `cfront-pplimits.json`'s sweep, through the
    memory-discipline harness, found a compile succeeding under an allocation that had failed.
  - What landed, oracle: `_fnv1a64_from` folds four bytes a step (the low 64 bits of a product depend only on the
    factors' low 64 bits, so one mask a step suffices); each op's prefix hash and each base spelling's number are
    memoized in bounded tables (`_VN_PREFIX`, `_VN_BASE`, cleared past 4 096 entries), and a function's spellings in
    a table of its own (`numbers`), kept apart between the first-writer and the last-writer pass, since a value past
    the depth cap of 96 is `cyc` by the pass that reaches it.
  - Outcomes (CPU time, the median of 7 runs under `prlimit`, Clang `-O2`; #786 / #797 / now): chain@500 17.9 /
    31.2 / 12.2 ms; chain@1800 96.5 / 288.9 / 29.7 ms; locals@1800 6 373 / 811 / 47 ms; the oracle's digest of the
    165-unit corpus (15 samples) 65.9 / 79.3 / 61.3 ms. The item's two cases: 8 000 chained locals (56 000 tokens)
    32.9 s on #797 and 0.83 s now (the oracle 7.1 s), 6 400 statements on one local 7.0 s and 0.14 s (the
    oracle 6.0 s), each digest-equal on both rails. Every fixture of the corpus and the scaling inputs -- 256 inputs on the
    four targets, the summary and `--emit-c`, 2 048 runs -- give #797's twin's bytes, status and diagnostics, built
    with Clang and with GCC alike; the oracle's digest of every corpus unit is unchanged.
  - Found, not fixed here: the parser's scans are still linear per name -- `lookup` over the environment, `mut_body`
    over the mutation table -- most of the 0.83 s 8 000 locals take; and a digest split independent of this slice:
    a global written inside a block that also declares a local hiding a parameter (`uint32_t x_2 = 7u; uint32_t
    f(uint32_t x, uint32_t y) { { uint32_t x = y * 3u; x_2 = x + 1u; } return x + x_2; }`) reads back as a fresh
    input (`in`) on the twin where the oracle forwards the written value, both emits running as the original; with
    a local in the global's place, or the write outside a block, the rails agree.
  - Faults: `tools/testing/faults/cfront-namecache.json`, 11 -- 9 on the twin, 2 on the oracle -- each caught by
    its own test: NC1 by the memory-discipline sweep, NC10 and NC11 (R21 reading a claim's `free` before its write,
    R21 reading one claim back) by `test_c_lifetime_verifier_reads_back_to_the_last_event_on_a_rid`, NC8 and NC9 by
    the ranking witness
    (`test_a_local_of_a_parameters_name_ranks_after_it_and_past_the_global_it_prefixes_on_both_rails`), the twin's
    linear emit by `test_the_twin_names_each_object_once_so_a_run_of_statements_emits_in_linear_time` (4 800 updates
    of one local against 600, the child's CPU time, the min of three runs, under 24 times; the first push measured
    2 400 against 300 in wall time under 16, and a loaded CI runner read 16.4 and 28.6). `cfront-pplimits.json`'s PO17-PO19, anchored in
    the lines the oracle's memo moved, were re-anchored, and `cfront-typedefscope.json`'s HT9 and HT11, anchored in
    the walk `spelled_each` replaced, onto the enumeration; its HT10 and HT15 injected into the walks the emit no
    longer takes -- a fault in a path nothing reads is caught by nothing -- and inject into the table's lookup now.
    Swept on the change, each table whole, one worker in a clean worktree: `cfront-namecache.json` 11 of 11,
    `cfront-pplimits.json` 43 of 43, `cfront-typedefscope.json` 30 of 30, `cfront-buf.json` 30 of 30.

  CF-GBLOCK (2026-10-03) closed the digest split CF-NAMECACHE found and left: the twin bound a global's resource
  with the block that first named it (`use_global` -> `env_add` at the block's depth) and dropped the binding with
  the block, so the next reference -- `{ g = y; } return g;`, an `if` body, a loop body -- bound a second resource
  for the same object; the canon read the written value back as an input (`in`) where the oracle, which holds one
  resource per global per function (`gres`), forwards it, and the digest split. The emits named both resources `g`
  and ran as the original, so no fixture's behaviour moved; the shape needs only a global first named inside a
  block and named again after it (a block-scope local hiding a parameter, which the first report blamed, is
  incidental). The twin now reuses the resource it gave the global (`global_res_of_name`: a read-only resource of
  the global's name that is no function designator -- a resource a speculative lowering dropped is not there to
  find, so the reuse survives every rollback), binding the name afresh for the block. Witness
  `test_a_global_first_named_inside_a_block_is_one_resource_on_both_rails`: five units -- the block, the `if`, the
  loop (`{ g = 0u; }` first, so the sum starts known), the global hidden by a block-scope `g`, the global named at
  function scope first -- summary and digest the oracle's on the twin and the twin's emits run as the originals;
  each unit assigns `g` on every path before reading it, since the equivalence harness's original and emit share
  one `g`. RED on the parent: the block, `if` and loop units' digests leave the oracle's (`in` for the read, the
  twin's claim graph two resources). Fault GB10 (`cfront-globals.json`): the reuse refused.

  CF-CALLARGS (2026-10-03) closed CF-PPLIMITS's second found-not-fixed item: the twin's claim held six reads, so
  `p_call` kept a call's first six arguments and dropped the rest, marking the claim `truncated` for the escape
  analysis to refuse -- but the compile succeeded, `ok=1`, with a digest the oracle's no longer, and the emit passed
  six arguments to a function of seven, which C refuses. Both rails now carry sixteen arguments and refuse a
  seventeenth alike (`a call of more than 16 arguments is not supported`): the twin's `BCIR_CALL_MAX_ARGS` and the
  oracle's `lower.MAX_CALL_ARGS`, one number, a claim's reads one more (`BCIR_CLAIM_MAX_RD`) for the callee value or
  the object base an indirect or member call reads first. Every call form -- direct, through a pointer, through a
  member, the atomic builtins -- reads its arguments through one parser (`p_call_args`), and the oracle's three sites
  through one helper (`_call_args`); the `call_dropped` flag and the `truncated` marking are gone from the twin, since
  nothing is dropped. A claim grew from 248 to 292 bytes. Witness
  `test_a_call_carries_sixteen_arguments_and_a_seventeenth_is_refused_on_both_rails`: calls of seven and sixteen,
  direct and through a function pointer, digest-equal on both rails and the twin's emits running as the originals;
  seventeen refused alike. RED on the parent: the call of seven lowers with six on the twin, its digest the oracle's
  no longer. Faults CL46 (the twin keeps six and drops the rest) and CL47 (the oracle lowers seventeen) in
  `cfront-calls.json`. The number had two more mirrors on the oracle besides `lower.MAX_CALL_ARGS`: the freeze's
  `CLAIM_MAX_RD` (`bcir/gem/handoff.py`), which refused a seventh read where the C rail now takes seventeen, and the
  escape analysis's, still six; a test reads the three out of `bcir_cir.h`
  (`test_a_claims_read_capacity_is_the_c_twins_on_every_rail`), and the freeze differential's `reads-longest` (17) and
  `too-many-reads` (18) graphs cross the bound on both rails. The escape analysis's witness of the capacity moved to
  the new boundary: a call of sixteen is analyzed, one of seventeen never lowers, and a claim widened past the
  capacity by hand is refused as the truncation was. Faults CL48, CL49.

  CF-CPPZERO (2026-10-03): the preprocessor's state (`CppState`, 2.3 MB of macro and line tables) was taken from the
  allocator and `memset` to zero, so every compile touched 2.3 MB before reading a byte of source: a trivial unit
  cost the twin 2.35 ms, of which that memset was a third and more. `bcir_host_allocate_zeroed` takes it zeroed --
  libc's `calloc`, whose large allocation is fresh pages the kernel already zeroed and the compile mostly never
  touches, or `allocate` then `memset` for an injected allocator, which promises nothing -- one allocation either
  way, as the fault-injecting allocators count it (the memory-discipline sweep is unchanged). A trivial unit now
  costs 1.30 ms. Witness `test_c_zeroed_allocation_is_zero_under_every_allocator_and_is_one_allocation`: an
  allocator that fills what it hands out with 0xAA and counts its calls, libc's, a NULL passed through, an invalid
  allocator refused. Fault B31 (`cfront-buf.json`): the injected allocator's memory handed back as it came.

  CF-PPSPLITS (2026-10-03) closed the preprocessor splits CF-PPLIMITS left. C's white space is a space, a tab, a
  new-line, a vertical tab and a form feed (6.4p3): the twin's preprocessor and lexer read a space and a tab alone, so
  `\f` in code was an expression's end and `\f#define` no directive, where the oracle lowered both -- one predicate
  now (`pp_space`, at the twelve sites that spelled the pair), the lexer skipping both, and the oracle's lexer widened
  to its preprocessor's set. Each preprocessor spells its output with single spaces, so a unit that goes through one
  never shows its lexer either character: the lexers are witnessed at their own entries, the twin's through
  `bcir_cfront_compile`, which takes preprocessed text (`test_c_the_frontend_reads_c_white_space_at_its_own_entry`;
  the first ppsplits sweep found PS2 uncatchable through the driver), the oracle's through `tokenize`. `#error text`
  is the compile's failure (6.10.5), its text macro-expanded, and a non-directive's execution is undefined (6.10p9):
  the twin ignored `#error`, `#warning`, `#pragma` and every unknown directive alike and lowered on; it refuses
  `#error` and an unknown directive in the oracle's words now (the oracle's reasons no longer name the file, which the
  driver's line does), and `#warning`, `#pragma` and the null directive lower on both. The twin held a logical line to
  8 190 bytes, a macro's replacement to 1 023 and an invocation to 16 arguments; the oracle held none, and neither
  held a function-like macro to as many arguments as parameters (6.10.3p4): the twin took `M(x, 1u, 2u)` for `M(a, b)`
  and the oracle that and seventeen arguments for sixteen parameters. The oracle holds the twin's three bounds now
  (`_MAX_LINE`, `_MAX_BODY`, `_MAX_ARGS`, in the twin's words) -- the line's read after the comments are gone, as the
  twin's reader reads it: CI's thorough tier caught the oracle refusing a 70 KB comment on one line -- and both hold
  the arity (`_check_arity`, the twin's check after its invocation reader): more arguments than parameters `too many
  macro arguments`, fewer `too few macro arguments`, a variadic macro's `...` free to be empty, `M()` of a macro of no
  parameters passing none. `#line` takes a decimal digit sequence (6.10.4p3) in 1..2147483647: `#line 12abc` read 12
  on the oracle and was `out of range` on the twin, `#line abc` was ignored on both; both refuse `#line number is not
  a decimal digit sequence` and `out of range` now. And no C token starts with `@`, `$`, a backquote or a stray `\\`:
  the twin's lexer made each a one-character punctuator for the parser to stumble on (`parse error: ;`); it refuses
  them where the oracle does, `unexpected character '@'`, spelled as Python's repr spells the character. Witnesses:
  five dual-rail tests, each unit lowering digest-equal on both rails or refused by both in one sentence -- the last
  byte within each bound lowers, the first past it is refused. A probe of the same family then found the line itself
  split. The oracle ended a logical line where `str.splitlines` does -- at a form feed, a `\x1c`, a NEL, a U+2028 --
  so `return x\x1c + 1u;` lowered there with the character gone and was `unexpected character` on the twin, a NEL in
  code vanished before the reader that refuses a character past ASCII saw it, and `}\f#define K 1u` was a directive on
  the oracle and text on the twin; the oracle read a space and a tab alone between a `#` and its name
  (`_DIRECTIVE_SPACE`), so `#\finclude <stdint.h>` was a null directive with its header unread; and the twin read a
  CRLF file's `\r` into the line, so `#if K == 5u\r` was `malformed` there and a `\\\r\n` splice none, where the
  oracle lowered. One rule on each rail now: a CRLF end of line is a new-line (translation phase 1 -- mapped before
  the splice on the oracle, read as one by the twin's line reader), a line ends at a new-line alone,
  `_DIRECTIVE_SPACE` is `pp_space`'s set, and a lone carriage return is white space on both (`pp_space`, the oracle's
  `_ASCII_SPACE`). A CRLF unit lowers to its LF twin's digest, the twin's longest line included; `\x1c` and `\x1e` in
  code are `unexpected character` alike and a NEL or a U+2028 `NONASCII` alike
  (`test_a_line_ends_at_a_new_line_alone_on_both_rails`; faults PS13-PS18). Found, not fixed: the parsers' reasons for
  an empty parenthesis differ -- `M(x, )` for `M(a, b)` is `unexpected PUNCT ')'` on the oracle and `expected
  expression` on the twin, both refusing -- as do the two for a lone `\r` between two directives on one line. Faults:
  `tools/testing/faults/cfront-ppsplits.json`, 18 -- 10 on the twin, 8 on the oracle; `cfront-pplimits.json`'s PO9,
  anchored in the oracle's old `#line` read, re-anchored onto the digit check.

---

## 4. Capability closure ledger migrated from the former master roadmap

The former master roadmap accumulated landing notes and checkmarks as well as future
work. This ledger preserves the durable historical conclusions without turning the
execution roadmap back into a changelog. “Landed” means code and deterministic tests
exist; it does not imply production deployment or hardware evidence.

| Program | Landed baseline | Work deliberately left open |
|---|---|---|
| Language and verifier | Python/C C-front twins, project/cross-TU mode, ABI and effect contracts, R1–R25, ordinary x86-64 assembly edges | Hosted-C completeness, `_Decimal*` reference support, reset/exception/paranoid entry, additional language frontends |
| Optimizer and backend | 12-axis K_BCIR, min-plus/RCSP/(max,+), GEM scheduling, C23 and resident LLVM/object paths, JVM/CIL/WASM bounded validation | Arbitrary-graph LLVM AOT, general native isel (gated), target-specific measured scheduling evidence |
| Machine/driver substrate | StreamPack v1–v3, device manifests, bank/move/event/DMA contracts, MC1/MC2 operator tools, direct RuntimeChannel hooks, metadata-only HAM routing/residency/replay, and strict context-shard activation | Resident UART/virtio/device drivers, physical HAM adapters, stable UAPI, Linux modules, native IPC, and physical-device qualification |
| Memory discipline | Freestanding/hosted/adapter classes, checked hosted allocator and fault injection, fail-every-allocation tests | Per-operation compiler arenas and further context migration as allocation-bearing surfaces expand |
| ML/model stack | Planned/streamed training, hosted safe pretraining and bounded alignment stages, deterministic corpus/BPE, provider-neutral contracts, model manifest/tokenizer/decoder, header-only cost/placement plans, exact static addresses, bounded hardware-policy training/search, adaptive/raw-byte/sequence-interface/growth references, GQA/KV cache, BCIRQ8, BCIRQ4T tensor compute, native Q8/Q4 conversion/projection and exact Q15 retrieval, and standalone-C TinyLlama parity | Production serving, executable hardware placement/rematerialization, architecture native/export promotion, whole-decoder Q4/additional formats, canonical 32M and useful byte-native/progressive training, balanced multilingual tokenizer expansion, large UniTok/Thunder builders, glyph/PCA artifacts, live providers, distributed/GPU execution, and hardware-qualified model/policy gates |
| Telemetry/control | Signal registry, BTLM codec, metric derivation, deterministic serializers, shared-ring baseline | Generated fixed-width C registry, source/session/generation/clock identity, live SPSC protocol and real transports |

### Retired AI-substrate research note

The three-week AI-substrate SOTA snapshot was audited against source and tests during
the 2026-07 documentation consolidation and then retired. Its conclusions
now have stable owners:

- **A1 precision:** exact-width `_BitInt(N)`, groupwise power-of-two Q8, BCIRQ8, and
  a bounded BCIRQ4T/SmoothQuant/AVX2 tensor path landed. Whole-decoder low-bit,
  model-level quality qualification, other targets, and any additional format remain in
  [`BCIR_ML_AI_INTEGRATION_ROADMAP.md`](machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md)
  §6 and require R17/provenance/drift gates.
- **B1 scheduling:** deterministic matmul tile/loop search, compute-vs-memory roofline,
  measured schedule artifacts, real OS/optional PMU counters, and selected-schedule MLIR
  landed. Two-target exhaustive evidence and reviewed promotion remain in the same roadmap.
- **B3 differentiation:** the hash-consed closed primitive set, reverse mode,
  transcendental VJPs, symbolic reverse-over-reverse, law op, C lowering, measured
  ordering, rematerialization, local mutation, and bounded loop/finite-call handling
  landed. Representative-graph qualification and the explicitly quarantined aliased/
  unbounded/dynamic cases remain open. The accepted mathematical description is monoidal/string-diagram/PROP
  rewriting—not “operad 2-cells.”
- **B5 libraries:** CBLAS, FFTW 1D/2D, LAPACK, GSL, SLEEF, and libcerf wrappers,
  link metadata, calling-side tuning, demand-driven provider probes, measurements, and
  evidence artifacts landed. Further libraries and target qualification remain workload-driven.

## 5. Where the detailed notes live now

- **Per-landing detail:** the GitHub PR record (#2–#647, plus subsequent changes; PR bodies carry Verification
  sections with exact gate outputs), `git log`, and the pre-consolidation revisions of
  `REPO_CURRENT_STATE_AUDIT.md` / `BCIR_MASTER_ROADMAP.md` /
  `machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md` (recoverable via git).
- **Current state:** [`STATUS.md`](STATUS.md) (generated counts),
  [`REPO_CURRENT_STATE_AUDIT.md`](REPO_CURRENT_STATE_AUDIT.md) (the honest snapshot),
  [`VISION_ALIGNMENT_AUDIT.md`](VISION_ALIGNMENT_AUDIT.md) (the dated pillar audit).
- **What's next:** [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) and its
  companions ([`BCIR_ML_AI_INTEGRATION_ROADMAP.md`](machine-learning/BCIR_ML_AI_INTEGRATION_ROADMAP.md),
  [`BCIR_DRIVER_KERNEL_ROADMAP.md`](kernel/BCIR_DRIVER_KERNEL_ROADMAP.md)).
