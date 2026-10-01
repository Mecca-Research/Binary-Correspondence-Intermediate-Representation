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
  Found, not fixed here (each a suggested follow-up):
  - the loop driver (`runtime/c/test_cfront_loop.c`) still reads the first 64 KiB of a source silently, and both
    twin drivers hold the preprocessed text in 64 KiB (a 6 000-global unit is refused there, loudly, and lowers on
    the oracle);
  - the twin's preprocessor refuses any token of 256 characters or more (a string literal, a stringize argument, a
    `__has_attribute` operand, a long number in `#if`), which the oracle takes;
  - a malformed `defined` (`#if defined(X`, `#if defined +`) lowers on the oracle and is refused by the twin; the
    oracle drops a NUL between tokens and lowers what surrounds it;
  - `bcir-cc -DXQ=1 -UXQ` fails (`macro name must be an identifier`): an undefined `-D` becomes an empty
    definition `define_macro` refuses;
  - `bcir_cfront_digest_with_allocator` returns a digest of a canon that ran out of memory as an ordinary value;
  - the twin reads only the ASCII start of a macro name: beside `#define caf 5`, `#ifdef café` keeps its group on
    the twin and skips it on the oracle.

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
    accepted by both rails, where C requires a diagnostic (6.8.4.2p3) and the emit does not compile;
  - a character constant `'\xff'` is -1 on every target on both rails; where plain `char` is unsigned (AArch64
    Linux) C gives 255;
  - an `enum` defined at block scope is refused by both rails, for different reasons, and so are an enumerator in
    a compound literal's dimension `(T[N]){...}` and a row pointer's `(*p)[N]`, where each rail reads a literal
    only;
  - `sizeof`, `_Alignof` and `(int)1.5` in an enumerator are refused on both rails (6.6p6 allows them);
  - a 3000-term `1 + 1 + ...` chain raises `RecursionError` in the oracle's parser, which the twin lowers;
  - the twin refuses `typedef T row[0];`, which the oracle takes.

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
  name`, the oracle: `expected 'IDENT', got OP '='`), where C reads the statement as an expression.

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
  Found, not fixed here (a suggested follow-up): five fixtures older than this branch -- `cfront_aostruct.c`,
  `cfront_signed.c`, `cfront_paste.c`, `cfront_structcall.c` and `cfront_assignexpr.c` -- overflow `int` in the
  original under the shared corpus's full-width random arguments (`_equiv`), so their equivalence runs are undefined;
  each passes on every platform CI runs today.

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
    with;
  - `#if` evaluates outside C's arithmetic on both rails: `#if -1 > 0u` and `#if 'a' == 97` take the `#else` branch
    on both, digest-equal (a silent miscompile; C reads -1 as `UINTMAX_MAX` there, and `'a'` as 97), and `#if
    0xFFFFFFFFFFFFFFFF == -1` raises a bare `ValueError` in the oracle where the twin refuses it as an overflow;
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
