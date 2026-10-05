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
