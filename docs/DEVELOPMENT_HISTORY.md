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
