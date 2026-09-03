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
> a live count. This revision is current through merged PR #751 (2026-09-03) and package version
> `0.2.0`. Sources are the GitHub PR record, first-parent history, implementation/tests,
> and pre-consolidation document revisions retained in git.

---

## 1. The development method

BCIR was developed almost entirely by AI coding agents under a single human operator,
with the recorded arc running from PR #2 through #638 between 2026-04-23 and 2026-07-15. Two agent fleets
are visible in the branch names: **OpenAI Codex** (`codex/*` branches — the initial
scaffolding, the `llvm-training/` corpus, and the closing integration-research docs) and
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
also later retired — its teaching value survives in `llvm-training/` as
explicitly-bannered historical material).

### #32–#152 — the `llvm-training/` corpus (2026-05-28 → 06-06)
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
- **2026-05-26 → 06-06:** the LLVM-first seed (#13–#31); the `llvm-training/` corpus
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
