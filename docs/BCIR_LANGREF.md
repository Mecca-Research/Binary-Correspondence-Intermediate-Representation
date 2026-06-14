# BCIR Language Reference — v0.1 (normative)

BCIR is the IR *law*; MLIR is the forge used to express, verify, transform, and
lower that law during bootstrap; LLVM/Clang are backends and interoperability
targets, **not** the conceptual center.

> **Source-of-truth rule.** BCIR semantics live in this document and in the
> dialect definitions under `mlir/`. The Python package under `bcir/` is the
> **executable conformance oracle** — it must agree with this document, never
> override it (see [`PARITY.md`](PARITY.md)). C++ may implement the engine; it may
> not become the definition of BCIR.

## 0. Stance

BCIR is a multi-level IR specification:
`syntax · types · attributes · operations · verification laws · rewrite laws ·
cost laws · execution laws · lowering contracts`. It is a **correspondence IR**:
it preserves the chain from semantic intent to physical execution and asks not
merely whether a program is structurally valid but whether the computation is
physically, temporally, and contractually **executable**.

## 1. The multi-level IR

| Level | Name | Question |
|---|---|---|
| BCIR-0 | Semantic Claim Graph | What computation/state transformation is intended? |
| BCIR-1 | Shaped Data Graph | What tensors/columns/buffers/sparse maps/records/layouts exist? |
| BCIR-2 | Registry / Placement Candidate Graph | Where can resources live; what domain constraints apply? |
| BCIR-3 | K_BCIR Correspondence Plan | Which realization path π is selected under H and Θ? |
| BCIR-4 | GEM Stream IR | What lane schedule, StreamPack, fences, prefetch contracts execute? |
| BCIR-5 | Target Lowering IR | LLVM / vector / GPU / SPIR-V / PTX / WASM / future BCIR-native binary. |

`K_BCIR(G | H, Θ) = min_π Σ_i T_i ⊗ f_i(π)` maps BCIR-2 → BCIR-3: a shortest-path
search over representation and machine-state space, not a one-shot
source→binary pipeline.

## 2. Central equation

The normative form is **constrained series-parallel**:

```
K_BCIR(G | H, Θ) = min_{π ∈ Legal(G,H)}  M(π, Θ)    subject to    R(π, Θ) ⪯ B(H, Θ)
```

- `G` — the goal graph (a BCIR program); `π` — a realization plan (lane/stride
  class, batching, schedule, prefetch).
- `M(π, Θ)` — the **schedule-aware price**: series composition (claims chained
  on one affinity domain, the decoupled GGG tail, successive waves and phases)
  accumulates with (min,+) ⊗; parallel composition (claims co-executing in a
  CT2 wave on distinct domains) combines with **max**. A transition's coupling
  `f_i` applies against its *actual* schedule predecessor — a fusion discount
  belongs only to claims that really run back-to-back. M is (max,+) over the
  wave/token DAG (`gem.overlap.price_scheduled`).
- `R(π, Θ)` — the additive resource ledger, `Σ_i T_i ⊗ f_i(π)` per dimension
  (`T_i` a 12-d cost vector; `⊗` element-wise Q8 coupling, **not** scalar
  multiply).
- `B(H, Θ)` — live budgets (thermal cap, power cap, …), Θ-dependent: a hot
  machine makes wide SIMD *infeasible*, not merely expensive.
- Solved exactly by label dominance over the layered DAG of **legal**
  candidates (RCSP, `kbcir.rcsp.optimize_constrained`); Pareto-optimal plans
  that no weight vector can reach are recovered by `kbcir.rcsp.pareto_plans`.
  All arithmetic is integer/Q8 and deterministic.

**Degenerate case (the scalarized default rail).** With no budgets (B = ∞) and
serial composition (one domain, textual chaining), M reduces to the weighted
sum and selection to the tropical (min,+) shortest path:

```
K_BCIR(G | H, Θ) = min_π  Σ_i  T_i ⊗ f_i(π)  =  min_π  C_H(π, Θ)
```

with `score = w(H,Θ,phase,policy) · (T_i ⊗ f_i(π))`. This degenerate form is
the default rail (`kbcir.realize.optimize`) and the worked-example constants
(vec16, score 7808) are pinned to it; the constrained rail reproduces it
exactly under an unbounded budget. Budget feasibility and scheduled-price
consistency (makespan + overlap_gain = serial) are verifier obligations under
law R9 (`bcir.kbcir.budget`, `bcir.kbcir.scheduled_price`, `-bcir-verify`).

**Temperature (the soft generalization).** The tropical (min,+) selection is the
zero-temperature limit of a log-sum-exp dynamic program:

```
F_T(G | H, Θ) = −T · log Σ_{π ∈ Legal} exp( −score(π) / T )   →   min_π score(π)   as T → 0
```

`F_T` is the Gibbs free energy over realization plans; at `T > 0` it yields a
*posterior over plans* (per-claim marginals, an expected cost vector) and is
**differentiable** — `∂F_T/∂w = E_π[C]`, the expected sufficient statistic — so
the optimizer becomes a learnable layer (`kbcir.softdp`). This is an L2/L3
offline organ (LangRef §13): learning happens at `T > 0`, then anneals and
**freezes** to a `T = 0` integer table for the certified hot path. At `T = 0` it
delegates to `optimize` and is bit-exact; the degenerate-case law (`F_T ≤`
hard score, with equality at `T = 0`) is a verifier obligation under R9
(`bcir.kbcir.soft_select`).

## 3–9. Laws (summary)

- **Module law (§3).** A module is a registry-governed execution universe;
  registries precede claims; plans derive from legal claims; a GEM stream may not
  exist without an originating BCIR plan.
- **Registry-first memory (§4).** Raw pointers are outlawed at the core level.
  `Address = (RID, layout, domain, offset, generation)`.
- **Claim law (§5).** The primitive object is the *claim*, not the instruction:
  `op + resources + contract + phase + cost + verification + ≥1 legal realization`.
- **Phase DAG (§6).** Execution order is a phase graph (acyclic), not textual
  order.
- **Lane law (§7).** Lanes are execution-geometry types: `U` unit/stride, `UX`
  cacheline-local, `T` tile, `GGG` gather/scatter (always legal, must be
  minimized), `A` atomic, `H` hazard/provenance.
- **K_BCIR cost (§8).** Cost is *in the IR* — a 12-d `costvec`
  (compute, memory, fabric, sync, compile, thermal, power, reliability, security,
  accuracy, contention, verification). Illegal paths are rejected before scoring;
  Pareto pruning precedes scalar selection; the selected path hydrates GEM.
- **GEM Stream IR (§9).** The StreamPack is the hot artifact; the BCIR graph is
  the dormant semantic artifact. A pack retains provenance and generation tags
  and is rehydrated (patch/repack/replan) on mismatch. Scheduling is
  duration-aware (`bcir.gem.schedule`): EFT waves with locality affinity and the
  bandwidth knee, or the `!bcir.token` DAG (pipelined phases, ABI v2
  double-buffer contracts).

## 10. Verifier laws (R1–R13)

R1 registry uniqueness · R2 registry resolution · R3 domain legality ·
R4 phase-DAG legality · R5 hazard legality · R6 lane legality · R7 bounds
legality · R8 cost completeness · R9 plan legality · R10 stream provenance ·
R11 generation validity · R12 lowering legality · **R13 policy provenance** —
every decision rule in force (gain schedule, cost table) carries a generation
tag and an admitting certificate: a promoted portfolio entry requires its
replay certificate, a calibrated profile must present its frozen table with
matching generation and constants, a regret ledger's books must balance. Rule
swaps are never silent. Encoded as IR via the `bcir.verify.*` op family. The
runnable full set lives in `bcir/verify`, one entry point per correspondence
artifact — `verify(module)` R1–R8(static), `verify_plan` R8–R9, `verify_pack`
R10–R11, `verify_lowering` R12, `verify_provenance` R13 — and the MLIR-native
`-bcir-verify` pass enforces the structurally checkable form of all thirteen
on the dialect.

## 11. Rewrite laws (the building-blocks engine)

Lane promotion (`GGG→UX→U(k)→U`), tile formation, layout (`AoS→SoA→AoSoA`),
prefetch introduction, GGG quarantine. A rewrite is legal **only if** it does not
increase the selected K_BCIR cost (or strictly improves legality) and the module
still passes R1–R12. Encoded via `bcir.opt.*`.

The **composition** engine that applies them is an e-graph / equality saturation
(`kbcir.egraph`, `bcir.egraph.extract`) — the realization of the liked/unliked-pair
model. **Liked pairs** are e-classes: an atom or a shared subexpression is a class
(the identity `a = a`, the memory module), and hashconsing finds common
subexpressions for free (CSE). **Unliked pairs** are operators and the rewrites
they enable: a rewrite proves two forms equal and *merges* their classes
(congruence closure), folding an unliked result toward a simpler liked attractor
(`x+0→x`, `1+1→2`, `a*b+a*c→a*(b+c)`). **Resolution** `Res(·)` is one round of
rewriting + congruence rebuild; **extraction** picks the min-cost representative
per class. Because extraction returns the minimum, the optimized cost is always
≤ the input cost — an R9 obligation.

**The Axiom of Memory Modules — `a = Lim(Res(U))`.** Resolution is *monotone* (a
merged class never un-merges) over a *bounded* lattice (finitely many e-nodes
under a terminating rule set), so by Knaster–Tarski/Kleene the iteration
`Res^k(U)` converges to a least fixpoint

```
Lim(Res(U)) = Res^∞(U) = the smallest X with Res(X) = X      (≡ saturation).
```

A **memory module** is the extraction of that fixpoint, *frozen* and
*generation-tagged*: `memory = Extract(Lim(Res(U)))` (`kbcir.memory`,
`bcir.kbcir.memory_module`). The **admissibility law** (the fixpoint witness) is
the bridge from the e-graph engine (§11) to the provenance spine (§13): an
artifact may be frozen into a generation **only if** resolution reached its
fixpoint —

```
saturated == True   ⇒   admissible as memory.
```

A *budget cutoff* is a partial `Res^k(U)`, `k < ∞`; freezing it pins a
non-canonical, still-improvable representative as "memory," and because a cutoff
is budget/order dependent while the fixpoint is canonical (confluence), two runs
that cut off differently need not agree — breaking the determinism the manifest
depends on. Idempotence `Res(Lim(Res(U))) = Lim(Res(U))` makes a memory module
its own attractor — the `a = a` identity at module scope — so re-resolving a
frozen module reproduces it; the verifier exploits this for tamper-evidence
(`verify.verify_memory` independently re-resolves rather than trusting the
recorded witness, the analog of `verify_manifest` recomputing the digest).
Witnessed by R13.

Nothing is globally immutable: a liked pair holds *within* a generation; across
generations it is an unliked pair resolved by rehydration, and the provenance
manifest pins identity. A frozen memory module's generation + content fingerprint
chain into that manifest (`manifest_for(..., memory=…)`), so an admissible
(saturated) extraction is itself part of a plan's commit hash.

## 12. Lowering contracts

BCIR-4 → BCIR-5 lowering is governed by R12: each lowered op preserves the BCIR
semantic (lane geometry, bounds, hazard, precision) or carries an explicit
discharge in `bcir.trace`. LLVM is the **first** backend, not the center.
Encoded via `bcir.isa.*` / `bcir.target.lower_contract`.

**Modular Mapping Functions (`kbcir.mapping`).** A lowering — and any
representation change — is a mapping function `f` between cost-bearing
representations, and R12 imposes two further laws on it:

- **Objective-support preservation.** `Supp(J)` is the set of cost dimensions on
  which the objective `J` is nonzero — *where the objective matters*. A legal map
  must carry that support forward,

  ```
  f(Supp(J)) ⊆ Supp(J')
  ```

  so a lowering may sharpen, rescale, or fuse a cost but may not silently **drop**
  a dimension that mattered (lose the thermal / security / accuracy / verification
  term) unless it carries an explicit discharge — the same escape R12 already
  grants bounds/hazard/precision. The objective's footprint is an invariant of
  legal lowering (`verify.verify_support_preservation`, R12).
- **Commutativity / path independence.** If two conversion paths reach the same
  target — a direct map `Φ` and a two-step `Ψ` then `Λ` — they must agree:

  ```
  Λ ∘ Ψ = Φ
  ```

  A result may not depend on which legal path produced it. This is the
  PARITY/manifest discipline generalized to **any** representation rail:
  oracle↔MLIR parity, manifest replay (`reproduces`), JSON round-trips, and any
  future rail are instances of one commuting-square law
  (`verify.verify_commutativity`, R12).

## 13. Learning placement (normative policy)

Learning and measurement enter BCIR only where decisions are slow enough to
amortize inference, reversible at the next checkpoint, and produce artifacts
the R-laws can check. The placement criterion is the **amortization
inequality** — place learning at a layer only if

```
E[improvement per decision]  >>  decision rate x inference cost
```

— stratified by timescale, **never by importance**:

- **L0 (the hot path — PROHIBITED).** No learned inference executes on the hot
  path: lane-promotion *application*, prefetch issue, bounds masks, the
  StreamPack ABI, stackify, fences. At hot-path rates even a nanosecond-scale
  inference swamps what it optimizes. Decisions are **compiled out**: the
  binary artifact carries decisions, never models. This prohibition is law,
  not guidance.
- **L1 (plan time — frozen tables only).** Learning and measurement supply
  *inputs to exact search, never the search*: cost tables T_i, tier factors,
  gather penalties, coupling factors — produced offline, **quantized to
  integer Q8, frozen, generation-tagged** (`bcir.kbcir.calibration`,
  `kbcir.microbench.CalibratedProfile`, `cal_gen` on the target capability).
  The table may be a point estimate (microbench) or a **Bayesian posterior with
  a certified conformal error bar** (`kbcir.bayescal`): a conjugate-Gaussian
  (VI-exact) posterior over each ratio + a distribution-free split-conformal
  `±δ` at a stated coverage, optionally inferred likelihood-free by **ABC** with
  the GEM/`optimize` forward model as the simulator. The frozen artifact is
  still Q8 integers plus an integer `δ`; the conformal guarantee lets later
  selection be made *robust* over the credible interval. Plan-time "inference"
  is a table lookup; determinism and the pinned scores are preserved by
  construction. The verifier gates table well-formedness — and the conformal
  guarantee (coverage in (0,1), `δ ≥ 0`, no interval from ≤ 1 sample) — under
  R8/R13. The loop is **closed and certified** (`kbcir.calibloop`): measure →
  freeze → apply → replan emits a `CalibrationCertificate` whose **win** is the
  measured cost of *not* recalibrating (the stale plan, faithfully rescored on
  the machine telemetry reports, minus the recalibrated optimum); it is
  admissible only when `cal_gen ≥ 1` and `win ≥ 0` (R13,
  `verify.verify_calibration`). Measurement stays offline (L2/L3); the frozen
  table and every downstream decision are integer and reproducible.
- **L2 (checkpoints — portfolio + replay gate).** Gain schedules (policy
  weight vectors, thresholds) adapt only at checkpoints, only as members of a
  **portfolio of frozen, generation-tagged policies**
  (`bcir.kbcir.portfolio`), selected at plan time by a router. The router may be
  the deterministic workload-class table (`classify`) or a **learned
  Mixture-of-Experts gate** — a GNN over the claim graph trained on the regret
  ledger (`kbcir.moegate`, `bcir.kbcir.moe_gate`). The gate is the *safe*
  learning regime: it only *selects among already-certified experts*, never
  emits a table or policy; it deploys **frozen** (Q8 integer routing,
  deterministic across hosts) and only behind an admitting replay certificate.
  A schedule swap or a gate deployment requires that **replay certificate**
  (`bcir.kbcir.replay_certificate`): counterfactual replay on logged Θ episodes,
  judged by the incumbent's scheduled metric M(π,Θ), zero regressions over ≥ 1
  episodes (verified under R9/R13). The network proposes a route; the verifier
  disposes. Shadow → canary → promote; never silent.
- **L3 (the meta-policy — measured, human-actuated).** Where the
  heuristic/learned boundary itself sits is a measured question: the **regret
  ledger** (`kbcir.regret`, `bcir.kbcir.regret_ledger`) continuously books each
  deployed rule's gap to the hindsight-best alternative under one neutral
  yardstick. The retune trigger is **not a magic threshold** but the **MDL /
  Bayesian-evidence** two-part code: a swap is recommended iff it shortens the
  total description length,

  ```
  ΔL = Σ_i regret_i/best_i  −  (k/2)·ln(N)  >  0
       \___ data fit (saving) _/   \__ BIC complexity _/
  ```

  i.e. the accumulated *relative* regret (the bits the deployed rule wastes)
  must outweigh the model-complexity penalty of specifying and certifying the
  swap (the large-sample Bayesian evidence, Schwarz 1978). Few episodes of small
  regret never flag — that would be overfitting noise — while sustained or large
  regret does. The verdict is a recommendation, never an actuation: a flagged
  rule is a *candidate* for retuning, the swap still goes through the L2 replay
  gate, and **R13 (policy provenance)** witnesses the chain *and the evidence* —
  a verdict is illegal unless it is consistent with its MDL margin (retune ⟺
  data_fit > complexity). Actuation stays human by policy; any future automation
  of the flip must run behind both the gate and R13.

**Provenance is the spine.** Every decision rule in force is frozen and
generation-tagged; a **provenance manifest** (`kbcir.provenance`,
`bcir.kbcir.provenance_manifest`) chains a plan's inputs and those generations
into a single content hash — the commit hash of a plan. Manifest equality ⇒ an
identical plan, so the constantly-updating computation DAG is reproducible and
debuggable: an immutable plan is a *committed* manifest (a closed branch), a
candidate under evaluation is an *open* branch, and `diff` reports which
generation moved between two runs. Nothing is globally immutable, but everything
is immutable *within its generation*. R13 (`verify.verify_manifest`) requires a
deployed plan's manifest to reproduce its recorded score and shape on replay.

The **memory module** (§11, `kbcir.memory`) is the e-graph's contribution to this
spine: a frozen, generation-tagged *saturated* extraction `a = Lim(Res(U))`. Its
admissibility law — `saturated == True ⇒ admissible`, the fixpoint witness as the
admitting certificate — is what lets it earn a generation tag at all; a budget
cutoff `Res^k(U)` may not be frozen. An admissible module's generation +
fingerprint chain into the manifest (`manifest_for(..., memory=…)`), and R13
(`verify.verify_memory`, folded into `verify_provenance`) independently
re-resolves the stored representative to confirm it is a genuine fixpoint before
admitting it. This ties the building-blocks engine (the e-graph) to the
version-DAG spine (the manifest) with one checkable law.

The legality laws (R1–R12), lane semantics, and hazard contracts are **never
learnable**: they are laws, not preferences.

## 14. The two-truth separation (MOPC)

What makes §13 *enforceable* rather than aspirational is that BCIR carries **two
distinct kinds of truth** and quarantines them apart (`kbcir.twotruth`):

- **Classical truth `v`** — deterministic, binary, generation-independent: the
  legality verdicts of the R-laws. A claim is legal or it is not; a manifest
  reproduces or it does not; a memory module is a fixpoint or it is not. There is
  no "0.7 legal." This is the only truth `verify.*` speaks (a `Diagnostic` carries
  no confidence).
- **Graded truth `(v, w)`** — a *graded proposition*: a value carried with a
  confidence `w ∈ [0,1]`. This is the learned/measured machinery — the softdp
  plan posterior (§2), the bayescal conformal interval (§13 L1), the regret
  ledger's evidence (§13 L3). It answers *which legal plan is best*, never
  *whether a plan is legal*.

**The quarantine (the single most important discipline, enforced not stated): a
graded proposition may inform but never become a legality verdict.** Graded truth
is kept out of the verifier. The only sanctioned crossing is a `decide` — an
explicit, *recorded* collapse of a graded proposition to a classical value at a
frozen threshold (the anneal/freeze of §2/§13 made auditable). The crossing is
never silent, and **R13** (`verify.verify_quarantine`) is the guard that no
confidence-weighted value reaches the R-laws except as the classical value of a
recorded decision. The graded algebra (`g_and`/`g_or`/`g_not` — "dynamic truth
tables that learn") lives entirely on the graded side: it proposes, and the
classical laws dispose. This is the safe way to import learned dynamic truth —
keep it out of the verifier.

## 15. The enriched-operad memory interface (the higher intelligence layer)

The memory module (§11, `a = Lim(Res(U))`) is already an **operad**: its e-nodes
are operations, the operators are the composition `γ`, the atoms are the identity
`η`, and the extraction tree is the operad's operation tree. The higher
intelligence layer **enriches** that operad with labels and indexes
(`kbcir.operad`, `O_L = ((O_L(n)), γ_L, η_L, L, I)`) to make memory navigable,
traceable, and queryable — without touching the deterministic spine:

- **Labeling `L`.** A hierarchical, descriptive label per operation
  (`L(op) = (L1, L2, …)`, e.g. `("MEMORY","op","mul")`). Composition preserves it,
  `L(γ_L(…)) = f_L(L(parent), L(children…))` (`f_label`).
- **Indexing `I`.** A **content-addressed** index — the FNV fingerprint of
  `(name, label, child indexes)` — kept consistent under composition
  `I(γ_L(…)) = f_I(…)` (`f_index`). Content addressing (**not** random UUIDs) is
  the discipline that keeps the layer deterministic: structurally identical
  operations get the *same* index, so CSE / the liked-pair identity `a = a` falls
  out for free and reproducibility is preserved.
- **Trace.** Reverse mapping from any operation to its constituent sources (the
  operation tree `T = (V,E)`, the `SourceMap`); rewrites are recorded as
  **2-cells** (the higher-category layer: transformations between operations).

**Where it sits.** Labels and indexes are *interpretive* metadata, quarantined on
the graded side of §14: they may **inform** planning, retrieval, and debugging but
are never read by the R-laws. The lower IR (StreamPack, realized plan) carries
decisions, not labels — so the layer is conditionally activatable
(`enable_labeling` / `enable_indexing`), matching the cost/benefit tiering: full
on the memory interface, selective in pipelines, off on the hot path. Its own
integrity (label consistency, content-addressed index uniqueness, mapping
integrity) is checkable under R13 (`verify.verify_enriched`) — the analog of
`verify_memory` for the enriched structure. `enrich_memory` lifts a frozen memory
module into this operad: the deterministic fixpoint, made intelligent.

## 16. Milestone map

1. LangRef v0.1 — this document. ✔
2. Declarative dialect definitions — `mlir/include/BCIR/*.td`. ✔ (tblgen-validated; compiled `bcir-opt` parses + verifies the pretty corpus on LLVM 18)
3. Verifier-first compiler. ✔ (laws R1–R13: the oracle runs the full chain — module R1–R7, plan R8–R9, stream R10–R11, lowering R12, provenance R13 — and the MLIR-native `-bcir-verify` enforces all thirteen structurally, negative-tested per law)
4. Rewrite laws. ◑ (MLIR-native `-bcir-promote-lanes` (GGG→UX); the rest authored as `bcir.opt.*` IR + run in the oracle; the **composition** engine is an e-graph / equality saturation (`kbcir.egraph`) whose saturated extractions freeze into generation-tagged **memory modules** `a = Lim(Res(U))` (`kbcir.memory`, R13: `saturated ⇒ admissible`))
5. K_BCIR planner — candidate-path/costvec/selected-path IR. ◑ (runnable in `bcir/`: the scalarized rail, the constrained RCSP/Pareto rail (`kbcir.rcsp`), and the (max,+) overlap price (`gem.overlap`); now MLIR-native too — `-bcir-select-realization` recomputes the min-plus `cost·weights` and reproduces the oracle's 7808 cool / 9472 under the thermal cap)
6. GEM hydration — GraphPlan/LanePlan/StreamPack IR. ◑ (runnable in `bcir/`: hydration, duration-aware EFT/token scheduling (`gem.schedule`), pipelined v2 packs; the MLIR-native GEM pipeline passes `-bcir-classify-lanes / -batch / -schedule / -lower-to-llvm` mirror the oracle stages and are cross-checked against it)
7. LLVM as first backend. ◑ (MLIR-native `-convert-bcir-to-llvm` lowers compute/barrier to the LLVM dialect; `-bcir-lower-to-llvm` checks the GEM StreamPack lowering contract (R12); oracle AOT (clang) + JIT (lli); plus a portable **C23 kernel backend** (`lower.c_kernel`) — width-driven, `restrict`, bounds-safe, library-first, R12-checked (`verify.verify_c_lowering`) — for any resident toolchain)
8. Physics-anchored calibration + learning placement (§13). ✔ (microbench harness → frozen Q8 tables (`kbcir.microbench`); policy portfolio + replay gate (`kbcir.portfolio`); the L0 prohibition is normative; the calibration loop is **closed + certified** (`kbcir.calibloop`: measure → freeze → replan → a generation-tagged certified replan win, R13); certificates verified under R8/R9/R13)
9. R13 policy provenance + the regret ledger. ✔ (`verify_provenance` / `-bcir-verify` R13; `kbcir.regret` — the boundary dashboard; the third-order loop is measured and certified, actuation human by policy)

Until the MLIR toolchain exists on this host, the oracle (`bcir/`, runnable via
`python -m bcir.run`) demonstrates Milestones 5–7 in miniature and is the
conformance reference for the dialects.

## 17. Thesis

> BCIR is a registry-first, phase-ordered, lane-typed, cost-governed
> correspondence IR. K_BCIR is the IR-level optimization calculus that selects
> legal physical realization paths. GEM is the execution IR that hydrates
> selected correspondence paths into streamed lane schedules. MLIR is the
> bootstrap framework used to define, verify, rewrite, and lower BCIR until BCIR
> has enough mass to become its own compiler toolchain.
