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
(`x+0→x`, `1+1→2`, `a*b+a*c→a*(b+c)`). **Resolution** is saturation (apply all
legal rewrites to a fixpoint or a budget — the bound on complexity generation)
followed by **extraction** of the min-cost representative per class. Because
extraction returns the minimum, the optimized cost is always ≤ the input cost —
an R9 obligation. Nothing is globally immutable: a liked pair holds *within* a
generation; across generations it is an unliked pair resolved by rehydration, and
the provenance manifest pins identity.

## 12. Lowering contracts

BCIR-4 → BCIR-5 lowering is governed by R12: each lowered op preserves the BCIR
semantic (lane geometry, bounds, hazard, precision) or carries an explicit
discharge in `bcir.trace`. LLVM is the **first** backend, not the center.
Encoded via `bcir.isa.*` / `bcir.target.lower_contract`.

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
  R8/R13.
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

The legality laws (R1–R12), lane semantics, and hazard contracts are **never
learnable**: they are laws, not preferences.

## 15. Milestone map

1. LangRef v0.1 — this document. ✔
2. Declarative dialect definitions — `mlir/include/BCIR/*.td`. ✔ (tblgen-validated; compiled `bcir-opt` parses + verifies the pretty corpus on LLVM 18)
3. Verifier-first compiler. ✔ (laws R1–R13: the oracle runs the full chain — module R1–R7, plan R8–R9, stream R10–R11, lowering R12, provenance R13 — and the MLIR-native `-bcir-verify` enforces all thirteen structurally, negative-tested per law)
4. Rewrite laws. ◑ (MLIR-native `-bcir-promote-lanes` (GGG→UX); the rest authored as `bcir.opt.*` IR + run in the oracle)
5. K_BCIR planner — candidate-path/costvec/selected-path IR. ◑ (runnable in `bcir/`: the scalarized rail, the constrained RCSP/Pareto rail (`kbcir.rcsp`), and the (max,+) overlap price (`gem.overlap`))
6. GEM hydration — GraphPlan/LanePlan/StreamPack IR. ◑ (runnable in `bcir/`: hydration, duration-aware EFT/token scheduling (`gem.schedule`), and pipelined v2 packs (`hydrate_pipelined`))
7. LLVM as first backend. ◑ (MLIR-native `-convert-bcir-to-llvm` lowers compute/barrier to the LLVM dialect; oracle AOT (clang) + JIT (lli))
8. Physics-anchored calibration + learning placement (§13). ✔ (microbench harness → frozen Q8 tables (`kbcir.microbench`); policy portfolio + replay gate (`kbcir.portfolio`); the L0 prohibition is normative; certificates verified under R8/R9)
9. R13 policy provenance + the regret ledger. ✔ (`verify_provenance` / `-bcir-verify` R13; `kbcir.regret` — the boundary dashboard; the third-order loop is measured and certified, actuation human by policy)

Until the MLIR toolchain exists on this host, the oracle (`bcir/`, runnable via
`python -m bcir.run`) demonstrates Milestones 5–7 in miniature and is the
conformance reference for the dialects.

## 16. Thesis

> BCIR is a registry-first, phase-ordered, lane-typed, cost-governed
> correspondence IR. K_BCIR is the IR-level optimization calculus that selects
> legal physical realization paths. GEM is the execution IR that hydrates
> selected correspondence paths into streamed lane schedules. MLIR is the
> bootstrap framework used to define, verify, rewrite, and lower BCIR until BCIR
> has enough mass to become its own compiler toolchain.
