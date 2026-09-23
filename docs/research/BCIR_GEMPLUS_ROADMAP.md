# GEM+ build roadmap

The build plan for GEM+, TMSAO certification, and the optimizer work behind them. Every slice
below names the measured numbers it must move, the comparative test that decides whether it
did, and what happens in each of the three possible outcomes.

Grounded in three documents, all in this tree:

- [`BCIR_TMSAO_ARCHITECTURE_AND_PERFORMANCE_REPORT.md`](BCIR_TMSAO_ARCHITECTURE_AND_PERFORMANCE_REPORT.md)
  — the measured baseline. Every target number here is quoted from it.
- [`BCIR_TMSAO_ASN1_JSON_DRIVER_PROPOSAL.md`](BCIR_TMSAO_ASN1_JSON_DRIVER_PROPOSAL.md) — the
  architecture proposal this refines into build steps.
- [`BCIR_SECURITY_AUDIT_2026-08-12.md`](BCIR_SECURITY_AUDIT_2026-08-12.md) and
  [`BCIR_SECURITY_AUDIT_2026-08-12b.md`](BCIR_SECURITY_AUDIT_2026-08-12b.md) — the correctness
  closure that had to land first, and the two items it deliberately left for this work.
- [`BCIR_GEMPLUS_TMSAO_STAGED_PLAN_2026-09-04.md`](BCIR_GEMPLUS_TMSAO_STAGED_PLAN_2026-09-04.md)
  — the 2026-09-04 review that re-staged this roadmap: the disposition of the earlier
  assessment's findings against the tree, today's re-measurement, the contracts the slices
  were missing (G11–G18), and the PR-sized sections in order.

---

## 0. The measurement discipline, and why it comes first

The instruction this roadmap is built to satisfy: *a comparative test per update; no gain means
investigate why; a gain means analyse its magnitude and how much more is still available.*
That is a loop, and a loop needs a fixed point to measure against, so the baseline is frozen in
code before any slice is written:

```
python tools/perf/gemplus_baseline.py --list      # 24 frozen metrics and their floors
python tools/perf/gemplus_baseline.py --compare   # re-measure and grade
```

### 0.1 Three kinds of row, and only two of them gate

Building the harness taught something that changes how every slice below is judged, and it was
learned by getting it wrong:

> The harness's first run called `optimize_scheduled.512` a **15.9% REGRESSION**. It was not
> one — the baseline host is a Ryzen 5 2600 under WSL with Python 3.10, and the measuring host
> was neither. The *same run* measured the slowdown **ratio** at 68.95 against a baseline of
> 69.2: **0.4% apart, on an unrelated machine.**

So metrics are classified by what survives a change of machine:

| kind | example | grades on any host? |
|---|---|---|
| `exact` | a solver's optimum, a fraction over a fixed corpus, a byte extent | **yes** — deterministic, zero variance |
| `ratio` | a ratio of two *timed* quantities | yes, with a wide band — cancels the machine's speed, not its variance |
| `wall` | an absolute millisecond | **no** — reported `INDICATIVE`, never blocks a slice |

The `ratio` band is 25% because the same host produced 68.95 and then 76.29 minutes apart. The
`exact` band is 2% because a solver's optimum does not vary at all.

**A slice is gated on its `exact` rows.** Wall rows are evidence for a human, not a gate. This
is not a lowering of standards — it is the only way the gate means the same thing on CI, on a
laptop, and on the report's host.

The confirmation that this classification is right: the §6.2 divergence fixture in the harness
reproduces the report's ratio as **1.9922178988** against the report's 51200/25700 = **1.9922**,
while its absolute costs are 4× the report's. The structure carries; the milliseconds do not.

### 0.2 The per-slice analysis protocol

Every slice ends with this, written into its PR:

**On GAIN** — three numbers, not one:
1. **Magnitude**: the measured improvement, on `exact` rows first.
2. **Remaining headroom**: `(incumbent − bound) / incumbent` from the harness. A 5% gain on a
   row with 90% headroom left is a different result from a 5% gain that closes the gap, and
   the certificate has to say which.
3. **What the residual gap is made of**: name the mechanism still costing the difference. "The
   remaining 40% is the digest recomputation G3 has not landed yet" is an answer; "further
   optimization possible" is not. This is what promotes a slice's claim from TMSAO-4 to
   TMSAO-2 — a gap is only a gap when its size is known.

**On NO-CHANGE** — a finding, not a pass. Exactly one of:
- **(a) mis-assigned** — the slice does not touch this path. Move the metric to the slice that
  does, and say which.
- **(b) cancelled** — the win happened and something else consumed it. The cost model is
  incomplete; find the consumer and add a metric for it.
- **(c) already at the bound** — the operation cannot improve. Prove it against the bound and
  **retire the row**, recording the proof. This outcome is a success and should be reported as
  one: a metric retired with a proof is a closed optimality question.

**On REGRESSION** — the slice does not land until it is explained. If the regression is a
deliberate trade (exactness bought with time, say), the trade is stated and a new metric is
added for the thing that got better.

### 0.3 Rows with no bound yet

Five rows have no lower bound computed:

```
static_memory.digest.2048        audit.kbcir-streampack.scale4
audit.mixed-wave-token-eft.scale4  audit.static-lifetime-planner.scale4
audit.iterative-phase-dag.scale4
```

**No optimality claim is available for these until a bound exists.** Computing one is
in-scope work, not paperwork: for the digest row the bound is an Ω(n) argument over the bytes
that must be hashed, and for the audit rows it is the critical path plus the work/capacity
bound from the lower-bound stack (§4). A slice that improves one of these without computing
its floor produces a TMSAO-4 claim and nothing better.

---

## 1. Scope identity: `S`

The proposal's

```
S = digest(P, H, W, Theta, A, B, O, M, U, G)
```

is what every certificate is *about*. Nothing below can be trusted before it exists, because
today's digest is provably incomplete — the 2026-08-12 audit showed that scaling a DRAM tier's
factors by 32 moves a plan's score from **51,200 to 1,574,912 with the digest unchanged**.

| Component | Content | Where it is today |
|---|---|---|
| `P` | program, input contract, R-laws, semantics, precision, admitted approximation | `hash_module` (claims in declared order since S0-D; the scope names the order as a component) |
| `H` | topology, ISA/capabilities, banks, links, capacities | `hash_target` (the memory hierarchy folded since S0-D, from `target.capability` `mem_tier_names` / `mem_tier_values` on the law rail) |
| `W` | workload shapes, input distribution, concurrency, SLOs, horizon | `kbcir.workload.Workload` (S2-E): shapes, batch, concurrency, service level, horizon, the expected counts of dynamic claims — declared, digested with the scope, held to the module |
| `Θ` | firmware, microcode, driver, OS, clocks, thermal, contention, wear | `hash_theta`, partial |
| `A` | admitted transformations, libraries, kernels, schedules, search boundary | implicit in `candidates_for` |
| `B` | capacity, security, reliability, temperature, power, policy caps | `Budget`, partial |
| `O` | objective relation: lexicographic, Pareto, robust, constrained, scalarized | scalarized only |
| `M` | measurement protocol, warm-up, sampling, counters, outliers, environment | `EncodingCostTable`, partial |
| `U` | uncertainty model, confidence/prediction coverage | intervals exist for JER J6 only |
| `G` | generations of profile, calibration, firmware, driver, model, artifacts | `cal_gen`, `map_gen`, `data_gen`, partial |

**The hard part is not the list, it is that `P` and `H` are recomputed in C++.**
`BCIRVerifyPass.cpp`'s `hashModuleFromIR` and `hashTargetFromIR` walk the IR field by field for
R13's cross-check. Widening either hash is a two-rail change that must land in one commit, or
the rails silently disagree about a content address — which is exactly why the security audit
fixed the *consequence* (`replay()` now compares the produced plan, not just the digest) and
left the hash itself to this roadmap.

---

## 2. Certificate classes

| Class | Permitted statement | What must exist |
|---|---|---|
| **TMSAO-1** | exact optimum over the declared finite model | complete candidate census + proof |
| **TMSAO-2** | best bounded result | valid lower bound + absolute and relative gap |
| **TMSAO-3** | best measured admitted realization | prediction interval + search coverage |
| **TMSAO-4** | heuristic incumbent | legality and reproducibility only — **no optimality claim** |

Every certificate carries: all scope hashes and generations; legality verdicts and replay
identities; candidate census or search coverage; incumbent `U` and lower bound `L`; absolute
gap `U−L` and relative gap `(U−L)/max(|U|,ε)`; solver, budget, stopping reason, deterministic
tie-break; the complete execution plan and target code identity; raw samples, intervals,
outlier policy and unavailable counters; rollback artifact and invalidation predicates.

**Everything BCIR emits today is TMSAO-4.** The audit's §6.6 finding is the reason: RCSP is
exact over the *layered candidate graph* only, and the composed problem — selection, schedule,
placement, memory — is not solved exactly by anything. Saying so is the precondition for
improving it.

The ladder is also the roadmap's success measure. **G4 is the first slice that can produce a
TMSAO-2 certificate**, because it is the first to compute a lower bound.

---

## 3. The slices

Ordered by dependency, not by size. Each names its gate.

### G0 — `ExecutionScopeV1` and the class ladder — **LANDED**

*Closes: the audit's open provenance item. Report P0. Unlocks: every later certificate.*

`bcir/kbcir/scope.py`. The versioned canonical serialization of `S`, plus the four
certificate classes and the gap arithmetic.

| Gate | Baseline | Result |
|---|---|---|
| `exact` A scope digest separates two plans that differ | collided (51,200 vs 1,574,912, one digest) | **distinct**, `diff` names `H` |
| `exact` Declared claim order changes the digest | collided (3,840 vs 4,352, one digest) | **distinct**, `diff` names `P` |
| `exact` An optimality class is refused over an undeclared model | n/a | TMSAO-1/2 refused without `P`, `H`, `A`, `O` |
| `exact` Equal scopes serialize to equal bytes | n/a | insertion order and set order both canonical |

**The design decision worth recording, because the obvious approach was wrong.** The direct
repair is to add the memory tiers to `hash_target`. That is a *cross-rail* hash:
`BCIRVerifyPass.cpp` recomputes it field for field for R13, and `TargetCapabilityOp` carries
no ODS attribute for the tiers, so widening it is a dialect change that must land on both
rails in one commit or they silently disagree about a content address.

So the two jobs were separated rather than merged. `hash_module`/`hash_target` keep doing
cross-rail agreement, unchanged and still correct at it; `ExecutionScopeV1` is the complete
identity certificates bind to, and it *contains* the target hash as one field. The gap closes
now, R13 keeps working, and the dialect change becomes optional rather than blocking.

**Still open from this slice**, deliberately and separately:

- ~~widening `hash_target` with a `DenseI64ArrayAttr` for the tiers plus the matching C++ walk,
  so the MLIR rail can recompute the same complete identity.~~ **Landed as S0-D** (2026-09-05):
  `mem_tier_names` / `mem_tier_values` on `target.capability`, the widened `hashTargetFromIR` / `hashModuleFromIR`
  (declared claim order), the emitter writing every hashed field, and the two-rail regression
  `test_hash_parity.py`.
- the `wall` row `static_memory.plan.2048` (301.02 ms): untouched here, and it is G3's.

### G1 — one canonical schedule artifact — **landed (S1-A, 2026-09-05)**

*Report P0.1. The audit's sharpest correctness finding.*

`price_scheduled` (fixed waves, round-robin bins) and `schedule_eft` (LPT/EFT with locality)
were two different algorithms, and the objective read one while the executor ran the other.

| Gate | Baseline | Target | Outcome |
|---|---|---|---|
| `exact` `pricing.eft.divergence` | **1.9922** | **1.0** — both read one artifact | **1.0 exactly** (GAIN, at the bound); `price_waves_legacy` still measures 1.9922 on the same fixture — the witness |
| `exact` Token execution and the priced schedule agree | not checked | identical slot assignment | **met**: `schedule_plan` is the artifact; `price_scheduled(...).schedule.slots == execute_tokens(...).slots` (and `schedule_eft`'s) on every corpus program, both modes |
| `wall` `audit.mixed-wave-token-eft.scale4` | 48.05 ms | no regression | 26.43 → 26.07 ms A/B on the same host, result digest identical (13/13 audit cases identical) |

What landed: `gem.schedule.schedule_plan` places the plan's own step costs by the hazard-
honoring LPT/EFT dispatch, with `concurrency.hazard_predecessors` — RAW/WAR/WAW plus the
`barriered`/`volatile` fence edges — built over every claim of a phase BEFORE the tail split
and the sparse tail dispatched on its own stream inside the same loop (assessment row 6: a
gather no longer starts before its producer finishes; a fence is overlapped by nothing);
`price_scheduled` reads that placement (`.schedule` is the artifact), `optimize_scheduled`
sweeps it, and `BCIRSchedule.h` is the law-rail twin shared by `-bcir-overlap`,
`-bcir-schedule-eft`, `-bcir-async` and `-bcir-power-rail` (`schedule_hazards.mlir`; the gate
checks the priced makespan equals the placed one on the corpus). The CSE credit now requires
the complete semantic identity and excludes atomic, barriered, volatile, effectful, sparse and
isolated-domain claims on both rails (row 7; `cost_model_cse_neg.mlir`). The schedule-
predecessor re-coupling of the old pricer is retired with it: durations are the plan's step
costs, the sum of the durations is the serial bound, and R9 holds by construction.

This is a **correctness** metric wearing a performance costume. Any value but 1.0 means
`M(π,Θ)` denotes two things, and no certificate above TMSAO-4 is possible while it does.

### G2 — incremental delta pricing — **landed (S2-A, 2026-09-13)**

*Report P1.4. The largest visible win in the baseline.*

`optimize_scheduled` recomputes the whole makespan per trial. Cache per-phase/bin
contributions and reprice only the affected chain.

| Gate | Baseline | Target | Headroom today |
|---|---|---|---|
| `ratio` `optimize_scheduled.slowdown.512` | 69.2× | ≤ 8× | **6.27× after S1-A**; **3.5× after S2-A** (A/B on one idle host, 6.4× → 3.5×: the price read from the placer's records instead of a second placement, the candidate map built once) — under the harness bound of 4×; the last of the fixed overhead is the serial chain's O(n²) hazard edges |
| `wall` `optimize_scheduled.512` | 1,703.66 ms | ≤ 200 ms | **83 ms after S1-A**, **60 ms after S2-A** (98 → 60 ms A/B on one host; 1,148 ms on the same host before S1-A; indicative) |
| `wall` `optimize_scheduled.256` | 435.73 ms | ≤ 100 ms | **29 ms after S1-A**, **23 ms after S2-A** (35 → 23 ms A/B; 287 ms before S1-A; indicative) |
| `exact` The plan chosen is unchanged | — | **identical assignment** | **met**: the delta search returns the reference sweep's assignment claim by claim, with the same step costs, price and artifact, on 1,344 (fixture, target, Θ, policy) cases — 12 corpus programs, the harness fixtures, the general-case and adoption fixtures and 40 generated hazard-bearing modules |
| `ratio` `optimize_scheduled.general.slowdown.512` (the general case: 256 step-shortening trials, 16 phases) | 44.5× (S1-D, this host) | ≤ 8× | **4.7×** (605 → 67 ms A/B on one host) |
| `exact` `sweep.replacement.fraction` (claims re-placed per trial / claims) | 1.0 | the affected phase (1/16) | **0.0625**, at the bound |

The last row is the one that matters. A faster sweep that picks a *different* plan has not
been made faster; it has been changed. Delta pricing must be an exact refactor of the same
search, proved by comparing assignments claim by claim.

S1-A moved the first three rows, with the mechanism named: the canonical pricer made a full
re-placement per trial cost ~8 ms at 512 claims (1,100× the serial pass — the harness refused
it as a regression), so the sweep now builds the hazard DAG once, re-prices only the two steps
an alternative touches, and places only alternatives that SHORTEN one of them (a step that only
lengthens cannot lower a placement of step costs except through a list-scheduling anomaly,
which is not a property of the plan). The assignment is identical to the exhaustive sweep's on
the 11 corpus programs, the 256- and 512-claim harness fixtures and 150 generated modules under
cool and hot Θ and two policies — and in every one of those cases both sweeps return the serial
optimum, because under the coupled cost model a wider lane is never longer for its own step and
the successor coupling does not distinguish vector widths. What remained for G2 was the general
case — many step-shortening trials — where each trial still re-placed the whole module.

S2-A closed it with the report's own mechanism, made exact. The general case is reachable with
the real cost model: under the ENERGY policy a 64-element claim's scalar realization is cheaper
for its own step than vec8, but the serial optimum picks vec8 when a large vector successor
shares a read operand (the locality discount on the successor outweighs the claim's own extra
cost), so the scalar alternative shortens its own step and the sweep must place it —
`bcir/tests/sweep_fixtures.general_fixture` has one such trial per pair of claims, 256 trials at
512 claims, and measured RED at 44.5× the serial pass with every trial re-placing all 512
claims. `gem.schedule.EftPlacer` records the base placement once — per phase: the span, the
residency at entry and exit, the pop order — and prices a trial as the cached prefix, the
changed phase replayed from the last checkpoint at or before the first pop the change can move
(the pop order is a function of the hazard edges and the durations alone: a shortened claim is
popped no earlier than before, a lengthened one at the first base pop whose key its new key
beats), and every later phase skipped with its cached span unless it touches a rid the replay
moved between streams. The dispatch itself is one predicate (`_PhaseDispatch.run`, of which
`_dispatch` is the one-shot form) so `schedule_eft`, `execute_tokens` and the replay cannot
drift, and the placer's `schedule()` is `schedule_eft`'s artifact on every fixture, trial and
adoption (6,840 random trials, 1,690 adoptions). The sweep also stopped paying for what it
already had: the candidate map `optimize` built and the final artifact the placer holds. Not
claimed: a sub-linear replay inside one phase of independent claims — there the LPT order puts
a lengthened successor first and the placement is inherently sequential (the single-phase
general fixture goes 44× → 27×); the phase-level mechanism is what the report named, and the
harness fixture's residual is the serial chain's O(n²) hazard edges, which the dispatch reads
once per placement.

### G3 — canonical digest computed once — **landed (S1-B, 2026-09-12)**

*Report P1.6. Cheap, and it unblocks measurement everywhere else.*

The profile found module hashing at ~0.65 s of a 4.8M-call run, with three independent hashes
of one immutable module: the planner hashes, the verifier hashes again, an independent client
hashes a third time. Replace recursive canonical flattening with an iterative stream, cache
immutable module identity, and expose an identity-bound API — **while keeping the verifier's
right to recompute at a trust boundary**, which is the whole reason the third hash exists.

| Gate | Baseline | Target | Outcome (A/B on one host; the baseline host is ~1.4× slower) |
|---|---|---|---|
| `exact` `static_memory.digests.2048` (full digests over plan + verify + client) | 3 | 1 | **1** — the planner mints the identity, its verifier and the client validate it by content |
| `wall` `static_memory.digest.2048` | 88.05 ms | ≤ 30 ms | 64.7 → **37.4 ms** here (projected ~50 ms on the baseline host: the per-byte FNV chain in the interpreter is the residual) |
| `wall` `static_memory.verify.2048` | 157.88 ms | ≤ 90 ms | 100.3 → **39.3 ms** identity-bound (71 ms recomputing); the bound is met; **15.8 ms** since S1-D's alias sweep |
| `wall` `static_memory.plan.2048` | 301.02 ms | ≤ 120 ms | 196.1 → **114.1 ms** from a fresh identity (70 ms with it cached); the bound is met; **84.3 ms** since S1-D's alias sweep |
| `exact` A cache cannot survive mutation | — | mutation invalidates; cross-module substitution refused | **witnessed**: a declared mutation drops the cache; an undeclared in-place edit is refused by the verifier's content check; module A's identity presented for module B is refused |

What landed: `provenance.canonical_stream` is the one iterative walk that produces the R13 item
sequence; `hash_module` chains it (bit-identical to the recursive flattening on the corpus,
60 generated modules and the 2,048-resource fixture, so no pinned digest moved on either rail);
`module_identity(module)` computes the digest once per module revision and caches it on the
module (`Module.revision`, bumped by `add_resource`, `add_phase` and `touch()`), and
`digest_of(module, identity)` is the verifier's side of the identity-bound API: it validates the
identity against the module's canonical stream — a complete content check at the cost of the walk
(3 ms at 2,048 resources against 37 ms for the digest) — and recomputes otherwise, refusing
under `strict`. `plan_static_memory` mints the identity once and hands it to its internal
verify; an external `verify_static_memory_plan(..., identity=...)` validates it; `build_manifest`
and `scope_for` read it; R13's `verify_manifest`, `replay` and `reproduces` recompute (`fresh`),
which is the trust boundary the third hash exists for. The law rail is unchanged: `-bcir-verify`
recomputes `hashModuleFromIR` from the IR by design.

The security half of that last row is not optional: a digest cache that survives a mutation is
the Class-B "vacuous check" defect from the audit, rebuilt. The cache here is keyed on the
revision and re-checked against the module's census, and no verifier trusts the revision: an
identity is accepted only when the module's content is exactly what it describes.

### G4 — bounded exact solvers and the lower-bound stack — **landed (S2-B, 2026-09-13)**

*Report P1.5. **The first slice that can emit TMSAO-2.***

Add dependency-free branch-and-bound for small fixtures, and the lower-bound stack: critical
path, work/capacity, hierarchical Roofline, communication cut, queue/network-calculus,
allocation peak/clique, occupancy, energy-at-minimum-work. Report the **maximum valid** bound
against the incumbent.

| Gate | Baseline | Target | Outcome |
|---|---|---|---|
| `exact` `eft.suboptimal.2domains` | 11.07% of 1,716 | 0% on the proof rail, or a stated gap on every instance | **0%** — every instance proved (stop reason `optimal`, at most 3,076 expansions); the heuristic alone still reads 190 / 1,716, kept as the witness row `eft.heuristic.suboptimal.2domains` |
| `exact` `eft.worst.2domains` | 1.1333× | 1.0 on the proof rail | **1.0** (the heuristic's 17/15 kept as the witness); the report's three-domain row added and closed the same way (18 / 1,716, 7/6 → 0 / 1.0) |
| `exact` `optimize_scheduled.quality` | 1.00696× | 1.0 on the proof rail | **1.0** — the one-sweep re-selection equals the exhaustive candidate enumeration on every module of `exact_fixtures.quality_corpus` (56 shaped and seeded four-claim modules × two policies); the report's 55,552 / 55,168 fixture was the retired wave pricer's and is not reconstructible under the canonical artifact |
| `exact` Every certificate carries `L`, `U`, and both gaps | absent | present | **present**: `gem.exact.certify_schedule` binds `L`, `U`, the heuristic's and the incumbent's gap, the stop reason, the budget and the bound stack to the `ExecutionScopeV1` digest; TMSAO-1 when the search closes, TMSAO-2 on a budget stop, TMSAO-4 with the reason when the scope is undeclared |
| `exact` `solver.unproved.fraction` / `solver.gap.p95` (Stage 2 exit) | 1.0 / 0.0625 (the heuristic's own p95 gap) | 0 / 0 | **0 / 0** over the pooled 2- and 3-domain corpus |

**The gap is the product, not the speed.** A slice that leaves the heuristic exactly as fast
and merely states how far from optimal it is has still moved BCIR from TMSAO-4 to TMSAO-2, and
that is a bigger step than any constant factor in this document.

What landed (S2-B): `gem.exact.exact_schedule`, a dependency-free branch-and-bound over one
phase's active schedules (phase barriers compose serially, so the module's optimum is the sum
of its phases' optima) under the artifact's own eligibility rules — the tail stream for sparse
claims, the knee for bandwidth claims — with the incumbent seeded by the heuristic's placement,
symmetry breaking on interchangeable empty streams and identical claims, a budget in node
expansions and, on a budget stop, the least bound over the subtrees never entered as `L`. The
bound stack at every node is the strongest of critical path, work over the streams' frontier,
bandwidth work over the knee and the tail's serial work, each named in the certificate. The
corpus of section 6.1 is pinned and reproduced exactly (`bcir/tests/exact_fixtures.py`: all
1,716 nondecreasing six-job multisets with values 1–8; `schedule_eft` gives the report's 190 /
1.0078 / 17:15 and 18 / 1.0013 / 7:6), and the solver is held to oracles that share no code with
it — the partition optimum on that corpus, the enumeration of every active schedule on 108
hazard-bearing tiny modules. `exact_selection` is the same discipline for the schedule-aware
candidate selection. The artifact is never replaced — the placement every rail reads and the
twins reproduce stays `schedule_eft`'s; the exact rail certifies it. Not claimed: proofs at
production scale (a budget stop states its gap), the remaining members of the stack the report
lists (hierarchical roofline, communication cut, queue calculus, occupancy, energy at minimum
work — bounds on quantities the schedule does not yet model), and the interval-graph memory DP
(still with G5's residual).

### G5 — schedule-aware liveness and bounded exact memory — **landed (S1-D, 2026-09-12)**

*Report P0.2 + P1.5. Carries a latent-correctness condition.*

`live_intervals` uses topological phase positions; `execute_tokens` legally overlaps phases.
The audit did **not** find a deployed corruption path — the static map is not currently fed
into token execution — and established the integration condition instead: *static memory must
be computed from the final schedule's intervals, or the verifier must prove the schedule
refines the phase order the plan used.*

| Gate | Baseline | Target | Outcome |
|---|---|---|---|
| `exact` The two-phase alias fixture | aliases at offset 0 | rejected, or disjoint storage | **both**: the phase-liveness plan held to the token placement is refused ("does not refine the phase order"); a plan computed from the placement gives the two disjoint storage, and one computed from the barriered placement still shares offset 0 because it is safe there |
| `exact` `memory.suboptimal.fraction` | 38.6% of 500 | 0% on the proof rail, or a stated gap | **0%** — every fixture of the corpus solved to a proved optimum within the budget (first-fit alone: 40.4% on this corpus, measured RED) |
| `exact` `memory.worst.ratio` | 1.6154× (21 vs 13 units) | 1.0 on the proof rail | **1.0** (first-fit alone: 1.6× on this corpus) |
| `exact` `memory.real.bytes` | 1,344 B | 832 B on the proof rail | **832 B** on the witness that reproduces the report's worst case through the real planner (first-fit: 1,344 B, the report's number) |

First-fit stays the predictable fast path (byte-identical to the historical layout under phase
liveness: the audit fixture's offsets did not move), and the verifier's alias law is an exact
sweep over the ticks with the live rows' addresses in one sorted list — A/B on one host against
the S1-C commit, `static_memory.verify.2048` 35.0 → 15.8 ms and `static_memory.plan.2048`
100.7 → 84.3 ms with all 13 audit result digests identical. What landed: `static_memory.schedule_intervals`
derives every resource's half-open liveness interval from the canonical placement (the first
touching slot's start to the last one's finish), `plan_static_memory(..., schedule=)` computes
the plan in that domain and binds it to the placement by digest, and every plan names its
`liveness` domain; `verify_static_memory_plan(..., schedule=)` refuses a phase-liveness plan the
placement does not refine and recomputes a schedule-liveness plan's intervals from it.
`exact_layout` is the bounded exact solver — a complete branch-and-bound over aligned offsets
below the first-fit incumbent, budgeted in candidate placements (work units, never seconds) —
and every bank summary records the concurrent-live lower bound, the extent, the gap and the
stop reason (`first-fit`, `optimal`, `budget`); the verifier re-runs the solver under the plan's
own budget rather than trust the stop reason or the offsets. The lifetimes travel in
`ExecutionPlanV1` v2 (the liveness byte and the tick tail, append-only, the lowest carrying
version emitted), the C twin and the codec refuse two lifetimes of one bank that alias, and
`verify_execution_plan` refuses a plan whose lifetimes do not cover its schedule. The report's
corpus is not in the tree; `bcir/tests/memory_fixtures.py` reproduces its shape (500
seven-resource fixtures) and its worst case exactly (21 vs 13 units, 1,344 vs 832 bytes). The
integration condition the report set — static memory from the final schedule's intervals, or a
verifier that proves the schedule refines the phase order — is met on both branches. Not
claimed: the exact solver at production scale (the 512-resource audit fixture stops on its
budget and keeps the first-fit incumbent with a stated gap — the interval-graph dynamic
programming the staged plan names for G4/G5 is the next lever) and any joint placement of
memory with the schedule (G6/G8).

### G6 — typed regions and the semiring registry — **landed (S2-D, 2026-09-13), affine first**

*Report P2. The architectural slice.*

A graph of typed regions, each with a strong local model **and a conservative expansion back
to claims**: affine/polyhedral, SDF/CSDF, timed-event (max-plus), tensor index-map, state
machine/Petri, equivalence graph, opaque claim DAG as the universal fallback. Plus the typed
objective registry — min-plus, max-plus, min-max, Boolean, lexicographic, Pareto, stochastic —
each verifying closure, identities, comparison semantics and overflow policy.

Every region must supply: a verifier, a conservative claim expansion, a cost/lower-bound
interface, and refusal conditions.

| Gate | Baseline | Target | Outcome (S2-D) |
|---|---|---|---|
| `ratio` `native.gather-avoidance` | 5.58× | **no regression** | 4.53× → 4.44× A/B on this host (medians of 3; the samples overlap) — no regression; the slice touches neither selection nor lowering |
| `ratio` `native.blocked-reduction` | 11.68× | **no regression** | 15.00× → 15.56× A/B — no regression |
| `ratio` `native.direct-stride` | 1.27× | **no regression** | 1.315× → 1.314× A/B — no regression |
| `ratio` `native.dense-parity` | 0.98–1.01× | stays in band | 1.003× → 1.004× A/B — in band |

The four `native.*` rows are **host-dependent ratios** and are graded only on the baseline host
(`Metric.host_dependent`; INDICATIVE elsewhere, reported and never blocking). A ratio cancels the
machine out only when both sides are timed in one process; these divide one compiled kernel by
another, so they measure this host's gather penalty rather than the code. S2-D first shipped them
graded against the report's host, where this machine's lower band tripped a REGRESSION verdict on
about one run in thirty — a gate firing on the machine. The same-host A/B above is how they are
read.
| `exact` Every region expands conservatively to claims | — | differential test per region | **met**: `regions.unexpanded.claims` 542 → 0 — `expand(region_graph(module))` is the module claim for claim on the 53-module corpus and the plan `optimize` selects over the expansion is the plan over the module; forged regions (non-consecutive claims, wrong maps, a model on an opaque region) are refused by the verifier |
| `exact` Every objective carries its laws | 2 names, no laws | verified before admission | **met**: `objectives.unverified` 2 → 0 — six entries admitted only through `verify_objective`; a lawless operator is refused |

What landed (S2-D): `kbcir.regions` — typed regions with a verifier, a conservative expansion,
a cost interface and refusal conditions, two kinds: **affine** (a maximal run of consecutive
claims of one phase whose accesses are 1-D affine maps with static trip counts, the local model
being the access maps and the dependence distances) and **opaque** (the universal fallback,
carrying the named refusal: `dynamic-trip-count`, `volatile`, `fence`, `atomic`, `sparse`,
`cacheline-indexed`, `tile`, `lane`, `stride`, `extent`, `count`); `region_graph` partitions
every phase, recognizes and re-verifies (on the corpus of 53 modules: 105 affine and 140 opaque regions), and
`region_floor` is the cost interface — for every claim the cheapest of its deforested candidates
under the most favourable coupling the access maps allow, a floor no realization of the region
can go below (never above the plan's score on the corpus × 2 targets × 2 Θ × 4 policies; tighter
by exactly the discount where no read is shared). `kbcir.objectives` — the typed objective
registry: `min_plus`, `max_plus`, `min_max`, `boolean`, `lexicographic`, `pareto`, each with its
combine/select/identities/order, the laws it claims and a declared overflow policy (`checked`
refuses outside i64, `saturate` clamps), proved by `verify_objective` before admission; one
layered relaxation `dag_best_path` reproduces the planner's min-plus path and the exact
scheduler's max-plus critical path to the digit. `gem.exact.certify_selection` and the dispatch
law's `path` kind (the report's §11.3 first row: the min-plus rail is exact) certify a plan's
selection — `L == U == optimum`, TMSAO-1 under a declared scope — and report the region graph's
structural floor with the coupling's price stated. Not claimed: the other region kinds the
report lists (SDF/CSDF, timed-event max-plus, tensor index maps, state machines, equivalence
graphs — opaque today, the conservative answer), a polyhedral depth above one (a tile claim is
opaque), a registry attribute on the law rail (`BCIR_Semiring` keeps the two shared names; a
wider attribute is a cross-rail change taken deliberately), and any silicon certificate from
the native rows on this host (its tenancy is `virtualized`).

The native rows are **guardrails, not targets**. They are what BCIR is for — the audit's §4.4
shows the wins come from preserving enough structure to avoid a gather or pick a blocked
reduction, not from graph representation itself. A regions refactor that speeds up the planner
and loses a 11.68× structural win is a net loss.

"Semiring" must not become a label that admits arbitrary operators without their laws.

### G7 — repair the native measurement rig — **landed (S0-F)**

*Report P0.3. Small, and it invalidated a number that was in use.*

The microbench used `(k * 16) % n` with a power-of-two `n`, so `gcd(n,16) = 16` and the walk
visited `n/16` unique elements — a nominal 32 MiB buffer with a **2 MiB** working set. It also
printed `native microbench (bare-metal)` under WSL and under this session's hypervisor alike.

| Gate | Baseline | Target | S0-F |
|---|---|---|---|
| `exact` Unique elements visited | n/16 | n (coprime stride or proved full-cycle permutation) | n: the walk runs the gcd(stride, n) cosets of ⟨stride⟩ in Z_n, each a full cycle, and a non-timed **census** counts the unique elements per regime (`unique=n/n/n` in the provenance; `strided_order` is the Python twin and the test pins the parent's n/gcd as the witness) |
| `exact` Provenance under virtualization | claims bare-metal | refuses the claim | the rig attests hypervisor flag, hypervisor nodes, DMI, WSL, container and the PMU event source, derives a **tenancy** (`virtualized` / `containerized` / `bare-metal` / `unproven`) and reserves "bare-metal" for no virtualization signal plus an exposed PMU; `CalibratedProfile.from_json` refuses a provenance whose claim the evidence does not attest and `calibrate_native(require_baremetal=True)` refuses every other tenancy |
| `exact` Report contents | ratios only | raw samples, intervals, working-set census, counter availability | `NativeEvidence`: one raw sample per repeat and its (min, median, max, MAD), the census and working-set bytes, PMU source, `perf_event_paranoid`, governor, RAPL, clocksource, the observed timer quantum, OS/arch/compiler; the reader re-derives the Q8 ratios from the medians and refuses a disagreeing summary |

A `native.*` row is a silicon certificate only where its evidence says `bare-metal`; on this
session's host it says `virtualized: hypervisor-flag`, which is the refusal made mechanical.

### G8 — data movement as a first-class transformation

*Report P2.10. HAM and Semantic Swap.*

A movement edge records source and destination bank, byte range, route, coherence action,
generation, event, overlap window, and whether it is direct, peer, staged, rematerialized,
compressed or evicted. The optimizer chooses movement and compute **jointly** — pricing
transfers after placement is a disconnected step and cannot be optimal.

Semantic Swap stays explicit and correctness-neutral: immutable weights may be dropped and
reloaded; recomputable activations may be rematerialized only from a replay-certified
producer; mutable state requires generation-checked writeback; approximation requires an
accuracy certificate; deadlines and backpressure prevent "optimal" plans that thrash storage.

Gate: new metrics, plus **no regression** on every row above.

### G9 — export declared alias facts to LLVM — **partly landed**

*New, from the [advanced-technique triage](BCIR_ADVANCED_TECHNIQUE_TRIAGE.md). Small, and it
fixes a false assertion the emitter was making.*

BCIR does not need an alias analysis. Every claim DECLARES its read and write RIDs, its
`hazard`, its `bounds` and its `volatile` flag — declared aliasing, which is strictly stronger
than any inferred result. The emitter was discarding all of it and writing `noalias` on every
pointer unconditionally, which is a *false* fact on any in-place graph: `Claim(rd=(1,2),
wr=(1,))` is `A[i] = A[i] + B[i]`, and A and C were both declared not to alias while both being
resource 1. `noalias` is an assertion LLVM reorders across, so that is undefined behaviour, not
a missed optimization.

| Gate | Before | Status |
|---|---|---|
| `exact` No `noalias` on a pointer pair sharing a RID | 3 of 3 on an in-place graph | **landed** — 1 of 3, and the disjoint case still gets all three |
| `exact` Alias scopes derived from the RID partition | absent | open |
| `exact` TBAA from the declared element type | absent | open |
| `exact` `volatile` carried through to LLVM | fenced in BCIR only | open |
| `ratio` `native.*` | see baseline | must not regress |

The landed half is the correctness half. The rest is upside: LLVM's OoO scheduling, load/store
reordering and vectorizer all improve on real alias facts, and BCIR is uniquely placed to supply
them because it has them by declaration rather than by inference.

### G10 — escape analysis and indirect-call target narrowing

*New, from the same triage. Follows G6 — escape analysis over typed regions beats it over the
opaque claim DAG.*

Neither ThinLTO nor CHA applies directly: a BCIR `Module` is whole-program by construction, so
there is no link step to be thin about, and there are no vtables. Two underlying capabilities
are missing and are worth having on their own.

- **Escape analysis.** `bounds_provenance` already records *why* a bounds contract is what it is
  (`declared_extent`, `recovered_count`, `snapshot_extent`, …). A resource whose provenance is
  `declared_extent` and whose RID never crosses a call boundary provably does not escape, which
  licenses stack placement or a bank-local arena slot instead of a heap resource — composing
  directly with the static-memory arena.
- **Indirect-call target narrowing.** `callee_sig` already carries a declared callee *type* on
  `c.call.indirect`. Narrowing it to a single admitted target over BCIR's own declared call graph
  turns an opaque effect edge into a known one, which unblocks fusion and reordering across it.

| Gate | Target |
|---|---|
| `exact` Resources proved non-escaping on the cfront corpus | a stated count, rising |
| `exact` Indirect edges resolved to exactly one target | a stated count, rising |
| `ratio` `native.*` | must not regress |


### G11 — `ExecutionPlanV1`: the plan as bytes — **landed (S1-C, 2026-09-12)**

*New, from the 2026-09-04 review. The canonical plan cannot be Python objects if the C twin,
the MLIR rail and a resident executor are to read it.*

An append-only StreamPack v4 record family carrying what G1 makes canonical: schedule slots
(claim, stream, bin, start, duration), lifetimes and addresses (from G5), movement edges (for
G8), and the per-resource generation vectors (R11, the closure item S0-2 — landed as the
StreamPack v4 record by S0-E, which the plan carries forward). A C twin decodes it;
BCAB gains a kind for it; the ASN.1 projection follows the StreamPack precedent. The pack stays
the executable; the plan is what the pack was derived from and what every reader prices.

| Gate | Target | Outcome (harness row, counted failures over fixed corpora, 0 at the bound) |
|---|---|---|
| `exact` `plan.abi.roundtrip` | Python encode → C decode → Python re-encode is byte-identical | **met**: 26/26 corpus plans (12 programs × 2 placements + the audit fixture × 2) — `plan.abi.mismatches` 26 → **0** |
| `exact` `plan.readers.agree` | pricing, token execution, static memory and StreamPack lowering read the plan and produce identical traces | **met**: the pricer's makespan and slots, both executors' placements, the re-placement from the plan's own step costs, the static-memory lifetimes and the pack hydrated from the plan's bytes are identical on every fixture — `plan.readers.disagreements` 27 → **0** |
| `exact` Stale generation vector | a pack whose plan carries an older vector for any resource is refused by both rails | **met**: registry moved / resource declared after minting / pack older than its plan, refused on the Python rail (R11) and the C rail (`BCIR_ERR_STALE`) — `plan.stale.accepted` 6 → **0** |
| `exact` Malformed plan | truncated, duplicated, out-of-order and unknown-claim records refused before publication | **met**: 39 (variant, rail) pairs — truncated, trailing, CRC, magic, version, flags, reserved, mode, knee, duplicated claim, stream, duration, makespan, lane, width, unsorted vector, lifetime alignment, movement kind on both rails; unknown claim, phase order and a missing claim on the rail that holds the module — `plan.malformed.accepted` 39 → **0** |

What landed: `gem.execution_plan` is the abstract value — one `PlanStep` per claim carrying the
realization (candidate, lane, width, the plan's own cost) and the canonical placement (stream,
start, duration) G1 made canonical, the static-memory planner's `Lifetime` rows, the G8
`MovementEdge` family (declared, carried and verified now; empty until G8 produces edges) and
the registry's generation vector; `plan_from_realization` mints it bound to the module (the
S1-B identity API) and the target, `schedule_of` / `realization_of` are the readers.
`abi.execution_plan_abi` is the frozen v1 wire format (`docs/kernel/BCIR_EXECUTION_PLAN_ABI.md`:
a 64-byte header with the `u64` fields 8-aligned, four length-prefixed record families, a CRC
trailer, the StreamPack's append-only discipline); `runtime/c/bcir_execution_plan.h` is the
freestanding C twin — `bcir_ep_verify` applies the same wire laws, `bcir_ep_check_generation_vector`
the R11 predicate, and `bcir_ep_check_pack` binds a pack to its plan by bytes (one segment per
step with the step's claim, phase, lane and width; identical vectors); `verify_execution_plan`
is the oracle's verifier (R9 structure against the module, R13 binding, the placement re-derived
from the plan's own costs, R11, the pack). BCAB gained kind 25 / format 13 with the full wire
verification in both readers and the MLIR `bcir.artifact.variant` kind; the `BCIR-ExecutionPlan`
ASN.1 module (`{ 1 3 6 1 4 1 62596 3 }`) projects it under DER, OER and JER with the native
octets surviving byte for byte. Indicative wall numbers on this host at the audit's 4,096-claim
fixture: the plan is 218 KB to the pack's 713 KB, encodes in 17.8 ms and decodes in 27.7 ms
(the pack: 40.0 / 77.2 ms); minting it — the placement plus the digest binding — costs 82 ms,
so a reader that holds the bytes reads the placement in a third of the time re-running the
dispatch takes (37 ms) and never re-derives the digest. Not claimed: a law-rail plan op (the
MLIR rail links the C decoder) and a C DER → native fast path for the plan.

### G12 — the dispatch law, work-unit budgets, resumable search — **landed (S2-C, 2026-09-13)**

*New. Turns "a solver portfolio" into a deterministic choice the certificate can name.*

A dispatch function from (region kind, instance size, requested certificate class, work budget)
to solver, recorded in the certificate with the budget, the stop reason and the bound source.
Budgets are counted in the solver's own work units — expansions, relaxations, evaluations —
never seconds, so a certificate class means the same thing on every host. Search state
(frontier, incumbent, bound, stop reason) is a content-addressed artifact; resuming from it
reproduces the same continuation. A plan diff (which claims moved, which bins changed, which
bound tightened) is the structured answer to "what is the residual gap made of".

| Gate | Target | Outcome |
|---|---|---|
| `exact` `dispatch.incumbent.first` | a legal incumbent exists at every interruption point of every solver | **met** — `dispatch.incumbent.missing` 26 → 0 over the interruption corpus (budget 0 included: the fast rail's answer stands, the exact layout no longer refuses a zero budget, the exhaustive selection is seeded with the sweep's assignment) |
| `exact` `solver.budget.units` | two runs with equal budgets and inputs produce identical plans and stop reasons | **met** — identical plans, stop reasons, expansions and state digests on the six-job corpus and the memory witness |
| `exact` Resume reproduces | continuing from a checkpoint equals the uninterrupted run | **met** — `search.resume.unavailable` 393 → 0: `exact_schedule`, `exact_layout` and `exact_selection` return content-addressed states (`SearchState`, `LayoutSearchState`, `SelectionSearchState`) and run(b₁) then resume(b₂) equals run(b₁+b₂), three legs included, on 1,118 splits of the six-job corpus, multi-phase random modules, the memory corpus and the selection corpus; a state refuses inputs it was not taken from |
| `exact` The learned ranker cannot remove a candidate | the census is identical with and without it | **met** — `dispatch.ranked` holds any ranker to a permutation of the census (`CensusError` otherwise) and `policy_ranking` orders the portfolio by the L2 gate's weights with every entry kept |
| `exact` Every certificate records its dispatch | absent | **met** — `dispatch.unrecorded` 12 → 0: `certify_schedule` dispatches through the law and records rail, solver, units, budget, stop reason, bound source and the class granted |

What landed (S2-C): `gem.dispatch` — the law as a table over (region kind, instance size,
requested class, work budget): a `TMSAO-4` request or a zero budget dispatches the fast rail
(`schedule_eft`, `first_fit_layout`, the one-sweep `optimize_scheduled`), a `TMSAO-3` request
the fast rail too because a measured-best claim needs `W` and the measured corpus (G13), and a
`TMSAO-1`/`TMSAO-2` request the proof rail (`exact_schedule`, `exact_layout`,
`exact_selection`) with the budget, expected to close up to the bounded size and to stop on its
budget above it; one runner per region kind returns the result with its `DispatchRecord`. The
exact scheduler's budget became the module's, consumed phase by phase in topological order (a
phase that closes hands the remainder on; one that exhausts it leaves the later phases pending
with the heuristic as their incumbent and the root stack as their bound), which is what makes
a checkpoint resume exactly; the memory and selection solvers keep their own units. `gem.diff`
is the plan and certificate diff — moves (stream, start, duration), re-selections (candidate,
width), re-pricings (the same candidate at another cost: the context coupling moved),
re-layouts (bank, offset), the makespans and the regret, the bounds when certificates are
given — the regret ledger generalized. Not claimed: a native solver owning a certificate
(G17), the measured rail (G13), and content-addressed *storage* of states (they are artifacts
with digests; the append-only store is G13's).

### G13 — the workload component `W` and the measured-candidate corpus — LANDED (S2-E, 2026-09-13)

*New. The scope table marked `W` "not modelled"; best-fit dispatch cannot fit work it cannot see.*

A declared workload descriptor — shapes, batch, concurrency, service-level requirement,
horizon — becomes the `W` component of `ExecutionScopeV1`, and the B1 measured schedule
artifacts (`schedule_artifact.py`) become the replay corpus and candidate database that
*informs* dispatch through the L2 replay gate (`portfolio.py`) and never decides legality.

| Gate | Target | Outcome (S2-E) |
|---|---|---|
| `exact` A scope digest separates two workloads | plans for `W₁` and `W₂` on one program carry distinct digests | **met** — `scope.workload.collisions` 36 → 0: `certify_schedule`, `certify_selection` and `certify_measured` take the declared workload, `W` is digested with the scope and `diff` names it; the same workload twice is the same scope |
| `exact` Replay-gate no-regression | a policy promoted by the corpus never loses to the incumbent on the logged episodes | **met** — `replay.subset.admitted` 24 → 0: the measured replay gate replays every logged episode of the scope or refuses, its certificate names the corpus head and the logged count, and the portfolio refuses a certificate over a subset; on the random corpus an admitted certificate is exactly "the candidate's pooled median never above the incumbent's on any logged episode" |
| `exact` The measured rail is dispatched over evidence | a TMSAO-3 request with a declared `W` and corpus evidence runs the measured rail | **met** — `dispatch.measured.unavailable` 12 → 0; without `W`, without evidence, or for a schedule or memory region the fast rail runs and the decision says which is missing |

What landed (S2-E): `kbcir.workload.Workload` — the declaration (shapes from the module's
resources, batch, concurrency, a `best-effort` / `latency` / `throughput` service level with its
budget or rate, horizon, a `static` or `dynamic` input distribution with the expected counts of
dynamic claims), integers and names only so the scope can digest it, validated on construction,
held to the module by `verify_workload`, and classed (`interactive` / `batch` / `nominal`) for
the L2 table; `scope_for(workload=)` and the three certifiers carry it as `W`. `kbcir.measured`
— the measured-candidate corpus: `MeasuredPlan` (the plan a policy selected for (program,
target, workload, Θ) as an assignment digest, the raw samples one per repeat, the PMU counters,
the host's attestation by the S0-F rig's rule, the source commit), `MeasuredCorpus`
(append-only: digests chained, the head the identity of the whole history, a corpus read back
refused when an entry was altered, removed or reordered or the chain forged, `extends` the only
relation two versions may have, duplicate evidence refused), `measure_plans` (B1's discipline
for whole plans: warm-up excluded, one lap per repeat under the OS counters, the PMU when
exposed), `from_schedule_artifact` (B1's matmul artifacts enter the same corpus with the
tenancy `unproven`, since B1 recorded none) and `replay_measured` — the L2 replay gate over
measured evidence: every logged episode of the scope, judged by the pooled median wall time, a
candidate without evidence on one episode refused rather than judged on the rest;
`ReplayCertificate` carries the corpus head and the logged count and is admitting only when it
covers the log. `PolicyPortfolio.select(theta, workload)` is now the table over (runtime class,
workload class): a runtime constraint outranks the workload; under a nominal runtime an
interactive workload takes the latency schedule and a batch one the throughput schedule.
`gem.dispatch`: the `measured` rail — a TMSAO-3 request with a declared `W`, a whole-plan
region (`path`, `selection`) and evidence in the corpus dispatches `measured_best` (units:
samples); `solve_measured` ranks the census by the corpus's pooled medians and returns the
measured-best policy's plan **re-derived from the planner** (evidence whose assignment digest
the planner no longer reproduces is stale and unused); `gem.exact.certify_measured` binds the
class to the scope with `W` and `M` inside it. The ladder now holds a measured claim to its
scope (`MEASURED_COMPONENTS` = P, H, W, M) and to the **two-target rule**: TMSAO-3 only when
the prediction interval comes from attested silicon on two materially different targets with
counters; on this host (`virtualized`) every measured certificate is TMSAO-4 and says so. Not
claimed: a silicon certificate (Stage 6); `W` in the cross-rail manifest digest
(`build_manifest` is recomputed by the MLIR verifier field for field — a manifest `W` is an
S0-D-pattern change on both rails); an input distribution beyond the expected counts of dynamic
claims; the append-only store as a service (it is a file with a chain, read and extended by a
library).

### G14 — control-plane record ABI — LANDED (S3-A, 2026-09-23)

*New, Stage 3. Lease, generation, quiescence, activation, rollback and cancellation were prose
and an `admit(map_gen, data_gen)` argument.*

Small, fixed, versioned records with a C twin and a DER/COER projection, carried across every
boundary (L2–L5 of the review's level table), so a stale generation is refused by bytes rather
than by convention. Capability-scoped handles and signatures are part of the first record, not
a later hardening.

| Gate | Target | Outcome (S3-A) |
|---|---|---|
| `exact` `control.record.bytes` | every record is bounded and versioned; an unknown version or a trailing byte is refused on both rails | **met** — `control.abi.mismatches` 29 → 0 (every kind, scope and reason survives the C twin byte for byte, its MAC accepted on both rails) and `control.malformed.accepted` 82 → 0 (one variant per wire law, refused on both rails with the same status) |
| `exact` Stale generation refused | activation with an older generation than the resident one fails at every boundary | **met** — `control.stale.accepted` 25 → 0: a record of every kind minted against a generation the plane has left, a pack or plan from an older registry (including a resource moved under unchanged maxima, and one with no vector) refused on both rails; the trusted loader and context-shard activation express the one `is_stale` predicate and `verify_control_record` holds a generation record to the module's registry (R11) |
| `exact` Quiescent switch | a generation switches only at a phase/event boundary; a switch requested mid-phase is deferred, never applied | **met** — `control.deferred.lost` 16 → 0: deferred (the resident generation unmoved, its bytes held in one pending slot), then applied exactly once at the boundary or refused there and reported; never applied early, twice, or lost. The naive row — "a mid-phase switch applied" — was already 0 on the parent (it *refused*), which is why the gate counts deferrals lost |

What landed (S3-A): `ControlRecordV1`
([`BCIR_CONTROL_PLANE_ABI.md`](../kernel/BCIR_CONTROL_PLANE_ABI.md)) — a 64-byte header (the
`expect` compare-and-swap witness, the capability, the boundary, the issuer's sequence, the
lease, the subject), one fixed body per kind (100 + 8..64 bytes, 192 declared), an HMAC-SHA256
MAC (the root key on a grant, `lease_key(root, id)` otherwise) and a CRC — the CRC a keyless
reader's corruption gate, the MAC authority. `bcir.abi.control_abi` is the codec (eleven wire
laws in one order, the encoder refusing what the decoder refuses); `bcir.gem.control` holds the
values, `registry_digest` (the generation vector's own bytes), `rollback_token`, `lease_key`,
the shared `is_stale` and **`ControlPlane`** — the resident state (generation, live artifact
and rollback target, registry, boundary counter, in-flight count, drain, an eight-lease table,
one pending switch) deciding each record applied, deferred or refused in the specification's
order (a refused record changes nothing), crossing phase and event boundaries, and admitting
packs and plans by the installed registry. `runtime/c/bcir_control_plane.{h,c}` is the
freestanding twin, decision for decision: two new statuses (`BCIR_ERR_CONTROL` 17,
`BCIR_ERR_MAC` 18, append-only), the shared SHA-256/HMAC from S3-A0, a `--api` harness for
the fail-closed laws no record reaches (a plane without a key verifies nothing), and a
structure-aware libFuzzer target that seals records under the plane's key and asserts the
plane's invariants after every operation. `bcir/asn1/BCIR-ControlPlane.asn1`
(`{ 1 3 6 1 4 1 62596 4 }`) projects the record under DER, canonical OER and JER — the body a
CHOICE whose alternative is the kind — and holds decoded values to the codec's laws.
`tools/perf/gemplus_baseline.py --group control` measures six exact rows over one fixture
module both rails and `tools/c/check_runtime.sh` grade; the parity traces compare every
verdict, refusal, status and resident state digest after every operation of 52 scenarios
(`control.decisions.nonconforming` 64 → 0, `control.traces.divergent` 52 → 0). No BCAB kind (a
bundle is immutable; a control record is a decision with a lifetime) and no MLIR source. Not
claimed: the BCIR UAPI (still unfrozen until UART and virtio-blk demonstrate their lifecycles);
a transport (G15 carries these records; a boundary here is a function call); a signature scheme
(HMAC under a shared root key proves possession of the key); capability enforcement beyond the
record; replay protection beyond sequences, the `expect` witness and lease ids that never recur.

### G15 — the live shared ring — LANDED (S3-B, 2026-09-23)

*New, Stage 3. The telemetry frame was frozen and byte-identical; the transport under it was not
built.*

A single-producer/single-consumer shared ring with head/tail, acquire/release publication,
per-slot sequence numbers, a declared overwrite or backpressure policy and exact loss
accounting — the "version-zero triple" of the 2026-09-03 analysis — carrying telemetry records
first and G14 control records second, with restart, stale-generation, wrap, saturation and
peer-death tests on both rails.

| Gate | Target | Outcome (S3-B) |
|---|---|---|
| `exact` `ring.loss.accounting` | records lost to overwrite are counted exactly; a consumer that falls behind sees the count, never a torn record | **met** — `ring.loss.unaccounted` 140 → 0 and `ring.torn.delivered` 140 → 0 over 70 scripted scenarios on both rails; on the parent a reader a lap behind received the survivors and no count, and a slot read mid-rewrite came back with the fields of two records and no error. The concurrent runs (threads; processes with a peer SIGKILLed and taken over) end with nothing torn or unaccounted (`ring.concurrent.violations` 6 → 0), race-free under ThreadSanitizer |
| `exact` Sequence continuity | gaps, reorders and duplicates are reported as the frame ABI already does for frames | **met** — `ring.sequence.misreported` 22 → 0: the intake and the BTLM decoder express one predicate (`SequenceTracker`; the frame decoder's counts unchanged over 20,000 random streams) |
| `ratio` `ring.throughput` | records per second against a `memcpy` floor of the same bytes, indicative on shared hosts | **measured** — 26× on the reference host (the baseline, the slice's first measurement). There it is dominated by thread placement: pinned per vCPU pair it read ~13–49× within one hour, and a minimal unchecked Lamport queue ~1.1–6× under the same pinning, so no ring change is attributable on that host. INDICATIVE (host-dependent); a faithful A/B needs pinned threads on an attested host |

What landed (S3-B): the version-zero triple. **TelemetryEnvelopeV0**
([`TELEMETRY_ENVELOPE_ABI.md`](../kernel/TELEMETRY_ENVELOPE_ABI.md)) — a 64-byte header carrying
source, session, generation, signal, sequence, the producer's own loss count and a clock with its
unit, a fixed payload per kind (a sample, or the frozen DataDNA record) and a CRC; wire laws in one
order on both rails, one spelling per record. **The generated signal table** — every built-in
registry definition as a 64-byte row, `runtime/c/bcir_signal_table.h` generated and drift-checked,
the BCIR/vendor/device ID ranges and the unknown-required-signal law. **The host intake** — the
wire laws, a bounded stream table, continuity classified before any refusal, unknown signals and
stale generations refused (`is_stale`, G14's predicate). **The live ring**
([`BCIR_LIVE_RING_ABI.md`](../kernel/BCIR_LIVE_RING_ABI.md)) — a region of seven one-writer
cache lines and `slot_count` slots: a per-slot seqlock under acquire/release publication,
BACKPRESSURE (refuse and count) or OVERWRITE (lose and count, exactly), double-buffered
accounting, and epochs with takeover of a peer proved dead; CONTROL rings are BACKPRESSURE with
slots that carry the control record's declared bound, and the 52 G14 scenarios decide identically
through one (`ring.control.divergent` 104 → 0). `bcir/gem/ring.py` is the oracle,
`runtime/c/bcir_ring.{h,c}` the freestanding C11 twin; four statuses appended
(`BCIR_ERR_TELEMETRY` 19, `_RING` 20, `_FULL` 21, `_BUSY` 22). Every scripted scenario decides
identically on both rails down to the region's bytes after every operation
(`ring.traces.divergent` 70 → 0), every malformed region and envelope is refused with its
declared status (`ring.malformed.accepted` 112 → 0), stale epochs and generations are refused
(`ring.stale.accepted` 10 → 0), and the table and corpus are byte-identical on both rails
(`ring.abi.mismatches` 29 → 0). The C gate adds ThreadSanitizer (and the race it must report
when the atomics are made plain), the seqlock-removed mutant it must fail, -O0 == -O3 == the
oracle, and a libFuzzer target over hostile and honest regions; a committed fault table injects
22 defects across both rails and every one is caught by its own row. The lease-key cache G14 deferred
landed with it (a lease's key derived once at the grant; leased records now verify as fast as
root-key ones). No MLIR source, no ASN.1 module and no BCAB kind: version zero carries no
compatibility promise. Not claimed: MPSC; death detection (a takeover needs the embedding's
proof); blocking or wake-up; authentication of what the ring carries; a transport beyond one host;
any v1 freeze before the UART and virtio-blk traces.

### G16 — data-plane hand-off

*New, Stage 3. The C++ seam is specified and single-node real; this makes its contract hold.*

Borrowed zero-copy views with explicit lifetime, generation gating at `admit()` against the
live registry generation, the dynamic-graph builder that freezes a fresh StreamPack per step
through the C/IR rail, and the manifest-of-shards format for graphs too large to ship as one
pack. The node level (MPI/NCCL) stays a declared stub until a cluster exists.

| Gate | Target |
|---|---|
| `exact` No copy where a borrow suffices | the hand-off moves bytes once; a view outliving its owner is refused |
| `exact` Generation gating | `admit()` refuses a pack older than the registry generation |
| `exact` Shard manifest | shards by digest reassemble to the whole-pack bytes |

### G17 — compact indexed planner, then native parity

*New, Stage 4. The profile's hot spot: Python objects per claim, immutable values rebuilt.*

Replace per-claim dataclass churn with compact indexed arrays behind the same API, proved by
byte-identical plans over the corpus; then the proposal's CXX2/CXX3 native planner, which may
own a certificate only after it reproduces the Python plan byte for byte over a generated
corpus — the same way the C twins earn their rails.

| Gate | Target |
|---|---|
| `exact` `planner.parity` | identical plan bytes before and after, and Python versus native |
| `exact` `planner.calls` | Python call count at scale 8 (6.06 M in the profile) reduced by a stated factor |
| `wall` `audit.kbcir-streampack.scale4` | indicative only |

### G18 — incremental re-verification and delta StreamPack

*New, Stage 4. Incremental plans are only honest if the laws are re-checked incrementally
**and** the incremental verifier is proven equal to the full one.*

Dependency-scoped re-verification of the R-laws after a delta plan, a delta StreamPack that
re-emits only changed segments, and the differential that keeps both truthful.

| Gate | Target |
|---|---|
| `exact` `verify.delta.identity` | delta verification equals full verification over the corpus, including every injected violation |
| `exact` Delta pack identity | a delta-emitted pack equals the full re-emission byte for byte |
| `ratio` `kbcir-streampack.delta` | incremental re-plan against the full re-plan, a stated mechanism |

---

## 4. The sublinearity question, answered precisely

The proposal asks for 4× workloads to grow "well below 4×". The honest statement, which the
report already makes and this roadmap adopts:

> If the algorithm must inspect or emit every item, **Ω(n) makes 4× the asymptotic floor.**
> Sublinear total time is possible only by changing the *admitted work*: incremental/delta
> plans, templates, compression, sparsity, memoization, resident state, or avoiding
> materialization. That is valuable — and it is a **different certificate**.

So the baseline rows split into two kinds, and they are graded differently:

**Ω(n)-floored rows.** `audit.iterative-phase-dag` (5.78× for 4× claims),
`audit.bounded-overwrite-ring` (4.03×), `q8-q4-blocks` (3.95×), `lstm-gru` (4.04×). These are
at or near their floor already. The available win is the **constant factor**, and a slice
claiming more than that is claiming something impossible. `4.03×` for 4× work is a *result*,
not a defect.

**Rows where the admitted work can change.** `optimize_scheduled` (quadratic → linear, G2),
the triple digest (three hashes → one, G3), `kbcir-streampack` at 71.67× for 64× claims
(incremental plans). These are where sublinear-in-the-old-work is genuinely available, and
each must say **which mechanism** bought it — delta, cache, or avoided materialization —
because the certificate class depends on it. A cached result is TMSAO-4 unless the cache's
invalidation predicate is part of the scope.

---

## 5. The learned-optimization boundary

ML may generate candidates, rank a frontier, predict residuals, choose measurement locations,
or forecast resource pressure. It may **not** decide legality, silently alter an in-flight
plan, or disable a check.

Promotion requires: a finite preverified candidate vocabulary; measured out-of-sample
evidence; hard capacity/policy checks; a quiescent generation switch; W^X and
signature/provenance checks; rollback and stale-generation rejection.

This is the same boundary the security audit enforced on the certificate: a model that ranks
candidates is inside the scope `A`, and its generation belongs in `G`. A model that changes
what is *legal* is outside the architecture entirely.

---

## 6. Order of work — the six stages

Re-staged on 2026-09-04 (the review in
[`BCIR_GEMPLUS_TMSAO_STAGED_PLAN_2026-09-04.md`](BCIR_GEMPLUS_TMSAO_STAGED_PLAN_2026-09-04.md)
§6–§7 carries the exit gates and the PR-sized sections). Stage-local ids `S0-n` are the
correctness-closure items that are prerequisites rather than GEM+ slices.

```
Stage 0  correctness closure remainder     S0-1 two-rail hash widening (B7)             <- LANDED (S0-D)
                                           S0-2 R11 per-resource generation vectors     <- LANDED (S0-E)
                                           S0-3 verify checkpoints in bcir-optimize/-hydrate + the inert fixtures  <- LANDED (S0-B)
                                           S0-4 EV1–EV3 in verify_all                 <- LANDED (S0-A)
                                           S0-5 module-scoped verifier walks             <- LANDED (S0-B)
                                           S0-6 shared structural-law corpus, both rails    <- LANDED (S0-C)
                                           S0-7 R9 re-derives offer/cost/feasibility; C R9 width/cost  <- LANDED (S0-A)
                                           S0-8 lowering tail contract; convolution overflow fixture  <- fixture + checked arithmetic LANDED (S0-C); tail contract LANDED (S0-G)
                                           S0-9 ODS→IRDL inventory gate                  <- LANDED (S0-B)
                                           S0-10 bcir-performance-audit rename + wording sweep  <- LANDED (S0-A)
                                           G7   native measurement repair          <- LANDED (S0-F)
Stage 1  one canonical plan and its ABI    G1 → G3 → G11 → G5               ALL LANDED: G1 (S1-A); G3 (S1-B); G11 (S1-C); G5 (S1-D)
Stage 2  best-fit solver portfolio         G2 → G4 (first TMSAO-2) → G12 → G6 → G13   ALL LANDED: G2 (S2-A); G4 (S2-B); G12 (S2-C); G6 (S2-D); G13 (S2-E)
Stage 3  IPC at every level                G14 → G15 → G16                  LANDED: G14 (S3-A); G15 (S3-B); next G16 (S3-C)
Stage 4  performance program               G17, G18
Stage 5  movement, alias, escape           G8, G9 remainder, G10
Stage 6  physical evidence                 two targets, PMU/energy — hardware-gated
```

G0 is landed. Stage 1 needs S0-1 and S0-2 (the identity the plan binds to and the vectors it
carries); Stage 2 needs G1 and G11; Stage 3 needs G11, and G14 precedes G16; Stage 4 needs G1
and G18's differential; Stage 5 needs G6; Stage 6 needs hardware. G3 and G7 are out of
dependency order on purpose: both are small, and each removes a false number from the table.

The [advanced-technique triage](BCIR_ADVANCED_TECHNIQUE_TRIAGE.md) records why the rest of the
standard advanced-compiler catalogue is absent: four techniques are already built, five are the
slices above under other names, and six belong to LLVM — where BCIR's job is to supply the
declared fact, not to reimplement the pass.

---

## 7. What this roadmap will not claim

- **No "TMSAO achieved" before G4 and multi-target evidence.** The report is explicit that
  optimality claims need repeatable evidence on at least one x86 CPU and one materially
  different target, with PMU counters, and this host has neither PMU nor RAPL.
- **No silicon certificate from a virtualized measurement.** G7 exists to make that refusal
  mechanical instead of a matter of discipline.
- **No optimality claim on a row with no lower bound.** Five rows are in that state today and
  they are listed in §0.3 rather than quietly graded.
- **No sublinear claim on an Ω(n) operation** without naming the admitted work that changed.
- **No IPC claim beyond one host.** Since G15 a second process runs the ring's contract over a
  shared mapping (and is SIGKILLed and taken over); a device has not, and the node level stays
  a declared stub until a cluster exists.
- **No cached or incremental result above TMSAO-4** unless its invalidation predicate is in the
  scope and the incremental verifier has been proved equal to the full one.

---

## 8. Current state, 2026-09-04

`tools/perf/gemplus_baseline.py --compare` on a host that is not the baseline's (Python 3.11,
no PMU):

| Row | Slice | Baseline | Today | Verdict |
|---|---|---:|---:|---|
| `pricing.eft.divergence` | G1 | 1.9922 | **1.0** (S1-A, 2026-09-05) | GAIN, at the bound — one artifact; the retired pricer still reads 1.9922 on the fixture as the witness |
| `optimize_scheduled.slowdown.512` | G2 | 69.2× | **6.27×** (S1-A) → **3.5×** (S2-A, 2026-09-13) | GAIN — under the 4× bound (A/B on one idle host); the residual is the serial chain's O(n²) hazard edges |
| `optimize_scheduled.general.slowdown.512` / `sweep.replacement.fraction` | G2 | 44.5× / 1.0 (S1-D, this host) | **4.7× / 0.0625** (S2-A, 2026-09-13) | GAIN, the fraction at the bound — 256 step-shortening trials each re-place one phase of sixteen; the assignment is the reference sweep's claim by claim |
| `static_memory.digests.2048` | G3 | 3 | **1** (S1-B, 2026-09-12) | GAIN, at the bound — one digest per plan-and-verify chain; the verifier still recomputes at a trust boundary |
| `plan.abi.mismatches` / `plan.readers.disagreements` / `plan.stale.accepted` / `plan.malformed.accepted` | G11 | 26 / 27 / 6 / 39 | **0 / 0 / 0 / 0** (S1-C, 2026-09-12) | GAIN, at the bound — the plan has bytes: every corpus plan survives the C twin, every reader reproduces its trace from the bytes, every stale and malformed fixture is refused on every rail that can see it |
| `memory.suboptimal.fraction` / `memory.worst.ratio` / `memory.real.bytes` | G5 | 38.6% / 1.6154× / 1,344 B | **0% / 1.0 / 832 B** (S1-D, 2026-09-12) | GAIN, at the bound — the bounded exact solver proves every corpus fixture and the witness reproduces the report's worst case; the two-phase alias fixture is refused against the token placement and gets disjoint storage when planned from it |
| five `wall` rows | G0–G3 | — | 1.5–3× faster; `optimize_scheduled.512` 1,148 → 83 ms and `static_memory.plan.2048` 196 → 114 → 84 ms (S1-D's alias sweep) A/B on one host | INDICATIVE — a faster host and interpreter, not evidence; the A/B is the same host |
| `eft.suboptimal.2domains` / `eft.worst.2domains` / `eft.mean.2domains` (+ the 3-domain rows) | G4 | 11.07% / 1.1333× / 1.0078 | **0 / 1.0 / 1.0** (S2-B, 2026-09-13) | GAIN, at the bound — the proof rail proves every instance; the heuristic's numbers are kept as witness rows |
| `optimize_scheduled.quality` / `solver.unproved.fraction` / `solver.gap.p95` | G4 | 1.00696× / 1.0 / 0.0625 | **1.0 / 0 / 0** (S2-B, 2026-09-13) | GAIN, at the bound — every certificate carries L, U and both gaps |
| `search.resume.unavailable` / `dispatch.incumbent.missing` / `dispatch.unrecorded` | G12 | 393 / 26 / 12 (the parent tree) | **0 / 0 / 0** (S2-C, 2026-09-13) | GAIN, at the bound — every proof-rail solver resumes exactly, has an incumbent at every interruption point, and every certificate names its dispatch |
| `regions.unexpanded.claims` / `objectives.unverified` | G6 | 542 / 2 (the parent tree) | **0 / 0** (S2-D, 2026-09-13) | GAIN, at the bound — every claim in a verified region whose expansion is the module; every objective admitted with its laws |
| `native.*` (four rows) | G6 | 5.58× / 11.68× / 1.27× / 0.98–1.01× (the report's host) | 4.44× / 15.56× / 1.314× / 1.004× on this host (S2-D), A/B against the parent 4.53× / 15.00× / 1.315× / 1.003× | no regression — measured through `bcir.bench` on a `virtualized` host: a guardrail, not a silicon certificate. Host-dependent, so INDICATIVE off the baseline host and read as the same-host A/B |
| `scope.workload.collisions` / `replay.subset.admitted` / `dispatch.measured.unavailable` | G13 | 36 / 24 / 12 (the parent tree) | **0 / 0 / 0** (S2-E, 2026-09-13) | GAIN, at the bound — `W` in every certificate's scope, a corpus certificate covers the log or is refused, the measured rail runs over evidence; every measured certificate on this `virtualized` host is TMSAO-4 by the two-target rule |
| `control.abi.mismatches` / `control.malformed.accepted` / `control.stale.accepted` / `control.deferred.lost` / `control.decisions.nonconforming` / `control.traces.divergent` | G14 | 29 / 82 / 25 / 16 / 64 / 52 (the parent tree) | **0 / 0 / 0 / 0 / 0 / 0** (S3-A, 2026-09-23) | GAIN, at the bound — the control plane has bytes: every record survives the C twin, every wire law refuses on both rails with one status, stale generations are refused at every boundary, a mid-phase switch is deferred and decided exactly once, and the two rails' traces are identical to the state digest |
| `ring.abi.mismatches` / `ring.malformed.accepted` / `ring.loss.unaccounted` / `ring.torn.delivered` / `ring.sequence.misreported` / `ring.stale.accepted` / `ring.traces.divergent` / `ring.control.divergent` / `ring.concurrent.violations` | G15 | 29 / 112 / 140 / 140 / 22 / 10 / 70 / 104 / 6 (the parent tree) | **0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0** (S3-B, 2026-09-23) | GAIN, at the bound — the transport has bytes and laws: nothing torn, nothing unaccounted, continuity as the frame ABI reports it, stale epochs and generations refused, the two rails identical to the region's bytes, and the same with the peers truly concurrent and dying |
| `ring.throughput` | G15 | 26× (the slice's first measurement, the reference host) | 24–44× unpinned, ~13–49× pinned per vCPU pair, on that host | INDICATIVE — host-dependent: two threads hand a 124-byte envelope across cores against one thread's `memcpy`; placement dominates (a minimal unchecked queue reads ~1.1–6× under the same pinning) |
| `verify.*` / `scope.*` | G0 | — | not measured | need the native rig or the digest fixtures |

Since S2-B BCIR emits TMSAO-1 and TMSAO-2 certificates on the proof rail; since S2-E the
measured rail exists and grants TMSAO-3 to nothing on this host — the two-target rule keeps
it hardware-gated (Stage 6).
