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

---

## 0. The measurement discipline, and why it comes first

The instruction this roadmap is built to satisfy: *a comparative test per update; no gain means
investigate why; a gain means analyse its magnitude and how much more is still available.*
That is a loop, and a loop needs a fixed point to measure against, so the baseline is frozen in
code before any slice is written:

```
python tools/perf/gemplus_baseline.py --list      # 22 frozen metrics and their floors
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
| `P` | program, input contract, R-laws, semantics, precision, admitted approximation | `hash_module`, incomplete: claims are sorted by id, so declared order is erased |
| `H` | topology, ISA/capabilities, banks, links, capacities | `hash_target`, **missing the memory hierarchy entirely** |
| `W` | workload shapes, input distribution, concurrency, SLOs, horizon | not modelled |
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

- widening `hash_target` with a `DenseI64ArrayAttr` for the tiers plus the matching C++ walk,
  so the MLIR rail can recompute the same complete identity. Not required for certificates
  now that they bind to the scope.
- the `wall` row `static_memory.plan.2048` (301.02 ms): untouched here, and it is G3's.

### G1 — one canonical schedule artifact

*Report P0.1. The audit's sharpest correctness finding.*

`price_scheduled` (fixed waves, round-robin bins) and `schedule_eft` (LPT/EFT with locality)
are two different algorithms, and the objective reads one while the executor runs the other.

| Gate | Baseline | Target |
|---|---|---|
| `exact` `pricing.eft.divergence` | **1.9922** | **1.0** — both read one artifact |
| `exact` Token execution and the priced schedule agree | not checked | identical slot assignment |
| `wall` `audit.mixed-wave-token-eft.scale4` | 48.05 ms | no regression |

This is a **correctness** metric wearing a performance costume. Any value but 1.0 means
`M(π,Θ)` denotes two things, and no certificate above TMSAO-4 is possible while it does.

### G2 — incremental delta pricing

*Report P1.4. The largest visible win in the baseline.*

`optimize_scheduled` recomputes the whole makespan per trial. Cache per-phase/bin
contributions and reprice only the affected chain.

| Gate | Baseline | Target | Headroom today |
|---|---|---|---|
| `ratio` `optimize_scheduled.slowdown.512` | 69.2× | ≤ 8× | 94.2% |
| `wall` `optimize_scheduled.512` | 1,703.66 ms | ≤ 200 ms | 98.7% |
| `wall` `optimize_scheduled.256` | 435.73 ms | ≤ 100 ms | 97.4% |
| `exact` The plan chosen is unchanged | — | **identical assignment** |

The last row is the one that matters. A faster sweep that picks a *different* plan has not
been made faster; it has been changed. Delta pricing must be an exact refactor of the same
search, proved by comparing assignments claim by claim.

### G3 — canonical digest computed once

*Report P1.6. Cheap, and it unblocks measurement everywhere else.*

The profile found module hashing at ~0.65 s of a 4.8M-call run, with three independent hashes
of one immutable module: the planner hashes, the verifier hashes again, an independent client
hashes a third time. Replace recursive canonical flattening with an iterative stream, cache
immutable module identity, and expose an identity-bound API — **while keeping the verifier's
right to recompute at a trust boundary**, which is the whole reason the third hash exists.

| Gate | Baseline | Target |
|---|---|---|
| `wall` `static_memory.digest.2048` | 88.05 ms | ≤ 30 ms |
| `wall` `static_memory.verify.2048` | 157.88 ms | ≤ 90 ms |
| `wall` `static_memory.plan.2048` | 301.02 ms | ≤ 120 ms |
| `exact` A cache cannot survive mutation | — | mutation invalidates; cross-module substitution refused |

The security half of that last row is not optional: a digest cache that survives a mutation is
the Class-B "vacuous check" defect from the audit, rebuilt.

### G4 — bounded exact solvers and the lower-bound stack

*Report P1.5. **The first slice that can emit TMSAO-2.***

Add dependency-free branch-and-bound for small fixtures, and the lower-bound stack: critical
path, work/capacity, hierarchical Roofline, communication cut, queue/network-calculus,
allocation peak/clique, occupancy, energy-at-minimum-work. Report the **maximum valid** bound
against the incumbent.

| Gate | Baseline | Target |
|---|---|---|
| `exact` `eft.suboptimal.2domains` | 11.07% of 1,716 | 0% on the proof rail, or a stated gap on every instance |
| `exact` `eft.worst.2domains` | 1.1333× | 1.0 on the proof rail |
| `exact` `optimize_scheduled.quality` | 1.00696× | 1.0 on the proof rail |
| `exact` Every certificate carries `L`, `U`, and both gaps | absent | present |

**The gap is the product, not the speed.** A slice that leaves the heuristic exactly as fast
and merely states how far from optimal it is has still moved BCIR from TMSAO-4 to TMSAO-2, and
that is a bigger step than any constant factor in this document.

### G5 — schedule-aware liveness and bounded exact memory

*Report P0.2 + P1.5. Carries a latent-correctness condition.*

`live_intervals` uses topological phase positions; `execute_tokens` legally overlaps phases.
The audit did **not** find a deployed corruption path — the static map is not currently fed
into token execution — and established the integration condition instead: *static memory must
be computed from the final schedule's intervals, or the verifier must prove the schedule
refines the phase order the plan used.*

| Gate | Baseline | Target |
|---|---|---|
| `exact` The two-phase alias fixture | aliases at offset 0 | rejected, or disjoint storage |
| `exact` `memory.suboptimal.fraction` | 38.6% of 500 | 0% on the proof rail, or a stated gap |
| `exact` `memory.worst.ratio` | 1.6154× (21 vs 13 units) | 1.0 on the proof rail |
| `exact` `memory.real.bytes` | 1,344 B | 832 B on the proof rail |

First-fit stays the predictable fast path. The exact solver runs when the fast layout violates
capacity, peak pressure crosses a threshold, the artifact is high-value, or a proof is asked
for. Every result records the concurrent-live lower bound, the achieved extent, and the gap.

### G6 — typed regions and the semiring registry

*Report P2. The architectural slice.*

A graph of typed regions, each with a strong local model **and a conservative expansion back
to claims**: affine/polyhedral, SDF/CSDF, timed-event (max-plus), tensor index-map, state
machine/Petri, equivalence graph, opaque claim DAG as the universal fallback. Plus the typed
objective registry — min-plus, max-plus, min-max, Boolean, lexicographic, Pareto, stochastic —
each verifying closure, identities, comparison semantics and overflow policy.

Every region must supply: a verifier, a conservative claim expansion, a cost/lower-bound
interface, and refusal conditions.

| Gate | Baseline | Target |
|---|---|---|
| `ratio` `native.gather-avoidance` | 5.58× | **no regression** |
| `ratio` `native.blocked-reduction` | 11.68× | **no regression** |
| `ratio` `native.direct-stride` | 1.27× | **no regression** |
| `ratio` `native.dense-parity` | 0.98–1.01× | stays in band |
| `exact` Every region expands conservatively to claims | — | differential test per region |

The native rows are **guardrails, not targets**. They are what BCIR is for — the audit's §4.4
shows the wins come from preserving enough structure to avoid a gather or pick a blocked
reduction, not from graph representation itself. A regions refactor that speeds up the planner
and loses a 11.68× structural win is a net loss.

"Semiring" must not become a label that admits arbitrary operators without their laws.

### G7 — repair the native measurement rig

*Report P0.3. Small, and it invalidates a number currently in use.*

The microbench uses `(k * 16) % n` with a power-of-two `n`, so `gcd(n,16) = 16` and the walk
visits `n/16` unique elements — a nominal 32 MiB buffer with a **2 MiB** working set. It also
prints `native microbench (bare-metal)` under WSL.

| Gate | Baseline | Target |
|---|---|---|
| `exact` Unique elements visited | n/16 | n (coprime stride or proved full-cycle permutation) |
| `exact` Provenance under virtualization | claims bare-metal | refuses the claim |
| `exact` Report contents | ratios only | raw samples, intervals, working-set census, counter availability |

Until this lands, no `native.*` row is a silicon certificate — only a structural comparison.

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

## 6. Order of work

```
G0  scope identity + class ladder        <- LANDED
G1  one schedule artifact                <- correctness; unblocks G2, G5
G3  digest computed once                 <- cheap; makes the rest measurable
G2  incremental delta pricing            <- largest visible win
G4  bounded exact + lower bounds         <- first TMSAO-2 certificate
G5  schedule-aware + exact memory        <- needs G1
G7  native measurement repair            <- small, invalidates a current number
G6  typed regions + semirings            <- architectural
G9  export declared alias facts          <- PARTLY LANDED; small, fixes a false assertion
G8  movement as first-class              <- needs G6
G10 escape analysis + call narrowing     <- needs G6
```

The [advanced-technique triage](BCIR_ADVANCED_TECHNIQUE_TRIAGE.md) records why the rest of the
standard advanced-compiler catalogue is absent: four techniques are already built, five are the
slices above under other names, and six belong to LLVM — where BCIR's job is to supply the
declared fact, not to reimplement the pass.

G0 and G1 are prerequisites. G3 is out of dependency order on purpose: it is small, and it
removes measurement noise that would otherwise contaminate every slice after it.

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
