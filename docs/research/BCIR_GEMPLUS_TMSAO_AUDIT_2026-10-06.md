# GEM+ / TMSAO audit: PR #758 to #808 (2026-10-06)

> Dated, non-normative. Baseline: `origin/main` at the merge of PR #808 (`5cd8e03`); the training
> PRs #768–#786 are out of scope. It compares the original plan as the maintainer stated it, the
> plan of record ([`BCIR_GEMPLUS_ROADMAP.md`](BCIR_GEMPLUS_ROADMAP.md) and
> [`BCIR_GEMPLUS_TMSAO_STAGED_PLAN_2026-09-04.md`](BCIR_GEMPLUS_TMSAO_STAGED_PLAN_2026-09-04.md))
> and the code and tests on `main`; where the docs and the code disagree, the code wins. The
> remaining work it names is the **completion ladder**, the normative order of which lives in the
> roadmap's §9. Counts live only in the generated [`STATUS.md`](../STATUS.md).

Status words: **Met** (built, with a witness test that fails on the defect), **Partial** (some
built, the rest named), **Missing** (nothing in the tree does it), **Defect** (found by this
audit, with a reproduction).

---

## 1. Verdict

**The roadmap as written is done. The original plan is not.**

Everything the roadmap committed to has landed and is gated: Stage 0, the G0–G18 slices
(Stages 1–5) and SP-ENC. The only open stage is Stage 6, the physical calibration, and it needs
hardware this project does not have. The frozen harness, re-run today on `5cd8e03`, reports
104 GAIN, 7 NO-CHANGE, 0 REGRESSION, 16 INDICATIVE and 0 NOT-MEASURED. None of the NO-CHANGE
rows is a miss. Five are parity or guard rows held at 0. The other two
(`optimize_scheduled.quality` and `eft.mean.2domains`) reached their bound of 1.0, but by less
than the 2% band an exact row must move.

The original plan asked for more than the 2026-09-04 re-staging carried into the roadmap, and
several of those items never became slices. Of the 22 items below (item 13 is split in two):

- **7 are met**;
- **9 are partial**;
- **6 are missing**: stochastic expected cost, SDF semantics, timed-event semantics, LLVM poison,
  type containment graphs, and semantic bounds derived at the point of application.

The audit also found three things no gate catches today:

1. **ASN.1 parameterization is not hygienic** (confirmed defect, §4.1). A template used inside
   another template picks up the outer template's dummy-parameter bindings. The meaning of a
   module then depends on the order its assignments are lowered in.
2. **Recursive ASN.1 types work only on BER/DER and PER** (confirmed gap, §4.2). OER, XER, JER
   and the JER plan compiler all refuse a self-referential type, and two of them give a misleading
   error.
3. **The garbage collector costs the Python oracle up to 68% of its hot-path time** (measured,
   §5). Those paths create no cyclic garbage at all, so the collector's work there is pure
   overhead.

The audit's own before/after run then found a fourth, in the instrument every call-count row is
read with:

4. **The call counts merged functions that share a label** (confirmed defect, §4.3, fixed with
   the ladder's AUDIT-0). `planner.calls` read 589,856 at scale 8 while the planner makes 655,393,
   and the 65,537 calls lost were the constructor calls its emission floor is made of.

Landing the ladder's ASN1-R slice found two more wrong-output defects, which the ladder now
carries as P0 slices:

5. **A component's own tag drops the tag of the type it names** (§4.4): `ticket [3] Ticket`, with
   `Ticket ::= [APPLICATION 1] SEQUENCE {...}`, goes out without the `[APPLICATION 1]`.
6. **NULL has two abstract values** (§4.5): BER/DER, PER and OER use `codec.NULL`, XER and JER
   use `None`, and each side refuses the other's.

Two places that guide future work are stale: the session digest and the systems-engineer skill
(§6).

**Since the audit.** The completion ladder (§9 of
[`BCIR_GEMPLUS_ROADMAP.md`](BCIR_GEMPLUS_ROADMAP.md)) closes these slice by slice. The item
table in §3 stays as measured on `5cd8e03`; what has changed since:

- **Item 1** (heuristics that fail 11% of the time): Partial → **Met**, by G19. A plan may state
  its placement, `exact_plan` mints the solver's optimum wherever it beats the dispatch, and the
  verifier holds such a plan to the legality of a placement rather than to the dispatch's:
  `eft.executed.suboptimal.2domains` 11.07% → 0 and `.3domains` 1.05% → 0.
- **Item 18** (containment graphs): Missing → **Met**, by ASN1-R (§4.2).
- **Item 21** (hygienic substitution): its confirmed defect is fixed by ASN1-H (§4.1). It stays
  Partial for what was never built: the §9.8 tagging environment across modules,
  ObjectFromObject (§15) and a parameterized object set used as a table constraint.
- **Item 17** (the oracle audit and migration): §5.3's first two recommendations landed with
  OR-GC -- the collector paused inside the pure hot paths, with the zero-cyclic-garbage witness
  and harness rows, and slots on `CostVector` and `Candidate`. The item stays Partial for the
  C ports (CXX3, CXX4).
- **Item 6** (a signed plan): (a) and (b) closed by G21 -- a PlanStatementV1 binds the plan's
  bytes, the scope, the module, the pack and the certificate by SHA-256 under an Ed25519
  signature, verified on both rails against a trust store with rotation, revocation and expiry.
  It stays Partial for (c) and (d): the objective's identity is bound only through the scope and
  the certificate, and the law rail has no plan op.
- **Item 8** (expected cost): Met by G22 -- the registry's `expectation` semiring, held to the
  semiring laws; the exact Markov expectation over loops (`kbcir.expectation`); measured branch
  probabilities (`BranchProfile`) recorded as the scope's `U`; and `compose.Cond` refusing an
  out-of-range probability instead of clamping it, its mean exact instead of floored per branch.
  The C front still lowers a `while` to one body iteration: its trip probability is not yet
  measured, so its loops carry no expectation of their own.
- **Items 11 and 12** (synchronous dataflow; timed events): Met by G23 / G24 -- a claim may
  declare fixed stream rates (`StreamRate`), a run of them is an `sdf` region with its
  repetition vector, a live schedule and its FIFO bounds, or a `timed` region with its max-plus
  cycle time (a lower bound on the period), and a run the model refuses is opaque with the
  reason named. Network calculus (arrival and service curves) is not built: item 12 asked for
  timed-event semantics and a cycle-time bound, which are; the law rail carries no stream
  attribute yet.
- **Item 15** (hard lower bounds): G25 adds the data-movement members -- the red-blue pebble
  I/O bound for an order (held to an exact optimal-I/O dynamic program, sound on every
  generated instance and equal to the optimum on 269 of 300) and the hierarchical roofline
  over it. Still Partial: the communication cut between affinity domains, queue and network
  calculus, occupancy and energy at minimum work are not built, and the pebble bound is for
  the order given, not over every order.
- **Item 3** (memory waste, at production scale): G26 makes the concurrent-live bound
  alignment-aware and gives the exact rail a best-fit incumbent portfolio; on the
  512-resource fixture the bound rises 5,400 -> 6,468 bytes and the layout falls 6,908 -> 6,820,
  so the stated gap is 352 bytes where it was 1,508 -- reduced and stated, not proved.
- **Item 14** (the depth of optimization priced by its value): G27 gives the dispatch law a
  value model -- `OptimizationValue` (expected executions, the declared price of a work unit)
  grants a proof-rail search at most `executions x gap / price` work units, none at a closed
  gap -- and delegates straight-line modules (one executable claim in the LLVM lowering
  subset) to LLVM: the scalar kernel, whose width LLVM's vectorizer picks, its output equal
  byte for byte to the planned kernel's under clang. Still Partial: the MLIR Transform
  export, the unprojected IRDL operations, polyhedral depth and the ILP/CSP rail are G28 and
  G30 to G32.
- **Item 9** (joint scheduling and memory): G28 solves a region's schedule and layout as
  one model on a dependency-free exact CSP (`kbcir.csp`, `kbcir.joint`): every start and
  offset together, `w_time x makespan + w_memory x extent` minimized, certified against a
  reference that enumerates every start vector. On 40 generated regions the parent's
  schedule-then-layout pipeline loses 26 objective units in all; the joint rail none. Met for
  one region (one phase) at small size; the placement axis (G8's movement) is not in the same
  model, and nothing larger than the brute-force-checked sizes is claimed.
- **Item 14**, continued: the CSP/ILP rail and the optional CP-SAT adapter (`hosted.cpsat`,
  held to the in-tree solver) are G28; MLIR Transform export, the unprojected IRDL operations
  and polyhedral depth remain (G30 to G32).
- **Item 13b** (a poison feature imported from LLVM): G29 imports poison as proved facts. A
  resource may declare its integer value range; the verifier proves `nsw` / `nuw` for an i32
  claim by exact interval arithmetic over the signed and unsigned readings; the lowering
  carries exactly those flags and the ranges as `!range` on its loads; R12 refuses a forged,
  dropped or widened fact; and LLVM's own range reasoning re-derives every exported flag (77
  over 120 generated claims, none missed, none extra). Met for i32 elementwise claims on the
  oracle's lowering; `freeze`, other widths, the C emitters and the law rail are not built.
- **Item 14**, continued: G32 projects 37 of the 38 unprojected IRDL operations -- the
  driver-subset core with the integer widths ODS declares, the GEM model seams, the ECN
  objects -- each round-tripped by stock `mlir-opt` and each constraint shown to refuse a
  mistyped program. `bcir.asm` stays declared unprojected for the reason IRDL itself gives: a
  constraint variable binds one type for a whole variadic group, and `asm`'s operands are
  heterogeneous. MLIR Transform export and polyhedral depth remain (G31, G30).
- **Item 14**, continued: G31 exports a K_BCIR decision to stock MLIR. The matmul's tiling
  plan (tile extents and loop order) becomes a Transform-dialect script --
  `tile_using_for` with the plan's tile sizes and its loop order as the `interchange` --
  beside a `linalg.matmul` payload of the planned shape, and stock `mlir-opt
  --transform-interpreter` applies it. Held to MLIR's own output, not to the script's text:
  the `scf.for` nest MLIR builds visits exactly the plan's tile origins in the plan's order,
  on 24 plans (the planner's own over 8 shapes, every loop order, ragged tiles); the inverse
  interchange a careless export would emit is caught on both 3-cycles. Met for the matmul's
  tiling; vectorization, packing and other plans are not exported. Polyhedral depth remains
  (G30).
- **Item 14**, continued: G30 takes the polyhedral model to depth two. A claim may declare
  its 2-D loop nest (`LoopNest`: extents and one affine map per operand); it is then a `nest`
  region whose model is the nest's exact dependence distances -- the access equation solved
  per source instance, never every instance pair -- and the interchange, tiling and
  per-loop parallel verdicts they imply. Held to a reference that enumerates every instance
  pair and simulates each order (interchange, five tilings, each loop parallel) on 160
  generated nests with aliasing, negative and row-sized strides: 160 disagreements in each
  of dependences and legality before (the tile claim was opaque), none after. Met at depth
  two on the oracle rail; deeper nests, skewing, fusion and the law rail are not built.
- **Defects 5 and 6** (§4.4, §4.5): fixed by ASN1-T and ASN1-N.

---

## 2. What landed, PR by PR (#758 to #808, training excluded)

| PR | Slice | What it closed |
|---|---|---|
| #758 | docs | the re-staged program: G0–G18 in six stages |
| #759 | S0-A | EV1–EV3 in `verify_all`; R9 re-derives the offer, every cost and feasibility; the C R9 width/cost contract |
| #760 | style | the tree formatted once under ruff, with one pinned ruff |
| #761 | S0-B | verifier checkpoints; the two inert MLIR fixtures executed; module-scoped walks; the ODS→IRDL inventory gate |
| #762 | S0-C | the shared structural-law corpus on both rails; the convolution overflow fixture |
| #763 | S0-D | the two-rail hash widening (memory hierarchy and declared claim order) |
| #764 | review fixes | 13 Codex findings on #761/#762, on both rails |
| #765 | S0-E | StreamPack v4: the per-resource generation vector (R11 on three rails) |
| #766 | S0-F + S0-G | **G7**, the native rig repair (the cache-microbenchmark misread); the runtime-`n` tail contract |
| #767 | S1-A | **G1**, one canonical schedule artifact (the pricing/execution disconnect) |
| #787 | S1-B + S1-C | **G3**, the digest computed once; **G11**, `ExecutionPlanV1` |
| #788 | S1-D, S2-A..E, S3-A | **G5** exact memory; **G2** delta pricing; **G4** exact solvers and the first TMSAO-2; **G12** the dispatch law; **G6** typed regions and the objective registry; **G13** the workload `W`; **G14** the control-record ABI |
| #789 | S3-B + S3-C | **G15**, the live SPSC ring; **G16**, the data-plane hand-off (Stage 3 complete) |
| #790 | S4-A + S4-B | **G17**, the compact planner and its native twin; **G18**, incremental re-verification and delta StreamPack |
| #791 | S5-A | **G9**, the declared alias facts carried to LLVM |
| #792 | SP-ENC | the StreamPack encoder compiled (7.6× fewer calls) |
| #793 | S5-B | **G10**, escape analysis and indirect-call narrowing |
| #794, #796, #798 | CF-* | cfront miscompile fixes on both rails |
| #795 | S5-C | **G8**, data movement as a first-class transformation; ExecutionPlan v3 |
| #797 | floors, G2 residual | every harness row's floor measured in the same run; the hazard frontier (130,816 → 511 edges); transport replay |
| #799 | PERF | the within-noise rows of the #786 vs #797 benchmark |
| #800 | TC23 | the GEM+ S slices judged on LLVM/Clang 23 and Node 24 |
| #801–#808 | BUILD-0..8b | CMake and BCIR Make over one source manifest |

---

## 3. Item-by-item audit of the original plan

### A. The issues before GEM+

| # | Item | Status | Evidence on `main` | What is missing |
|---|---|---|---|---|
| 1 | Heuristics that fail 11% of the time | **Partial** | G4 (#788). `gem.exact.exact_schedule` proves every instance of the report's 1,716-instance corpus (`eft.suboptimal.2domains` 11.07% → **0** on the proof rail, worst ratio 1.1333 → 1.0, `solver.unproved.fraction` 1 → 0). Every certificate carries L, U, both gaps, the stop reason and the budget, bound to the scope digest. | **The heuristic still runs.** The exact solver certifies the heuristic's placement; it does not replace it (`bcir/gem/exact.py` docstring). When `verify_execution_plan` is given the target, its R13 re-derives the placement with the canonical heuristic dispatch (`schedule_plan`) and refuses any other, so an optimal placement would be rejected. The 190/1,716 suboptimal placements (held by `test_gemplus_baseline.py:259`) are now measured and stated, not removed from what executes. |
| 2 | Quadratic scaling | **Met** | G2 (#788) plus the hazard frontier (#797), G17 and G18 (#790). `optimize_scheduled.slowdown.512` 69.2× → **2.49×** today; dispatch edges 130,816 → **511** (at the proved floor); general case 44.5× → **3.82×**; a one-claim delta costs **1.06%** of a rebuild (473 calls vs 10,855,666). Assignments are identical claim by claim on 1,344 cases. | A new superlinear source the roadmap does not track: the garbage collector, at large module sizes (§5). |
| 3 | Memory waste | **Met** (opt-in) | G5 (#788). `plan_static_memory(layout="exact")` is a budgeted branch-and-bound behind first-fit: `memory.suboptimal.fraction` 38.6% → **0**, worst 1.6154× → **1.0**, the report's witness 1,344 B → **832 B**. Lifetimes come from the placement's own ticks. | The default is still first-fit. At production scale (512 resources) the exact solver stops on its budget and keeps first-fit, with the gap stated. The interval-graph dynamic program the plan names as "the next lever" is not built. |
| 4 | CPU cache microbenchmark misread | **Met** | G7 (#766). The walk visits every element (census `unique=n/n/n`, `strided_order` twin). The rig attests its tenancy and refuses a bare-metal claim on a virtualized host. The reader refuses a summary its raw samples do not support. | — |

### B. The missing composition invariants

| # | Item | Status | Evidence | What is missing |
|---|---|---|---|---|
| 5 | Pricing schedule, execution schedule, static memory layout and token scheduler disagreed | **Met** | G1 (#767). `pricing.eft.divergence` 1.9922 → **1.0**. `schedule_plan` is the one artifact both pricers and both executors read (`test_schedule_plan_is_the_one_artifact_the_price_and_the_executors_read`). G5 derives lifetimes from that placement. G11 (#787): every reader reproduces its trace from the plan's bytes (`plan.readers.disagreements` 27 → 0). | — |

### C. The unified execution plan

| # | Item | Status | Evidence | What is missing |
|---|---|---|---|---|
| 6 | A content-addressed, cryptographically signed plan that dictates the schedule, memory layout, data transfers, final artifact digest, objective verification, static memory planner and token executor | **Partial** | `ExecutionScopeV1` (G0, a SHA-256 scope digest); `ExecutionPlanV1` (G11, #787) with explicit steps and slots, lifetimes (v2, #788), movement edges (v3, #795) and the generation vector; a freestanding C decoder and verifier; BCAB kind 25; the `BCIR-ExecutionPlan` ASN.1 module; `verify_execution_plan` (lifetimes must cover the schedule). Activation is authorised by `ControlRecordV1`: the plan's SHA-256 under HMAC-SHA256 (G14, #788). | **(a) Not signed.** The plan carries a CRC-32 trailer. Authority is a shared-key HMAC on the control record, which `docs/kernel/BCIR_CONTROL_PLANE_ABI.md` itself says is "not a signature scheme". No asymmetric signature, key distribution, rotation or revocation exists anywhere (0 files). **(b) Not cryptographically bound.** The plan names its module and target by the 63-bit FNV-1a R13 hashes (`module_hash`, `target_hash`). The SHA-256 scope digest is not in the plan. **(c) Incomplete.** The plan does not carry the StreamPack/artifact digest, the objective identity or its certificate (L, U, gap, solver); those are separate JSON artifacts. **(d)** There is no law-rail plan op; the MLIR rail links the C decoder. |

### D. The portfolio solver

| # | Item | Status | Evidence | What is missing |
|---|---|---|---|---|
| 7 | Tropical min-plus, max-plus and Boolean algebra | **Met** (oracle) | `kbcir.objectives` (G6, #788) admits min_plus, max_plus, min_max, boolean, lexicographic and pareto only after `verify_objective` proves their laws, each with a declared overflow policy. `dag_best_path` reproduces the planner's min-plus path and the exact scheduler's max-plus critical path. `kbcir/tropical.py` (pre-#758) adds the Kleene closure and Karp's minimum mean cycle. | The law rail spells only `min_plus` and `max_plus` (`BCIR_Semiring`). |
| 8 | Expected cost from branch probabilities (stochastic models) | **Missing** | Only the pre-#758 `kbcir/compose.py`: `Cond.prob_then_milli` weights two branches. It is declared rather than measured, defaults to 500, and silently clamps an out-of-range value. | No registered stochastic or robust objective. No profile-derived branch probabilities. No Markov expectation over loops. Scope component `U` is still "intervals for JER J6 only". `tropical.py` deliberately offers no expected-cost objective, "because the cost model does not carry branch probabilities". |
| 9 | Optimally solve joint scheduling and memory allocation layouts | **Partial** | Each part is exact on small instances, separately: the schedule by G4's branch-and-bound, the layout by G5's. G8 (#795) plans movement and compute jointly, reaching the exact optimum on 11 fixtures (`movement.excess` 597,840 → 0, TMSAO-1). | No joint schedule × placement × memory solver: memory is planned after the schedule, from its intervals. The 2026-09-04 dispatch table's "schedule + placement + memory jointly" row and the CXX3 joint solvers are not built. ("Join scheduling" in the plan is read as joint scheduling.) |
| 10 | Expand GEM into a DAG of typed regions | **Partial** | `kbcir.regions` (G6, #788). Every region has a verifier, a conservative expansion (the identity on the carrier), a cost floor and named refusals. `regions.unexpanded.claims` 542 → 0. | **Two kinds only**: affine (1-D maps, static trip counts) and opaque. `RegionGraph` is a per-phase partition with no edges between regions and no nesting. The layer is read-only: regions bound and certify a plan but never change a decision. |
| 11 | Synchronous dataflow semantics for fixed-rate streams | **Missing** | Nothing in the code (no repetition vectors, balance equations or bounded FIFOs). | An SDF/CSDF region kind. Fixed-rate claims fall to opaque today. |
| 12 | Timed-event semantics for time-critical processes | **Missing** | The max-plus objective, Karp's minimum mean cycle, and the pre-#758 R19/R20 timing laws exist. | No timed-event region, no cycle-time bound, no network calculus. |
| 13a | Fallback to unguaranteed optimality | **Met** | The fast rail at budget 0 gives a TMSAO-4 incumbent; the opaque region is the universal fallback; `bcir-cc --fallback` hands unsupported C to LLVM. | — |
| 13b | A poison feature imported from LLVM | **Missing** | No poison or `freeze` semantics in any rail. The word "poison" appears only for poisoned telemetry. | Either import poison/freeze as a law (claims whose result is poison-on-overflow, with `nsw`/`nuw` on lowering) or record it as out of scope. |
| 14 | Budget the optimization depth by the complexity of the math and the value of the operation | **Partial** | G12 (#788), `gem.dispatch`: a table over four region kinds × three rails (fast, proof, measured). Budgets are counted in work units, never seconds; solver state is resumable and content-addressed; an incumbent exists at every interruption. | The caller picks the class and the budget: there is **no value-of-optimization model**. Sub-items: **LLVM for simple sequential code**: only the capability fallback above; nothing routes simple code to LLVM because deep optimization would not pay. **MLIR Transform dialect**: missing (0 code hits). **IRDL**: the projection exists and is gated, but 38 operations are still unprojected. **Polyhedral**: 1-D affine only; a tile claim is opaque. **CSP / ILP / CP-SAT**: missing (0 code hits; the proposal planned an optional hosted adapter). **Hong–Kung red-blue pebble bound**: missing (0 code hits). |
| 15 | Hard mathematical lower bounds | **Partial** | G4's bound stack: critical path, work/capacity, bandwidth/knee, tail-serial. G5's concurrent-live bound; G6's region floor. Since #797, every harness row's floor is measured in the same run, so no optimality claim lacks a bound. | Hierarchical roofline, communication cut / red-blue pebble I/O, network-calculus or queue bounds, occupancy, and energy at minimum work. The roadmap's own G4 text lists them as not claimed. |
| 16 | The four-tier ladder (exact, bounded, measured, heuristic) | **Met** (TMSAO-3 hardware-gated) | The classes and gap arithmetic since G0. TMSAO-1 and TMSAO-2 have been issued on the proof rail since S2-B (#788). The measured rail (G13) exists. | TMSAO-3 has never been granted. The two-target rule needs two physical targets with PMU counters, and this host is virtualized. Stage 6 has not started. |

### E. The Python oracle audit and C/C++ migration

| # | Item | Status | Evidence | What is missing |
|---|---|---|---|---|
| 17 | A complete audit of the Python oracle, migrating hot low-level work to C/C++ (micro-allocations, object overhead, GC, attribute dicts), and the Cython question | **Partial** | Native twins exist: the planner `bcir_kplan.c` (byte-identical over 3,290 cases; 3.17 ms vs about 47 ms in Python at 4,096 claims), the control plane, ring, hand-off, shard manifest, ExecutionPlan decoder, StreamPack, the cfront twin and `bcir_make.c`. An AI-only boundary audit exists (2026-07-19, before #758). | **The whole-oracle inventory (the proposal's CXX0) was never done.** Hydrate, the delta chain, the exact solvers, dispatch and regions are Python-only. §5 gives today's measurements and the recommendation, which is C, not Cython. |

### F. ASN.1 / JSON

| # | Item | Status | Evidence | What is missing |
|---|---|---|---|---|
| 18 | Containment graphs for nested structures | **Missing**, with a confirmed gap | The JER plan compiler (`jer_plan.py`) unrolls nesting recursively and refuses at depth 64. It builds no type graph. `graph.py` (P1) is a node table for *programs*, not a containment graph for *types*. | Recursive types fail on five of six rails (§4.2). Needed: a type containment graph with cycle (SCC) detection, cyclic plan nodes, and recursion support on PER, OER, XER and JER. |
| 19 | Lifetime bounds for memory guarantees | **Partial** | The J3 C twin allocates nothing: capacity comes from the caller, and `needed` reports the capacity that would have sufficed. J1 enforces limits before parsing. P3 projects R21 lifetimes. | No schema-derived static memory bound. Only BOOLEAN, NULL, ENUMERATED and a fixed-size BIT STRING are bounded (`_bounded`), so no "a valid document of this schema needs at most N bytes" guarantee exists. |
| 20 | Application and semantic value mapping, deriving bounds from the point of application | **Missing** | `jer_plan.py` explains why plain X.697 makes value constraints invisible to the encoding. | Under BCIR's own canonical JER profile, a value constraint does bound the token: `INTEGER (0..255)` has at most 3 digits, `OCTET STRING (SIZE (4))` exactly 10 octets. None of this is derived, at the type or at the point a parameterized type is applied. ECN value mappings exist but feed no bound. |
| 21 | Parameterized structures and hygienic substitution | **Partial**, with a confirmed defect | X.683 is built: substitution is structural over the AST and memoized on the actuals. | **Not hygienic** (§4.1). Also not built: the §9.8 tagging environment across modules, ObjectFromObject (§15), and a parameterized object set used as a table constraint. |

---

## 4. Defects and gaps found by this audit

The reproductions below were run against `5cd8e03`; ASN1-H and ASN1-R carry them as witnesses.

### 4.1 X.683 substitution captures names (confirmed, fixed by ASN1-H)

`bcir/frontends/asn1/lower.py` `_instantiate` installs each object-set actual into the
module-wide `self.object_sets` table under the dummy's name while the body is lowered. That is
dynamic scoping. A nested template lowered during that window resolves its own references through
the outer template's bindings, and `_instantiations` memoizes the captured result under a key
that leaves the environment out.

```asn1
ATTRIBUTE ::= CLASS { &id OBJECT IDENTIFIER UNIQUE, &Type } WITH SYNTAX {&Type IDENTIFIED BY &id}
Other ATTRIBUTE ::= { {UTF8String IDENTIFIED BY {2 5 4 3}} | {BOOLEAN IDENTIFIED BY {2 5 4 99}} }
Inner2 {ATTRIBUTE:P} ::= SEQUENCE { type ATTRIBUTE.&id ({Ghost}), value ATTRIBUTE.&Type ({Ghost}{@type}) }
Outer2 {ATTRIBUTE:Ghost} ::= SEQUENCE { a Inner2 {Ghost} }
```

| Use | Result | Correct |
|---|---|---|
| `Inner2 {Other}` directly | refused: "table constraint names object set 'Ghost', which this module does not define" | yes |
| `Outer2 {Other}` (lowers `Inner2` inside `Outer2`'s window) | **accepted, with `Other`'s 2 rows** | no: `Inner2`'s `Ghost` is unbound |

The shadowing variant gives the same capture. There, `Inner`'s body names a module-level
`Supported` (1 row) and is used inside `Outer {ATTRIBUTE:Supported}`:

| Order of assignments | `UseOuter.a` rows | `UseInner` rows |
|---|---|---|
| `UseInner` alone | — | 1 |
| `UseOuter` alone | 2 | — |
| `UseOuter` then `UseInner` | 2 | **2** (cached capture) |
| `UseInner` then `UseOuter` | 1 | 1 |

So the module's meaning depends on assignment order.

**Fix.** Substitute object-set actuals in the AST (the way `TableConstraintNode` already is),
never through a shared table. Include the resolved environment in the memo key. Add the two
repros above, plus an order-independence check, as witnesses.

### 4.2 Recursive types work only on BER/DER and PER (confirmed, fixed by ASN1-R)

```asn1
Node ::= SEQUENCE { label INTEGER (0..255), children SEQUENCE (SIZE (0..4)) OF Node }
```

The front end lowers the self-reference to a `_LazyType`.

| Rail | Result |
|---|---|
| BER/DER (`module.encode`) | round trip OK (`300c800101a1073005800102a100`) |
| JER `encode_jer` | `Asn1Error: no encoding for schema type _LazyType` |
| JER plan `compile_plan` | refused with "19.2.4 forbids an open type as an alternative of an unwrapped choice", which **misattributes** the cause |
| OER | `Asn1Error: no OER encoding for _LazyType` |
| XER | refused as if it were an open type, which **misattributes** the cause |
| PER | round trip OK (`PER: expected bytes` in the first version of this table was the reproduction's own error: it passed `decode_per` its arguments in the wrong order; PER already had a private resolver for the reference) |

This is the repository's most common defect shape: a mechanism landed on two rails of six (L14).
It is also exactly the plan's containment-graph item.

ASN1-R found three more while building the fix. A recursive CHOICE
(`Expr ::= CHOICE { lit INTEGER, neg Expr }`) did not lower at all. Nothing bounded a recursion:
a 256-deep value encoded under DER and a 1,000-deep one raised `RecursionError`, and a crafted
input of a few kilobytes did the same to the PER, OER, XER and JER decoders. And `decode_jer` let
a deeply nested JSON text raise `RecursionError` from the parser, recursive schema or not.

Reviewing the fix found four more, all in the front end. A recursive instance of a CHOICE
template (`E {T} ::= CHOICE { lit T, neg E {T} }`) did not lower. `A ::= B` with `B ::= A`
lowered with no type behind it. A recursive reference to a type with an assignment-level tag
lost the tag. And a contained subtype naming an alias of a recursive type was refused as one of
itself. The same review found the two defects of §4.4 and §4.5, which are not about recursion.

### 4.3 The call counts merged functions that share a label (confirmed, fixed)

Found by the audit's own tool. The first before/after run of the AUDIT-0 commit graded
`sched_eft@4` as nondeterministic: the same tree counted 34,368 calls in one round and 36,415 in
the next. The garbage collector was the first suspect, and it was wrong: the row counts 36,415
with the collector disabled. A per-function diff of the two profiles named the cause, one row:
`<string>:2:__init__`, called 2,048 times in one process and once in the other.

cProfile keeps one entry per code object. `pstats.Stats` re-keys those entries by their label,
(file, first line, name), so two functions that share a label become one row, and the entry
written second replaces the first. On CPython 3.11 every dataclass-generated `__init__` has the
label `("<string>", 2, "__init__")`. `pstats.Stats(profile).total_calls` therefore kept one
class's constructor calls and dropped every other class's. Which class survived depended on
where the code objects were allocated, which is why the count moved between processes with
nothing changed.

Every call-count row read its count this way: the before/after audit's hot-path rows, and the
frozen harness's `planner.calls`, `kbcir-streampack.delta.calls`, `kbcir-streampack.full.calls`
and `streampack.encode.calls` through `planner_fixtures.call_count` and `delta_fixtures._profiled`.

| Row (scale 8) | `pstats` total | Exact | Floor |
|---|---:|---:|---:|
| `planner.calls` | 589,856 | 655,393 | 65,539 |
| `kbcir-streampack.delta.calls` | 472 | 489 | 13 |
| `kbcir-streampack.full.calls` | 3,490,034 | 3,689,720 | 166,923 |
| `streampack.encode.calls` | 989,245 | 989,245 | 1 |
| the parent planner (`realize_reference`) | 3,419,168 | 3,549,217 | |

The undercount was not noise. The emission floors count one constructor call per record the
output must carry, and those are exactly the calls the merge dropped. The headroom above each
floor was therefore understated. G17's factor is 5.42× (5.38× at scale 4), not the 5.80× it
quoted. The stated gate of 5× still holds.

The fix is one predicate (L14), `bcir/tests/call_counts.py`. It sums cProfile's entries per code
object before any label can merge them, and pauses the collector across the counted call. Every
fixture counts through it, and `tools/perf/ab_audit.py` loads it by path from its own checkout,
so both trees of an A/B are read with one instrument.

`planner.calls`'s frozen baseline is re-counted over the parent's planner, which
`realize_reference` keeps verbatim. Counted the old way, that code reproduces the frozen
3,419,172 to within four calls, so the recount measures the same code. The delta and full-chain
baselines came from the parent tree and keep their numbers. They are labelled as `pstats`
totals: merging only drops calls, so each is a lower bound on its exact count, and a GAIN graded
against it is understated, never overstated.

---

### 4.4 A component's own tag drops the tag of the type it names (confirmed, fixed by ASN1-T)

```asn1
K DEFINITIONS EXPLICIT TAGS ::= BEGIN
  Ticket ::= [APPLICATION 1] SEQUENCE { tkt-vno [0] INTEGER }
  AP-REQ ::= SEQUENCE { ticket [3] Ticket }
  Holder ::= SEQUENCE { ticket Ticket }
END
```

`[3] Ticket` tags the type `Ticket`, which is itself tagged (X.680 §31), so X.690 nests both
tags: `30 0b a3 09 61 07 30 05 a0 03 02 01 05`. The front end gives
`30 09 a3 07 30 05 a0 03 02 01 05`: the `[APPLICATION 1]` is gone. `Holder`, whose component
carries no tag of its own, is right (`61 07 ...`). The type model carries one tag per component,
and a component's own tag replaces the one the assignment gave the type. The shape is Kerberos'
(RFC 4120's `ticket [3] Ticket`). A related gap was already recorded: the direct encode of a
tagged assigned type omits its tag (`LoweredModule.assigned_tags`). Both need the same thing, a
tagged type the type model can hold.

ASN1-T found the rest of the family while building that. Every place a tagged type can stand
other than an untagged component dropped a tag:

| Shape | X.690 | The front end gave |
|---|---|---|
| `SEQUENCE OF [0] INTEGER`, value `{1, 2}` | `30 0a a0 03 02 01 01 a0 03 02 01 02` | `30 06 02 01 01 02 01 02` |
| `A ::= [1] B`, `B ::= [2] INTEGER`, A directly | `a1 05 a2 03 02 01 05` | `02 01 05` |
| the same A as an untagged component | `a1 05 a2 03 02 01 05` | `a1 03 02 01 05` |
| `[1] IMPLICIT T`, `T ::= [5] CHOICE {...}` | `a1 03 02 01 03` | refused, as if T were an untagged CHOICE |

It also found the write plan's OER emitter writing an untagged CHOICE alternative's index as its
tag, on both of its rails. `CHOICE { a INTEGER, b BOOLEAN }` under EXPLICIT TAGS went out as
`80 01 03` where the oracle writes `02 01 03`. The C twin agreed with the Python emitter, which
is why the differential between them never saw it.

### 4.5 NULL has two abstract values (confirmed, fixed by ASN1-N)

| Rail | takes `codec.NULL` | takes `None` | decodes NULL to |
|---|---|---|---|
| BER/DER | yes | refused ("no ASN.1 universal type is mapped to NoneType") | `codec.NULL` |
| PER, OER | yes | yes | `codec.NULL` |
| XER, JER | refused ("a NULL value is None, got NULL") | yes | `None` |

A value one rail decodes is refused by another rail's encoder. `codec.Asn1Null` exists because
`None` means "absent" in the value model, and PER's decoder was moved to it for that reason; the
text rails never were. This is the semantic value mapping of item 20 failing at its smallest
value. ASN1-N also found PER and OER encoding a value of any type as a NULL: they put nothing on
the wire for one and never looked at the value, so `5` went out as a NULL on both.

## 5. The Python oracle: what the measurements say about migration

Measured on this container: 4 vCPU, virtualized, CPython 3.11.15. Treat the absolute times as
indicative only. The portable part is the ratio of GC on to GC off within one process. The
fixture is the harness's audit module.

### 5.1 How much of the time is the cyclic garbage collector

| Hot path | 4,096 claims: GC on / off | GC share | 32,768 claims: GC on / off | GC share | Collections (gen 0/1/2) at 32k |
|---|---|---|---|---|---|
| planner (`realize.optimize`) | 47.0 / 29.2 ms | 38% | 617 / 326 ms | **47%** | 541 / 49 / 2 |
| StreamPack `hydrate` | 10.5 / 8.0 ms | 24% | 282 / 91 ms | **68%** | 299 / 27 / 2 |
| StreamPack `decode` | 25.9 / 22.8 ms | 12% | 397 / 235 ms | **41%** | 338 / 30 / 1 |
| R8/R9 `verify_plan` | 30.6 / 28.6 ms | 7% | 516 / 318 ms | **38%** | 509 / 46 / 2 |
| `schedule_eft` | 21.6 / 18.9 ms | 12% | 323 / 222 ms | **31%** | 175 / 16 / 1 |
| StreamPack `encode` (compiled by SP-ENC) | — | none | — | none | 0 / 0 / 0 |
| R-law `verify` | — | none | — | none | 0 / 0 / 0 |

The last two rows triggered no collections; their small on/off differences are timing noise.

Each full collection walks the whole live heap, so the GC share grows with module size. With GC
on, hydrate grows 27× for 8× the claims; with GC off it grows 11×.

**Each of these six paths leaves zero cyclic garbage per call.** That was measured: GC disabled,
one call, then `gc.collect()` returns 0. Reference counting already frees everything they
allocate, so the collector's work inside them buys nothing.

`gc.freeze()` after the module is built recovers 41–65% of the GC cost. Pausing the cyclic
collector inside the call recovers all of it.

### 5.2 Attribute dicts

The non-test oracle has 771 `@dataclass` decorators, of which 7 use `slots=True`, plus 25
`__slots__` declarations. The hot value types `CostVector` and `Candidate` are frozen dataclasses
with a per-instance `__dict__`. A 6-field frozen dataclass costs 168 B per instance with a dict
and 120 B with slots (−29%), and constructs in 555 vs 470 ns (−15%). The 3.11 floor allows
`slots=True`.

### 5.3 Recommendation: C, not Cython

- **Pause the cyclic GC inside the oracle's pure entry points** (planner, hydrate, decode, plan
  verification, schedulers), re-enabling it in `finally`. Add a witness that each call leaves
  zero cyclic garbage, and a harness row. The results stay byte-identical, so it needs no parity
  gate. It is the largest, cheapest win available: up to 3.1× on hydrate at 32k claims.
- **Use `slots=True` on the hot frozen dataclasses.** This is the plan's "dictionary of attributes" cost.
- **Send throughput work to the C twins that already exist, and port the next ones:**
  StreamPack hydrate and the G18 delta chain first (CXX4), then the exact solvers (CXX3) when
  certificates are needed at production scale.
- **Keep Python as the oracle** for the R-laws, research ML and test generators. The proposal's
  §8.2 explicitly says not to migrate these.
- **Do not use Cython.** It would add a build-time dependency to a core that is dependency-free
  by rule. It would be a third realization needing its own parity gate. And it keeps the object
  model these measurements blame. The C twins already carry differential parity gates, which is
  how a realization earns its rail here.

Sizes for context: the non-test oracle is 141,994 Python lines (130,804 more in tests);
`runtime/c` is 56,306 lines, `runtime/cpp` 3,572 and the MLIR rail 16,276.

---

## 6. Documentation debt

- `.claude/context/BCIR_DIGEST.md` (lines 64–69) says "everything BCIR emits is TMSAO-4 until
  G4's lower-bound stack" and lists G1–G8/G10 and G11–G18 as open. All of them have landed. This
  paragraph is hand-written, so `build_digest.py --check` cannot see it.
- `.claude/skills/bcir-systems-engineer/SKILL.md` has the same stale claim in §7 (line 213) and
  §9 (lines 294 and 304).
- `docs/REPO_CURRENT_STATE_AUDIT.md` (header dated 2026-07-22) has no GEM+ section. Item 9 is
  still true for target certificates, but it does not say that TMSAO-1 and TMSAO-2 now exist on
  the proof rail.

---

## 7. Remaining work, in priority order

**P0: correctness**

1. **ASN1-HYGIENE.** Lexically scoped X.683 substitution and an environment-aware memo key, with
   the §4.1 witnesses.
2. **ASN1-RECURSION.** A type containment graph with SCC detection. Recursion support on PER,
   OER, XER and JER, and cyclic nodes in the JER plan. Until then, honest refusals that name
   recursion rather than open types.

Found while landing ASN1-R, and P0 for the same reason:

- **ASN1-TAGS** (§4.4). A tagged type the type model can hold, so a component's own tag over a
  tagged type keeps both tags, and a tagged assignment encodes with its tag.
- **ASN1-NULL** (§4.5). One abstract value for NULL, `codec.NULL`, on every rail.

**P1: TMSAO honesty and speed**

3. **EXEC-EXACT.** Let a proved-optimal placement be the plan that runs. R13 would accept any
   legal placement (hazards, eligibility, the plan's own price) instead of only the heuristic's,
   and the readers would execute the slots from the plan's bytes. This is what makes the "11%"
   go away in execution, not only in the certificate.
4. **ORACLE-GC.** The scoped GC pause with its zero-cyclic-garbage witness, plus slots on the hot
   dataclasses; measured as harness rows.
5. **PLAN-SIGN.** Put the SHA-256 scope digest and the pack digest in the plan (an append-only v4
   record). Then either add an asymmetric signature (for example Ed25519) with a key policy, or
   record "MAC-only" as the declared boundary.

**P2: portfolio breadth** (each one needs a region or objective with its laws, an expansion, a
floor and refusals)

6. A stochastic objective: declared or measured branch probabilities, filling the scope's `U`.
7. An SDF/CSDF region and a timed-event (max-plus cycle-time) region.
8. A red-blue pebble / communication-cut bound, a hierarchical roofline, and the interval-graph
   dynamic program for memory at scale.
9. A value-of-optimization term in the dispatch law, including a route that sends simple
   sequential code straight to LLVM.
10. An optional hosted CP-SAT/ILP adapter as a small-instance oracle; a joint
    schedule × memory solver.
11. MLIR Transform export; the 38 unprojected IRDL operations; polyhedral depth above one; a
    decision on LLVM poison/freeze (a law, or a recorded boundary).
12. The CXX0 inventory as a committed document; native hydrate and the native delta chain.
13. ASN.1: semantic bounds under the canonical JER profile; the §9.8 tagging environment.

**P3: hardware.** Stage 6, two physical targets with PMU counters, which TMSAO-3 needs.

**Docs.** Reconcile the digest and the skill (done with the ladder's first commit); add a GEM+
section to the current-state audit.

---

## 8. How this was checked

- `python3 tools/perf/gemplus_baseline.py --compare --json …` on `5cd8e03`: 155 s, exit 0;
  104 GAIN, 7 NO-CHANGE (5 guards held at 0; 2 rows at their bound of 1.0, moved less than the
  2% exact band), 0 REGRESSION, 16 INDICATIVE, 0 NOT-MEASURED.
- Source reads of the roadmap, the staged plan, the 2026-07-31 report, the ASN.1/JSON proposal,
  the JER and JSON-program roadmaps, and the modules cited above.
- Repository-wide `git grep` for each named technique (excluding `training/`), checking code as
  well as docs.
- Reproductions run against `5cd8e03`. The capture and recursion modules of §4 become the RED
  witnesses of the ladder's ASN1-H and ASN1-R slices, and the collector measurements of §5 those
  of OR-GC, so each finding is checked by the slice that closes it rather than by a script kept
  outside the tree.
- The audit changed no tracked file; this document and the ladder are the first commit that does.
