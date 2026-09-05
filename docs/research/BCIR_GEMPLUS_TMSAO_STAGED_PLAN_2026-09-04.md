# GEM+ / TMSAO: assessment review and re-staged program (2026-09-04)

> Dated, non-normative. Baseline: `origin/main` at the merge of PR #757 (`7c851878`). The
> normative slice definitions, gates and order of work live in
> [`BCIR_GEMPLUS_ROADMAP.md`](BCIR_GEMPLUS_ROADMAP.md); this document records the review that
> produced its 2026-09-04 revision — what an earlier assessment found, what the tree has closed
> since, what it lacked, and why the program is staged the way it now is. Counts live only in
> the generated [`STATUS.md`](../STATUS.md).

## 0. Inputs

1. The assessment under review: the 2026-07-31 core/performance audit
   ([`BCIR_TMSAO_ARCHITECTURE_AND_PERFORMANCE_REPORT.md`](BCIR_TMSAO_ARCHITECTURE_AND_PERFORMANCE_REPORT.md),
   baseline PR #688), its 2026-08-04 ASN.1/ECN update (PR #706), the architecture proposal at
   PR #739 ([`BCIR_TMSAO_ASN1_JSON_DRIVER_PROPOSAL.md`](BCIR_TMSAO_ASN1_JSON_DRIVER_PROPOSAL.md)),
   and the consolidated text supplied for this review, which adds a P−1 → P4 stage plan and a
   list of P0/P1/P2 blockers found by a read-only audit of that era's tree.
2. What landed after it: the two 2026-08-12 security audits
   ([`BCIR_SECURITY_AUDIT_2026-08-12.md`](BCIR_SECURITY_AUDIT_2026-08-12.md),
   [`…12b.md`](BCIR_SECURITY_AUDIT_2026-08-12b.md)), GEM+ G0 and half of G9 with the frozen
   baseline harness (#740–#748), the assurance rails (#749), the whole-repository analysis
   (#751, [`BCIR_SYSTEM_ANALYSIS_2026-09-03.md`](BCIR_SYSTEM_ANALYSIS_2026-09-03.md)), the LLVM 23
   move (#752) and the dependency-audit arc (#753–#757).
3. Today's re-measurement of the 22 frozen rows (`tools/perf/gemplus_baseline.py --compare`,
   this host: Python 3.11.15, no PMU; §3).
4. The three emphases the revision was asked to serve: **core algorithms that fit the work in
   front of them** (a solver portfolio with a dispatch law, not one algorithm), **complete
   inter-process communication with optimization at every level** (the planes and the per-level
   contracts), and **measured performance above the current version** (the harness rows and the
   mechanisms that move them).

## 1. Executive summary

The assessment's architecture is right and the tree has been converging on it: keep GEM as the
conservative carrier, promote it to typed regions, keep min-plus as a primitive rather than a
master algorithm, bind every claim to one content-addressed scope, price one canonical plan,
solve with a bounded portfolio, and label every result with a certificate class. Nothing in
this review reverses that.

What changed is the ground under it. Of the seven P−1 blockers the assessment names, four are
closed and three are partly closed with the remainder scoped (§2.1). The scope identity it
called for exists (`ExecutionScopeV1`, G0). The two rows that decide whether the optimizer is
telling the truth about its own objective are **unchanged today**: `pricing.eft.divergence` is
still 1.9922 and `optimize_scheduled.slowdown.512` is 65.5× (§3). Every certificate BCIR emits
is still TMSAO-4, exactly as the roadmap says.

What the assessment lacked is not a slice but a set of *contracts* the slices need to cross
rails and processes honestly (§4): a binary plan ABI, a control-plane message ABI, a live
shared ring, deterministic solver budgets, incremental re-verification, resumable search
state, a workload model, and cross-rail solver parity. Without them, "one canonical plan"
would exist only as Python objects, "anytime" would mean wall-clock on whichever host ran it,
and "IPC at every level" would remain prose.

So the program is re-staged into six stages (§6). Stage 0 finishes the correctness closure the
assessment demanded first. Stage 1 lands the one canonical plan *and its ABI*. Stage 2 builds
the best-fit solver portfolio around the first lower bounds. Stage 3 completes IPC at every
level. Stage 4 is the performance program proper, measured on the frozen rows plus the new
ones. Stage 5 is movement, alias and escape. Stage 6 is physical evidence and is
hardware-gated. Each stage exits on `exact` rows; §7 lists the PR-sized sections in order.

## 2. Disposition of the assessment against `main` at #757

Status vocabulary: **closed** (a witness exists in the tree), **partly** (the consequence is
closed, the cause is scoped to a slice), **open** (scoped to a stage below), **not re-verified**
(the tree may or may not have moved; a Stage 0 check owns it), **outside** (a GitHub or
maintainer setting, not repository content).

### 2.1 P−1 — blockers before trusted artifacts or certificates

| # | Finding (assessment) | Status | Evidence / owner |
|---|---|---|---|
| 1 | Atomic opcodes lose atomic semantics in candidate generation (scalar atomic → U vec16; random atomic → GGG gather) | **closed** | `bcir/kbcir/realize.py` — an `ATOMIC_OPCODES` claim has exactly one realization, itself, on the atomic lane; the stride-class dispatch no longer applies to it |
| 2 | R9 accepts a forged, unavailable, zero-cost candidate | **closed (S0-A)** | `bcir/verify/__init__.py` R9 re-derives the planner's *actual* offer (`fused_candidates`, not `candidates_for`: the old re-derivation rejected the planner's own plan on every fused consumer — 3,840 of 4,096 claims of the audit fixture), every step's realized cost through the one predicate the planner prices with (`realize.edge_cost`), and budget feasibility; `verify_all` and every caller pass the scope they planned with |
| 3 | The graph → plan → StreamPack → artifact chain is fail-open (`attested` = R12 only; empty pack passes R10; pack not bound to its result; R11 stores only maximum generations) | **closed (S0-E)** | `bcir/api.py` — `attested` now means the whole chain (`verify` + `verify_plan` + `verify_c_lowering`); the empty-pack R10 vacuity was closed as a Class-B defect in the 2026-08-12 audit. S0-E landed StreamPack **v4**: `hydrate` emits one `(rid, map_gen, data_gen)` per declared resource (`generation_vector`, an append-only record after the trace stream, `n_gens` at header offset 40, the header tags pinned to the vector's maxima), and R11 per resource is one predicate on three rails — `verify_pack`, the C `bcir_sp_check_generation_vector` / `bcir_sp_execute_checked_vector`, and `-bcir-verify` over the `generations` triples — with the ASN.1 projection (`generations [10]`, projection version 2). RED first: the parent accepted a resource that moved under the maxima and a resource declared after hydration on all three rails (the oracle over `vector_add`, the law rail over `verify_generation_vector.mlir`'s cases, the C maxima API over the mutator's `stale_vector`); each is now refused, a vector-less v1–v3 artifact is stale against any registry that declares resources, and the maxima-only API keeps that RED as a gated witness. `topo_gen` stays the constant 1 by design: topology identity is bound through the manifest's `m_module` (R13, S0-D), documented in `BCIR_STREAMPACK_ABI.md` §v4 |
| 4 | Provenance equality does not imply identical plans (hash omits the memory hierarchy and declared order; `replay` compared only the digest) | **closed (S0-D)** | G0 landed `ExecutionScopeV1` (`bcir/kbcir/scope.py`): the certificate binds to the complete scope, and `replay()` compares the produced plan. S0-D widened the two cross-rail hashes on both rails in one commit: `hash_target` folds the memory hierarchy (`target.capability` gains `mem_tier_names` / `mem_tier_values` — the tier names, a channel's own included, and four i64 per named tier; absent = the default hierarchy; `hashTargetFromIR` recomputes it), `hash_module` folds the claims in DECLARED order (`hashModuleFromIR` walks them textually), the emitter writes every hashed field (`target_name`, `scalable`, `cal_gen`, the tier arrays, `base_weights`, all eight Θ pressures, `opcode`, `offset`, every declared phase), and `bcir/tests/test_hash_parity.py` is the differential regression (six targets × four modules with their real manifests accepted; a swapped claim order or a scaled tier refused as an m_module / m_target mismatch) |
| 5 | LLVM/C lowering miscompiles accepted claims (offsets and strides ignored; blanket `noalias`; unmasked vector tail for runtime `n`) | **closed (S0-G)** | `bcir/lower/llvm.py` refuses nonzero `offset` and non-unit strides instead of lowering them to `A[i]`; G9 removed the blanket `noalias` on shared RIDs. S0-G closed the tail: the defect stood (the vector loop stepped to the runtime `n` itself, so vector_add at width 16 called with n = 1031 wrote C[1031..1039]; a non-divisible compile-time count was legalized to scalar instead). A vector kernel now bounds its loop by `n & -W` and finishes the remainder in a scalar epilogue at the selected width; R12 holds the mask, the epilogue and its declaration, and the self-check harness drives every kernel (AOT, JIT, WASM) with `count + 7`, a sub-width count and 0 behind canaries — the parent's kernel fails it |
| 6 | The decoupled GGG tail can overlap conflicting or barriered work (cross-stream RAW/WAR/WAW and fence edges not built before the split) | **open** | `bcir/gem/schedule.py` still schedules the sparse tail as a serial chain overlapping the waves. Owned by G1: hazard edges across streams are constructed before the stream split, with the barriered-writer/main-reader fixture as the negative witness |
| 7 | CSE can merge semantically different or effectful claims (signature = op + read versions; precedes the barrier guard) | **open** | The oracle's deforestation/CSE coupling lives in `realize.py`'s fused-candidate path. G1 adds the complete semantic identity to the signature and excludes atomic, volatile, barriered and effectful claims categorically, with negative tests |

### 2.2 P1 — correctness and parity

| # | Finding | Status | Evidence / owner |
|---|---|---|---|
| 8 | MLIR module scope violated (root-global maps in `-bcir-verify`, selection and GEM passes) | **closed (S0-B)** | one scope predicate in `BCIRPassSupport.h` (`forEachScope` / `walkScope`): `-bcir-verify`, `-bcir-select-realization` and the GEM batch/schedule/lower passes are anchored at `bcir.module`, with operations outside any module as the one outer scope (a module-free file verifies as before). Witnesses `verify_module_scope.mlir` (a claim cannot resolve another module's resource; two modules may own RID 10; the outer scope is not vacuous) and `select_module_scope.mlir` (namesake paths priced per module: 9472 and 7808, where the root-global map had reported a false mismatch). `-bcir-rcsp` (the constrained selection) is scoped through the same predicate. Of the passes still walking from the root, eleven already dispatch per `bcir.module` by name (alloc-pool, bundle, cim/dvfs, compose, cost-model, explain, overlap, plan, rcsp-plan, schedule-eft, sense) and the rest build no name maps (cache-contention, layout-pivot, the GEM cost and lowering passes) |
| 9 | Phase identity and ordering inconsistent across rails (dangling/duplicate phase ids; five different orders in use) | **closed (S0-C)** | R4 refuses a duplicate phase id and a dependency on an undeclared phase on both rails, and ONE canonical phase order (dependency-first, roots in declaration order: `model.topological_phase_ids` / `BCIRPassSupport.h` `canonicalPhaseOrder`) ranks the cost-model columns, `-bcir-schedule`, `-bcir-overlap` and `-bcir-schedule-eft` — the numeric-id sort put a phase before the phase it depends on (`schedule_phase_order.mlir`); the structural corpus carries the identity cases |
| 10 | Advertised verifier checkpoints missing from `bcir-optimize` and `bcir-hydrate` | **closed (S0-B)** | `bcir-optimize` is verifier-checkpointed on entry and on the plan it emits; `bcir-hydrate` on entry (`BCIRPasses.cpp`). `pipeline_checkpoints.mlir` runs an illegal module under both and expects the same R2 at the door; both pipelines ran it to completion before. Since the #761 review fixes the trailing checkpoint READS the plan: R9 re-derives `kbcir.plan_width` / `plan_cost` / `plan_score` from the scope through the planner's own functions (`verify_plan_annotations.mlir`: the emitted plan accepted, four corruptions refused) |
| 11 | Two compiled fixtures inert (`verify_timing_lifetime.mlir`, `cost_model_barrier.mlir` not invoked) | **closed (S0-B)** | both execute in `check_passes.sh` (R19–R21 under `-verify-diagnostics`; the ASM3b barrier cost under FileCheck), and `bcir/tests/test_mlir_fixture_inventory.py` reconciles every `mlir/test/passes/*.mlir` against the runner scripts both ways in the quick tier: it named exactly the two, plus the three S0-B witnesses, before the runner did |
| 12 | MLIR R13 incomplete (skips absent IR objects; no arity/order check; certified constants unchecked) | **closed (S0-C)** | the defect stood on all three counts. `-bcir-verify` R13 now holds the artifact record to the oracle's shape (names and generations of one arity, sorted and unique, counted by `n_artifacts`; the digest folds that sequence), refuses a component hash for an object the enclosing module does not carry (a bare manifest record stays digest-only), holds a calibrated capability's `gather_penalty`/`base_overhead`/`mem_unit` to its certificate, and tags the generation rule R13; the oracle's promoted-entry rule covers every entry (`structural_corpus.mlir`, `verify_provenance.mlir`) |
| 13 | Verifier-legal convolution can overflow signed arithmetic in the GEM lowerer | **closed (S0-C)** | the defect stood: a verifier-legal one-tile conv (M = 2⁶², N = 4) lowered to a `gem.block` with `count = 0`. The `gem.conv` verifier and `conv.check_conv` bound the output element count M·N and the im2col work M·N·K to the signed 64-bit wire domain, and `-bcir-lower-gem-conv` computes every tile origin and count with checked arithmetic (`lower_gem_conv_overflow.mlir`); the S0-8 tail contract for runtime-`n` remains open |
| 14 | C R9 trusts caller-provided costs; scalar planner writes `count` as the lane width, hydration requires a power of two | **closed (S0-A)** | `bcir_plan_func` emits width 1 and cost = base × n; `bcir_verify_plan` re-derives every cost through the header-inline `bcir_plan_base_cost` and requires a power-of-two width; the planner and the hydrator compose on any count (C witness in `test_c_runtime.py`) |
| 15 | EV1–EV3 exist but `verify_all` never invokes them | **closed (S0-A)** | `verify(module)` carries EV1–EV3 (module laws; vacuous over the eventless corpus), so `verify_all` and `is_legal` do too |
| 16 | Telemetry replay evidence is not transport replay evidence (static ids as sequencing; no integrity witness retained) | **partly** | The BTLM v1 frame ABI carries `seq`, per-frame CRC and resync, byte-identical on both rails (`docs/kernel/TELEMETRY_FRAME_ABI.md`); a *live* ring with per-slot sequence and loss accounting is G15 |
| 17 | Python structural legality weaker than MLIR (widths, alignments, shapes, zero/negative strides) | **closed (S0-C)** | `verify(module)` holds the registry (R1: positive shape extents, an element count within signed 64-bit, a power-of-two `align`) and the access pattern (R7: non-negative count and offset, positive `stride_k`, an extent within signed 64-bit — the zero stride no longer folds to 1) to the op verifiers' rules, and `TargetProfile` validates at construction like `target.capability`; the corpus runs the same cases on both rails |

### 2.3 P2 — portability, packaging, governance

| # | Finding | Status | Evidence / owner |
|---|---|---|---|
| 18 | IRDL coverage incomplete with no ODS→IRDL inventory gate | **closed (S0-B)** | `mlir/irdl/MANIFEST.json` declares the unprojected subset (38 of 133 operations: the driver-subset core ops, ECN, and the GEM model ops that landed after the projection) and the one naming rule (IRDL admits only `[a-z0-9_]`, so `gem.stream_pack` is projected as `gem_stream_pack`); `tools/irdl/check_inventory.py` reconciles ODS ↔ projection ↔ manifest both ways (undeclared gap, stale entry, ghost, orphan, naming collision; vacuous inventories refused), runs first in `check_corpus.sh` without a toolchain and in the quick tier (`test_irdl_inventory.py`, one injected fault per witness). Coverage itself (projecting the 38) is now a visible, gated gap rather than an unknown |
| 19 | Address width target-insensitive (i32 accepted, later `inttoptr`) | **closed (S0-C)** | the defect stood (an i32 `volatile_load` address under `x86_64-avx512` verified clean). `-bcir-verify` R12: under a declared target the address of a `volatile_load`/`volatile_store`/`atomic_rmw`/`atomic_cas` must equal the target's pointer width, from ONE triple→width table (`kbcir.cost.pointer_width` / `pointerWidthOfTriple`; `verify.verify_address_width` is the oracle twin); the op-level ≥ 32-bit floor stays for target-less IR |
| 20 | MAP/ROP mishandle non-RAM resources (default domain RAM) | **closed (S0-C)** | the defect stood (an HBM-only MAP or ROP program was refused by R3; a device register could not be addressed). Both front-ends derive a claim's domain from the resources it touches (`model.derived_claim_domain`: an isolated domain it touches, else its destination's), mark isolated-domain claims volatile and accept `hazard`; a device-register access without an ordered hazard is refused by R3/R5 on both rails. The isolated-domain rule became ONE rule on both rails: the resource side must match, an MMIO claim may carry tier operands (all 38 cfront MMIO claims have that shape), and NVM is a memory tier (the HAM fabric stages it into VRAM; the law rail's "NVM cell" isolation had no fixture) |
| 21 | M5 descriptors need construction-time validation | **closed (S0-C)** | the ETL descriptors (`BinaryField`/`BinaryRecord`/`BinaryFormat`, `EventStream`/`EventKind`/`Event`, `State`/`Transition`/`Transducer`, `TokenRule`/`Grammar`) validate at construction — a zero width, a negative offset, overlapping or overrunning fields, an unknown endianness or stream kind, a transition to an undeclared state, a duplicate `(state, symbol)` — and the dialect's `binary.*`, `event.*`, `fsm.*`, `parse.*` ops carry the same rules as op verifiers; the corpus holds them together |
| 22 | Wheel incomplete; CI never tests the built wheel | **closed** | #749: the suite runs from the installed package, package data ships, and an exclusion is scoped to the dependent test (law L21) |
| 23 | `main` unprotected | **outside** | repository settings; recommended to the maintainer, not repository content |
| 24 | Private vulnerability reporting unavailable | **outside** | `SECURITY.md` names the route; enabling it is a repository setting |
| 25 | No tags, releases, attestation, SBOM | **outside** / open | release governance is a maintainer decision; the dependency audit's inventory and `tools/security/audit/` are the SBOM substrate when it is taken |
| 26 | License terminology; CER described as non-canonical | **closed (S0-A)** | README states the license is source-available and non-commercial, not OSI open source; the LangRef says CER is excluded by profile (its indefinite-length form), not for want of canonicality |
| 27 | `bcir-tmsao-audit` is a performance-audit precursor, not a certificate | **closed (S0-A)** | `bcir-performance-audit` is the command; `bcir-tmsao-audit` stays as an alias that prints that GEM+/TMSAO certificates are not implemented |

### 2.4 Documentation

The LangRef split the assessment asked for has happened: `docs/BCIR_LANGREF.md` is the normative
reference alone, the proposal and reports live under `docs/research/`, and the law-range drift
it noted is now machine-checked (`tools/docs/check_law_range.py`, #752).

### 2.5 The assessment's stage plan against the roadmap

| Assessment stage | Where it is now |
|---|---|
| P−1 correctness closure | Stage 0 below: items 2–17 that remain, plus G7 |
| P0 scope and evidence schemas | G0 **landed** (scope identity, class ladder, gap arithmetic); the frozen baseline harness exists; the plan's *binary* form is G11 |
| P1 canonical GEM+ plan | Stage 1: G1, G3, G11, G5 |
| P2 bounded TMSAO | Stage 2: G2, G4, G12, G6, G13 |
| P3 physical evidence | Stage 6, hardware-gated; G7 is pulled forward into Stage 0 because it invalidates a number in use |
| P4 drivers / wire deployment | the driver/kernel roadmap's D-ladder; Stage 3 supplies the planes it needs |

## 3. Measurement today

`tools/perf/gemplus_baseline.py --compare` on this host (Linux 6.18, Python 3.11.15, no PMU;
the baseline host was a Ryzen 5 2600 under WSL with Python 3.10):

| Row | Slice | Baseline | Today | Verdict |
|---|---|---:|---:|---|
| `pricing.eft.divergence` | G1 | 1.9922 | 1.9922 | **NO-CHANGE** — the objective still denotes two schedules |
| `optimize_scheduled.slowdown.512` | G2 | 69.2× | 65.5× | **NO-CHANGE** (inside the 25% `ratio` band) — the quadratic sweep is untouched |
| `optimize_scheduled.512` | G2 | 1,703.66 ms | 1,143.17 ms | INDICATIVE (faster host and interpreter, not a slice) |
| `audit.kbcir-streampack.scale4` | G2 | 275.22 ms | 126.94 ms | INDICATIVE |
| `audit.static-lifetime-planner.scale4` | G0 | 566.35 ms | 220.48 ms | INDICATIVE |
| `audit.mixed-wave-token-eft.scale4` | G1 | 48.05 ms | 24.96 ms | INDICATIVE |
| `audit.iterative-phase-dag.scale4` | G3 | 34.62 ms | 10.83 ms | INDICATIVE |
| 14 rows | G4, G5, G0, G6 | — | not measured | need the exact oracles (`eft.*`, `memory.*`), the digest fixtures, or the native rig |

Two facts to carry forward. First, the NO-CHANGE rows are the roadmap's outcome (a): the slices
that own them have not landed, so the rows are correctly assigned and simply waiting. Second,
the INDICATIVE rows being 1.5–3× faster here is *not* evidence: `wall` rows never grade off the
baseline host, and this comparison is the demonstration of why. The rows that will grade a
slice are the `exact` and `ratio` ones, and the new rows in §6 are classified the same way.

## 4. What the assessment lacks

Each of these is a contract or a mechanism the slices need in order to be honest across rails,
processes and hosts. None is a new architecture; each is the missing piece that lets an
existing slice claim what it says.

1. **Deterministic solver budgets.** "Anytime, within budget" is only host-portable if the
   budget is counted in the solver's own work units — node expansions, label relaxations,
   candidate evaluations — never in seconds. A certificate that says "TMSAO-2 after 10⁶
   expansions, stop reason: budget" means the same thing on CI and on the report's host; one
   that says "after 2 s" does not. This is the `exact`/`ratio`/`wall` discipline applied to the
   solver itself, and it is what makes G4 and G12 gateable. (G12)
2. **A binary plan ABI.** The canonical plan cannot be Python objects if the C twin, the MLIR
   rail and a resident executor are to read it. `ExecutionPlanV1` is an append-only StreamPack
   v4 record family — schedule slots, lifetimes and addresses, movement edges, generation
   vectors — with a C twin, a BCAB kind and an ASN.1 projection, byte-identical across rails.
   The pack stays the executable; the plan is what the pack was derived from and what every
   reader prices. (G11)
3. **A control-plane message ABI.** Lease, generation, quiescence, activation, rollback and
   cancellation are prose and an `admit(map_gen, data_gen)` argument today. They become small,
   fixed, versioned records with a C twin and a DER/COER projection, so a stale generation is
   refused by bytes at every boundary rather than by convention. (G14)
4. **A live shared ring.** The telemetry frame is frozen and byte-identical, but there is no
   live SPSC ring with head/tail, acquire/release publication, per-slot sequence numbers and a
   declared overwrite/backpressure policy with exact loss accounting — the "version-zero
   triple" the 2026-09-03 analysis named. It is the transport for telemetry first and for the
   control records second. (G15)
5. **The workload component `W`.** The scope table marks `W` "not modelled". Best-fit dispatch
   needs shapes, batch, concurrency, service-level requirement and horizon as declared inputs,
   and the measured-candidate database (`schedule_artifact.py`, B1) as the replay corpus that
   informs — never decides — the choice. (G13)
6. **Incremental re-verification.** Incremental plans (G2, G18) are only honest if the
   R-laws are re-checked incrementally *and* the incremental verifier is proven equal to the
   full one over a corpus. A delta plan verified by a delta verifier that has never been
   differentially tested is a Class-B vacuous check waiting to happen. (G18)
7. **Resumable, content-addressed search state.** The frontier, incumbent, bound and stop
   reason are an artifact; resuming from it must reproduce the same continuation. Without this
   "anytime" cannot be checkpointed or audited. (G12)
8. **Plan and certificate diff.** The per-slice analysis protocol asks what a residual gap is
   made of. That needs a structured comparison of two plans under one scope — which claims
   moved, which bins changed, which bound tightened — the regret ledger generalized. (G12)
9. **Dominance, symmetry breaking and the interval structure of memory.** The exact solvers
   are dependency-free Python at CI scale; they are feasible only with dominance pruning,
   symmetry breaking on identical tasks and bins, and, for memory, dynamic programming over the
   interval graph's clique structure rather than raw backtracking. (G4, G5)
10. **Cross-rail solver parity.** A native planner (the proposal's CXX2/CXX3) may own a
    certificate only after it reproduces the Python solver's plan byte for byte over a generated
    corpus, the same way the C twins earn their rails. (G17)
11. **Security at the plane boundaries.** Capability-scoped generation handles, signed
    artifacts and W^X are already policy in the driver roadmap; the IPC slices carry them from
    the first record, not as a later hardening. (G14–G16)
12. **Measurement without a PMU.** Most hosts that will run these gates have no counters. The
    harness measures deterministic proxies (work units, bytes, unique elements) and refuses the
    silicon claim; G7 makes the refusal mechanical. The two-target rule stands: no TMSAO-3
    without two materially different physical targets with counters. (G7, Stage 6)

## 5. The three emphases

### 5.1 Core algorithms that fit the work: the dispatch law

One algorithm cannot be best for a 6-claim atomic region, a 4,096-claim elementwise stream and
a fixed-rate StreamPack pipeline. The portfolio is chosen by a **dispatch law**, deterministic
and recorded in the certificate:

| Question | Region / size | Solver | Certificate it can reach |
|---|---|---|---|
| Fast legal incumbent | any | list/EFT, first-fit | TMSAO-4 |
| Layered additive choice | candidate DAG, any size | min-plus DP; RCSP under budgets; Pareto frontier | TMSAO-1 over the layered graph, TMSAO-4 composed |
| Schedule + placement + memory jointly | ≤ small (work budget) | branch-and-bound with dominance, symmetry breaking, incremental bounds | TMSAO-1 |
| Same, medium | bounded work budget | beam / anytime B&B with the lower-bound stack | TMSAO-2 (gap stated) |
| Same, large | any | delta-priced local search from the EFT incumbent; bound tracked | TMSAO-2 if a bound exists, else TMSAO-4 |
| Critical path, throughput, deadlines | timed-event region | max-plus / event graph, network calculus | TMSAO-2 bound source |
| Regular loops | affine region | index maps, dependence polyhedra, tiling/fusion legality | TMSAO-2 bound source + candidates |
| Fixed-rate streams | SDF/CSDF region | repetition vectors, static schedules, bounded FIFOs | TMSAO-1 over the region |
| Algebraic alternatives | equivalence region | bounded e-graph saturation + extraction | candidate source only |
| Static memory | interval graph, small | exact clique/DP placement | TMSAO-1 |
| Static memory, large | interval graph | first-fit / best-fit with the concurrent-live bound | TMSAO-2 |

Rules: every solver publishes its first legal incumbent before improving it; budgets are work
units; tie-breaks are deterministic; a learned ranker may order candidates inside `A` and is
recorded in `G`, and may not remove one without a proof; a region that cannot verify its
refusal conditions falls back to the opaque claim DAG. The certificate names the solver, the
budget, the stop reason and the bound source.

### 5.2 IPC at every level

The proposal's five planes, mapped to what exists and what each stage adds:

| Plane | Carries | Today | Open |
|---|---|---|---|
| Data | StreamPack v1 (+v2/v3), BCAB bundles, tensors | frozen ABI, C twin, byte-identity gates | `ExecutionPlanV1` records (G11); borrowed zero-copy views and a manifest-of-shards format (G16) |
| Control | lease, generation, quiesce, activate, rollback, cancel | prose; `admit(map_gen, data_gen)`; quiescence in the driver fixtures | the control record ABI (G14) |
| Telemetry | frames and readings | BTLM v1 frame ABI (byte-identical), `TelemetryIntegrity`, signal registry T1 | the live SPSC ring with loss accounting (G15) |
| Evidence | certificates, provenance, measurements, calibration | content-addressed JSON artifacts (`scope.py`, `provenance.py`, `schedule_artifact.py`, calibration) | append-only store semantics and the plan/certificate diff (G12, G13) |
| Inspect / plan | reads and transactional plan operations | Python API and CLI, embeddable and deterministic | stays a library; no service is introduced |

And the levels a plan crosses, each with the contract that carries it:

| Level | Boundary | Contract | Status |
|---|---|---|---|
| L0 | planner ↔ verifier, same process | identity-bound digest API; the verifier keeps its right to recompute at a trust boundary | G3 |
| L1 | Python oracle ↔ C twin | StreamPack / plan bytes, differential parity | plan bytes: G11 |
| L2 | C++ orchestrator ↔ C runtime | borrowed views with explicit lifetime; generation gating at `admit()`; transactional builders | G16 |
| L3 | host ↔ device | `bcir_channel_open/claim/map/submit/sync/next_event/close` (`runtime/c/bcir_runtime_channel.h`), loopback and simulator; HAM movement edges | verbs exist; movement as a first-class edge is G8 |
| L4 | process ↔ process | SPSC shared rings for telemetry and control records | G15, G14 |
| L5 | node ↔ node | the MPI/NCCL orchestrator, `shard()` real, dispatch a declared stub | stays a stub until a cluster exists; the manifest-of-shards format (G16) is the honest preparation |

"Optimization at each level" means: L0 hashes once; L1 moves bytes, never objects; L2 copies
nothing it can borrow; L3 prices movement jointly with compute; L4 is lock-free with declared
backpressure; L5 ships digests and shards, not graphs. The invariants that hold across all of
them: records are bounded and versioned; a stale generation is refused by bytes; no plane
carries a legality verdict except the verifier's own output; evidence is append-only.

### 5.3 Performance above the current version

Measured, on the frozen rows and the rows below, with the mechanism named:

| Mechanism | Rows | Slice |
|---|---|---|
| Delta pricing replaces makespan recomputation (identical assignment) | `optimize_scheduled.slowdown.512` 65.5× → ≤ 8×; `optimize_scheduled.{256,512}` | G2 |
| One digest, identity-bound, mutation-invalidated | `static_memory.{digest,verify,plan}.2048` | G3 |
| Compact indexed planner (arrays, not per-claim dataclasses), byte-identical plans | `audit.kbcir-streampack.scale4`; a new `planner.calls` exact row | G17 |
| Incremental re-plan + delta StreamPack | `kbcir-streampack.delta` (new, `ratio`) | G18 |
| Native planner after parity | `wall` rows, indicative; `exact` parity row | G17 (CXX2/3) |
| Structural wins preserved through the regions refactor | `native.*` guardrails | G6 |

## 6. The re-staged program

The normative definitions are in the roadmap; this is the shape and the exit gates.

| Stage | Slices | Exit gate (`exact` rows) |
|---|---|---|
| **0 — correctness closure remainder** | S0-1 … S0-10 (§7), G7 | every item has a negative witness on its rail; the two inert fixtures execute; `verify_all` includes EV1–EV3; the native rig refuses the bare-metal claim; no new certificate class yet |
| **1 — one canonical plan and its ABI** | G1, G3, G11, G5 | `pricing.eft.divergence` = 1.0; pricing, token execution, static memory and StreamPack lowering read one artifact and produce identical traces; `plan.abi.roundtrip` byte-identical Python ↔ C; the two-phase alias fixture rejected |
| **2 — best-fit solver portfolio** | G2, G4, G12, G6, G13 | first TMSAO-2 certificate; `solver.exact.coverage` and `solver.gap.p95` reported over the corpus; identical assignment under delta pricing; every region expands conservatively (differential per region); `native.*` no regression |
| **3 — IPC at every level** | G14, G15, G16 | one artifact generation flows plan → control → data → telemetry → evidence on the loopback/simulator with generation handles end to end; stale generations refused at every boundary; `ring.loss.accounting` exact; `control.record.bytes` bounded |
| **4 — performance program** | G17, G18 | rows moved with the mechanism named and the plan unchanged (`planner.parity`, `verify.delta.identity` exact) |
| **5 — movement, alias, escape** | G8, G9 remainder, G10 | joint movement/compute pricing; alias scopes, TBAA and `volatile` to LLVM; non-escaping resources counted |
| **6 — physical evidence** | two-target calibration, PMU/energy | TMSAO-3 on two materially different physical targets; hardware-gated, never simulated |

Dependencies: Stage 1 needs S0-1 and S0-2 (the identity the plan binds to and the generation
vectors it carries); Stage 2 needs G1 (one artifact to solve over) and G11 (a plan the exact
solvers can emit); Stage 3 needs G11 (the records the planes move) and G14 precedes G16; Stage
4 needs G1 and G18's differential; Stage 5 needs G6; Stage 6 needs hardware.

## 7. The sections, in order

Each is one PR, one gate, one analysis paragraph. Stage 0 first, smallest first.

| Section | Content | Depends on |
|---|---|---|
| **S0-A** (Python + C) — **landed** | S0-4 EV1–EV3 into `verify_all`; S0-7 R9 re-derives the offer, every cost and budget feasibility from the scope, and the C R9 width/cost contract; S0-10 `bcir-performance-audit` with the compatibility alias and the CER/license wording sweep | — |
| **S0-B** (MLIR) — **landed** | S0-3 verify checkpoints in `bcir-optimize`/`bcir-hydrate` and the execution-inventory gate that runs the two inert fixtures; S0-5 module-scoped walks; S0-9 the ODS→IRDL supported-subset manifest and parity check | LLVM 23 host or CI's rail |
| **S0-C** (both rails) — **landed** | S0-6 the shared structural-law corpus: widths, alignments, shapes, strides, phase identity and ordering, address width, non-RAM domains, M5 descriptors, MLIR R13 arity/order, the convolution overflow fixture — one corpus, two runners, every mismatch a finding | S0-B |
| **S0-D** (two-rail, one commit) — **landed** | S0-1 `hash_target`/`hash_module` widening: ODS attributes for the memory tiers and declared order plus the matching C++ walks, with the differential regression | S0-B |
| **S0-E** (ABI) — **landed** | S0-2 R11 per-resource generation vectors as a StreamPack v4 append-only record, its C twin and ASN.1 projection, landed with `BCIR_STREAMPACK_ABI.md` | — |
| **S0-F** — **landed** | G7 native measurement rig repair: the full-cycle strided walk with a counted census, raw samples with their statistics, and a host attestation from which the rig derives its tenancy — "bare-metal" only with proof; the reader refuses a table whose summary or claim its evidence does not support | — |
| **S0-G** — **landed** | S0-8 the LLVM kernel's runtime-`n` tail contract: the vector loop over `n & -W` plus a scalar epilogue at the selected width, R12 holding the mask and the epilogue, the self-check harness driving every kernel with a non-divisible count, a sub-width count and zero behind canaries | — |
| **S1-A** | G1 one canonical schedule artifact, with cross-stream hazard edges before the tail split and the CSE exclusions as negatives | S0-D, S0-E |
| **S1-B** | G3 digest computed once, with the mutation-invalidation witness | — |
| **S1-C** | G11 `ExecutionPlanV1` records, C twin, BCAB kind, ASN.1 projection | S1-A, S0-E |
| **S1-D** | G5 schedule-aware liveness and bounded exact memory | S1-A, S1-C |
| **S2-A** | G2 incremental delta pricing (identical assignment) | S1-A |
| **S2-B** | G4 bounded exact solvers and the lower-bound stack — the first TMSAO-2 | S1-C |
| **S2-C** | G12 the dispatch law, work-unit budgets, resumable state, plan diff | S2-B |
| **S2-D** | G6 typed regions and the objective registry, affine first | S2-C |
| **S2-E** | G13 the workload model `W`, the measured-candidate database, the replay-gate integration | S2-C |
| **S3-A** | G14 control-plane record ABI | S1-C |
| **S3-B** | G15 live SPSC ring (telemetry, then control) | S3-A |
| **S3-C** | G16 data-plane hand-off: borrowed views, generation gating, the dynamic-graph builder, the manifest-of-shards format | S3-A |
| **S4-A / S4-B** | G17 compact planner and native parity; G18 incremental re-verification and delta StreamPack | S2-A, S1-B |
| **S5** | G8, G9 remainder, G10 | S2-D |
| **S6** | physical calibration | hardware |

## 8. Decision rules and what this plan will not claim

- **Correctness before speed, always.** A row that moves while a plan changes has not moved.
- **No certificate above TMSAO-4 before Stage 2**, and none above TMSAO-2 without two physical
  targets with counters.
- **No IPC claim beyond the simulator** until a device or a second process runs the contract;
  the node level stays a declared stub until a cluster exists.
- **No sublinear claim on an Ω(n) row** without naming the admitted work that changed, and no
  cached result above TMSAO-4 unless its invalidation predicate is in the scope.
- **No optimizer decision by a learned organ.** It orders candidates inside `A`, is recorded in
  `G`, and never removes one without a proof.
- **Governance items** (branch protection, private vulnerability reporting, releases and
  SBOM) are the maintainer's; this program supplies the substrate, not the settings.

## 9. Sources

The two in-tree assessment documents and the roadmap named in §0; the 2026-08-12 audits; the
2026-09-03 system analysis; the driver/kernel, HAM, telemetry and C++ hand-off documents under
`docs/kernel/` and `docs/languages/`; and, for the solver portfolio, the references the
proposal already lists (MLIR Transform and Affine, VPlan, OR-Tools CP-SAT, equality saturation,
GraphBLAS, the Roofline model, Ptolemy SDF).
