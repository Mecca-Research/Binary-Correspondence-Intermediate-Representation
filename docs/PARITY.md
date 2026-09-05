# Python ↔ MLIR parity contract

The Python package `bcir/` is the **executable conformance oracle**; the MLIR
dialect family under `mlir/` is the **law**. They must agree. This file is the
cross-map and the invariants that keep them in lockstep for package version `0.2.0`.
Separate tables below cover Python ↔ C artifact/runtime parity; not every concept has,
or should have, a representation on every rail.

## Enum value parity (normative)

Integer enum values are identical in `bcir/model/lanes.py` /
`bcir/kbcir/cost.py` and `mlir/include/BCIR/BCIRAttrs.td`:

| Enum | Values |
|---|---|
| Lane | U=0, UX=1, T=2, GGG=3, A=4, H=5 |
| StrideClass | Scalar=0, Unit=1, Strided=2, Cacheline=3, Tile=4, Random=5 |
| Domain | RAM=0, VRAM=1, NVM=2, MMIO=3, CXL=4, HBM=5 |
| HazardMode | Unique=0, Atomic=1, Barriered=2 |
| Verify | None=0, Bounds=1, Exact=2, Hash=3 |
| Bounds | Strict=0, Masked=1, AssumedSafe=2 |
| MemTier | L1=0, L2=1, L3=2, DRAM=3, HBM=4, CXL=5, SSD=6 |
| Access | Flat=0, HAM=1 |
| Asn1Class | Universal=0, Application=1, Context=2, Private=3 (X.690 Table 1: the identifier octet's high bits) |
| Asn1Rules | Ber=0, Cer=1, Der=2 |
| Asn1Tagging | Implicit=0, Explicit=1 |

The K_BCIR cost vector is **12-d** in both, same dimension order:
`compute, memory, fabric, sync, compile, thermal, power, reliability, security,
accuracy, contention, verification`.

## Concept parity

| Concept | Oracle (`bcir/`) | Law (`mlir/`) |
|---|---|---|
| target descriptor H | `kbcir.cost.TargetProfile` (alias `HProfile`) | `bcir.target.capability` |
| memory hierarchy | `kbcir.cost.Tier` / `MemoryHierarchy` | `bcir.mem.tier` / `bcir.mem.ham` / `bcir.mem.cxl_swap` |
| resource | `model.graph.Resource` (`access`, `priority`) | `bcir.resource` |
| claim | `model.graph.Claim` | `bcir.claim` |
| phase DAG | `model.graph.Phase` | `bcir.phase` |
| cost vector | `kbcir.cost.CostVector` (12-d) | `#bcir.costvec<...>` |
| policy / weights | `kbcir.weights.Policy` | `bcir.kbcir.policy` |
| candidate path | `kbcir.realize.Candidate` | `bcir.kbcir.path` |
| cost algebra (candidate costs) | `kbcir.cost._cost` / `realize.candidates_for` / `_stride_penalty` | **`-bcir-cost-model`** (`BCIRCostModel.cpp`, C++23): recomputes the 12-d candidate costs from `bcir.claim` + `bcir.target.capability` (constexpr tier table + seeded constants); reproduces vec16 @ 7808 / gather @ 528384 / tile @ 126976 from the claim alone (`cost_model.mlir`) |
| verification cost (12th axis) | `realize._verify_cost` (the verify-contract discharge cost: `none`/`bounds` free, `exact`/`hash` O(n); width-independent) | `BCIRCostModel.h::verifyCostFor` (mirror); `cost_model_verify.mlir` (bounds → verification 0 @ 7808, exact → verification 1024 @ 8832); a tradeable RCSP resource (`test_verify_cost.py`) |
| min-plus select | `kbcir.realize.optimize` + `semiring` | `bcir.kbcir.select` (`#bcir.semiring<min_plus>`) |
| min-plus **plan** (coupled shortest path) | `kbcir.realize.optimize` + `semiring.dag_shortest_path` + `_context_factor` | **`-bcir-plan`** (`BCIRPlanPass.cpp`, C++23): the layered tropical shortest path over the fused candidate columns with the path-based shared-input fusion; reproduces `optimize` for *all* modules (7808 / corpus 1015808·101888·1595520; `plan.mlir`, `gem_corpus.mlir`) |
| budget B(H,Θ) (RCSP) | `kbcir.rcsp.Budget` / `optimize_constrained` | `bcir.kbcir.budget` + `bcir.kbcir.select` `budget`; **`-bcir-rcsp`** (`BCIRPasses.cpp`, C++23) recomputes the budget-feasible label-DP argmin (reproduces 9472 under the 700 cap) and cross-checks the declared selection |
| Pareto front | `kbcir.rcsp.pareto_plans` (label dominance) | **`-bcir-rcsp`** computes the front over (score, thermal, power) by label dominance, annotates `kbcir.pareto_size` (the {vec16, vec8} front = 2; scalar dominated) — `mlir/test/passes/rcsp.mlir` |
| plan-level RCSP (accumulated budget) | `kbcir.rcsp.optimize_constrained` (label DP across the plan) | **`-bcir-rcsp-plan`** (`BCIRRcspPlanPass.cpp`, C++23): the accumulated-budget label DP over the fused columns; narrows one claim to fit a plan-wide cap (thermal≤2000 → {16,8} @ 17280; `rcsp_plan.mlir`) |
| scheduled price M(π,Θ) | `gem.overlap.price_scheduled` / `optimize_scheduled` | `bcir.kbcir.scheduled_price` (R9) + **`-bcir-overlap`** (`BCIROverlapPass.cpp`): recomputes the (max,+) wave makespan (matmul 253952 / gain 761856, corpus; `overlap.mlir`); **`-bcir-overlap-optimize`** ports `optimize_scheduled` — the makespan-driven re-selection sweep (adopt the per-claim alternative that strictly lowers makespan, re-priced serially for R9); matches the oracle's `(makespan, serial)` on all 11 corpus programs (`overlap_optimize.mlir`) |
| soft/differentiable select | `kbcir.softdp.softselect` / `free_energy` (T=0 ⇒ `optimize`) | `bcir.kbcir.soft_select` (R9: F ≤ score; T=0 ⇒ F == score) |
| duration-aware schedule | `gem.schedule.schedule_eft` (LPT+EFT+locality+knee) | `bcir.gem.schedule` mode `eft` (R9) |
| token-DAG execution | `gem.schedule.execute_tokens` (pipelined phases) | `bcir.gem.schedule` mode `tokens` (R9) |
| bandwidth knee | `gem.schedule.bandwidth_knee` / `TargetProfile.mem_channels` | `bcir.target.capability` `mem_channels` |
| pipelined/routed StreamPack (ABI v2/v3) | `gem.streampack.hydrate_pipelined` / append-only codec | `bcir.gem.stream_pack` `pipeline_depth`, `bcir.gem.prefetch` `buffers`, and v3 segment dispatch/channel metadata (R10) |
| L1 frozen cost table | `kbcir.microbench.CalibratedProfile` (`cal_gen`, Q8 ratios) | `bcir.kbcir.calibration` + capability `cal_gen` (R8) |
| L1 Bayesian + conformal table | `kbcir.bayescal` (`gaussian_update` VI / `conformal_delta` / `bayes_calibrate` / `abc_calibrate`) | `bcir.kbcir.calibration` `coverage_milli`/`random_delta_q8` (R8/R13) |
| L2 policy portfolio | `kbcir.portfolio.PolicyPortfolio` (class-table selection) | `bcir.kbcir.portfolio` (R9) |
| L2 replay gate | `kbcir.portfolio.replay_gate` / `ReplayCertificate` | `bcir.kbcir.replay_certificate` (R9: admitted ⇒ zero regressions) |
| L2 learned MoE gate | `kbcir.moegate` (`train_gate` GNN / `freeze` Q8 / `gate_replay_gate`) | `bcir.kbcir.moe_gate` (R13: routes certified experts, admitted ⇒ zero regressions) |
| search accelerator | `kbcir.accel` (`optimize_ordered` B&B / `train_ranker` / `accelerator_certificate`) | `bcir.kbcir.search_accel` (R13: admitted ⇒ zero mismatches; same optimum) |
| provenance manifest | `kbcir.provenance` (`build_manifest` / `replay` / `reproduces` / `verify_manifest`; `_fnv`/`_digest`/`hash_module`/`hash_target`/`hash_theta`/`hash_policy`) | `bcir.kbcir.provenance_manifest` (R13: deployed ⇒ reproduced; manifest equality ⇒ identical plan). When the op carries the four component hashes + artifacts, `-bcir-verify` **recomputes the digest** byte-identically to `provenance._digest`, and **cross-checks every component hash** against the IR — `m_module` from the `bcir.module` (resources/claims incl. the `opcode`), `m_target` from the `target.capability` (incl. `target_name`/`scalable`), `m_theta` from the `kbcir.theta`, `m_policy` from the `kbcir.policy` unfolded `base_weights` — each recomputed byte-identically to the oracle's `hash_*`. So a manifest can be re-pointed at neither a different goal graph, target, Θ, nor policy (`verify_provenance.mlir`; pinned by `test_provenance.py`) |
| building-blocks engine (e-graph) | `kbcir.egraph` (`EGraph` / `optimize_expr` / `saturate` / `shared_blocks`) | `bcir.egraph.extract` (R9: optimized_cost ≤ original_cost) |
| memory module (fixpoint) | `kbcir.memory` (`MemoryModule` / `freeze` / `freeze_module` / `is_idempotent` / `memory_artifacts`) + `verify.verify_memory` | **`bcir.kbcir.memory_module`** op + first-class `-bcir-verify` **R13** (`BCIRVerifyPass.cpp`): admissible iff `saturated ∧ generation ≥ 1` (`a = Lim(Res(U))` is a saturated, generation-tagged fixpoint; `fingerprint` chains into the manifest); negative-tested in `verify_laws_deep.mlir` |
| two-truth quarantine (MOPC) | `kbcir.twotruth` (`Graded` / `Decision` / `decide` / `is_classical` / `g_and`/`g_or`/`g_not`) + `verify.verify_quarantine` | *(graded-side interpretive layer; **no MLIR rail — quarantined off the spine by design**)* a graded `(v,w)` may inform but never *be* a legality verdict, so it must not become a dialect op; the only crossing is a recorded `decide`, guarded Python-only by `verify_quarantine` |
| modular mapping function | `kbcir.mapping` (`support` / `MappingFunction` / `CommutingSquare`) + `verify.verify_support_preservation` / `verify_commutativity` | `bcir.target.lower_contract` + **`-bcir-verify` R12**: the lowering-discharge contract attr **and** the objective-support law `f(Supp(J)) ⊆ Supp(J')` (the op's optional `source_support`/`target_support`/`discharges`; reproduces `mapping.py::dropped`, `verify_laws_deep.mlir`). The commuting-square `Λ∘Ψ = Φ` is a runtime path-equivalence (covered by the provenance digest + the generated parity campaign), not a static law |
| enriched-operad memory | `kbcir.operad` (`EnrichedOperad` / `EnrichedOp` / `TwoCell` / `enrich_memory` / `f_label` / `f_index`) + `verify.verify_enriched` | *(higher interpretive layer; no MLIR rail — quarantined off the spine)* R13: content-addressed labels+indexes over `a = Lim(Res(U))` (CSE = liked pair; `Trace` integrity) |
| L1 cost throttle | `kbcir.throttle` (`AmortizationCertificate` / `certify` / `ThrottleReport`) | `bcir.kbcir.amortization` (R13: L0 ⇒ zero inference; gain ≥ cost ≤ budget) |
| L3 regret ledger | `kbcir.regret.RegretLedger` / `measure_regret` / `boundary_report` | `bcir.kbcir.regret_ledger` (R13: books balance, rule resolves) |
| MDL/evidence retune | `kbcir.regret` `data_fit_nats`/`complexity_nats`/`evidence_margin` (BIC: ΔL = Σ regret/best − (k/2)ln N) | `bcir.kbcir.regret_ledger` `data_fit_milli`/`complexity_milli`/`verdict` (R13: retune ⟺ data_fit > complexity) |
| policy provenance R13 | `verify.verify_provenance` | `bcir.verify.policy_provenance` + `-bcir-verify` R13 (promotion coverage, table correspondence) |
| StreamPack | `gem.streampack.StreamPack` | `bcir.gem.stream_pack` |
| lane segment | `gem.streampack.LaneSegment` (incl. `dispatch` "core"/"pim") | `bcir.gem.lane_segment` (incl. `dispatch` StrAttr) |
| CIM/PIM dispatch (R14) | `gem.cim` (`annotate_cim` / `cim_decision`: a large reduction offloads to PIM when bytes-not-moved beat the surcharge+overhead) | `-bcir-lower-to-llvm` **R14**: `dispatch = "pim"` legal only on a `reduce.*` op; `mlir/test/passes/gem_passes_neg.mlir` (positive + negative) — dual-rail parity |
| DVFS clock (R15) | `gem.dvfs` (`clock_for` Q8; downclock memory-bound, overclock compute-bound, hold under a Theta cap) | `bcir.gem.lane_segment.clock_q8` (append-only) + `-bcir-lower-to-llvm` **R15**: clock in [64, 512]; a `pim` (memory-bound) segment must not overclock; `gem_passes_neg.mlir` (positive + 2 negatives) |
| allocator placement (R16) | `kbcir.allocator` (`place` / `Placement`: hot→SRAM only when it fits, gains-only) | `bcir.resource.placement` (append-only `BCIR_MemTier`) + `-bcir-lower-to-llvm` **R16**: an L1 placement must be ≤ 64 KiB, L2 ≤ 4 MiB (static `product(shape)*4`); `gem_passes_neg.mlir` (positive + negative) |
| verifier R1–R13 | `verify.{verify,verify_plan,verify_pack,verify_lowering,verify_provenance}` | `bcir.verify.*` ops + the `-bcir-verify` pass (R1–R13) |
| verifier **R14–R16** (smart lowering) | `verify.{verify_cim (R14), verify_dvfs (R15), verify_allocator (R16), verify_smart_lowering}` — dual-rail with the law, negative-tested per law | **first-class `-bcir-verify` R14/R15/R16** (`BCIRVerifyPass.cpp`; `verify_laws_deep.mlir` positive + negative) **and** the `-bcir-lower-to-llvm` checkpoint (`gem_passes_neg.mlir`); `dispatch="pim"` only on `reduce.*`; clock_q8 ∈ [64,512] + no pim overclock; L1 ≤ 64 KiB / L2 ≤ 4 MiB |
| verifier **R17** (accuracy contract) | `verify.verify_accuracy` (a claim's static Q8-ULP error bound vs its declared tolerance; `reduce.*` count ULP naive / 1 compensated) | **first-class `-bcir-verify` R17** (`BCIRVerifyPass.cpp`, consuming `#bcir.precision<…, exact, tol>`; `verify_accuracy.mlir` positive + negative) — the law that forces the compensated realization at a tight tolerance |
| compensated reduction | `kbcir.precision.compensated_reduce_q8` (residual-carry MAC, bit-identical to int64-exact) | **C** `lower.c_kernel.emit_compensated_reduce_c` (`precision="compensated"`); self-check compiles+runs under C11+C23 (`test_precision_lowering.py`) |
| bundle (joint) optimization | `kbcir.bundle.optimize_bundled` (jointly reorder input-sharing claims; bounded, dependency-preserving; `BundleCertificate` per gain) | **`-bcir-bundle`** (`BCIRBundlePass.cpp`, C++): detects input-sharing bundles and reorders the cost columns, re-pricing each legal intra-bundle order via the shared `PlanAnalysis` min-plus; annotates `kbcir.bundle_gain` / `bundle_order` (`bundle.mlir`, `bundle_reorder.mlir`) — the real matmul gain the pairwise shortest path misses |
| proof-carrying records | `kbcir.proof` (`explain`/`replay`/`reduce`; `DecisionRecord` = R13 digest + per-claim rationale + rewrite certificates) | CLI `bcir.run --explain/--replay/--reduce`; replays bit-for-bit from the same inputs or diffs (`test_proof.py`) |
| deterministic executor | `gem.execute` (topological phase order, ascending claim id within a phase, per-phase telemetry) | **C** `runtime/c/bcir_exec.c` (`bcir_sp_execute`): freestanding, Python↔C dispatch-order + telemetry parity (`test_c_executor.py`, `check_runtime.sh`) + libFuzzer (`fuzz_exec.c`) |
| verifier differential (illegal modules) | `kbcir.differential.{gen_illegal_module, check_verifier, gen_illegal_plan, check_plan_verifier, _artifact_law_misses, run_verifier_campaign}` — the original differential fault-injects the scoped R1–R18 oracle entry points (R1/R18 construction guards included). R19–R23 use their dedicated optional-metadata/GEM seam tests; ASN.1 parity and negative fixtures cover R24; the same pair covers R25 (`test_asn1_ecn_law_parity.py` + `verify_ecn.mlir`), which additionally pins every rule to the fixture that trips it, so a law deleted from the pass fails the parity gate rather than quietly reducing coverage. | **`-bcir-verify` implements the current R1–R25 set** (`BCIRVerifyPass.cpp`), with negative `-verify-diagnostics` fixtures under `mlir/test/passes/verify*.mlir`; generated `STATUS.md` inventories tags but does not claim execution |
| overlap law net (the C++ port's net) | `kbcir.differential.check_overlap` + `gem.overlap.price_scheduled` (R9: makespan + gain == serial == score, 0 ≤ makespan ≤ serial) | `bcir.kbcir.scheduled_price` VerifyPass R9 — the invariant the deterministic-optimizer-core C++ port must reproduce |
| verifier **R24** (ASN.1 encoding-rule legality) | `asn1.schema` (OPTIONAL/DEFAULT, tag application) + `asn1.der` (clause 10+11) — enforced when a VALUE is encoded | **first-class `-bcir-verify` R24** over the `bcir.asn1.*` schema ops — enforced when the TYPE is written, before any value exists (`verify_asn1.mlir`: 1 positive + 13 negatives); enum/OID/diagnostic parity pinned by `test_asn1_law_parity.py` |
| ASN.1 / X.690 codec | `asn1.{tags,length,tlv,values,der,codec}` (clause 8 contents, clauses 10+11 restrictions, BER→DER rewrite) | **C** `runtime/c/bcir_asn1.{h,c}` (freestanding, non-recursive, explicit-stack walk); dual-rail node-tree + BER-verdict + DER-verdict differential (`test_c_asn1.py`), 12 000 mutants clean |
| ASN.1 subtype constraints | `asn1.constraints` — X.680 cl. 49–51 with the effective value/size constraint of X.696 §8.2.7/§8.2.8; extensible constraints report NO bounds (§8.2.2 g) | **R24** gained three diagnostics — empty value constraint, empty SIZE, negative SIZE lower bound — with `constraint_low/high` and `size_low/high` on `bcir.asn1.type`; both rails' notion of "effective" pinned by `test_asn1_law_parity.py` |
| ASN.1 open type (X.681) | `asn1.schema.OpenType` + `frontends.asn1` parsing of X.681 §9/§11/§12 and `ANY DEFINED BY`; the contained value is carried as its complete encoding, untouched | *(interop rail)* RFC 5280 `SubjectPublicKeyInfo` decodes and re-encodes **152/152** host trust-store certificates byte-for-byte — the gate phase A could not pass (`test_asn1_frontend.py`) |
| ASN.1 X.696 OER | `asn1.oer` — BASIC-OER + CANONICAL-OER over the SAME `asn1.schema` model the DER rail uses (clauses 8–32); COER out, BASIC-OER in | *(interop rail)* validated **byte-for-byte against X.696 Annex A's own 95-octet worked example**, which exercises §18.2 canonical SET order, §16.2.3 the presence bitmap and §17.2 the quantity field; corpus projection round-trips and measures 76.4 % of DER (`test_asn1_oer.py`) |
| ASN.1 DER -> native fast path | *(reconstruction is C-side; the Python rail is `asn1.streampack.value_to_pack` + `abi.encode`)* | **C** `runtime/c/bcir_asn1_streampack.{h,c}` -- freestanding, allocation-free, two bounded passes; byte-identical to `abi.encode` on all 12 corpus programs and all three native versions (`test_c_asn1_streampack.py`); refuses BER-only spellings and re-validates every blessed artifact through `bcir_sp_verify_semantic`; fuzzed as the eighth trust boundary |
| ASN.1 X.680 front-end | `frontends.asn1.{lexer,ast,parser,printer,lower}` — module text -> `asn1.schema` model; `bcir-asn1c` CLI | *(compiler rail)* round-trip law `parse(print(parse(t))) == parse(t)`; the `BCIR-StreamPack` module is COMPILED from `bcir/asn1/BCIR-StreamPack.asn1` and byte-identical to the hand-built model on all 12 corpus programs; RFC 5280 `AuthorityKeyIdentifier` re-encodes 37/37 real trust-store extensions (`test_asn1_frontend.py`) |
| ASN.1 SET / SET OF / CHOICE | `asn1.schema.{Set,SetOf,Choice}` — X.690 11.6 canonical SET OF order, tag-directed SET decode, X.680 29.1/29.3/31.2.7 CHOICE laws | ODS already declared `set`/`set_of`/`choice` kinds on `bcir.asn1.type`; the oracle now expresses them, closing the gap |
| ASN.1 StreamPack projection | `asn1.streampack` (the `BCIR-StreamPack` module + DER projection; additive, native octets frozen) | *(interop rail)* laws A1 faithful / A2 canonical / A3 additive / A4 normalizing in `test_asn1_streampack.py`; `docs/BCIR_ASN1_X690_ABI.md` |
| trust-boundary fuzz | `kbcir.fuzz` (`run_fuzz`, seeded by `gen_module`): StreamPack codec, ROP/MAP/ETL front-ends, calibration JSON, the MLIR emitter — valid round-trip + graceful malformed rejection | *(host fuzz; C/C++ libFuzzer + ASan/UBSan is the toolchain-rail remainder)* |
| real-silicon energy/thermal | `bcir.silicon.{rapl_available, read_rapl_uj, RaplSampler, read_thermal_millideg, thermal_pressure, silicon_dna}` (RAPL package energy + on-die temp; honest `None` in a sandbox) | *(host capability; the physical drivers of Θ.thermal/power that flip vec16→vec8)* |
| measured replan win (CT4) | `kbcir.calibloop.{measured_replan, MeasuredReplanCertificate}` + `bcir.run --silicon` (real PMU+RAPL+thermal → train+freeze `LinearCalibrator` → replan → certify; provenance-tagged real-vs-synthetic) | *(host measurement; closes the calibration loop's physical half on a bare-metal rig)* |
| GEM pipeline (classify→select→batch→schedule→lower) | `kbcir.realize.optimize` / `gem.{hydrate,schedule,execute}` (the oracle stages) | `-bcir-classify-lanes / -bcir-select-realization / -bcir-batch / -bcir-schedule / -bcir-lower-to-llvm` (`mlir/lib/BCIRPasses.cpp`); `-bcir-select-realization` recomputes the min-plus `cost·weights` and reproduces 7808/9472 (`mlir/test/passes/gem_passes{,_neg}.mlir`) |
| Python→MLIR emitter (the bridge) | `lower.mlir.to_mlir` / `plan_view` (oracle plan → GEM-pipeline IR; context-resolved candidate costs) | the emitted IR is exactly what `-bcir-classify-lanes … -bcir-lower-to-llvm` consume; `--emit-corpus` freezes `mlir/test/passes/gem_corpus.mlir` |
| generated differential parity | `kbcir.differential` (`gen_module` + `law_select` (independent per-claim argmin) + `check_module` + `shrink` + `run_campaign`); `bcir/tests/test_differential.py` | `bcir-opt -bcir-select-realization` on `gem_corpus.mlir` (the C++ argmin recomputes the oracle's per-claim score for the widened corpus) |
| widened corpus (real workloads) | `examples.{matmul_tiled, scan, multi_histogram}` (`examples.CORPUS`) — real tiled matmul / multi-stage scan / map-reduce histogram | `mlir/test/passes/gem_corpus.mlir` (the GEM select pipeline on the AVX-512 profile, FileCheck-pinned scores) |
| six-target capability matrix | `kbcir.differential.emit_target_matrix` (`MATRIX` programs × the six TARGETS, each with its `bcir.target.capability` seeds); `bcir/tests/test_target_matrix.py` | `bcir-opt -bcir-plan -bcir-overlap -bcir-rcsp-plan` on `mlir/test/passes/target_matrix.mlir` recomputes the oracle's per-target plan score / makespan+gain / constrained optimum **from the capability alone** (avx512/sve/rvv 7808, avx2 9472, neon 12800, ptx 6976; GPU gather 266240 vs 528384; `--emit-matrix`, drift-gated) |
| memory tier id | `kbcir.cost.MemTier` | `BCIR_MemTier` (`BCIRAttrs.td`) |
| partial LLVM AOT/JIT subset | `lower.llvm` (clang/lli; exactly one 2-read/1-write add/sub/mul claim, otherwise reject) | `bcir.target.lower_contract`; not arbitrary-graph lowering |
| C kernel backend (C23) | `lower.c_kernel` (`emit_kernel_c` / `emit_header_c` / `emit_selfcheck_c` / `compile_and_run_c`) + `verify.verify_c_lowering` | `bcir.target.lower_contract` (R12: selected width → loop — a *floor* at the full hardware lane (idiomatic loop), a *cap* when sub-maximal (a thermal/power throttle); `restrict`, bounds tail, precision; portable C23 for any resident toolchain) |
| library facade (embeddable) | `bcir.api` (`build_artifact` / `compile_kernel` / `KernelArtifact`) | *(host library surface)* plan → C source + ABI header + metadata + R12 attestation + provenance digest; AOT or driver-embedded |
| bare-metal calibration | `runtime/c/bcir_microbench.c` + `kbcir.microbench.calibrate_native` | feeds the frozen `CalibratedProfile` schema with real cache latency (closes the loop's conservative half) |
| measured-evidence rail | `bcir.bench` (`compare` / `measure` / `Comparison`) | *(host measurement)* times BCIR's selected realization vs the scalar baseline; reports the measured speedup (honest, not pinned) |
| gather avoidance (measured) | `bcir.bench.compare_gather` + `lower.c_kernel.emit_gather_kernel_c` (`--bench-gather`) | *(host measurement)* direct vs the avoided gather form; the `gather_penalty` realized (~6× on silicon, random indices) — the cost model vindicated |
| gather/blocked reduction (end-to-end) | `examples.gather_reduce` (`reduce.gather`) + `realize.candidates_for` (blocked vs gather) + `lower.c_kernel.emit_reduce_c` + `bench.compare_reduce` (`--bench-reduce`) | `mlir/examples/gather_reduce_ct1.mlir` (the law: blocked vs gather paths, min-plus selects blocked); *(host)* blocked == gather sum (correct), ~16× faster |
| R7 reduction-write extent | `verify` R7 (`op.startswith("reduce.")` ⇒ write extent 1, not count) | `-bcir-verify` R7 (`getOp().starts_with("reduce.")`); `verify_laws.mlir` reduction pair (clean reduction + non-reduction overrun) — dual-rail parity |
| strided gather-avoidance | `examples.saxpy_strided` + `lower.c_kernel.emit_strided_c` + `bench.compare_strided` (`--bench-strided`) | *(host)* the cost model picks direct strided over gather; ~1.4× faster (the gather-instruction overhead) — a non-reduction gather avoidance |
| latest-LLVM rail | — | `.github/workflows/ci.yml` `mlir-rail-validate` matrix tracks the latest release (LLVM 23, gating; 22 kept in the matrix for one cycle) from apt.llvm.org; scripts auto-resolve the toolchain version-agnostically (highest `/usr/lib/llvm-*/bin/{FileCheck,mlir-opt,mlir-tblgen}`) |
| real-signal probes | `bcir.silicon` (`cache_topology`/`tier_capacities`, `cpufreq_info`, `CounterSampler`, `read_hw_counters`/`perf_counters_available`, `summary`) | *(host capability)* read-only `/sys` cache + cpufreq + `getrusage` + `perf_event_open` PMU; honest about what the host exposes (no PMU / no cpufreq actuation ⇒ reported, not faked — see `docs/kernel/HARDWARE_VALIDATION.md`) |
| DVFS actuation (attempt + gate) | `gem.dvfs.actuate` (writes `scaling_setspeed`, reads `scaling_cur_freq` back) | *(host)* sets the real clock when a `userspace` governor + privilege exist; otherwise a dry-run `ActuationResult` naming the missing capability — never faked |
| RL allocator on real tiers | `kbcir.allocator.place(tier_capacity=silicon_tier_capacity())` | *(host)* fast-tier capacities = the machine's real L1/L2/L3; measured L1≈1 ns vs DRAM≈166 ns justifies hot→SRAM (`test_silicon`) |
| synchronized zero-copy telemetry ring | `telemetry.TelemetryRing` + `silicon.sample_into_ring` (real OS counters) | *(host)* fixed-buffer publication/read transitions are serialized and benchmarked against equivalently validated JSON; CI asserts integrity and exercises both paths, but sets no scheduler-sensitive speedup floor |
| phase-aware DVFS on real cpufreq | `gem.dvfs.quantize_to_silicon` (Q8 clock → real frequency; actuation gated on `userspace` governor + privilege) | *(host)* anchors to the real nominal; actuation honestly reported unavailable in sandboxes |
| C-side zero-copy ring (real bridge) | `lower.memory_model.emit_ring_header_c` (C producer: atomic release-store to fixed offsets, `hazard_to_ordering`) + `telemetry.parse_shared_ring` | *(host)* C writes / Python reads the **same mmap** with no syscall/serialization — measured round-trip in `test_persistent_oracles` |
| a-priori telemetry gating | `kbcir.accel.FrozenRanker.confidence` (top-2 z-margin) + `kbcir.sensing.sense_by_ranker` | *(planning; off-rail — leans on a learned ranker margin)* instruments only columns the ranker cannot resolve (narrow margin); complements the a-posteriori `RegretSensor.sense` |
| a-posteriori telemetry sensing | `kbcir.sensing.RegretSensor.sense` (per-path `cv_milli` over observed costs; deterministic CV-threshold + budget gate) | **`-bcir-sense`** (`BCIRSensePass.cpp`): per-segment `cv_milli = 1000·stdev/mean` over the `bcir.trace.data_dna` cycles (population variance, floor-isqrt), ranked `(-cv, segment)`, gated to `high`/`low`/`off`; matches the oracle exactly (`sense.mlir`) |
| predictive pool allocation | `kbcir.allocator.pool_plan` / `live_intervals` (liveness interval-partitioning) | *(planning)* disjoint-lifetime tensors share an arena ⇒ `peak_bytes ≤ naive_bytes`; gains-only modeled footprint win |
| timeline DVFS (power rail) | `gem.schedule.schedule_power_rail` (per-Slot clock over the placed timeline) | *(modeled)* downclocks memory-bound slots to data-arrival bounds; energy figure modeled (no RAPL — `docs/kernel/HARDWARE_VALIDATION.md`) |
| CIM/PIM spatial partition | `lower.c_kernel.optimize_spatial` / `is_pim_target` (a `pim` ISA-feature target binds reductions to memory) | reuses the **R14** law (`dispatch="pim"` only on `reduce.*`); modeled transport-saved, next-phase needs a real PIM target |
| multi-claim fusion | `examples.fused_chain` + `gem.overlap.price_scheduled` / `optimize_scheduled` | `bcir.kbcir.scheduled_price` + `-bcir-batch` (the (max,+) overlap: makespan < serial Σ for ≥2 claims) |
| operand fusion (two kinds) | `realize._context_factor` (shared-input cache reuse, path-based) + `realize.fused_candidates` (producer→consumer **deforestation**, dependency-based, shared by optimize/RCSP/overlap/accel/softdp) | *(oracle cost model)* shared-input + producer→consumer write→read fusion both discount memory; ~12% lower plan score on a producer→consumer chain, no width churn |
| CSE / duplicate elimination | `realize.fused_candidates` value numbering (same op + same operand-versions ⇒ a copy, no recompute) — the egraph's liked-pair (`egraph.module_exprs`/`shared_blocks`) finally priced into the plan | *(oracle cost model)* a duplicate claim is priced as a copy (~15% cheaper); a write between duplicates bumps the version and withdraws the credit (sound); no-op on single-claim programs (7808 intact) |
| trained calibrator + broker | `kbcir.calibrate` (`train_calibrator` → `FrozenCalibrator`) + `telemetry.Broker` | the §13 freeze: an online model trained offline, frozen to deterministic Q8; the live pub/sub broker feeds the loop (closes the calibration loop's learned half) |
| budget feasibility (RCSP) | `kbcir.rcsp` (`feasible` / `plan_resources` / `optimize_constrained`) + `api.build_artifact(budget=…)` | `bcir.kbcir.budget` + `bcir.kbcir.select` `budget` (R9): `R(π,Θ) ⪯ B`; BCIR emits the *feasible* plan (vec8) where the naive max-width (vec16) violates the cap — a correctness property |
| scope-aware R9 (cost re-derivation + plan feasibility) | `verify.verify_plan(module, result, h, theta=…, policy=…, budget=…)`: every step's realized cost re-derives through the planner's own DAG edge weight (`realize.step_cost` — one predicate for planner and verifier, over the planner's own offer `fused_candidates`), and the accumulated `R(π,Θ) ⪯ B` via `rcsp.plan_resources`; without the scope R9 checks candidate membership and coverage only | `-bcir-verify` R9: the selected path is drawn from the candidate set, realizes its claim, satisfies every cap of the budget it names, and the scheduled price is consistent with its serial bound; `-bcir-select-realization` recomputes the min-plus `cost·weights` it scores |
| event phases (EV1–EV3) | `kbcir.events.check_event_phases`, run by the canonical `verify(module)` (EV1 asynchronous entry, EV2 explicit arming, EV3 the interrupt-context ordering seam) | `-bcir-verify` R3/EV1–EV3 over `bcir.phase` `event` (`mlir/test/passes/verify_event_phases.mlir`) |
| concurrency/affinity (CT2) | `gem.schedule_concurrent` | `bcir.gem.lane_segment` `affinity`/`unroll` |
| ROP/MAP front-ends (CT3) | `frontends.{rop,map}` | `bcir.parse.*` / `bcir.binary.*` |
| data-DNA telemetry (CT4) | `telemetry.DataDNA` + `kbcir.calibrate` | `bcir.trace.data_dna` |
| calibration loop (closed) | `kbcir.calibloop` (`close_loop` / `measure_and_close` / `rescore_plan` / `CalibrationCertificate`) + `verify.verify_calibration` | `bcir.kbcir.calibration` (R13: measure → freeze → replan; `cal_gen ≥ 1` ∧ `win ≥ 0`; the measured cost of not recalibrating) |
| JIT (CT5) | `lower.jit` (lli) | per-target `bcir.target.lower_contract` |
| StreamPack ABI | `abi.streampack_abi` (frozen v1, append-only v2/v3 codec) | `runtime/c/bcir_streampack.h` (spec) + `bcir_runtime.c` (decode) + **`bcir_encode.c`** (`bcir_sp_reencode`, byte-identical re-encode; `test_c_encoder.py`) |
| WASM (Phase 7) | `lower.wasm` (clang→wasm + node) | per-target `bcir.target.lower_contract` |
| stackify (Phase 7) | `lower.stackify` (→ wasm/jvm/cil) | foundation for `bcir.target.lower_contract` encoders |
| C runtime (Phase 8) | `runtime/c/bcir_runtime.{h,c}` decodes `abi.streampack_abi` | `runtime/c/bcir_streampack.h` (C23: `restrict`/`[[nodiscard]]`/frozen-ABI `static_assert`; fuzzed under libFuzzer+ASan/UBSan via `runtime/c/fuzz_streampack.c`) |
| named pass pipelines | `bcir.run` CLI stages | **MLIR** `registerBCIRPipelines`: `bcir-audit` / `bcir-optimize` / `bcir-hydrate` / `bcir-lower-llvm` / `bcir-aot`, every one verifier-checkpointed on entry and `bcir-optimize` again on the plan it emits (`pipeline_checkpoints.mlir`); `bcir-aot` is **partial AOT preparation** and may leave mixed BCIR/GEM/LLVM IR |
| module scope | `verify(module)`, `optimize(module, …)`: one module at a time by construction | `-bcir-verify`, `-bcir-select-realization`, `-bcir-rcsp` and the GEM batch/schedule/lower passes are anchored at `bcir.module` through one predicate (`BCIRPassSupport.h` `forEachScope` / `walkScope`); a multi-module file verifies and prices each module in its own symbol table, and operations outside any module are the one outer scope (`verify_module_scope.mlir`, `select_module_scope.mlir`) |
| structural laws (S0-6) | `verify(module)` R1/R4/R7 well-formedness, `derived_claim_domain` (MAP/ROP), `verify_address_width`, and construction-time rules on `TargetProfile`, the ETL descriptors, `ProvenanceManifest` and `check_conv` | the op verifiers (`bcir.resource`/`claim`/`target.capability`/`gem.conv`; the `binary.*`/`event.*`/`fsm.*`/`parse.*` descriptors) and `-bcir-verify` R3 (isolated domains), R4 (phase identity), R12 (address width under a declared target), R13 (artifact record, absent objects, certified constants) — ONE corpus, `bcir/verify/structural_corpus.py`: the quick tier runs it on the oracle, `structural_corpus.mlir` (generated, drift-gated) runs it under `check_passes.sh`; every mismatch a finding |
| phase order | `model.topological_phase_ids` (dependency-first, roots in declaration order) — the planner, R9, the GEM scheduler, `overlap._makespan` | `BCIRPassSupport.h` `canonicalPhaseOrder` ranks the cost-model columns, `-bcir-schedule`, `-bcir-overlap` and `-bcir-schedule-eft` (`schedule_phase_order.mlir`: ids declared out of dependency order) |
| live state Θ (cost coupling) | `kbcir.cost.Theta` + `weights()` fold | `bcir.kbcir.theta` op (thermal/power/...) — `-bcir-plan`/`-bcir-overlap` apply the multiplicative thermal coupling under hot Θ (matmul hot 1159168; `theta_hot.mlir`) |
| async tokens (Phase 8) | `gem.async_tokens` (fork/await plan) | `bcir.async.fork` / `bcir.async.await` (`!bcir.token`) |
| memory model (Phase 8) | `lower.memory_model` (hazard→ordering) | `BCIR_MemOrdering` + barrier `ordering` → `llvm.fence` |
| per-target codegen (Phase 9) | `codegen.*` (llc → ARM/RISC-V/PTX/eBPF/C) | `bcir.target.lower_contract` (one per target) |

## Python ↔ C artifact and runtime parity

These contracts terminate at byte layouts or observable behavior rather than MLIR
operations. Their governing prose is the LangRef or the corresponding ABI document.

| Surface | Python/reference rail | C/production rail and gate |
|---|---|---|
| BCAB v1 artifact bundle | `bcir.abi.artifact_bundle` deterministic codec, integrity/identity checks, compatibility envelope and selector; `bcir-bundle` validates before listing, extraction, hex or delegated disassembly | `bcir_artifact_bundle.c` allocation-free borrowed reader/selector plus the C++ borrowed-view wrapper; identical malformed-input refusal and priority/specificity/feature/ID selection; MLIR records the validated directory/selection metadata, not payload bytes |
| BCAB ASN.1 projection | `asn1.artifact_bundle` + compiled `BCIR-ArtifactBundle.asn1`; DER-out/BER-in and COER-out/BASIC-OER-in, with native byte-identity transcodes and atomic CLI conversion | R24 carries the additive `native = "artifact_bundle"` projection; generic freestanding C X.690 validates DER and the existing C BCAB reader validates reconstructed native bytes. A schema-specific C transcoder is not claimed |
| StreamPack v1–v3 | `bcir.abi.streampack_abi` encode/decode and semantic validation | `bcir_runtime.c` + `bcir_encode.c`; byte-identical re-encode, exact-version, CRC, bounds, trailing-byte, dispatch/channel, and malformed-corpus gates |
| BTLM telemetry frame v1 | `bcir.telemetry_frame` strict codec, resync, continuity/wrap evidence | `bcir_telemetry_frame.c`; exact layout/CRC/re-encode and corruption rejection in `test_telemetry_frame.py` and the C gate |
| Signal and metric meaning | `bcir.signal_registry`, `telemetry_metrics`, deterministic exporters | C carries the frame codec today; a generated fixed-width signal table and live transport remain open and therefore are not claimed as parity |
| BCIRQ8 v1 | `frontends.models.weights_io` deterministic write/read, canonical tensor order, CRC/hash/bounds checks | `bcir_q8_model.c`; identical header/directory interpretation, strict rejection, allocator injection, and failure-state tests in `test_model_weights_io.py` / memory-discipline gate |
| Q8/Q4 conversion | `kbcir.quantize` and `lowbit.PackedQ4Tensor` define symmetric codes, power-of-two groups, nibble order, and subnormal behavior | `bcir_ai_kernels.c`; exponent/code bytes are identical across empty, odd, boundary, random, and extreme finite fixtures |
| Llama Q8 greedy decode | `decode.head_logits`, full/KV decode, Q8 drift/NLL and artifact readback; untied head and checkpoint `rms_norm_eps` are shared | `bcir_llama.c` / `bcir_llama_cli.c`; toy tied/untied parity plus pinned real-model generated-ID and final-logit parity |
| Exact optimization-memory retrieval | `optimization_memory.query_optimization_memory` applies hard facts then squared-Q15 distance and SHA tie-break | `bcir_ai_q15_topk` via `NativeOptimizationIndex`; exact match/distance ordering after the same Python hard-fact filter |
| Bounded model measurement | `run_bounded_model_microbench` remains the readable diagnostic oracle | `bcir_ai_microbench.c` / `run_bounded_model_microbench_native`; same bounded operation/format envelope with strict timestamp-free output; timings are evidence, not semantic equality |
| Hosted allocation behavior | Python harness enumerates failure points and validates public outcomes | `bcir_host_alloc.h` injection; failed growth preserves the original, outputs remain valid, and repeated cleanup is safe |
| RuntimeChannel direct ABI | Transport-neutral device/event/DMA contracts provide reference inputs | append-only C v1 hook table and bounded loopback; future Linux/native adapters must reproduce direct behavior before they can claim parity |
| K_BCIR plan step contract | `kbcir.realize.optimize` steps carry the chosen candidate (lane, width) and a realized cost that `verify_plan` re-derives under the scope | `bcir_plan.c` emits `width = 1` (the scalar realization) and `cost = base × ⌈count/width⌉` from the header-inline `bcir_plan_base_cost`; `bcir_verify.c` R9 refuses a width that is not a power of two and a cost that does not re-derive from it; `bcir_hydrate.c` hydrates a NULL plan at width 1 (`test_c_planner_width_contract_and_r9_rederives_costs`) |

BCIRQ8’s normative byte contract is [`BCIR_LANGREF.md`](BCIR_LANGREF.md#16-bcirq8-v1-decoder-artifact-contract)
§16. BCAB is owned by
[`BCIR_ARTIFACT_BUNDLE_ABI.md`](kernel/BCIR_ARTIFACT_BUNDLE_ABI.md); StreamPack and telemetry bytes are owned by
[`BCIR_STREAMPACK_ABI.md`](kernel/BCIR_STREAMPACK_ABI.md) and
[`TELEMETRY_FRAME_ABI.md`](kernel/TELEMETRY_FRAME_ABI.md). A clean skip for a missing
compiler, architecture, or model cache is not parity evidence for that execution path.

## Python ↔ C frontend twin (`runtime/c/`)

The C frontend (`runtime/c/bcir_cfront.c` + the `bcir-cc` driver) is the **production twin** of the
Python prototype in `bcir/frontends/cfront/`: once a stage is validated in the oracle, the real
implementation lives in C and a dual-rail gate keeps the two byte/structurally identical. The gate is
`bcir/tests/test_c_cfront.py` (structural-summary parity + Clang behaviour-equivalence) **and**
`tools/c/check_runtime.sh` (the CI-gated fixture list + the per-feature blocks below); both run the
*same* source through both rails and diff the result.

`_parity_check_fixture` diffs the two C-frontend rails on **four** axes, each the floor the previous
one cannot see: (1) the **claim-summary** parity (the RID-independent structural summary equals the
oracle's); (2) **behaviour-equivalence** under Clang of *both* the twin's and the oracle's own emitted C
on seeded-random inputs; (3) the **bounds-guard count** — both rails promote the same accesses to
`masked`, so they emit the same number of `BCIR_CHK` guards; and (4) the **storage-extent** parity
(`_array_extents`): the sorted multiset of dim-product array-declaration sizes must match. (4) closed a
real blind spot — an *over-sized* backing array keeps the same per-element stores, the same observable
behaviour, **and** the same `BCIR_CHK` guard count, so it slipped through (1)–(3); a multi-dim compound
literal once sized `_cl[10]` vs the correct `_cl[6]` and passed every prior check. Normalizing to the
dim **product** makes a flat `[4]` and a nested `[2][2]` compare equal, so the check is robust to
decl-form differences between the rails (`test_storage_extent_parity_catches_oversizing` pins the gap
closed).

| Oracle (`bcir/frontends/cfront/`) | C twin (`runtime/c/`) | Gate |
|---|---|---|
| lowering: integers, struct/union/bitfields, casts, control flow, atomics, funcptr dispatch, `<math.h>`, hex-float, array compound literals (1-D + multi-dim, scalar + aggregate-element, inferred/designated outer dims) | `bcir_cfront.c` (claim-graph parity + faithful emit) | `check_runtime.sh` fixture loop; `test_c_cfront.py` |
| `abi.py` `TargetABI`/`TARGETS` (`--target`) | `bcir_cfront.c` target table + `cc_abi`/`p_type` (long/ptr/`size_t` per data model) | `#abi` block; `test_abi_target_matrix_dual_rail` |
| `pipeline.compile_with_fallback` (route-to-LLVM) | `bcir-cc --fallback` (clean 0 / dirty 1 / fallback 2) | `#fallback` block; `test_fallback_contract_dual_rail` |
| `diagnostics.py` renderer (caret layout) | `bcir_diag.c` `bcir_diag_render` | `#diag` block; `test_diagnostic_renderer_dual_rail` |
| `DiagnosticReport.to_json` | `bcir_diag_to_json` (matches `json.dumps(indent=2)`) | `test_diagnostic_json_dual_rail` |
| `FixIt` hints (verb + `repr`) | `bcir_diag.c` `py_repr` + fix-it emit | `test_diagnostic_fixits_dual_rail` |
| line-map `origin` (`In file included from …` / `includedFrom`) | `bcir_diag.c` origin path | `test_diagnostic_include_stack_origin_dual_rail` |
| parser recovery (`DiagnosticReport.render` over N) | `bcir_diag_report_render` | `test_diagnostic_error_recovery_report_dual_rail` |
| scalar file-scope global read + write | `bcir_cfront.c` (`c.copy` to the global rid, emitted by name) | `cfront_global_rw.c`; `test_scalar_globals_read_write_dual_rail` |
| `pipeline.own_footprint` + `commute` | `bcir_cfront_effects` (`bcir-cc --emit-effects`) | `#effects` block; `test_effect_commutation_analysis_dual_rail` |

**Delegated (the two-truth line — in the oracle only, by design):** the IPO **cost** model
(`compose.summarize`/`plan_composite`'s `worst`/`expected`/`reused`) and all K_BCIR cost/plan/RCSP
machinery (`Theta`/policy/`plan_composite`) — the C twin has no cost analogue; `long double` layout
beyond the size/align axes; IEEE float *math* and the calling convention (the resident backend's
job, as for the MLIR rail). These are the same delegations the MLIR rail makes, not gaps in the port.

## Worked-example parity

`vector_add` (n=1024) on the AVX-512 profile under cool Θ selects the `vec16`
realization with K_BCIR **score = 7808** in both:

- Oracle: `python -m bcir.run vector_add --target x86_avx512` (and
  `bcir/tests/test_kbcir.py::test_vector_add_cool_selects_vec16_score_7808`).
- Law: `mlir/examples/full_vec_add_ct1.mlir` `bcir.kbcir.select ... score = 7808`.

A hot Θ replans both to `vec8` (AVX-512 downclock). Per-target π* differs by
lane width: x86_avx512→16, x86_avx2→8, arm64_neon→4, nvidia_ptx→32.

The **constrained rail** pins a second worked example: under a 700 thermal/power
budget, vec16 (thermal 1088) is infeasible and the budgeted optimum is `vec8`
with K_BCIR **score = 9472** in both:

- Oracle: `optimize_constrained(..., Budget.of(thermal=700))`
  (`bcir/tests/test_rcsp.py`); CLI `python -m bcir.run vector_add --budget thermal=700`.
- Law: `mlir/examples/full_vec_add_ct1.mlir` `bcir.kbcir.select ... budget = @thermal_cap, score = 9472`.

The **scheduled price** pins the degenerate overlap example: a single-claim plan
prices at `makespan == serial == 7808`, `overlap_gain = 0`
(`bcir.kbcir.scheduled_price @overlap_price` ↔ `gem.overlap.price_scheduled`).

## Generated, adversarial parity (the proof, not the hope)

Curated pins prove parity on *one* program. `bcir.kbcir.differential` proves it on
*generated* ones: a structured/adversarial module generator (`gen_module` — every
stride class, lane, memory tier, fusion/deforestation/CSE shape, reduction, gather,
HAM table, across one or two phases) feeds **both rails** and **diffs** the selected
realization, the per-claim + total score, the constrained (RCSP) budget feasibility,
and the deterministic schedule order. The two rails are genuinely distinct
algorithms over the one cost model:

| Rail | Algorithm | Code |
|---|---|---|
| Oracle | min-plus **shortest path** over the coupled candidate DAG | `kbcir.realize.optimize` |
| Law | per-claim min-plus **argmin** over declared candidate costs | `kbcir.differential.law_select` (mirrors `-bcir-select-realization`) |

`law_select` reads only the IR-level `lower.mlir.plan_view` (the same structure
`to_mlir` emits), never the oracle's objects, so an agreement is evidence — not a
restatement. The candidates are emitted at their oracle **context-resolved** costs
(the fusion/thermal coupling `f_i(pi)` baked in), so the per-claim argmin reproduces
the globally-coupled plan. `run_campaign` sweeps thousands of generated modules ×
the six targets × {cool, hot, mem_bound} × {latency, throughput, energy}; a mismatch
is **shrunk** (`shrink`) to a minimal witness. `bcir/tests/test_differential.py`
asserts zero divergence over ≥1500 modules, plus the metamorphic laws (ID-renaming
preserves the score; a tighter budget never raises the capped dimension; emit→read
is lossless).

The **widened corpus** — real tiled matmul (`matmul_tiled`, register-resident
K-accumulation → deforestation), a multi-stage scan (`scan`), and a map/reduce
multi-claim histogram (`multi_histogram`) — carries this parity across all six
targets and is emitted to `mlir/test/passes/gem_corpus.mlir` (regenerate with
`python -m bcir.kbcir.differential --emit-corpus`; a drift gate keeps the committed
artifact equal to a fresh emission). When the MLIR toolchain is present,
`bcir-opt -bcir-select-realization …` recomputes the same per-claim scores on that
file (the ultimate cross-rail check; `tools/wsl/check_passes.sh`), and
`bcir/tests/test_differential.py::test_bcir_opt_recomputes_the_corpus_when_available`
runs it inline.

## How parity is enforced today

`bcir/tests/` pins the exact scores and per-target widths (runnable with
`python -m bcir.tests.run_all`, no third-party deps), **and** the generated
differential campaign above proves the oracle↔law selection agreement on randomized
modules. When the MLIR toolchain is available, the `mlir/examples` + `mlir/test/irdl`
corpus round-trips through `bcir-opt` / stock `mlir-opt` and must carry the same
constants, and `mlir/test/passes/gem_corpus.mlir` recomputes the widened corpus.
Two inventory gates keep that corpus honest without a toolchain (S0-B, quick tier on every
host): `bcir/tests/test_mlir_fixture_inventory.py` reconciles every `mlir/test/passes/*.mlir`
against the runner scripts both ways (a fixture nothing runs, a runner reference to no
fixture), and `tools/irdl/check_inventory.py` reconciles the ODS dialect, the IRDL projection
and `mlir/irdl/MANIFEST.json` — every operation projected under the one naming rule or
declared unprojected with its reason, no stale entry, ghost, orphan or naming collision.
The structural laws themselves are held to ONE corpus (S0-6, `bcir/verify/structural_corpus.py`):
every case is a rail-neutral spec with the law family and the diagnostic each rail must
produce; the quick tier runs the oracle over all of it, `mlir/test/passes/structural_corpus.mlir`
is generated from it (`--check` refuses drift) and executes under `check_passes.sh`, and when
`bcir-opt` is built the test drives both rails and asserts zero findings. A legal case refused,
an illegal case admitted, a refusal for another reason or under another law — each is a finding.

The current **R1–R25** MLIR law set is negative-tested per law with
`-bcir-verify -verify-diagnostics`: `verify_laws.mlir` (R1–R7),
`verify_laws_deep.mlir` (R8–R16), `verify_accuracy.mlir` (R17),
`verify_callgraph.mlir` (R18), `verify_timing_lifetime.mlir` (R19–R21),
`verify_shape_dtype.mlir` (R22–R23), and the generated `structural_corpus.mlir` (the S0-6
structural cases of R1–R4, R6–R8, R12, R13 and the descriptor op verifiers). The Python oracle covers each applicable surface
through its verifier and dedicated timing/lifetime/GEM-seam tests; the C frontend's
documented twin remains scoped to R1–R18. Generated `STATUS.md` is a static fixture
inventory, not an execution claim. The pretty ODS corpus must also stay clean under the
full `-bcir-verify` (`tools/wsl/check_passes.sh`, CI `mlir-rail-validate`).

The **GEM pipeline passes** carry the same dual-rail discipline:
`mlir/test/passes/gem_passes.mlir` FileCheck-pins the recomputed plan (the
min-plus `cost·weights` reproduces the oracle's 7808 cool / 9472 under the cap),
and `gem_passes_neg.mlir` negative-tests the cross-check (a declared selection
that is not the true argmin, or a StreamPack segment that breaks the R12 lowering
contract, is rejected) via `-verify-diagnostics`.

The **bundle joint-reorder**, **proof-carrying explain**, and **compositional func/if
op family** extend the same rail. `-bcir-bundle` reuses the proven cost machinery
(`BCIRCostModel.h`'s `fusedColumns` + `planChosen`): it reorders the cost-model columns
so an input-sharing bundle is contiguous, re-runs the min-plus shortest path for every
legal intra-bundle order, and annotates the re-priced `kbcir.bundle_gain` /
`kbcir.bundle_order` — the law-rail twin of `bundle.optimize_bundled` (an interleaved
bundle recovers the shared-input fusion discount the pairwise plan misses;
`mlir/test/passes/bundle_reorder.mlir`). `-bcir-explain` is the law-rail port of
`proof.explain`: per claim it annotates the candidates the optimizer weighed (their widths
+ scalarized costs), the chosen width, and the coupled edge score, plus any fusion credit;
per module the plan total — reproducing the pinned 7808 on `vector_add`
(`explain.mlir`). `-bcir-replay` is the law-rail port of `proof.replay`: it recomputes a fresh
plan from the IR (the same cost machinery, factored into a shared `freshRecord`) and diffs it
against the declared `kbcir.explain_*` record — the module total and, per claim, the chosen width
+ edge score — annotating `kbcir.replay_reproduced` (and `replay_mismatches` when diverged). A
faithful record reproduces; a tampered edge score is flagged with the exact
`replay (w16/7808) != recorded (w16/9999)` divergence (`replay.mlir`; the total-mismatch wording
is byte-identical to `ReplayResult.mismatches`, pinned by `test_proof.py`). The R13 provenance
digest gate of `proof.replay` is the separate `-bcir-verify` provenance recheck; this pass adds the
decision-record half. The `kbcir.func` / `kbcir.call` / `kbcir.cond` op family gives
`compose.py`'s region tree first-class MLIR form and round-trips through `bcir-opt`
(`compose_ops.mlir`). **`-bcir-compose`** then computes the compositional cost on the law
rail (`compose.plan_composite`): a region's direct `bcir.claim` leaves are priced by the
shared K_BCIR cost model (`fusedColumnsFromClaims` + `planChosen` — a Leaf reproduces the
oracle's 7808), `Seq` sums, `kbcir.cond` is the worst-case max + the probability-weighted
expected, and `kbcir.call` is an **inter-procedural summary** — a `kbcir.func` is planned
**once** over its formals (memoized) and a call whose actuals are cost-compatible (same
`compose._cost_key`: domain / element-count / access) reuses that summary, else the body is
re-priced with the actuals substituted; it annotates `kbcir.compose_worst` / `compose_expected`
/ `compose_reused` per func, reproducing `plan_composite`'s 23432 / 18747 (`compose_cost.mlir`)
and the reuse-vs-re-price 10624 / reused 1 (`compose_summary.mlir`; pinned by `test_compose.py`).
With a `kbcir.budget` present, each Leaf is priced by the **constrained label DP**
(`cm::planConstrained` = `rcsp.optimize_constrained`) so the compositional plan respects
`min M(π,Θ) s.t. R(π,Θ)⪯B`: a thermal cap re-prices wide vec16 to the feasible vec8 (9472) or
marks the func `kbcir.compose_feasible = false` when nothing fits (`compose_budget.mlir`). It
also ports `compose.effect`/`independent`/dynamic: each func is annotated with its read/write
footprint (`kbcir.effect_reads`/`effect_writes`, folded through calls' substitution), each call
with `kbcir.commutes_with_prev` (disjoint footprints commute -- the RAW/WAR/WAW test), and
`kbcir.compose_dynamic` (a dynamic-shape leaf makes the plan a worst-case bound;
`compose_effect.mlir`). A
generated **compose differential** (`test_compose_differential.py`) fuzzes the metamorphic laws
(determinism, worst≥expected, unbounded-budget degeneracy, RCSP monotonicity, summary
consistency) over randomized region trees.

The **GEM scheduling decisions** are recomputed, not just verified: `-bcir-cim` ports
`gem.cim.cim_decision` (core-vs-PIM cost for a reduction → `kbcir.cim_offload`/`cim_core_cost`/
`cim_pim_cost`; offload at count 4096, not 1024 — `cim.mlir`) and `-bcir-dvfs` ports `gem.dvfs`
(per-phase compute:memory intensity → a Q8 clock → `kbcir.dvfs_class`/`dvfs_clock`; a
bandwidth-bound `vector_add` phase downclocks to 192 — `dvfs.mlir`; constants pinned by
`test_cim.py`). R14/R15 still verify the declared dispatch/clock is *legal* (defense in depth);
these derive what it *should* be. `-bcir-schedule-eft` ports `gem.schedule.schedule_eft` — the
duration-aware (HEFT-lite) wave scheduler: LPT priority + earliest-finish placement + locality
+ the bandwidth-knee clamp, annotating `kbcir.sched_domain`/`sched_start`/`sched_finish` per
claim + `sched_makespan`/`sched_knee` (two shared-read compute claims run parallel on domains
0/1, makespan 7808 — `schedule_eft.mlir`; pinned by `test_schedule.py`). `-bcir-async` ports
`gem.async_tokens.async_plan` + `schedule.execute_tokens` — the `!bcir.token` fork/await DAG drives
a single cross-phase dispatch (no phase barriers), so a later-phase independent claim overlaps an
earlier one (software pipelining); annotates `kbcir.async_awaits`/`async_domain`/`async_start`/
`async_finish`/`async_makespan` (a phase-1 independent claim starts at 0, makespan 15616 vs
2·7808 — `async.mlir`; pinned by `test_schedule.py`). `-bcir-power-rail` ports
`gem.schedule.schedule_power_rail` — a per-slot DVFS overlay on the EFT *placed timeline* (the join
of `-bcir-schedule-eft` and `-bcir-dvfs`): each scheduled slot is classified by its base
compute:memory mix and gets a per-slot Q8 clock for its real `[start,finish)` interval (memory-bound
slots downclock to 192, keying off the slot interval rather than `-bcir-dvfs`'s per-phase totals),
annotating `kbcir.rail_class`/`rail_clock` per slot + `rail_energy_saved` per module (two memory-bound
slots on the 7808/5888 timeline both downclock, modeled energy saved 3424000 — `power_rail.mlir`;
pinned by `test_schedule.py`). `-bcir-alloc-pool`
ports `allocator.pool_plan` — liveness-based memory pooling (resources with disjoint live ranges
share an arena, greedy left-edge), annotating `kbcir.pool_id` per resource + `pool_naive_bytes`/
`pool_peak_bytes`/`pool_saved` (A/D and B/E share arenas, C its own → peak 12288 vs naive 20480 —
`alloc_pool.mlir`; pinned by `test_persistent_oracles.py`).

The **R18** call-graph law (`-bcir-verify`) is the law-rail twin of
`compose.plan_composite`'s rejections — every `kbcir.call` must resolve to a `kbcir.func`
(the oracle's `KeyError`) and the call graph must be acyclic (the oracle's `RecursionError`,
pinned oracle-side by `test_compose.py::test_undefined_call_and_recursion_are_rejected`;
law-side by `verify_callgraph.mlir`). All are wired into `tools/wsl/check_passes.sh`
(CI `mlir-rail-validate`).
