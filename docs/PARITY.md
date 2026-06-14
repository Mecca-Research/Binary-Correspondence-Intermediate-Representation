# Python ↔ MLIR parity contract

The Python package `bcir/` is the **executable conformance oracle**; the MLIR
dialect family under `mlir/` is the **law**. They must agree. This file is the
cross-map and the invariants that keep them in lockstep.

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
| min-plus select | `kbcir.realize.optimize` + `semiring` | `bcir.kbcir.select` (`#bcir.semiring<min_plus>`) |
| budget B(H,Θ) (RCSP) | `kbcir.rcsp.Budget` / `optimize_constrained` | `bcir.kbcir.budget` + `bcir.kbcir.select` `budget` (feasibility: `-bcir-verify` R9) |
| Pareto front | `kbcir.rcsp.pareto_plans` (label dominance) | RCSP labels over the same candidate DAG |
| scheduled price M(π,Θ) | `gem.overlap.price_scheduled` / `optimize_scheduled` | `bcir.kbcir.scheduled_price` (consistency: `-bcir-verify` R9) |
| soft/differentiable select | `kbcir.softdp.softselect` / `free_energy` (T=0 ⇒ `optimize`) | `bcir.kbcir.soft_select` (R9: F ≤ score; T=0 ⇒ F == score) |
| duration-aware schedule | `gem.schedule.schedule_eft` (LPT+EFT+locality+knee) | `bcir.gem.schedule` mode `eft` (R9) |
| token-DAG execution | `gem.schedule.execute_tokens` (pipelined phases) | `bcir.gem.schedule` mode `tokens` (R9) |
| bandwidth knee | `gem.schedule.bandwidth_knee` / `TargetProfile.mem_channels` | `bcir.target.capability` `mem_channels` |
| pipelined StreamPack (ABI v2) | `gem.streampack.hydrate_pipelined` / `abi` v2 codec | `bcir.gem.stream_pack` `pipeline_depth`, `bcir.gem.prefetch` `buffers` (R10) |
| L1 frozen cost table | `kbcir.microbench.CalibratedProfile` (`cal_gen`, Q8 ratios) | `bcir.kbcir.calibration` + capability `cal_gen` (R8) |
| L1 Bayesian + conformal table | `kbcir.bayescal` (`gaussian_update` VI / `conformal_delta` / `bayes_calibrate` / `abc_calibrate`) | `bcir.kbcir.calibration` `coverage_milli`/`random_delta_q8` (R8/R13) |
| L2 policy portfolio | `kbcir.portfolio.PolicyPortfolio` (class-table selection) | `bcir.kbcir.portfolio` (R9) |
| L2 replay gate | `kbcir.portfolio.replay_gate` / `ReplayCertificate` | `bcir.kbcir.replay_certificate` (R9: admitted ⇒ zero regressions) |
| L2 learned MoE gate | `kbcir.moegate` (`train_gate` GNN / `freeze` Q8 / `gate_replay_gate`) | `bcir.kbcir.moe_gate` (R13: routes certified experts, admitted ⇒ zero regressions) |
| search accelerator | `kbcir.accel` (`optimize_ordered` B&B / `train_ranker` / `accelerator_certificate`) | `bcir.kbcir.search_accel` (R13: admitted ⇒ zero mismatches; same optimum) |
| provenance manifest | `kbcir.provenance` (`build_manifest` / `replay` / `reproduces` / `verify_manifest`) | `bcir.kbcir.provenance_manifest` (R13: deployed ⇒ reproduced; manifest equality ⇒ identical plan) |
| building-blocks engine (e-graph) | `kbcir.egraph` (`EGraph` / `optimize_expr` / `saturate` / `shared_blocks`) | `bcir.egraph.extract` (R9: optimized_cost ≤ original_cost) |
| memory module (fixpoint) | `kbcir.memory` (`MemoryModule` / `freeze` / `freeze_module` / `is_idempotent` / `memory_artifacts`) + `verify.verify_memory` | `bcir.kbcir.memory_module` (R13: `saturated ⇒ admissible`; `a = Lim(Res(U))`, idempotent, chains into the manifest) |
| two-truth quarantine (MOPC) | `kbcir.twotruth` (`Graded` / `Decision` / `decide` / `is_classical` / `g_and`/`g_or`/`g_not`) + `verify.verify_quarantine` | `bcir.kbcir.two_truth` (R13: graded `(v,w)` may inform but never *be* a verdict; the only crossing is a recorded `decide`) |
| modular mapping function | `kbcir.mapping` (`support` / `MappingFunction` / `CommutingSquare`) + `verify.verify_support_preservation` / `verify_commutativity` | `bcir.target.lower_contract` (R12: `f(Supp(J)) ⊆ Supp(J')`; `Λ∘Ψ = Φ` — the parity discipline generalized) |
| enriched-operad memory | `kbcir.operad` (`EnrichedOperad` / `EnrichedOp` / `TwoCell` / `enrich_memory` / `f_label` / `f_index`) + `verify.verify_enriched` | *(higher interpretive layer; no MLIR rail — quarantined off the spine)* R13: content-addressed labels+indexes over `a = Lim(Res(U))` (CSE = liked pair; `Trace` integrity) |
| L1 cost throttle | `kbcir.throttle` (`AmortizationCertificate` / `certify` / `ThrottleReport`) | `bcir.kbcir.amortization` (R13: L0 ⇒ zero inference; gain ≥ cost ≤ budget) |
| L3 regret ledger | `kbcir.regret.RegretLedger` / `measure_regret` / `boundary_report` | `bcir.kbcir.regret_ledger` (R13: books balance, rule resolves) |
| MDL/evidence retune | `kbcir.regret` `data_fit_nats`/`complexity_nats`/`evidence_margin` (BIC: ΔL = Σ regret/best − (k/2)ln N) | `bcir.kbcir.regret_ledger` `data_fit_milli`/`complexity_milli`/`verdict` (R13: retune ⟺ data_fit > complexity) |
| policy provenance R13 | `verify.verify_provenance` | `bcir.verify.policy_provenance` + `-bcir-verify` R13 (promotion coverage, table correspondence) |
| StreamPack | `gem.streampack.StreamPack` | `bcir.gem.stream_pack` |
| lane segment | `gem.streampack.LaneSegment` | `bcir.gem.lane_segment` |
| verifier R1–R13 | `verify.{verify,verify_plan,verify_pack,verify_lowering,verify_provenance}` | `bcir.verify.*` ops + the `-bcir-verify` pass (R1–R13) |
| GEM pipeline (classify→select→batch→schedule→lower) | `kbcir.realize.optimize` / `gem.{hydrate,schedule,execute}` (the oracle stages) | `-bcir-classify-lanes / -bcir-select-realization / -bcir-batch / -bcir-schedule / -bcir-lower-to-llvm` (`mlir/lib/BCIRPasses.cpp`); `-bcir-select-realization` recomputes the min-plus `cost·weights` and reproduces 7808/9472 (`mlir/test/passes/gem_passes{,_neg}.mlir`) |
| memory tier id | `kbcir.cost.MemTier` | `BCIR_MemTier` (`BCIRAttrs.td`) |
| lowering (AOT) | `lower.llvm` (clang) | `bcir.target.lower_contract` |
| concurrency/affinity (CT2) | `gem.schedule_concurrent` | `bcir.gem.lane_segment` `affinity`/`unroll` |
| ROP/MAP front-ends (CT3) | `frontends.{rop,map}` | `bcir.parse.*` / `bcir.binary.*` |
| data-DNA telemetry (CT4) | `telemetry.DataDNA` + `kbcir.calibrate` | `bcir.trace.data_dna` |
| calibration loop (closed) | `kbcir.calibloop` (`close_loop` / `measure_and_close` / `rescore_plan` / `CalibrationCertificate`) + `verify.verify_calibration` | `bcir.kbcir.calibration` (R13: measure → freeze → replan; `cal_gen ≥ 1` ∧ `win ≥ 0`; the measured cost of not recalibrating) |
| JIT (CT5) | `lower.jit` (lli) | per-target `bcir.target.lower_contract` |
| StreamPack ABI (Phase 7) | `abi.streampack_abi` (v1) | `runtime/c/bcir_streampack.h` |
| WASM (Phase 7) | `lower.wasm` (clang→wasm + node) | per-target `bcir.target.lower_contract` |
| stackify (Phase 7) | `lower.stackify` (→ wasm/jvm/cil) | foundation for `bcir.target.lower_contract` encoders |
| C runtime (Phase 8) | `runtime/c/bcir_runtime.{h,c}` decodes `abi.streampack_abi` | `runtime/c/bcir_streampack.h` |
| async tokens (Phase 8) | `gem.async_tokens` (fork/await plan) | `bcir.async.fork` / `bcir.async.await` (`!bcir.token`) |
| memory model (Phase 8) | `lower.memory_model` (hazard→ordering) | `BCIR_MemOrdering` + barrier `ordering` → `llvm.fence` |
| per-target codegen (Phase 9) | `codegen.*` (llc → ARM/RISC-V/PTX/eBPF/C) | `bcir.target.lower_contract` (one per target) |

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

## How parity is enforced today

`bcir/tests/` pins the exact scores and per-target widths (runnable with
`python -m bcir.tests.run_all`, no third-party deps). When the MLIR toolchain is
available, the `mlir/examples` + `mlir/test/irdl` corpus round-trips through
`bcir-opt` / stock `mlir-opt` and must carry the same constants.

The verifier laws are negative-tested **per law on both rails**: the oracle in
`bcir/tests/test_verify.py` (R1–R13 across module/plan/pack/lowering/provenance
artifacts) and the dialect in `mlir/test/passes/verify_laws.mlir` (R1–R7) +
`verify_laws_deep.mlir` (R8–R13) via `-bcir-verify -verify-diagnostics`. The
pretty ODS corpus must stay clean under the full `-bcir-verify`
(`tools/wsl/check_passes.sh`, CI `mlir-rail-validate`).

The **GEM pipeline passes** carry the same dual-rail discipline:
`mlir/test/passes/gem_passes.mlir` FileCheck-pins the recomputed plan (the
min-plus `cost·weights` reproduces the oracle's 7808 cool / 9472 under the cap),
and `gem_passes_neg.mlir` negative-tests the cross-check (a declared selection
that is not the true argmin, or a StreamPack segment that breaks the R12 lowering
contract, is rejected) via `-verify-diagnostics`.
