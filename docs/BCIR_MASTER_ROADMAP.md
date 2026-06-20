# BCIR Master Roadmap

> **Status: the single, current, authoritative roadmap.** This document consolidates
> and supersedes the former strategy/blueprint/plan notes — `BCIR_STRATEGY_AND_ROADMAP.md`,
> `BCIR_LOWERING_PLAN.md`, `BCIR_BLUEPRINT.md`, `BCIR_Codex_Blueprint.md`,
> `BCIR_Full_LLVM_Build_Blueprint.md`, and `BCIR_LLVM_IR.md` (all removed in the
> consolidation; their unique content is folded in here). It pairs with the docs that
> stay separate because they are **normative reference / evidence / governance**, not
> roadmap:
> - [`BCIR_LANGREF.md`](BCIR_LANGREF.md) — the IR language + R1–R18 law spec (normative).
> - [`BCIR_STREAMPACK_ABI.md`](BCIR_STREAMPACK_ABI.md) — the frozen binary ABI (v1 + append-only v2).
> - [`PARITY.md`](PARITY.md) — the active oracle↔law cross-map (dual-rail enforcement).
> - [`BCIR_NATIVE_OBJECT_GATE.md`](BCIR_NATIVE_OBJECT_GATE.md) — the native-isel decision gate.
> - [`HARDWARE_VALIDATION.md`](HARDWARE_VALIDATION.md) — what is measured vs blocked on real silicon.
> - [`CLANG_COMPARISON.md`](CLANG_COMPARISON.md) — the measured BCIR-vs-Clang evidence.
> - [`REPO_CURRENT_STATE_AUDIT.md`](REPO_CURRENT_STATE_AUDIT.md) — the dated snapshot + changelog.
> - [`BCIR_Repo_Structure.md`](BCIR_Repo_Structure.md) — directory layout + build entry points.

---

## 1. What BCIR is (positioning)

BCIR is a **cost-governed planning + verification layer above LLVM**, designed to live
inside a driver/runtime — not a Clang replacement. It models what LLVM does not:

- **Cost as a first-class IR object** — a 12-dimensional integer/Q-fixed cost vector
  per realization, scalarized under a policy; the optimizer is a tropical (min-plus)
  shortest path with (max,+) wave overlap and resource-constrained search (RCSP).
- **Θ-feasibility** — live runtime state (thermal/power/contention) changes what is
  *legal*, not just what is fast: a thermal cap can make max-width vec16 infeasible and
  the feasible vec8 the correct plan.
- **A principled ML boundary** — the **two-truth quarantine** (L0–L3): learned/graded
  machinery is quarantined off the deterministic decision/execution path and may only
  *inform* it through frozen Q8 artifacts; it can never *become* a legality verdict.
- **Provenance + reproducibility as obligations** — every plan carries a replayable
  provenance manifest (R11/R13); the same inputs reproduce the same plan bit-for-bit.

**Niche-first philosophy.** "Forcing a refactor of Clang/LLVM" is a research result,
not a milestone. The goal is to demonstrate *on silicon* that cost-as-IR + Θ-feasibility
+ frozen-learned planning beats LLVM on workloads LLVM handles poorly — irregular
memory (gather/scatter avoidance), multi-target placement, and power/thermal-capped
kernels — while *matching* LLVM on dense kernels (which it must, since it delegates
instruction selection to the resident backend). The measured evidence is in
`CLANG_COMPARISON.md`: **match** on dense (0.98–1.00×), **win** on intent the backend
lacks (gather avoidance 6.0×, reduction order 14.1×, strided 1.33×), plus budget
feasibility as a correctness win.

**Three highest-ROI shapes:** (1) cost-governed scheduling/placement + verification
above LLVM; (2) the planning brain of an AI-accelerator / heterogeneous-SoC
runtime/driver; (3) a principled ML-in-compilers research vehicle.

---

## 2. Current state at a glance (measured)

| Fact | **Now** |
|---|---|
| Oracle conformance tests (`python -m bcir.tests.run_all`) | see **[`docs/STATUS.md`](STATUS.md)** (generated count — single source of truth), incl. the generated differential (now incl. a compose-rail metamorphic campaign) + verifier + fuzz |
| Deterministic **optimizer core** on the MLIR/C++ rail | **COMPLETE** — cost model, fusion/CSE/deforestation, min-plus plan, (max,+) overlap, per-claim + plan-level RCSP, all bit-exact vs the oracle |
| GEM C++ passes (classify/select/batch/schedule/lower) | all implemented (`mlir/lib/passes/`) |
| Verifier laws | **R1–R18** all first-class in `-bcir-verify` (R1–R17 dual-rail with the Python oracle + the `-bcir-lower-to-llvm` checkpoint; **R18** compositional call-graph integrity — callee resolution + no recursion — for the `kbcir.func/call/cond` family). R13 also **recomputes** the manifest digest + cross-checks `m_theta` against the IR |
| Named pass pipelines | `bcir-audit` / `bcir-optimize` / `bcir-hydrate` / `bcir-lower-llvm` / `bcir-aot` with verifier checkpoints |
| Θ context op | `bcir.kbcir.theta` — the C++ plan matches the oracle under **hot** Θ (matmul hot 1159168), not just cool |
| Six-target capability matrix | all six TARGETS cross-checked on the MLIR rail (`target_matrix.mlir`) — the law plans per-target from the capability seeds alone |
| C23 in the runtime + kernels | `_BitInt(N)` exact-width Q-fixed lanes + `#embed` frozen Q8 tables (both with C11 fallbacks) |
| **Plug-in C compiler rail** (`runtime/c/`, `bcir-cc`) | **Freestanding driver-subset C23 compiler *candidate*** — a no-Python stack (preprocessor → frontend → claim-graph IR → R1–R18 verify → plan → hydrate → execute), a cc-like driver, a data-model **ABI matrix** (`--target`), a Clang-grade **diagnostics engine**, an **`--fallback`** route-to-LLVM contract, a module-scope **effect/commutation analysis** (`--emit-effects`), and real register-map / UART / DMA driver fixtures — every stage **Python↔C parity-gated**. *Not yet* a hosted C23 / Clang-GCC replacement; the remaining hard-compiler work is **§5.9** |
| Trust-boundary fuzz | Python (`kbcir.fuzz`) + **libFuzzer + ASan/UBSan** on the StreamPack **and** ETL-binary C decoders (500k runs in CI) |
| Native object emission | decision gate documented (DEFERRED); the warranted slice (C → resident compiler → real eBPF/x86-64 object) is closed and ELF-verified |
| LLVM version policy | tracks the **latest LLVM/MLIR release (22), gating** (from apt.llvm.org; toolchain auto-resolved) |
| Perf vs Clang | **measured** (`CLANG_COMPARISON.md`): match on dense, 1.3–14× on irregular memory |
| Calibration loop | **closed on host** (microbench → `FrozenCalibrator` → R13 replan) + real-signal wiring (`bcir.silicon`); a *measured* bare-metal replan win is the one deferred item |

---

## 3. The MLIR / C / C++ / Python placement map (the two-truth line)

The port boundary is BCIR's own **L0–L3 / two-truth line** and is not negotiable:

> **Deterministic + integer/Q-fixed on the decision/execution path → C++/MLIR (law)
> or C (runtime). Graded / float / train-time → Python that *freezes* to Q8.**

> ### The prototype-then-port discipline (NON-NEGOTIABLE)
>
> The Python oracle is a **prototype rail**, not the destination. A capability is *prototyped*
> in the oracle (cheap iteration, the conformance reference), then — **once it is validated, the
> oracle work STOPS and the real implementation is built on the production rail**: the **MLIR/C++
> law** for plan-time decisions, or **C in `runtime/c/`** for the runtime + the plug-in C compiler.
> Every port is gated by a **Python↔production parity test** (oracle == law / oracle == C),
> exactly as the optimizer core was ported to `-bcir-*` passes and GEM hydrate/execute were ported
> to `bcir_encode.c`/`bcir_exec.c`.
>
> **This applies to the C frontend.** The L1–L8 ladder + C.2 in `bcir/frontends/cfront/` is the
> *prototype*. The production plug-in C compiler is **`runtime/c/bcir_cfront.c`** (the C twin),
> built stage-by-stage with a parity gate (`bcir/tests/test_c_cfront.py`, `tools/c/check_runtime.sh`).
> Do **not** keep extending the oracle frontend as if it were the product — port each validated
> stage to C and let the prototype freeze. *Started:* the register-map slice (integer expressions,
> struct/bitfield layout, volatile/MMIO) lowers identically on both rails (`claims=23 mmio=1 bf=3
> …`); the remaining stages (arrays/calls, control flow, preprocessor, full ABI) port next.

| Subsystem | Today | **Target home** | Status |
|---|---|---|---|
| Dialect / ODS (the law's vocabulary) | `mlir/include/BCIR/*.td` | MLIR | ✅ |
| Verifier **R1–R13** | `-bcir-verify` + `bcir.verify` | MLIR/C++ + Python (oracle ref) | ✅ dual-rail (R13 **recomputes the provenance digest** AND **cross-checks every component hash** — `m_module`/`m_target`/`m_theta`/`m_policy` — against the in-IR module/capability/theta/policy, byte-identical to `provenance.hash_*`; trusts neither the digest nor the input identities) |
| Verifier **R14/R15/R16** (CIM dispatch / DVFS clock / alloc placement) | `-bcir-verify` + the `-bcir-lower-to-llvm` checkpoint + oracle | MLIR/C++ + Python | ✅ **first-class `-bcir-verify` laws** (dual-rail with `verify.{verify_cim,verify_dvfs,verify_allocator}`) |
| K_BCIR **cost model** (`_cost`, candidates, stride/tier) | `-bcir-cost-model` (C++23) | MLIR/C++ | ✅ **ported** (bit-exact: vec16 7808, gather 528384, tile 126976) |
| K_BCIR **fusion / CSE / deforestation** | `-bcir-cost-model` | MLIR/C++ | ✅ **ported** (7808 / 5888 / 5100) |
| K_BCIR **min-plus shortest path** (`optimize`) | `-bcir-plan` | MLIR/C++ | ✅ **ported** (per-target, hot/cool Θ) |
| K_BCIR **overlap (max,+)** | `-bcir-overlap` | MLIR/C++ | ✅ **ported** (makespan/gain) |
| K_BCIR **RCSP / Pareto** (per-claim + plan-level) | `-bcir-rcsp`, `-bcir-rcsp-plan` | MLIR/C++ | ✅ **ported** (9472, {16,8}@17280) |
| **Bundle** (joint) optimization (`bundle.optimize_bundled`) | `-bcir-bundle` | MLIR/C++ | ✅ **detection + joint-reorder** — reorders the cost columns, re-prices the min-plus path, annotates `kbcir.bundle_gain`/`bundle_order` (`bundle_reorder.mlir`) |
| **Proof-carrying record + replay** (`proof.explain` / `proof.replay`) | `-bcir-explain` / `-bcir-replay` | MLIR/C++ | ✅ **ported** — per-claim candidates weighed + chosen width/score, per-module total, as IR annotations (`explain.mlir`); plus the replay recheck that recomputes a fresh plan and diffs it against the declared `kbcir.explain_*` record (`kbcir.replay_reproduced`/`replay_mismatches` — `replay.mlir`) |
| **Compositional** semantics (`compose.plan_composite`) | `kbcir.func` / `kbcir.call` / `kbcir.cond` + `-bcir-compose` | MLIR/C++ | ✅ **complete** — op vocabulary + **R18 call-graph law** + **`-bcir-compose`**: Seq sum / Cond max+expected / Leaf optimize / Call **inter-procedural summary** / **RCSP-constrained** under a `kbcir.budget` / **alias-effect** footprint (`kbcir.effect_reads/writes`) + sibling-call independence (`commutes_with_prev`) + **dynamic-shape** bound (`compose_dynamic`). Reproduces plan_composite's worst/expected/reused/feasible/effect/dynamic (`compose_cost`/`_summary`/`_budget`/`_effect.mlir` + a generated compose differential) |
| GEM classify / batch / schedule / lower | `-bcir-*` passes | MLIR/C++ | ✅ |
| **CIM/PIM dispatch + DVFS clock DECISION** (`gem.cim` / `gem.dvfs`) | `-bcir-cim` / `-bcir-dvfs` | MLIR/C++ | ✅ **recomputed** (core-vs-PIM offload cost; per-phase intensity → Q8 clock) — the law derives the decision, not just R14/R15-verifies a declared one (`cim.mlir`, `dvfs.mlir`) |
| **EFT schedule + async pipelining + power rail** (`gem.schedule` / `gem.async_tokens`) | `-bcir-schedule-eft` / `-bcir-async` / `-bcir-power-rail` | MLIR/C++ | ✅ **ported** — phase-barriered EFT waves (LPT + earliest-finish + locality + knee), the cross-phase fork/await pipelined schedule (later-phase independent claims overlap earlier ones), AND a per-slot DVFS overlay on the placed timeline (classify+clock each slot's interval, downclock memory-bound slots — the join of EFT + DVFS); `schedule_eft.mlir`, `async.mlir`, `power_rail.mlir` |
| **Allocator pool-plan** (`allocator.live_intervals`/`pool_plan`) | `-bcir-alloc-pool` | MLIR/C++ | ✅ **ported** — liveness-based pooling (disjoint live ranges share an arena, greedy left-edge); annotates per-resource pool_id + peak/naive/saved bytes (`alloc_pool.mlir`) |
| GEM **hydrate** (plan → StreamPack **bytes**) | **C** `runtime/c/bcir_encode.c` + Python `abi.streampack_abi.encode` | **C** (the encoder) | ✅ **ported** — `bcir_sp_reencode` is byte-identical to the Python encoder (v1 + v2) |
| GEM **deterministic executor** (decode → drive kernels) | **C** `runtime/c/bcir_exec.c` + Python `gem.execute` | **C** (hot path) | ✅ **ported** — Python↔C dispatch-order + telemetry parity + libFuzzer |
| **Plug-in C compiler / frontend** (C source → claim graph) | **C** `runtime/c/bcir_cfront.c` (IR: `bcir_cir.h`) + Python prototype `bcir/frontends/cfront/` | **C** (the driver-embeddable compiler) | ✅ **the full L1–L8 ladder ported to C + Python↔C parity-gated**: **L1** integer expr, **L2** struct/bitfield layout, **L3** pointers/arrays (GEP), **L4** functions + call graph (**R18** in C), **L5** volatile/MMIO, **L6** control flow, **L7** a real C preprocessor (`bcir_cpp.c`), **L8** ABI (struct return-by-value, `__attribute__((packed))`/`aligned`, layout cross-checked vs Clang). A full **R1–R18 verifier** (`bcir_verify.c`: R1–R8 incl. R6 lane↔stride, R9 plan, R10–R11 StreamPack, R12, R13 provenance digest, R17, R18), **§5.8 atomics/fences** (`ATOMIC_*`/`BARRIER` on lane A), a **C.2 attestation** stamped on the emit, a **faithful C emitter** (Clang-behaviour-equivalent), and the **closed loop** `C → bcir_cpp → bcir_cfront → bcir_plan → bcir_hydrate → bcir_exec` (no Python). Now also a Clang-grade **diagnostics engine** (`bcir_diag.c`), a cross-platform **data-model ABI matrix** (`--target`), an **`--fallback`** route-to-LLVM contract, a module-scope **effect/commutation analysis** (`--emit-effects`), and scalar globals (read+write). The IR now grows with no fixed `BCIR_MAX_*` ceilings (any number of functions / params / calls / resources / claims). The road to a full C23 *replacement* is the ordinary hard-compiler work in **§5.9** (full integer promotions + usual arithmetic conversions, designated/compound initializers, object/debug/unwind emission, conformance suites) |
| StreamPack ABI **decoder** | **C** `runtime/c/bcir_runtime.c` | **C** (frozen ABI) | ✅ CRC-gated parity + libFuzzer |
| ETL binary-record decoder | **C** `runtime/c/bcir_binrec.c` | **C** | ✅ parity + libFuzzer |
| Portable kernel emission (C23 + `_BitInt`/`#embed`) | Python `lower.c_kernel` | C output (emitter may become C++) | ✅ |
| Telemetry ring (zero-copy) | **C** producer + Python reader | **C** | ✅ |
| Real-signal probes / DVFS actuation | Python `bcir.silicon`, `gem.dvfs.actuate` | **C/C++** (runtime) | read ✅; actuation gated (needs bare-metal) |
| **Learned organs** (bayescal/softdp/moegate train/calibrate SGD/regret) | Python | **Python** (freeze to Q8) | by design — porting them violates the quarantine |
| Enriched operad / memory-module fixpoints / two-truth | Python | **Python** unless load-bearing for plan-time caching | research-side |

**One-line rule:** *anything deterministic and integer on the decision/execution path
→ MLIR/C++ (law) or C (runtime); anything float/learned/train-time → Python that
freezes to Q8.*

---

## 4. What's done (the landed work)

**The deterministic optimizer core is fully on the MLIR rail (C++23)** — the headline
of the last cycle. Each stage is bit-exact against the oracle and gated by
`check_passes.sh` + the generated differential harness:

```
-bcir-cost-model (cost + fusion/CSE)  →  -bcir-plan (coupled min-plus optimize)
   →  -bcir-overlap (max,+ M(π,Θ))  →  -bcir-rcsp / -bcir-rcsp-plan (constrained search)
```

- **Cost algebra + fusion/CSE/deforestation** recompute every claim's 12-d cost from
  `bcir.claim` + `bcir.target.capability` (a `constexpr` tier table + the seeded
  constants), so the law no longer trusts emitter-baked path costs.
- **Coupled shortest path** reproduces the oracle's `optimize` on every module and, via
  the `bcir.kbcir.theta` context op, under hot Θ too.
- **Six-target capability matrix** (`target_matrix.mlir`): each program is emitted once
  per TARGET; `-bcir-plan`/`-overlap`/`-rcsp-plan` recompute the oracle's per-target
  result from the capability alone — avx512/sve/rvv 16→7808, avx2 8→9472, neon 4→12800,
  ptx 32→6976; the GPU's coalesced gather halves histogram (266240 vs 528384).
- **Verifier R1–R18**, negative-tested per law (R1–R17 dual-rail on the Python oracle + the
  MLIR rail; **R18** — compositional call-graph integrity — on the MLIR rail), plus a verifier
  *differential* (`gen_illegal_module` + `run_verifier_campaign`) that fault-injects each
  law and confirms the verifier catches it.
- **C23 where it pays:** `_BitInt(N)` exact-width Q-fixed lane kernels (the place a
  standard int promotes or has no type at all) and `#embed` frozen Q8 tables, both with
  preprocessor-selected C11 fallbacks; bit-identical across `-std=c23`/`-std=c11`.
- **Trust-boundary fuzz:** libFuzzer + ASan/UBSan on the StreamPack **and** ETL-binary
  C decoders; the Python `kbcir.fuzz` covers the StreamPack codec, ROP/MAP/ETL
  front-ends, `etl.binary`, calibration JSON, and the MLIR emitter.
- **Native object gate** (`BCIR_NATIVE_OBJECT_GATE.md`): documented GO/STOP criteria;
  the warranted slice (emit C → resident compiler → real eBPF/x86-64 ELF object) is
  closed and ELF-verified — no hand-rolled isel.
- **Calibration software path** closed end-to-end (`bcir.silicon` reads real PMU + RAPL
  + on-die thermal; `kbcir.calibloop.measured_replan` trains+freezes a `LinearCalibrator`
  and certifies the win) — it degrades honestly in a sandbox and lights up on a rig.
- **Adaptive "smart" layer** (8 deterministic, opt-in, gains-only capabilities): RL
  allocator, compute-in-memory dispatch (R14), persistent e-graph with telemetry pivot,
  JIT shape specialist, uncertainty-gated sensing, zero-copy telemetry ring, fuzzy MoE
  routing, phase-aware DVFS — all off the default plan/emit path (`test_perf` guard).
- **Precision module** (`kbcir.precision`, opt-in): Q8-ULP error unit, integer interval
  error bounds, a compensated Q8 reduction (bit-identical to int64-exact), stability
  diagnostics as two-truth `Graded` signals.

The earlier-cycle spine is also done: the 5 C++ GEM passes, the multi-version LLVM
matrix, the R13 provenance manifest, the generated adversarial Python↔MLIR differential,
the widened corpus (real tiled matmul / scan / multi-claim histogram), the StreamPack
freeze + freestanding C decoder, the portable C23 kernel backend, and the measured
Clang comparison.

---

## 5. The forward roadmap (what's next)

### 5.1 Oracle → MLIR / C++ (plan-time law) — remaining ports

The optimizer core is ported. What is still Python-only **and** belongs on the law rail:

1. ✅ **R14 (CIM/PIM dispatch) + R15 (DVFS clock) + R16 (allocator placement) as
   first-class `-bcir-verify` checks. DONE.** All three are now enforced by the dedicated
   `-bcir-verify` pass (`BCIRVerifyPass.cpp`, dual-rail with `verify.{verify_cim,
   verify_dvfs,verify_allocator}`), with positive/negative `.mlir` cases in
   `verify_laws_deep.mlir`; they remain enforced at the `-bcir-lower-to-llvm` checkpoint
   too (defense in depth). Verifier dual-rail symmetry is complete — every law R1–R18 is
   checkable by `-bcir-verify` alone (R1–R17 dual-rail; **R18** call-graph integrity on the
   MLIR rail).
2. ✅ **The `verification` cost dimension (the 12th cost axis). DONE.** It now has a real
   producer on both rails (`realize._verify_cost` / `BCIRCostModel.h::verifyCostFor`,
   cross-checked by `cost_model_verify.mlir`): the cost of discharging a claim's verify
   contract. `none`/`bounds` are free (the bounds check fuses into the access the memory
   axis already prices, so every existing pinned score is unchanged); `exact` (recompute
   + compare) and `hash` (digest every output) carry an O(n) cost. It is width-independent
   (the contract is a property of the claim, not the lane), so it never perturbs the
   per-claim selection but *is* a tradeable plan resource — an RCSP cap on `verification`
   can make a claim infeasible (`test_verify_cost.py`). exact/hash were previously unused,
   so this is purely additive.
3. ✅ **A C++ `-bcir-verify` law-for-law differential. DONE.** Every law **R1–R18** now
   has a negative `-verify-diagnostics` case in the committed `verify_laws*.mlir` /
   `verify_accuracy.mlir` / `verify_callgraph.mlir` (run under `bcir-opt` in CI), and a coverage gate
   (`test_verify_differential.py`) guarantees no law silently loses its toolchain-rail
   negative case — the systematic complement to the oracle-rail `run_verifier_campaign`.

**New: the accuracy contract (R17).** `verify.verify_accuracy` and the MLIR `-bcir-verify`
R17 law (consuming the `#bcir.precision<…, exact, tol>` attr) are the dual-rail accuracy
contract: a claim with a declared tolerance must realize within its static Q8-ULP error
bound — a `reduce.*` over `count` terms is bounded by `count` ULP naive but 1 ULP
compensated, so a tight tolerance forces the compensated realization.

**MLIR-22 completion (2026-06-17).** The rail moved to LLVM/MLIR 22 and was made fully
locally-validatable (true-22 via conda-forge — `tools/local/`). A dual-rail completeness scan
confirmed the law rail now mirrors the oracle's **entire deterministic spine** (84 ODS ops,
23 passes, **R1–R18** all first-class in `-bcir-verify` + negative-tested). Landed since:
the **`bcir.kbcir.memory_module`** op + R13 admissibility (`saturated ∧ generation ≥ 1`;
closes the PARITY gap that previously claimed the op); cross-pass `PlanAnalysis` sharing across
all 9 optimizer passes; bytecode round-trip; `hasVerifier` parse-time structural checks on
`resource`/`gem.lane_segment`/`claim`/`target.capability`; and the C++ standard at `-std=c++2c`.

**Remaining law-rail gaps — CLOSED (2026-06-18).** The completeness scan's three narrow gaps
were built and dual-rail-verified against the oracle on true MLIR 22:
1. ✅ **Overlap re-selection sweep. DONE.** `-bcir-overlap-optimize` (`BCIROverlapPass.cpp`) ports
   `gem/overlap.py::optimize_scheduled`: from the serial optimum it sweeps each claim once in
   column order and adopts the legal alternative that strictly lowers the scheduled makespan
   (first-best tie-break), re-pricing serially so R9 holds. The reusable `computeMakespan` was
   extracted (shared with `-bcir-overlap`). Matches the oracle's `(makespan, serial)` on all 11
   corpus programs — including the real overlap gains (fused_chain 7808<13696, matmul 253952<1015808)
   — and is a no-op where the serial optimum is already makespan-optimal (`overlap_optimize.mlir`).
2. ✅ **MOPC R12 support-preservation. DONE.** `bcir.target.lower_contract` carries optional
   `source_support`/`target_support`/`discharges`; `-bcir-verify` **R12** now enforces
   `f(Supp(J)) ⊆ Supp(J')` (identity dim-map) unless discharged — reproducing
   `kbcir/mapping.py::dropped` (`verify_laws_deep.mlir`). The commuting-square `Λ∘Ψ = Φ` stays a
   runtime/**differential** property (path-equivalence over inputs, not a static structural check)
   — it is the PARITY discipline already enforced by the provenance digest + the generated parity
   campaign, so it is not a static `-bcir-verify` law by design.
3. ✅ **Telemetry sensing gate. DONE.** `-bcir-sense` (`BCIRSensePass.cpp`) ports
   `kbcir/sensing.py::RegretSensor.sense`: per-segment `cv_milli = 1000·stdev/mean` over the
   `bcir.trace.data_dna` cycles (population variance, floor-isqrt — the exact integer formula),
   ranked by `(-cv_milli, segment)`, assigning `high`/`low`/`off` under the threshold+budget gate.
   Matches the oracle exactly (`sense.mlir`). The a-priori `sense_by_ranker` variant stays off-rail
   (it leans on a learned ranker margin).

The law rail now mirrors the oracle's **entire deterministic spine** with no known buildable gaps:
85 ODS ops, 25 passes, R1–R18, the optimizer core (plan / RCSP / overlap + the re-selection sweep /
bundle / schedule), the smart-lowering laws, provenance, compose, the memory-module fixpoint, the
MOPC support law, and the sensing gate. What remains Python-only is by-design off-rail
(quarantine/learned organs, §5.6) or inherently host-side (measurement, fuzz, toolchain).

### 5.2 Oracle → C (run-time hot path) — remaining ports

The C runtime has the decoders (StreamPack, ETL-binary) and now the executor:

1. ✅ **The deterministic StreamPack executor** (`gem/execute.py` → `runtime/c/bcir_exec.c`).
   **DONE.** A freestanding `bcir_sp_execute` decodes the pack and dispatches its claims
   in GEM order — topological phase order (first appearance in the pack), then ascending
   claim id within a phase — invoking an optional per-claim kernel callback and collecting
   per-phase telemetry, with no libc and caller-owned memory. Python↔C dispatch-order +
   telemetry parity (`test_c_executor.py`, `check_runtime.sh`) and a libFuzzer + ASan/UBSan
   harness (`fuzz_exec.c`). The StreamPack is now a no-Python hot artifact a driver runs
   end-to-end.
2. ✅ **The StreamPack encoder in C** (`runtime/c/bcir_encode.c`). **DONE.** A freestanding
   `bcir_sp_reencode` parses a pack and re-serializes it through value-based write
   primitives, **byte-identical** to `bcir.abi.encode` across the corpus and both ABI
   versions (v1 + v2 pipeline/double-buffer tails) — the full C round-trip, so a
   driver-resident hydrate emits the artifact with no Python and no libc. Parity gate
   `test_c_encoder.py` + `check_runtime.sh`; libFuzzer + ASan/UBSan harness `fuzz_encode.c`.
3. ✅ **The `precision="compensated"` C-kernel. DONE.** `lower.c_kernel.emit_compensated_
   reduce_c` lowers the residual-carry Q8 MAC (`kbcir.precision.compensated_reduce_q8`):
   the dropped low 8 bits are carried forward, so the result is **bit-identical to the
   int64-exact reduction** (vs the naive per-term-truncating form, which drifts up to `n`
   ULP). Self-check compiles + runs under C11 and C23 (`test_precision_lowering.py`).

### 5.3 New deterministic features (not ports) — for C++/MLIR

1. ✅ **Multi-claim bundle (joint) optimization. DONE.** `kbcir.bundle.optimize_bundled`
   is the genuine combinatorial step beyond the pairwise coupling: it finds the clusters
   of claims that share a read operand and jointly searches the intra-phase order
   (bounded, exhaustive, dependency-preserving) that minimizes the plan score — recovering
   the fusion discount the pairwise shortest path misses when sharers are interleaved.
   Correctness-preserving (only mutually-independent same-phase claims are reordered, never
   across a conflicting pair) and a no-op where there is nothing to join; it emits a
   `BundleCertificate` per improving bundle (a proof-carrying search record). A real **12%
   gain on the tiled-matmul corpus**, scores otherwise pinned (`test_bundle.py`).
2. ✅ **Proof-carrying optimization records. DONE.** `kbcir.proof` (CLI `bcir.run
   --explain` / `--replay FILE` / `--reduce`): `explain` builds a replayable
   `DecisionRecord` — the R13 provenance digest + the per-claim *rationale* (the
   candidates the optimizer weighed and the one it chose) + the bundle rewrite
   certificates; `replay` reproduces it bit-for-bit from the same inputs (the digest gate
   + the per-claim decisions) or reports exactly what diverged; `reduce` minimizes a module
   to a legal witness. Round-trips through JSON (`test_proof.py`).
3. ◑ **Compositional semantics. DEEPENED.** `kbcir.compose` extends planning past
   straight-line kernels along the central equation's own series-parallel grain — a region
   tree: `Seq` (series, sum cost), `Cond` (control flow: worst-case **max** over branches +
   a probability-weighted **expected** cost), `Call`/`Function` (reuse via inline argument
   substitution; recursion rejected for bounded compile time), and `dynamic` claims (count
   as a static upper bound, worst-case priced — the plan holds for any actual ≤ the bound).
   It reuses `optimize` for the leaves, so a `Leaf([vector_add])` prices to exactly 7808.
   The deepening adds **alias/effect modeling** (`Effect`/`effect`/`independent`: the
   read/write footprint folded through calls, and the RAW/WAR/WAW test that decides whether
   two calls commute) and **inter-procedural summary costs** (`summarize`/`FunctionSummary`:
   a function is planned **once** over its formals and every cost-compatible call reuses that
   cost instead of re-planning the body — sound, because the reuse is gated on the actuals
   matching the formals' cost-keys, else it falls back to inline; bounds compile time to
   O(functions + call-sites)) (`test_compose.py`). On the law rail, the **func/if op family**
   (`kbcir.func` / `kbcir.call` / `kbcir.cond`) gives the region tree first-class MLIR form
   (`compose_ops.mlir`). The compositional cost stays the oracle's conformance reference.

### 5.4 Measured real-silicon calibration (DEFERRED — the top differentiator)

The software path is merged and certified on host, and now **push-button**:
`tools/silicon/measure_replan.sh` runs the whole probe→read-PMU/RAPL/thermal→fold-Θ→
replan→certify loop and prints the **measured** win on a rig (provenance=real) or degrades
honestly otherwise (synthetic, no fabricated number). The probe now **enumerates the three
gating signals** — a hardware PMU (`perf_event_open`), RAPL energy, and a cpufreq userspace
governor — and prints an explicit **rig-ready: YES/NO** verdict that names any missing
signal, so the requirement to fire the win is crisply specified, never implicit. It is
**CI-exercised in degrade mode** (`test_silicon_runbook.py`: the probe enumerates all three
signals; `--require-real` fires exactly when all three are present) so the rig path can never
silently rot and a sandbox run can't masquerade as a measured result. The one thing no
architecture substitutes for is the *measured* (not synthetic) replan win itself — it needs
a bare-metal rig with `intel_pstate=passive` + a userspace governor + RAPL exposed (the
exact rig is in `HARDWARE_VALIDATION.md`). The measured win **fires the moment** such a host
runs the runbook; that converts "optimal-w.r.t.-a-model" into evidence and is the single most
valuable next result once a rig is available.

### 5.5 Native backend (DEFERRED — gated)

BCIR-native instruction selection stays deferred behind the documented decision gate
(`BCIR_NATIVE_OBJECT_GATE.md`): every seeded target has a resident LLVM backend, so the
GO criteria (G1 no resident backend + G2 measured ≥2× economics) are unmet. Revisit for
a bare PIM/CIM controller or a driver-resident eBPF JIT under a latency SLA.

### 5.6 Stays Python (the quarantine)

The learned organs (`bayescal`, `softdp`, `moegate` training, `calibrate` SGD, `regret`
ledger, `microbench`), the offline calibration, the enriched operad / memory-module
fixpoints, the conformance **oracle** itself, and the generators (`differential.gen_module`,
`fuzz`) stay Python by design and emit generation-tagged frozen Q8 artifacts. Porting
them would violate the two-truth quarantine. L2 portfolio offline learning (Bayesian
optimization) and a production Kafka broker deployment are operational/research items
that stay on the Python/ops side.

### 5.7 The plug-in-compiler roadmap — C frontend → drivers → ML ops → C++/Python → ecosystem

The next strategic arc turns BCIR from a cost-governed planning/verification layer above LLVM into a
**plug-in compiler for whole paradigms**: a real program (C first, then Python, then C++) is parsed
into the claim graph, planned + verified by the K_BCIR spine, lowered through the resident backend,
and executed by GEM across the heterogeneous **channel** tower (`docs/HETEROGENEOUS_CHANNELS.md`). It
is **dependency-ordered** — each phase unlocks the next, and building out of order (drivers before a
*verifiable* C backend) creates technical debt. Every phase keeps the dual-rail discipline (prototype
in the Python oracle, port to the MLIR law, lockstep) and the channel separation (each backend
isolated, unified K_BCIR/GEM execution). The current ROP/MAP/ETL seams (`frontends.{rop,map}`,
`bcir.binary.*`) are deliberately narrow DSL/binary front-ends; the paradigm frontends below are the
larger effort.

#### Phase C — A solid C frontend + backend (the immediate next priority; it gates Phase D)

The keystone. Drivers, opcode tables, and the Hardware Description Layer all need a working,
*verifiable* C path — generating C from BCIR and checking it against source makes importing Linux
kernel tables / register maps / PCIe / ACPI clean; building drivers first would be debt.

##### C.0 — C23 as the substrate (relearned from ISO/IEC 9899:2024; § refs verified against the text)

The C path targets **C23**, not C11/C17. C23 removes the awkward/non-portable workarounds the
per-pattern kernels lean on today (`__builtin_*`, `xxd`-generated arrays, implementation-defined
`enum`/sign representations) and — more importantly — makes it realistic to express *the oracle
itself*, not just kernels, in C. Two adoption tiers:

**Tier 1 — main-lowering enablers (adopt as the C path is built):**

| C23 feature | § | What it unlocks for the port |
|---|---|---|
| `_BitInt(N)` bit-precise ints | 6.2.5 / 6.7.2.5 | Exact-width Q-fixed accumulators + RID/opcode/stride bit-fields matching the StreamPack wire format (two's-complement now *guaranteed*, Annex M). |
| `<stdckdint.h>` `ckd_add/sub/mul` | 7.20 | Overflow-checked cost accumulation in min-plus / (max,+) / RCSP — the biggest correctness win for the cost algebra; portable, replaces `__builtin_*_overflow`. |
| `<stdbit.h>` + endian macros | 7.18 / 7.18.2 | `__STDC_ENDIAN_NATIVE__` collapses the StreamPack encode/decode `#ifdef`s to one path (Python↔C parity); `stdc_count_ones`/`bit_width`/`bit_ceil` for Pareto bitsets + buffer rounding. |
| `enum : underlying-type` | 6.7.2.2 | ABI-stable opcode/phase/lane/kind enums (size+representation fixed for the wire format); values wider than `int`. |
| `constexpr` objects | 6.6 / 6.7 | Opcode/policy/channel-descriptor tables + verifier-law constants validated *at translation time* (true cost-model metaprogramming). |
| `static_assert` (1-arg) | 6.7.11 | `static_assert(sizeof(StreamHeader)==N)` / `offsetof` / `BITINT_MAXWIDTH` wire-format + width invariants. |
| `typeof` / `typeof_unqual` | 6.7.8 | Portable generic containers (min-plus priority queue, Pareto set, CSE table) without per-type duplication. |
| `nullptr` / `nullptr_t` | 6.3.2.4 / 7.21.2 | Unambiguous null in the pointer-heavy claim graph + `_Generic`/variadic dispatch. |
| `[[nodiscard]] [[fallthrough]] [[maybe_unused]] [[noreturn]]` | 6.7.12 | `[[nodiscard]]` forces handling of validator `bool`s + `ckd_*` flags (laws must not be dropped); clean opcode-dispatch `switch`. |
| `unreachable()` | 7.21.1 | Dead-path assertion after exhaustive opcode dispatch + "proven impossible by law Rk" points. |
| `#embed` (+ `__has_embed`) | 6.10.3 | Bake calibration tables / golden StreamPacks / provenance digests into the binary — no runtime I/O (keeps the runtime freestanding + deterministic). Already used for the frozen Q8 tables, with C11 fallbacks. |
| binary literals `0b` + digit separators `'` | 6.4.4.1 | Readable opcode/flag masks + stride/budget constants matching the Python source. |

**Tier 2 — build-later advanced (turn C into a runtime/AI/metaprogramming platform):**

- **A graph runtime environment in C** (`gem/` ported): `<threads.h>` + `<stdatomic.h>` for a parallel
  RCSP/Pareto search and a concurrent e-graph fixpoint; `[[unsequenced]]`/`[[reproducible]]` (§6.7.12.7)
  to let the host C compiler legally fuse/CSE the pure cost kernels — mirroring K_BCIR's own
  fusion/deforestation at the C level; `call_once` (now mandatory) for one-time channel-table init.
- **AI-style processes in C** (fixed-point inference / table-driven nets): `_BitInt(N)` exact Q-fixed
  activations + `ckd_*` saturating MAC + `constexpr`/`#embed` quantized weight tables; `_Decimal*` /
  IEC 60559 (`<tgmath.h>`, Annex F/H) for reproducible Bayesian/conformal calibration under the R17
  accuracy contract.
- **Metaprogramming**: `typeof` + `_Generic` + X-macros + `constexpr` to *generate the StreamPack
  (de)serializer and the R1–R18 law table from one declarative spec* — the closest C gets to the
  oracle's single-source-of-truth design.

**Porting caveats (from the standard text):** `realloc(p,0)` is now **UB** (audit the arena
allocator); `_BitInt` widths cap at `BITINT_MAXWIDTH` (§5.2.4.2.1 — probe it, don't assume ≥128);
gate every Tier-1 header on `__STDC_VERSION__ == 202311L` (+ the per-header `__STDC_VERSION_*_H__`
macros) with C11 fallbacks, exactly as the Q8 `#embed` path already does.

##### C.1 — A usable C frontend (the *input* seam) as a **staged conformance ladder**

A Clang-compatible parser + semantics for a useful C subset → the *same* claim graph the oracle
reasons over (so R1–R18 + the cost model apply unchanged). Built as an escalating ladder of language
stages — **each stage is only "done" when it has all six artifacts** (the dual-rail discipline,
applied to C):

1. C source fixture · 2. claim-graph golden · 3. the K_BCIR plan · 4. the emitted C output ·
5. behaviour equivalence against Clang on a harness · 6. an R1–R18 verifier checkpoint.

> **This `bcir/frontends/cfront/` ladder is the oracle PROTOTYPE.** Per the prototype-then-port
> discipline (§3), the production plug-in C compiler is **`runtime/c/bcir_cfront.c`** (C, the
> driver-embeddable twin). The register-map slice is already ported + Python↔C parity-gated; the
> remaining stages port stage-by-stage. Stop extending the prototype as if it were the product.

**Prototyped (the full L1–L8 ladder)** (`bcir/frontends/cfront/`, `bcir.frontends.cfront.compile_unit`) — a
real recursive-descent C **preprocessor** + lexer/parser → the claim-graph model
(`Resource`/`Claim`/`Phase`), the K_BCIR plan, an arbitrary-claim-graph C emitter (straight-line +
real `if`/`while` control flow, `memcpy`-based alignment-safe member access), the `plan_composite`
call-graph (R18) checkpoint, a `bcir-explain` artifact, the **C.2 verified-C attestation**
(R12/R13/R17/R18 stamped on each emitted function) + a reusable self-check artifact
(`emit_selfcheck`), and a seeded-random Clang behaviour-equivalence harness (toolchain-gated).
`python -m bcir.frontends.cfront <file.c> [--explain|--selfcheck]`.

| Stage | C surface | status |
|---|---|---|
| L1 | fixed-width integer expressions (`_BitInt`/`<stdint.h>`) | ✅ |
| L2 | structs / unions / explicit layout (Clang-compatible offsets) | ✅ |
| L3 | pointers / arrays → GEP-equivalent claim mapping | ✅ |
| L4 | functions + the call graph → **R18** (recursion + undefined-callee rejected) | ✅ |
| L5 | `volatile`/MMIO → `Domain.MMIO` resources (ordered/`barriered`) + bitfield mask/shift claims | ✅ (the register-map/MMIO MVP) |
| L6 | control flow — `if`/`else` → `compose.Cond`, bounded `while` (mutable named locals) | ✅ |
| L7 | preprocessor — object/function/variadic `#define` (+ `#`/`##`, `__VA_ARGS__`/`__VA_OPT__`), `#if`/`#ifdef`/`#elifdef`, predefined macros (`__FILE__`/`__LINE__`/`__DATE__`/`__TIME__` + `__STDC__`/`__STDC_VERSION__`/`__STDC_HOSTED__`), `#line`, `_Pragma`, `#include`, C23 `#embed` (→ const globals) | ✅ |
| L8 | ABI — struct return-by-value, `__attribute__((packed))`/`aligned`, layout cross-checked against Clang's `sizeof`/`offsetof` | ✅ |

With the C ladder complete, **Phase C is effectively done** (modulo full-C breadth, C.3): a vendor
register-map header now ingests through L7 → L5 → an R1–R18-clean plan with `bcir-explain`,
behaviour-equivalent to Clang. Next: **Phase D** — the first real driver behind a `channel.json`
plugin (generate/JIT a channel's kernel from an imported register map), closing the heterogeneous-
tower loop.

##### C.1-MVP — the first milestone: a register-map + MMIO file (driver/kernel-relevant C)

The MVP targets exactly the C drivers/kernels need: fixed-width integers, structs/unions, arrays +
pointers, functions, simple control flow, bitfields, `volatile`, MMIO-like resource mapping, and
explicit layout/ABI checks. **Success criterion:** a small C file containing a *register-map struct*
+ *MMIO-style access* lowers to —

1. BCIR resources / claims / phases;
2. an **R1–R18-clean** plan;
3. C output (or an LLVM-backed artifact);
4. behaviour equivalence against Clang on a test harness;
5. a provenance manifest (R13);
6. `bcir-explain` output explaining the chosen realization.

##### C.2 — Generalized C output backend (the *output* seam, already partly built)

Today `lower.c_kernel` emits *per-pattern* kernels (elementwise / gather / reduce / qfixed / strided
/ compensated) and `codegen/` lowers via `llc` per the channel triple. Generalize from per-pattern
kernels to an **arbitrary claim graph**:

- multi-claim functions + call boundaries;
- structs + ABI layout; MMIO / `volatile` lowering; atomics / fences;
- dynamic shapes + guards; multi-channel lowering decisions (the channel-plugin routing contract);
- generated self-check harnesses; **R12 / R17 / R18 attestation**.

The closed loop this delivers — the gate before drivers (Phase D):

```
C input → claim graph → K_BCIR plan → verified C output → Clang behaviour check
```

##### C.3 — Full C (the multi-month horizon)

Clang-compatible parser, complete ABI support, the full preprocessor (macros, `#include`,
conditional compilation, `#embed`, `__VA_OPT__`, `__has_include`), full standard-library compatibility.

#### Phase M — Selective ML operations (in parallel with Phase C, but throttled — not at full speed)

Add ML-specific ops + passes (tensor ops, attention patterns, quantization, layout/packing) *after*
the core lowering + C support are stable enough not to be destabilized. Prototype each in the **Python
oracle first** (cheap iteration, the conformance reference), then port to the **MLIR law** — the exact
dual-rail discipline every existing op followed. It advances alongside C but must never block or
destabilize the keystone.

#### Phase D — Drivers, opcode tables & hardware integration (next major milestone, after C)

The Hardware Description Layer. With verifiable C in hand:

- **Import Linux kernel tables** — register maps, driver structures, PCIe/ACPI data — into BCIR
  (clean now that C can be generated + verified against the source).
- **BCIR-native ISA / opcode / registry representations** — the binary opcode tables, modeled the way
  the claim graph models compute.
- **A dedicated `drivers/` (or `targets/`) folder = a semi-separate JIT kernel generator**, *wired
  into* the compiler but architecturally separable (a deliberate decoupling — the kernel generator
  evolves without churning the core). This is where each hardware **channel** gets its **real
  driver**: the channel *declares* the backend through the now-stable **plugin boundary** — a
  `channel.json` manifest (`bcir/channel_plugin.py`: target-profile schema, codegen identity, runtime
  signal-provider contract, execution-capability set, calibration-artifact ref, provenance flag) — and
  the driver *generates/JITs* its kernels and supplies its measured calibration that replaces the
  modeled profile. Phase D closes the heterogeneous-tower loop: the modeled `fpga_systolic` /
  `nvme_stream` / `hbm_pim` channels become driver-backed, and a new accelerator joins the tower by
  shipping a manifest (`register_from_manifest` / `discover_plugins`) — no core edit.

#### Phase F — Full language frontends (deferred until the core is rock-solid)

- **C++** — the hardest: templates (two-phase lookup + instantiation), exceptions, RAII, the STL, move
  semantics, the complex ABI (Itanium mangling, vtables, EH tables) — Clang-level completeness. Defer
  until the C path + core are rock-solid; depends on the C frontend + a far richer claim-graph type
  system.
- **Python** — start as a **transpiler / lifter** (the analyzable subset — array/numeric code — →
  claim graph) rather than a full frontend, then grow toward full parser + semantics, the dynamic
  features (classes, decorators, generators, `async`/`await`), a CPython-compatibility layer (the
  C-API / object model), and ecosystem integration (NumPy/pandas array semantics on the resource/claim
  model — the natural fit for cost-as-IR). Cheaper to *start* (the transpiler) than the C++ work.

#### Phase L — The ML library + ecosystem (the payoff, on top of the compilers + kernels)

Once C support + lowering + drivers + kernel infrastructure are in place, build the ML library *on
top* of them — applying the same **"take and compress only what we need"** strategy used for the Linux
C files: a *compressed extraction* into the BCIR claim-graph + channel model (not a wholesale
dependency) from GCC, TensorFlow, PyTorch, Apache, pandas, NumPy, scikit-learn, XGBoost, JAX, Keras,
XLA, SPIR-V, ONNX, Cassandra, hybrid SQL/NoSQL, and vector databases. Each contributes the kernels /
ops / data structures BCIR needs, lowered to the same K_BCIR plan + GEM execution and orchestrated
across the channel tower.

**The through-line.** Frontends produce claim graphs; drivers populate channels; the ML library
composes them — all decomposing to the *same* binary K_BCIR optimization + GEM Binary-Graph execution,
hardware-agnostic by construction. Native instruction selection stays gated
(`BCIR_NATIVE_OBJECT_GATE.md`): the frontends feed the resident backend + the per-channel JIT
generator, never a hand-rolled isel.

### 5.8 Oracle → C: the plug-in C compiler's remaining ports + missing infra

The C frontend is porting from the oracle prototype to `runtime/c/` (§3 row). Beyond the ladder
stages, the loop **`C input → claim graph → K_BCIR plan → verified C output → Clang behaviour check`**
needs these components that **do not yet exist in `runtime/c/`** (researched against the oracle +
the existing C twins):

- ✅ **Ladder stages L6–L8 ported** — L6 control flow (`if`/`while` → structured body + `compose.Cond`),
  L7 a real C preprocessor (`bcir_cpp.c`: object/function/variadic macros, conditionals, `#include`, C23
  `#embed`), L8 ABI (struct return-by-value, `packed`/`aligned`, layout cross-checked vs Clang) all lower
  on the C twin with the six-artifact gate + Python↔C parity. The full L1–L8 ladder is complete (§3 row).
- ✅ **Verifier R1–R18 in C (`bcir_verify.c`)** — the runnable LangRef laws over the claim graph +
  plan + StreamPack, the C twin of `bcir/verify`: R1–R8 module/claim laws (incl. **R6** lane↔stride
  legality), R9 plan legality (`bcir_verify_plan`), R10–R11 StreamPack well-formedness
  (`bcir_verify_pack`, over the hydrated bytes), R12 lowering-contract, **R13 provenance digest**
  (`bcir_provenance_digest`, FNV-1a over the claim graph), R14–R16 vacuous for the scalar subset, R17
  accuracy (integer/Q-fixed exact), R18 call-graph integrity. `bcir_cfront` runs R1–R8+R18 at compile
  time; the **R9/R10–R11 verdicts are checked in the closed loop** (`test_cfront_loop.c`).
- ✅ **K_BCIR planner in C (`bcir_plan.c`)** — a compact, freestanding scalar planner (per-claim
  realization width + an integer cost; total cost) lands the *plan* in `runtime/c/`. The full cost
  model / min-plus / RCSP stays on the MLIR/C++ law rail; this is the driver-embeddable seam that
  drives hydration. *Future:* richer cost model / a `bcir-opt` bridge for the full optimizer.
- ✅ **Claim-graph → StreamPack hydration in C (`bcir_hydrate.c`)** — the `gem.hydrate` step (plan →
  StreamPack segments), freestanding + bounds-checked. **The loop now closes with no Python:**
  `C source → bcir_cfront → bcir_plan → bcir_hydrate → bcir_exec` runs the compiled artifact end to
  end (`test_cfront_loop.c`; gated in `tools/c/check_runtime.sh` + `test_c_cfront.py`).
- ✅ **C.2 attestation in C** — `bcir_cfront` stamps the emitted verified-C with an attestation header
  naming the discharged laws (R1–R8 + R18 clean, R9/R10–R11 checked in the loop, R12 lowering-contract,
  R17 accuracy) and the **R13 provenance digest** — the same digest the compile→execute loop reports
  (a reproducible manifest across the two C entry points; the oracle does this in `pipeline.py`).
- ✅ **Atomics/fences** — `__atomic_fetch_add/sub/xor` → `ATOMIC_ADD/SUB/XOR` and
  `__atomic_thread_fence`/`__sync_synchronize` → `BARRIER`, lowered on **lane A** (R6 admits lane A for
  a scalar atomic counter as well as a RANDOM scatter-atomic; the atomic/barriered hazard discharges
  R5), emitted back as the matching seq-cst builtins, and **behaviour-equivalent under Clang** on
  independent copies of the same seeded counter (`cfront_atomic.c`, both rails `ok=1`).
- ✅ **Compare-and-swap** — `__sync_val_compare_and_swap` / `__sync_bool_compare_and_swap` →
  the `CMPXCHG` opcode: a 3-read claim (ptr, expected, desired) on lane A, emitted back as the
  matching `__sync` CAS builtin, behaviour-equivalent under Clang (`cfront_cmpxchg.c`). *Still to
  port:* **dynamic shapes** (`compose` dynamic bound guards).
- ✅ **Multi-channel lowering decision in C** (`bcir_channel.c`) — the C twin of `bcir/channels`'
  routing seam: a `channel.json` reader + `bcir_claim_required_caps` / `bcir_channel_suits` /
  `bcir_channel_route` (the cost-free plan-time backend pick — most-specialized eligible channel,
  tie-broken by name), so a driver routes each claim to its backend with no Python. Python↔C
  parity-gated against the new `route_claim` (`test_c_channel.py`; `channels/example_{cpu,tpu,pim}`
  exercise the plugin/universal/legacy paths). The full K_BCIR **cost**-based pick (`orchestrate`)
  stays on the cost-model rail; this is the eligibility + static route a driver makes first.
- **Type-model breadth** — ✅ `typedef` (scalar/pointer/aggregate aliases, incl. `typedef struct
  {...} N;`), ✅ `enum` (enumerators folded to their integer values at parse time), and ✅ full
  `union` layout (members overlap at offset 0; size = the widest member) all lower on both rails,
  parity- + Clang-equivalence-gated (`cfront_typedef.c`, `cfront_enum.c`, `cfront_union.c`). ✅
  **Function pointers** (a `typedef`'d `RET (*name)(PARAMS)` passed as a parameter and called
  indirectly — the HAL dispatch pattern; the indirect call lowers to a `c.call.indirect` claim that
  R18 treats as an opaque external edge, while direct calls in the same function still travel the
  call graph; `cfront_funcptr.c`), ✅ the **ternary operator** `?:` (`cfront_ternary.c`), and ✅
  **multi-dimensional arrays** (a 2D array parameter `uint32_t m[4][8]` decays to a flat element
  pointer with a recorded shape, and `m[i][j]` flattens row-major to the linear index `i*8 + j` —
  reusing the 1D pointer/index/load machinery on both rails; `cfront_array2d.c`, runs the full
  execute loop), and ✅ **function-pointer struct members** (the HAL dispatch table -- `o->fn(args)`
  fuses the member access + call into one `c.call.imember:<field>` claim emitted verbatim as
  `o->fn(args)`, so no 8-byte function-pointer value rides in the 4-byte value model; R18-opaque;
  `cfront_dispatch.c`) now lower on both rails. ✅ **array-of-row pointer declarators**
  (`uint32_t (*m)[8]` — the row pointer a 2D array decays to; modeled as the equivalent multi-dim array
  param so `m[i][j]` flattens row-major to `i*8 + j`, reusing the 2D machinery; `cfront_widerow.c`) —
  the vendor-header declarator form, now lowering on both rails.
- ✅ **Phase D — real register-map headers driven end-to-end** — vendor-style headers + drivers
  ingested with no hand-written claim graph, through the full `C → bcir_cpp → bcir_cfront → verify →
  emit → bcir_plan → bcir_hydrate → bcir_exec` loop, both rails agreeing and the emit
  Clang-behaviour-equivalent. Two complementary drivers cover the real-driver surface:
  - **`cfront_driver.{h,c}`** — a DMA-channel map: the *read + decode + call-graph* path (`#include`
    + field macros, typedef/enum/union/bitfields, volatile MMIO *loads*, struct pointers, an R18
    call graph; `claims=30 mmio=1 bf=3 call=2 ok=1`).
  - **`cfront_driver_uart.{regs.h,_uart.c}`** — a UART map: the *write + control-flow* path that
    real drivers live on — MMIO register **writes** (`u->BRR/CR/DR =`) and a **bounded status-poll
    loop** (L6 `while`+`if`), plus the union/bitfield/enum/typedef ABI (`claims=22 bf=4 ok=1`). Every
    function (incl. the MMIO + control-flow `uart_send`) is Clang-equivalent; the straight-line entry
    executes R9/R10–R11 clean.

  Together they are the demonstration the L1–L8 + verifier/type/atomics/channel work was built toward.

> Channels are already a real plugin boundary (`bcir/channel_plugin.py`: target-profile schema,
> runtime signal-provider contract, codegen identity, calibration artifact, execution-capability set,
> simulator/model/provenance flag — #262); the C-side consumer of a `channel.json` is the
> multi-channel lowering decision above.

### 5.9 Oracle → C: the road from a driver-subset frontend to a C23 *replacement* compiler

The plug-in C path is now a **driver-oriented, no-Python, production C rail** — far past "C lowering"
or a kernel emitter. In the current repo it is a freestanding stack: a C **preprocessor**
(`bcir_cpp.c`), a C **frontend** (`bcir_cfront.c`), a C **claim-graph IR** (`bcir_cir.h`), an
**R1–R18 verifier** (`bcir_verify.c`), a **planner** (`bcir_plan.c`), **StreamPack hydration**
(`bcir_hydrate.c`), a deterministic **executor** (`bcir_exec.c`), and the **`bcir-cc`** cc-like
driver — plus target-ABI modeling, a wide C23/embedded-driver language surface, a full Clang-grade
**diagnostics engine** (`bcir_diag.c`: caret renderer, JSON, fix-its, include-stack origin, recovery
reports), an **`--fallback` route-to-LLVM contract**, a module-scope **effect/commutation analysis**
(`--emit-effects`), real register-map / UART / DMA driver fixtures, and Python↔C parity gates on
every stage (`docs/PARITY.md` § *Python ↔ C frontend twin*).

It is best described as a **freestanding embedded/driver-subset C23 compiler *candidate*** with a
widening language surface and a fallback contract — a *productionizing* freestanding driver/kernel-
subset compiler with strong BCIR parity gates, real driver examples, and a cc-like front end. It has
crossed the threshold from *prototype* to **usable compiler substrate for controlled driver
development**. It is **not yet** a complete *hosted* C23 compiler or a general Clang/GCC replacement:
the remaining work is the classic hard-compiler work — full C semantics, a full hosted environment,
full ABI/object generation, conformance suites, and integration with a resident backend/linker
toolchain — **not** the BCIR optimizer (that is already complete on the MLIR/C++ law rail). We
complete it **systematically, one PR-sized chunk at a time**, in four phases.

**Phase 1 — Driver-subset compiler, productionized** (the near-term milestone — make the existing
subset behave like `cc` on a small multi-file driver project):
- ✅ **Fix the oracle-rail CLI include-path gap** (`python -m bcir.frontends.cfront`): the file's own
  directory is on the search path, so a driver with sibling `#include "regs.h"` headers compiles
  directly — plus `-I` / `-D` / `-U` / `-std=` / `-E` / `-o` (the Python preprocessor resolves headers
  from disk via a search path, not just an in-memory mount).
- ✅ **`bcir-cc` — the production C compiler driver** (`runtime/c/bcir_cc.c`, the cc-like front over
  the full C rail `bcir_cpp_run_ex → bcir_cfront → bcir_plan → bcir_hydrate`): `-I` (multi-dir) / `-D`
  / `-U` / `-std=` / `-E` / `-o` + `--emit-c` (verified C + C.2 attestation) / `--emit-claimgraph` /
  `--emit-pack` (the entry's hydrated StreamPack, `BSPK`). The C preprocessor gained the dual-rail
  twin of the oracle's search-path/defines (`bcir_cpp_run_ex`: multi-dir `#include`, `-D` predefines,
  **macros persist across nested includes** like `cpp.py`). A driver with sibling headers builds via
  a normal compile command — `bcir-cc runtime/c/cfront_driver_uart.c` → `ok=1`. CI-gated
  (`test_bcir_cc_driver_compiles_and_emits_artifacts` + `tools/c/check_runtime.sh`).
- ✅ **Precise unsupported-construct diagnostics** (`bcir_diag.c`) — a full Clang-grade diagnostics
  engine on the C rail: a caret/underline source renderer, machine-readable JSON, fix-it hints,
  `In file included from …` include-stack origin, and multi-diagnostic panic-mode recovery reports,
  each **byte-identical to the oracle's `diagnostics.py`** (the `#diag` blocks in `check_runtime.sh`;
  `test_diagnostic_*_dual_rail`). The frontend names exactly what it can't do, with spans + notes.
- *Remaining Phase 1:* provenance-manifest artifact emission (the R13 digest is already stamped in the
  C.2 attestation; a standalone `--emit-manifest` is the gap). **Exit:** a small multi-file driver
  project builds via a normal compile command and passes the behaviour check — **met on the production
  rail** (the diagnostics engine + `bcir-cc` driver close it).

**Phase 2 — Freestanding C23 compiler (embedded/kernel subset).** Close the concrete language gaps real
headers/drivers hit. ✅ **ternary `?:`** — lexed (`?` added to the oracle punct set; the C lexer's
single-char fallback already had it), parsed as a conditional expression (right-associative, layered
over the binary grammar on both rails), lowered to a scalar `c.select` claim, and emitted as the real
`(cond ? a : b)` — Clang-behaviour-equivalent (`cfront_ternary.c`, both rails `claims=13 ok=1`). ✅
**function pointers** — a `typedef`'d `RET (*name)(PARAMS)` passed as a parameter and called
indirectly (HAL dispatch); the indirect call lowers to a `c.call.indirect` claim (reads: the pointer
value then the actuals), R18 leaves it an opaque external edge while direct calls in the same function
still resolve through the call graph, and the emit calls through the pointer verbatim — both rails
`funcs=2 claims=2 call=2 ok=1`, Clang-equivalent (`cfront_funcptr.c`). ✅ **integer casts**
(`(type)expr` — a cast binds at the unary level on both rails; in the 32-bit-unit value model a
narrowing cast to an unsigned fixed-width type masks, exactly matching Clang's integer promotion, so
it lowers to a `c.cast:<width>` claim and emits `(type)expr`; `cfront_cast.c`, both rails `claims=12
ok=1`, executes the full loop). ✅ **the `signed` type specifier** (`signed char` / `signed int` /
`signed long`, as a declarator and a cast): the C twin recognized `unsigned` but not `signed` in its
declaration- and cast-type-start detection, so `signed char sc = (signed char)a` was rejected (routed
to fallback) while the oracle accepted it -- a rail disagreement found by the dual-emit sweep. Fixed by
adding `signed` to the twin's scalar-type table (a modifier; the base sets the width). `signed` alone
(signed int with no base) stays a fallback on both rails. `#signedty`/`cfront_signed.c`. *Open (a
separate, pre-existing twin miscompile the sweep also surfaced):* a **signed comparison against an
integer literal** -- `x < 0` for a signed `x` -- compiles to an unsigned compare on the twin (it types
int literals `uint32_t`, so `int32_t < uint32_t` promotes to unsigned), affecting plain `int` too.
✅ **multi-dimensional arrays** (a 2D array parameter
`uint32_t m[4][8]` decays to a flat element pointer + a recorded shape; `m[i][j]` flattens row-major
to `i*8 + j` (Horner) on both rails, reusing the 1D index/load machinery; `cfront_array2d.c`, both
rails `claims=21 ok=1`, runs the full execute loop) + ✅ **function-pointer struct members**
(the HAL dispatch table: `o->fn(args)` fuses member access + call into one `c.call.imember:<field>`
claim emitted verbatim as `o->fn(args)` -- no 8-byte funcptr value rides in the 4-byte value model,
so no typed temporaries needed; R18-opaque; `cfront_dispatch.c`, both rails `claims=3 call=2 ok=1`).
✅ **array-of-row pointer declarators**
(`(*m)[8]` — the row pointer a 2D array decays to, modeled as the equivalent multi-dim array param;
`cfront_widerow.c`); *still to port:* the comma
operator, `typeof`, compound literals, integer promotions + usual arithmetic
conversions; ✅ **pointer dereference** (`*p` -- a one-read deref load, previously unsupported on the
C rail entirely -- and `*(p + i)`, the pointer-arithmetic spelling of `p[i]`, routed through the
index/load machinery on both rails; `cfront_deref.c`); ✅ **store through a pointer** (`*p = v` / `*p
OP= v` / `*(p + i) = v` -- the write counterpart; the C twin parsed `*p` only as a read, so a store
through a pointer failed. Now both rails lower a deref store -- an offset-0 `imm=[0,size]` store for
`*p` (the member-store shape), the indexed `p[i]` store for `*(p + i)`, a load+op+store for `OP=` --
verified on independent buffers since these mutate through `p`; `#ptrstore`/`cfront_ptrstore.c`); the
rest of pointer-arithmetic completeness;
✅ **`sizeof`** (`sizeof(type)` / `sizeof expr` folds to
a compile-time constant -- the type/operand's static size, operand not evaluated; both rails agree
via the shared scalar table + struct/union layout, Clang-equivalent, `cfront_sizeof.c`) + ✅
**`_Alignof`/`alignof`** (the type's alignment from the same layout model, type-name form only;
`cfront_alignof.c`, both rails `claims=9 ok=1`, runs the loop); still
`typeof`, the comma operator; ✅ **logical `&&` / `||`** (the condition idiom -- the C rail parsed
only the bitwise `&` / `|`, so a user-written `a && b` did not parse; added at the correct precedence,
emitted verbatim with Clang short-circuit; `cfront_logic.c`) + ✅ **unary `+`** (a no-op, on both
rails); ✅ **increment / decrement** (`i++` / `++i` / `i--` / `--i` in
statements + `for` clauses -- the value-discarded form desugars to `i = i ± 1` on both rails, the
loop-counter idiom; `cfront_incdec.c`); the rest of control flow — ✅ **`for`** (desugared onto
the existing `while` machinery on both rails: `init; while(cond){ body; step }`, the step lowered at
the loop-body end; `cfront_for.c`), ✅ **`do/while`** (a `WhileNode` `test_at_end` flag / the C
`c.loop.test` marker placed at the loop-body bottom -- body runs at least once), ✅ **`break`** (a
`BreakNode` / `c.break` marker emitted as `break;`, correct in every loop form), and ✅ **`continue`**
(a per-loop `goto __cont_<id>;` + a `__cont_<id>:` label placed at the loop's continue point -- before
the `for` step / the `do/while` bottom test / at the `while` body end -- so it runs the step in a
`for`, which the naive `while(1)` desugar would skip; `cfront_continue.c`), and ✅ **`switch`/`case`**
(desugared to a nested if/else-if chain on both rails: a clause's labels OR together for the shared
`case A: case B:` pattern, a top-level `break;` terminates the clause, `default` is the final `else`;
enum cases fold to their values; `cfront_switch.c`), all Clang-equivalent, and ✅ **interleaved
top-level declarations** (a `typedef`/`enum`/`struct`/`union` defined *between* functions now parses
on the C rail too -- one top-level loop with a `try_top_decl()` helper, matching the oracle's already
interleaving `parse_unit`; `cfront_interleave.c`), and ✅ **`goto` + labels** (the driver
error-cleanup pattern -- `goto done;` / `done:;` carried as emit-only markers like break/continue,
which already lower to a goto, so they emit verbatim and stay Clang-equivalent; the mutable
accumulator is a real C local so skipped updates match; `cfront_goto.c`), and ✅ **empty statements**
(a bare `;` -- the body of `for(...);` / `while(...);` / `if(c);` and stray `;;` between statements --
consumed on both rails as a no-op that emits no claim, so behaviour is unchanged + Clang-equivalent;
previously both rails rejected it (`unexpected ;`); `#emptystmt`/`cfront_emptystmt.c`), and ✅ **`static` local
variables** (static storage duration -- persists across calls, a once-only constant initializer baked
into the `static T name = init;` declaration so it lowers no init claim; the driver counter/accumulator
pattern; `cfront_static.c`, both rails `claims=2 ok=1`, runs the loop), and ✅ **file-scope lookup
tables** (a `static const T NAME[N] = {...}` global indexed at runtime -- the driver calibration /
jump-table pattern; the global lowers to a read-only data resource, an access `NAME[i]` is an indexed
load emitted by name, and the global is referenced (not redeclared, defined in the source);
`cfront_global.c`, both rails `claims=5 ok=1`, runs the loop), and ✅ **compound assignment**
(`name OP= expr` -- the register / bit-manipulation idiom `reg |= MASK`, `flags &= ~BIT`; each
desugars to `name = name OP expr` = a binary op + a copy on both rails, the C lexer gaining the
`+= -= *= /= %= &= |= ^=` tokens; `cfront_compound.c`, both rails `claims=11 ok=1`, runs the loop),
and ✅ **MMIO register read-modify-write** (`dev->reg OP= expr` -- the set/clear-control-bits idiom,
the most common driver operation; a compound assignment to a volatile struct member is an ordered
MMIO load + a binary op + an ordered MMIO store; `cfront_rmw.c`, both rails `claims=8 mmio=3 ok=1`),
and ✅ **MMIO bitfield write** (`r->field = v` for a named bitfield -- read the storage unit, insert
the masked bits (`c.bf.set`), store back; bitfield reads (`c.bf.get`) already worked, this completes
the write side register maps need; `cfront_bitfield.c`, both rails `claims=5 mmio=2 bf=1 ok=1`), and
✅ **bitfield compound-assignment** (`r->field OP= bits` -- read the field via `c.bf.get`, op, re-insert
via `c.bf.set`, store; one unified member-assign path now covers plain/bitfield x plain/compound;
`cfront_bfcompound.c`, both rails `claims=15 mmio=5 bf=3 ok=1`);
and ✅ **C11 `<stdatomic.h>` atomics** (the `_Atomic` type qualifier parses + round-trips like
`volatile`, and the generic functions `atomic_fetch_add`/`atomic_fetch_sub`/`atomic_fetch_xor` /
`atomic_load` / `atomic_store` / `atomic_exchange` on an `_Atomic` object lower to the BCIR ATOMIC
opcodes on lane A with the atomic hazard -- emitted as the C11 functions themselves, which accept an
`_Atomic*` where the GCC `__atomic_*` builtins do not; `cfront_atomic11.c` / `cfront_atomic_xchg.c`,
both rails `ok=1`, run the loop; C11 compare-exchange remains -- it needs the address-of operator);
and ✅ **scalar file-scope globals — read *and* write** (a `static`/file-scope scalar global both
read and assigned across functions, lowered to a named read-only data resource with `c.copy`-to-global
writes, referenced by name in the emit; `cfront_global_rw.c`, both rails parity- + Clang-gated), with
its module-scope **effect/commutation analysis** (the C twin of `pipeline.own_footprint` + `commute`:
per-function read/write global footprints, callee effects folded transitively, the pairwise commute
matrix — `bcir-cc --emit-effects`, byte-identical to the oracle; `cfront_effects.c`). **Still to
port** — the genuinely-remaining Phase-2 language/infra work:
  - *language:* ✅ **the comma operator in the for-step** (`for(...; ...; i++, j--)` — two-pointer /
    reversal loops + parallel-counter updates; each comma-separated step element, inc/dec or plain /
    compound assignment, runs in order, on both rails; the twin's `p_simple` gained scalar/pointer
    compound-assign to match, closing a pre-existing single-`a += b`-step gap too;
    `#commastep`/`cfront_commastep.c`); the comma operator in general expression position (blocked on
    the twin, which has no assignment-in-expression), `typeof`, compound literals; ✅ **full integer promotions + the usual
    arithmetic conversions** — **dual-rail** (oracle `ctype_model.promote_int`/`usual_arith_int`/
    `int_literal_type` + `lower._bin_result_type`; C twin `tempi`/`rid_int`/`uac_i`/`lit_int_type` +
    `tty`/`ctype_str` rendering the true fixed-width type, with `is_signed` threaded through the resource
    model): every temp is typed by its true (width, signedness), so a signed `int` divide / remainder /
    right-shift / comparison emits signed C (not the old flat `uint32_t`) and `int + long` widens to
    64-bit — behaviour-equivalent to Clang over the *full* signed range, the case the old unsigned-32
    model got wrong (`test_integer_promotions_and_uac_oracle`; `#intpromote` runs the twin's `--emit-c`
    against Clang on 300k full-range inputs). *Scalar* operands are covered; pointer-element signedness
    (the twin's pointer model carries pointee *width* but not yet signedness) is a follow-on. ✅
    **array element stores** (`a[i] = v` / `a[i] OP= v` through a pointer/array param -- the driver
    buffer-fill / scatter idiom) now lower on the C twin too (a 3-read `c.store`, emitted as `a[i] = v`;
    the oracle already had them), dual-rail parity + a source-vs-twin buffer differential
    (`#astore`/`cfront_arraystore.c`); ✅ **`<<=` / `>>=` shift-compound-assign** (the lexer emits them
    as 3-char tokens; a shared `is_compound_op`/`compound_binop` desugars `lv OP= e` → `lv = lv OP e` on
    scalar / member / array lvalues -- `#shiftassign`/`cfront_shiftassign.c`); the rest of
    pointer-arithmetic completeness — ✅ **pointer mutation** (`p++` / `++p` / `p += n` / `p -= n` on a
    pointer lvalue lowers to a single `c.ptradd`/`c.ptrsub` claim, emitted `p += n;` so C scales by the
    element size — the old integer-typed desugar truncated the pointer; dual-rail, `#ptrarith`/
    `cfront_ptrarith.c`); (`p - q` pointer difference works as an integer result) + an object/provenance
    model; ✅ **cross-clause
    `switch` fallthrough** -- both rails now emit a *real* C `switch` (`cast.Switch`/`SwitchNode` +
    folded case labels, the discriminant lowered once, `break` preserved) instead of the old if/else-if
    desugar, so a `case` without `break` falls through exactly as C specifies (and an MMIO discriminant
    is read once, not per case) -- `#switchfall`/`cfront_switchfall.c`; ✅ **designated initializers**
    for a file-scope **dispatch / jump table** —
    `static const T NAME[N] = {[OP]=v, ...}` with enum-indexed designators and a gap that zero-fills
    (§6.7.10) — now parse on both rails (oracle `cparse._global`; the twin references the table by name,
    defined in the source, so the emit is Clang-equivalent — `#designated`/`cfront_dispatch_table.c`);
    ✅ **local aggregate
    initializers** (`struct cfg c = {.baud=9600}` / `union` / `T a[N] = {…}`, positional + `.field=` /
    `[i]=` designators) — lower to a `= {0}` zero baseline + a store per initialized member/element
    (reusing the member/array store path, so uninitialized members zero-fill, §6.7.10): the oracle does
    struct/union/array (`cast.AggInit`/`cparse._init_value`/`lower._agg_init`;
    `test_local_aggregate_initializers_oracle`), and the **C twin** now does **struct/union**
    (`agg_init` + a `zinit` resource flag → `= {0}`; `#aggregate`/`cfront_agginit.c`) **and arrays** —
    ✅ a **local array declarator** `T a[N]` (a scalar-element resource of count N, emitted `T a[N]`,
    element access via the index load/store, `arr_init` for the positional + `[i]=` aggregate
    initializer; `#localarr`/`cfront_localarray.c`) — both dual-rail parity + Clang-equivalent.
    ✅ **multi-declarator declarations** (`T a = x, b, c = z;` — several comma-separated declarators
    off one type-specifier, incl. the canonical two-variable loop init `for(unsigned i = 0u, j = n;
    …)`; each declarator lowers to its own storage + copy, identical to separate decls, so the emit is
    Clang-equivalent; `#multidecl`/`cfront_multidecl.c`. The oracle types a per-declarator `*`/`[]`
    shape per declarator; the twin folds `*` into the specifier, so `int *p, q;` is rejected there
    rather than mis-typed — a follow-on. The comma *operator* in a for-step (`i++, j--`) now lowers on
    both rails — see `#commastep` above).
    ✅ **multi-declarator struct/union members** (`unsigned x, y, z;` — several members off one
    type-specifier, including multi-declarator bitfields `unsigned a:3, b:5;`; each lays out exactly as
    if written on its own line, so offsets / `sizeof` + member access match Clang on both rails. The
    member-declaration twin of the multi-declarator locals above — `#structmulti`/`cfront_structmulti.c`,
    differential asserts `sizeof` + per-member round-trip == Clang).
    ✅ **nested struct member access** (`o.pos.lo` / `dev->ctrl.flags` — a struct-in-struct / sub-
    register-block; the oracle already flattened the `.`/`->` chain to a single offset access, and the
    C twin's `field` now carries a sub-struct index so a descent helper accumulates the byte offset
    through each hop — read, plain / compound store, and nested bitfields all match. `#nestmember`/
    `cfront_nestmember.c`, differential asserts `sizeof` + nested read/write == Clang. *Follow-on:*
    pointer-member chains `o.p->v` and nested funcptr dispatch).
    *Follow-on:* **compound literals** (`(struct S){…}`);
  - *storage/linkage:* ✅ **`extern`** (recognized as a storage-class specifier on both rails -- consumed
    like `static`; an `extern T g;` global is referenced by name, defined in another TU, so the emit is
    Clang-equivalent once linked against the definition -- `#extern`/`cfront_extern.c`); ✅
    **`_Thread_local` / `thread_local`** (recognized + consumed like `static`; a thread-local global
    behaves as a global under the deterministic single-thread executor -- `#threadlocal`/
    `cfront_threadlocal.c`); tentative definitions, internal/external linkage, broader (aggregate /
    non-const-init) globals,
    extern function prototypes;
  - *memory model:* ✅ **`restrict`** (`restrict` / `__restrict` / `__restrict__` recognized as a
    pointer qualifier on both rails and consumed -- the value model carries no aliasing facts, so the
    emit is behaviour-equivalent; the perf-sensitive driver/DSP idiom -- `#restrict`/`cfront_restrict.c`);
    broader alias/effect propagation (the module-scope analysis above is the seed), atomic
    compare-exchange, a fuller C memory model;
  - *infra:* ✅ **scalable IR allocation (no fixed `BCIR_MAX_*`)** — the C IR's per-unit function list
    and every per-function array (params, calls, static locals, resources, claims) now grow
    geometrically (`bcir_cir.h`/`bcir_cfront.c`); the old `BCIR_MAX_PARAMS 8` / `BCIR_MAX_CALLS 32` /
    `BCIR_MAX_FUNCS 16` + the 256-resource / 4096-claim per-function caps are gone, so a real
    translation unit of any size lowers. Gated by a cap-busting unit (43 functions, a 12-param
    function, a 40-call aggregator, a 7500-claim function) that compiles clean and matches the oracle
    (`#scale` in `check_runtime.sh`; `test_scalable_ir_no_fixed_ceilings`), valgrind-clean across the
    realloc paths. The twin's **parser-state** caps are gone too: ✅ struct defs (was `s[16]`),
    file-scope globals (was `gv[16]`), typedefs (was `td[64]`), enum constants (was `ec[256]`) and
    locals (was `env[256]`) all grow geometrically (reused across compiles via a save/restore around
    the static `CC`), so a real header (20 structs / 25 globals / 300 locals) lowers and matches the
    oracle (`#pscale`; `test_scalable_parser_state_no_fixed_caps`; valgrind-clean over multi-compile
    reuse). *Still:* a fuller per-target calling-convention/varargs/aggregate ABI on top of the landed
    **data-model layout matrix** (`--target`, §5.8), real object/dependency output via a resident
    backend, and the per-struct member array (`f[64]`, embedded; guarded) + token buffer (`16384`).

  **Exit:** BCIR compiles a freestanding embedded C test suite + a nontrivial driver codebase with no
  hand-written claim graphs. ✅ **Composition checkpoint:** `cfront_integration.c` -- a realistic driver combining
typedef + enum + an MMIO register-map struct, a `switch` over an enum status, a `static` fault
counter, a `goto` cleanup path, integer casts, a 2D bank lookup, and an inter-procedural call graph
-- is ingested with no hand-written claim graph; the two rails agree on the entry's summary and
*every* function is Clang-behaviour-equivalent, proving the Phase-2 features compose, not just pass
in isolation. ✅ **Register-map composition checkpoint:** `cfront_regdriver.c` -- a realistic device
driver exercising the whole register surface together (a `switch` over a status field, a bitfield
write, a bitfield read, a register read-modify-write, a file-scope lookup table, an `enum`, and a
`static` counter; both rails `claims=34 mmio=5 bf=1 ok=1`, Clang-equivalent) -- proves the register-map
chunks compose into a real driver.

**Phase 3 — Hosted C23 compiler candidate.** libc-header compatibility; full preprocessor (predefined
macros — `__FILE__`/`__LINE__`/`__DATE__`/`__TIME__` + `__STDC_HOSTED__`, the `#line` directive, the
`_Pragma` operator, variadic macros (`__VA_ARGS__`/`__VA_OPT__`), and the
`__has_include`/`__has_attribute`/`__has_builtin`/`__has_c_attribute` feature-test macros **done**
(dual-rail; `__DATE__`/`__TIME__` frozen by `SOURCE_DATE_EPOCH`, `_Pragma`/`#pragma` lowering no-ops,
`__FILE__` carries the driver's real source path, `__has_include` resolves against the search path,
`__has_attribute` reports the L8 ABI attributes); C-twin `__has_embed` eval, the full translation
phases next); lexer/parser breadth — **string + character literals** (dual-rail: the lexer tokenizes
`"..."` with escape decoding; `sizeof "..."` folds to the char-array length; a literal materializes as
an anonymous read-only `char[]` global that decays to `const char *` — indexing reads a byte, the bare
literal is the pointer — Clang-equivalent via inlining the literal in the emit. **Done since:**
**character constants** `'c'` (a single char is its byte value as a signed char, an escape decodes to
one byte, and a multi-character `'AB'` packs big-endian like Clang/GCC — folded to a `c.const`); a
**string-literal table** in the C twin that holds the full spelling out-of-band (so a literal of any
length inlines faithfully — the old 32-byte resource-name cap is gone) with **dedup** (identical
literals in a function share one global); and **adjacent-literal concatenation** `"a" "b"` (C
translation phase 6 — `sizeof` folds across the pieces, which stay adjacent in the emit so a hex/octal
escape never merges with the next piece's leading digit); and **wide/UTF literal prefixes**
`L`/`u`/`U`/`u8` on character + string literals (a bare prefix letter stays an identifier; a prefixed
character constant keeps its code-point value; a prefixed string has the element width of its character
type — `wchar_t`/`char32_t` = 4, `char16_t` = 2, `char`/`u8` = 1 — so `sizeof` scales and the prefix is
preserved in the emit). ✅ **floating-point (minimal core)** — `float`/`double` types, decimal float
literals (`1.5`/`1e10`/`.5`/`3.14f`, with the `f`/`F`/`l`/`L` suffix), the four arithmetic operators
(which propagate the wider float type) + comparisons (which stay int), and float params/locals/returns
(`cfront_float.c`). The design keeps the **two-truth line intact**: a float value lowers to a
type-annotated *scalar* claim (dataflow + an integer cost) and the emit renders **real** float C, so
the actual IEEE-754 math is delegated to the resident backend (BCIR never computes or decides on a
float value) — the deterministic integer/Q-fixed plan + executor core is untouched, and no float
reassociation. The one emit change is threading each temp's real type (float/double) instead of
assuming `uint32_t`. The pp-number lexer was also corrected to span float exponents/suffixes on both
rails. ✅ **int↔float conversions** — explicit casts `(float)i` / `(double)i` / `(int)f` (a cast to a
floating type yields a float temp; `(int)f` truncates) and the implicit usual-arithmetic conversions in
mixed int/float expressions (`cfront_floatcast.c`); the equivalence harness feeds integer scalars below
2³¹ so the unsigned value model agrees in sign with a signed int→float cast. ✅ **hex-float literals**
(`0x1.8p3` etc. — lexed + folded on both rails; `cfront_hexfloat.c`), ✅ **`<math.h>` calls** (the
library-call surface: a `<math.h>` function lowers to a type-annotated scalar claim emitted as the real
call, IEEE math delegated to the resident backend — incl. 64-bit-integer *results* like `llround` and
pointer out-params like `frexp`/`modf`; `cfront_mathh{,_long,_mixed,_ptr}.c`), and ✅ **preprocessor
comment stripping** (translation phase 3, on both rails — a `* /` inside a comment no longer glues to
`*/`; `cfront_comments.c`) are all landed + dual-rail-gated. *Next (float follow-ons):* `long double`,
`_Complex`, `_Decimal`. Then: variadic functions +
varargs ABI; system headers + compiler builtins; debug/unwind info; linker/build-system integration;
Csmith + GCC-torture differential gates. **Exit:** BCIR compiles meaningful hosted C and either matches
Clang or emits a clear unsupported-feature diagnostic.

**Phase 4 — General replacement compiler.** Several Phase-4 foundations have already landed on the
production rail: ✅ **Clang-grade diagnostics** (spans, include-stack notes, fix-its, error recovery,
machine-readable JSON — `bcir_diag.c`, dual-rail), ✅ a **cross-platform data-model ABI matrix**
(`--target`, six targets, layout cross-checked vs Clang — §5.8), ✅ a module-scope **alias/effect
analysis** (`--emit-effects`), and ✅ an **LLVM-backend fallback contract** (`bcir-cc --fallback`:
clean 0 / dirty 1 / route-to-LLVM 2, the supported subset pinned to the oracle). ✅ **native struct
member arrays** `s.arr[i]` (a DMA buffer / FIFO / packet body -- the 1-D array member): originally a
silent miscompile (the oracle emitted `b[idx]`, indexing the struct, while reporting `is_clean`), first
made safe by routing to fallback (#380), and **now lowered faithfully on both rails** -- the access
carries the index AND a `(member byte offset, element size)` imm, so the element lands at `&base +
member_off + i*elem_size` (the emit a memcpy at that offset), distinct from a plain `base[idx]` (no imm)
and a plain member (no index). Read, write, compound, and a narrowing `uint8` element all match Clang +
structural parity (`#memberarray`/`cfront_memberarray.c`, differential asserts `sizeof` + access). The
oracle `_LV` already carried both `byte_off` and `idx`; the twin's `field` gained an `arr_count`. A
follow-up closed an oracle-emit bug the aggregate sweep found: a member array at **offset 0** (the first
member) had its access gated on `byte_off` *truthiness*, so it collapsed to an invalid `struct[idx]` --
the `_LV` now carries an explicit `member` flag so the `(offset, size)` imm rides even at offset 0. It
hid because the per-fixture differential compiles the *twin's* emit; a new test
(`test_member_array_oracle_emit_is_clang_equivalent`) compiles the **oracle's** emit via
`compile_unit(check_clang=True)`, pinning it directly -- and the blind spot is now closed **corpus-wide**:
`test_python_c_parity_and_equivalence_across_fixtures` compiles + runs *both* the twin's and the
oracle's emitted C against the source for every fixture, so an oracle-emit regression in any fixture is
caught (it passes today across the whole corpus, confirming every oracle emit compiles + matches Clang).
✅ **multi-dimensional** member arrays too -- `s.m[i][j]` (a grid / matrix in a register block) up to
**3 dims**: the access descends the nested array to the scalar element + the per-dim sizes, flattens
row-major (`r*cols + c`, the same Horner the twin uses for array params), and lands the element at
`&s + member_off + lin*elem_size`; the twin's `field` gained `nadims`/`adims`. *Still fallback (both
rails, pinned by `_FALLBACK_PROBES`):* a **pointer** member indexed `s.ptr[i]` (the loaded pointer is a
4-byte temp in the value model, truncating an 8-byte pointer) and a **>3-dimensional** member array
(the dim table holds 3).
✅ **native multi-dimensional *local* arrays** `T m[A][B]` (a small grid / matrix scratch buffer) up to
3 dims: originally a silent miscompile (the local kept only the outer dim, so the emit was mis-sized
`m[A]` and `m[i][j]` collapsed to `m[i + j]` -- colliding cells -- while reporting `is_clean`), first
made safe via fallback (#381), and now lowered faithfully on both rails -- a **flat** resource of `A*B`
elements carrying the per-dim shape, so `m[i][j]` flattens row-major to `m[i*B + j]` (declared `m[A*B]`,
the same memory layout) using the existing index Horner and the array-access machinery the twin already
had for params. The twin gained a multi-dim local declarator + a shared `array_index` flatten on the
store path; `#localmd`/`cfront_localmd.c` (2-D + 3-D fill/read, differential == Clang). A **>3-D** local
defers to fallback (the dim table holds 3); a 2-D array *param* (a decayed row pointer) was already fine.
✅ **integrity fix — unique C identifiers for reused local names (both rails emit)**: the lowering
flattens scopes, so two source locals that shared a name in disjoint scopes -- the everyday `for(unsigned
i = ...)` in two separate loops -- became distinct rids both named `i`. Both emitters declared `i` twice
at function scope: a C **redefinition**, so the emit did not compile though the unit was `is_clean`. Each
emitter now assigns the second-and-later occurrence a `_N` suffix (`i`, `i_2`, ...) for both the
declaration and every reference -- a fresh variable per scope, which is behaviour-exact for disjoint
reuse (a naive name-merge would corrupt a shadowed value). `#loopreuse`/`cfront_loopreuse.c`, differential
== Clang on both rails. ✅ **and the deeper scope fix — for-loop variable scope**: a `for(unsigned i =
...)` declares `i` scoped to the loop, but the init was lowered into the *enclosing* scope (no push/pop),
so a post-loop read of a shadowed name resolved to the loop var (`return s + i` reading the counter, not
the param) -- a miscompile the `#loopreuse` emit fix had turned from a build error into a silent wrong
answer. Both rails now save/restore the name environment around the loop (oracle: snapshot `self.env`;
twin: an `nenv` mark), so the loop scope is popped and outer bindings are restored. `#loopscope`/
`cfront_loopscope.c` (param-shadow + outer-shadow, differential pins the post-loop value == Clang).
✅ **and the general case — bare-block scope**: a `{ unsigned x = ...; }` compound statement is a scope
too; both rails now save/restore the name env around every `{ ... }` (oracle in `_block`, twin in
`p_block`), so a nested block shadowing an outer `x` read after the block no longer leaks (was a silent
miscompile, `m[0][1]`-class). The oracle also lowers a bare block **inline** now (a `Block` AST node)
instead of wrapping it in an always-true `if(1)`, which dropped a spurious const claim and made the bare-
block claim count match the twin -- so the case is gate-able. `#blockscope`/`cfront_blockscope.c`.
✅ **standing guard for the whole class — the Clang-equivalence fuzzer now spans control flow + scope**:
all of the above (and the #loopreuse / member-array / multi-dim-local finds) were latent because the
front end only validated `is_clean`, never *"does the emit compile and run like Clang?"* on programs
with loops, blocks, or shadowing -- the random generator (`cfuzz.gen_program`) emitted only a single
`return <expr>`. It now also builds an accumulator through random statements: `for` loops that reuse a
counter, bare blocks, and a loop/block that shadows a parameter read after the construct. So
`fuzz_valid(check_clang=True)` (gated, `test_cfront_fuzz`) exercises lowering, the emit, and name
scoping every run; a regression in this class surfaces as a fallback / MISMATCH / build failure rather
than a silent miscompile.
**Remaining Phase-4
work:** a *broad* calling-convention/object ABI matrix (on top of the landed data-model layout); the
optimizer correctness/differential story still owed by the C rail — a full **cost-model bridge into the
MLIR/C++ law rail**, arbitrary claim-graph lowering, RCSP/thermal planning through C, cost-based
multi-channel orchestration, **IPO** (the inter-procedural cost model stays oracle/backend-side today,
by the two-truth line — see `docs/PARITY.md`), broader provenance through optimization; real
**object/debug/unwind emission + linker/build-system integration**; performance-regression +
security/fuzzing programs; user docs + toolchain integration. **Exit:** a usable C23 compiler
distribution, not only a research substrate.

> Conformance scales with the phases: Csmith-style random differential, a GCC-torture / LLVM-C subset,
> WG14/C23 feature tests, preprocessor torture, an ABI matrix (x86-64 / AArch64 / RISC-V), sanitizer
> builds, lexer/parser/preprocessor fuzzers, real-world project builds, and miscompile-reduction tooling.

---

## 6. Next build steps (concrete, prioritized)

In recommended order — each is gated by the generated differential harness + FileCheck
+ the C-runtime parity/fuzz scripts:

1. ✅ **`-bcir-verify` R14 + R15 + R16** (§5.1.1). **DONE** — first-class verifier laws.
2. ✅ **The C StreamPack executor** (§5.2.1). **DONE** — `runtime/c/bcir_exec.{h,c}`.
3. ✅ **The C StreamPack encoder** (§5.2.2). **DONE** — `runtime/c/bcir_encode.c`, byte-
   identical to `bcir.abi.encode` (full C round-trip).
4. ✅ **The verify-cost dimension** (§5.1.2). **DONE** — the 12th axis has a producer on
   both rails (exact/hash O(n); bounds/none free).
5. ✅ **`precision="compensated"` C-kernel + the R17 accuracy law** (§5.2.3 / §5.1). **DONE**
   — dual-rail accuracy contract.
6. ✅ **The C++ `-bcir-verify` law-for-law differential** (§5.1.3). **DONE** — every law
   R1–R18 has a toolchain-rail negative case + a coverage gate.
7. ✅ **Multi-claim bundle (joint) optimization** (§5.3.1). **DONE** — a real 12% matmul
   gain, with search certificates.
8. ✅ **Proof-carrying records** (§5.3.2). **DONE** — `bcir.run --explain/--replay/--reduce`.
9. ✅ **Compositional semantics** (§5.3.3) — **DONE on both rails, end to end.** Oracle
   `kbcir.compose` (Seq/Cond/Call/Function + dynamic shapes + alias/effect modeling + summary
   costs + **RCSP budgets**); law rail `kbcir.func`/`call`/`cond` + the **R18 call-graph law** +
   **`-bcir-compose`** — Seq sum / Cond max+expected / Leaf optimize / Call **inter-procedural
   summary** (plan once, reuse cost-compatible calls, else re-price) / **RCSP-constrained** under
   a `kbcir.budget` (`min M s.t. R⪯B` over the tree — `compose_feasible`), reproducing
   plan_composite's worst/expected/reused/feasible, with a generated compose-rail differential.
   The Tier-1 remainder is **DONE** too: `-bcir-compose` annotates the alias-effect footprint
   (`kbcir.effect_reads/writes`, folded through calls), sibling-call independence
   (`kbcir.commutes_with_prev` = disjoint footprints commute), and the dynamic-shape bound
   (`kbcir.compose_dynamic`; the claim op carries `dynamic`) — `compose_effect.mlir`.
10. ◑ **MLIR ports of the new oracle capabilities** — bundle **detection + joint-reorder**
    ported (`-bcir-bundle`: it now reorders the cost-model columns so a bundle is contiguous,
    re-runs the min-plus shortest path for every legal intra-bundle order, and annotates the
    re-priced `kbcir.bundle_gain` / `bundle_order`), the **proof-carrying decision record
    as IR annotations** (`-bcir-explain`: per claim the candidates weighed + chosen width/
    score, per module the plan total — `proof.explain` on the law rail), and the **replay
    recheck** (`-bcir-replay`: recompute a fresh plan and diff it against the declared
    `kbcir.explain_*` record — `proof.replay` on the IR, annotating `kbcir.replay_reproduced` /
    `replay_mismatches`).
11. **(When a rig is available)** the measured real-silicon replan win (§5.4) — the
    software path is push-button (`tools/silicon/measure_replan.sh`) and CI-exercised in
    degrade mode; the probe enumerates the three gating signals and prints a **rig-ready**
    verdict, so the win lights up the instant a bare-metal host with PMU + RAPL + a userspace
    governor runs the runbook (`HARDWARE_VALIDATION.md`). The top differentiator.
12. **➡ THE NEXT MAJOR MILESTONE — a solid C frontend + backend (§5.7 Phase C).** The keystone
    that unlocks drivers + the Hardware Description Layer (Phase D): a usable Clang-compatible C
    **frontend** (a useful subset → claim graph; the *input* seam that does not exist yet), then a
    generalized, self-verifying C **lowering/codegen** for an arbitrary claim graph (today
    `lower.c_kernel` is per-pattern). Build this *before* drivers — generating + verifying C from
    BCIR is what makes importing Linux kernel tables / register maps / PCIe / ACPI clean. **Phase
    M** (selective ML ops: tensor/attention/quantization, oracle-first then MLIR) runs in parallel
    but throttled. Then **Phase D** (drivers/opcode tables — a semi-separate `drivers/` JIT kernel
    generator that gives each hardware **channel** its real driver), **Phase F** (C++, then a
    Python transpiler→frontend), and **Phase L** (the ML library, *compressed* from the
    GCC/PyTorch/TF/NumPy/… ecosystem onto the claim-graph + channel model).

---

## 7. Release ladder (reconciled)

✅ done · ◑ in progress · ☐ next

- **0.2 — reproducible compiler** (✅ effectively complete): 5 C++ GEM passes, multi-version
  LLVM matrix, R13 provenance manifest, generated differential parity, the widened
  corpus, the full optimizer-core C++ port, named pipelines, the six-target matrix,
  initial + C fuzzing, and a generated-status + broken-link + retired-path doc-governance CI
  gate (`tools/docs/`, see [`docs/STATUS.md`](STATUS.md)).
- **0.3 — measured adaptive compiler** (◑): real-hardware CT4 evidence (§5.4) + durable
  telemetry (schema registry, backpressure, a live broker in CI behind a fake producer)
  + compile-time/peak-memory regression budgets. *Landed:* the non-flaky perf-budget gate
  (`bcir.perf_budget` + `tools/perf/check_budgets.py`) now carries the five **Clang-comparison
  budgets** (dense streaming / dense L1 *match bands*; gather / reduction / strided *win floors*) —
  strict on correctness + measurement validity, perf floors/bands bare-metal-only, the documented
  6.0×/14.1×/1.33× references tracked in a JSONL trend log (never asserted in CI); named test tiers
  (`run_all --tier {quick,c-runtime,silicon-degrade,thorough}`); and the **hardware-channel plugin
  boundary** (`bcir/channel_plugin.py` — a `channel.json` manifest format so FPGA/NVMe/HBM-PIM
  extensions register without touching the core).
- **0.4a — proof-carrying (mechanism)** (✅): replay records + per-claim certificates +
  `bcir.run --explain`/`--replay`/`--reduce` are implemented and tested (§5.3.2).
- **0.4b — proof-carrying (contract)** (☐): a *stable* certificate schema (versioned, with a
  decode/upgrade path), an external replay-CLI contract (a third party can re-check a record
  without the producing build), and certificate upgrade tests across schema revisions.
- **1.0** (☐): stable language/ABI policy; no known Python↔C++ divergence (generated +
  fuzzed); ≥2 real hardware targets with measured evidence; R1–R17 dual-rail symmetry
  (§5.1.1); one external frontend; published benchmark methodology; upgrade tests; a
  clear native-backend decision (the gate, §5.5).

---

## 8. Risk register

- **Substrate/intelligence inversion** — a rich learned/categorical stack over a backend
  that can't yet codegen on real silicon and tables that aren't yet measured. *Mitigation:*
  §5.4 (real-silicon calibration) is the top priority; the quarantine keeps the learned
  side off the deterministic path until measured.
- **Multi-rail divergence** ("the law trails the oracle") — *largely mitigated:* the
  whole optimizer core + R1–R17 are dual-rail and cross-checked by the generated
  differential + the committed corpus/matrix under real `bcir-opt`. Remaining: §5.1.1/§5.1.3.
- **Validation realism** ("green ≠ competitive") — *mitigated:* the measured Clang
  comparison + the multi-version matrix + 540 tests. Remaining: continuous perf
  regression gating (0.3).
- **Complexity / bus factor** — the adaptive + learned organs are large. *Mitigation:*
  lazy imports keep the simple plan→emit path light (`test_perf` guard); every organ is
  opt-in and off the default path.

---

## Appendix A — capability tracks & the build history (what's built)

BCIR's capability model (the CT tracks, from the original blueprint), all oracle-done
and law-authored:

- **CT1 — memory hierarchy:** L1/L2/L3/DRAM/HBM/CXL/SSD tiers with Q8 bandwidth/latency
  factors (frozen in `runtime/c/bcir_q8_tables.h`); HAM (O(log n) access) and CXL
  semantics.
- **CT2 — concurrency / wave scheduling:** affinity domains, (max,+) overlap, the
  decoupled GGG tail.
- **CT3 — front-ends:** ROP (declarative) + MAP (macro-assembly) → claims; the M5 ETL
  (events / FSM transducer / parser / binary record decode).
- **CT4 — calibration / measurement:** microbench → frozen Q8 cost tables → R13 replan;
  real-signal probes (`bcir.silicon`).
- **CT5 — learning organs:** the L1–L3 learned stack (bayescal / portfolio / MoE gate /
  accelerator / softdp / regret / e-graph / two-truth / operad), each frozen to Q8.

The implementation arrived in phases (a condensed history; the dated detail lives in
`REPO_CURRENT_STATE_AUDIT.md`): the StreamPack freeze + freestanding C runtime; the
compiled `bcir-opt` on LLVM 18/19 with R1–R17; the learning/intelligence organs; the
oracle optimization pass (hot/cold locked); the MLIR-native GEM pipeline; the calibration
loop; the portable C23 kernel backend; the measured Clang comparison; the generated
differential + widened corpus; the full deterministic optimizer-core C++ port; the
six-target capability matrix; C23 `_BitInt`/`#embed` + ETL-binary C fuzz; and the
native-object decision gate.

**Non-regression invariants (design law, from the blueprints — must always hold):**
determinism (the same inputs reproduce the same plan bit-for-bit); back-compat (the
StreamPack ABI is append-only; old readers stay correct — `BCIR_STREAMPACK_ABI.md`); the
lowering never invents LLVM instructions (only legal load/store/add/atomicrmw/cmpxchg/
fence/calls); **atomics are never rewritten** into barrier+load+binop+store; the IRDL
projection stays C++-free; and the two-truth quarantine is never crossed (no learned
inference on the hot path, L0).

## Appendix B — what was consolidated / removed

Folded into this document and **removed**: `BCIR_STRATEGY_AND_ROADMAP.md` (strategy →
§1, §8), `BCIR_LOWERING_PLAN.md` (the lowering status → §3, §4), `BCIR_BLUEPRINT.md`
(capability tracks / phase inventory → Appendix A), `BCIR_Codex_Blueprint.md` +
`BCIR_Full_LLVM_Build_Blueprint.md` + `BCIR_LLVM_IR.md` (the old C++ `ir/`-tree rebuild
work-orders — obsolete since the `ir/` skeleton was retired into the `bcir/` oracle +
`mlir/` law; their forward items survive in §5). **Kept** (normative / evidence /
governance, not roadmap): `BCIR_LANGREF.md`, `BCIR_STREAMPACK_ABI.md`, `PARITY.md`,
`BCIR_NATIVE_OBJECT_GATE.md`, `HARDWARE_VALIDATION.md`, `CLANG_COMPARISON.md`,
`BCIR_Repo_Structure.md`, and `REPO_CURRENT_STATE_AUDIT.md`.
