# BCIR Master Roadmap

> **Status: the single, current, authoritative roadmap.** This document consolidates
> and supersedes the former strategy/blueprint/plan notes — `BCIR_STRATEGY_AND_ROADMAP.md`,
> `BCIR_LOWERING_PLAN.md`, `BCIR_BLUEPRINT.md`, `BCIR_Codex_Blueprint.md`,
> `BCIR_Full_LLVM_Build_Blueprint.md`, and `BCIR_LLVM_IR.md` (all removed in the
> consolidation; their unique content is folded in here). It pairs with the docs that
> stay separate because they are **normative reference / evidence / governance**, not
> roadmap:
> - [`BCIR_LANGREF.md`](BCIR_LANGREF.md) — the IR language + R1–R16 law spec (normative).
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
| Oracle conformance tests (`python -m bcir.tests.run_all`) | **540**, incl. the generated differential + verifier + fuzz campaigns |
| Deterministic **optimizer core** on the MLIR/C++ rail | **COMPLETE** — cost model, fusion/CSE/deforestation, min-plus plan, (max,+) overlap, per-claim + plan-level RCSP, all bit-exact vs the oracle |
| GEM C++ passes (classify/select/batch/schedule/lower) | all implemented (`mlir/lib/passes/`) |
| Verifier laws | **R1–R16** all first-class + dual-rail in `-bcir-verify` (+ the `-bcir-lower-to-llvm` checkpoint + the Python oracle ref) |
| Named pass pipelines | `bcir-audit` / `bcir-optimize` / `bcir-hydrate` / `bcir-lower-llvm` / `bcir-aot` with verifier checkpoints |
| Θ context op | `bcir.kbcir.theta` — the C++ plan matches the oracle under **hot** Θ (matmul hot 1159168), not just cool |
| Six-target capability matrix | all six TARGETS cross-checked on the MLIR rail (`target_matrix.mlir`) — the law plans per-target from the capability seeds alone |
| C23 in the runtime + kernels | `_BitInt(N)` exact-width Q-fixed lanes + `#embed` frozen Q8 tables (both with C11 fallbacks) |
| Trust-boundary fuzz | Python (`kbcir.fuzz`) + **libFuzzer + ASan/UBSan** on the StreamPack **and** ETL-binary C decoders (500k runs in CI) |
| Native object emission | decision gate documented (DEFERRED); the warranted slice (C → resident compiler → real eBPF/x86-64 object) is closed and ELF-verified |
| LLVM version policy | multi-version matrix **LLVM 18 + 19, both gating** |
| Perf vs Clang | **measured** (`CLANG_COMPARISON.md`): match on dense, 1.3–14× on irregular memory |
| Calibration loop | **closed on host** (microbench → `FrozenCalibrator` → R13 replan) + real-signal wiring (`bcir.silicon`); a *measured* bare-metal replan win is the one deferred item |

---

## 3. The MLIR / C / C++ / Python placement map (the two-truth line)

The port boundary is BCIR's own **L0–L3 / two-truth line** and is not negotiable:

> **Deterministic + integer/Q-fixed on the decision/execution path → C++/MLIR (law)
> or C (runtime). Graded / float / train-time → Python that *freezes* to Q8.**

| Subsystem | Today | **Target home** | Status |
|---|---|---|---|
| Dialect / ODS (the law's vocabulary) | `mlir/include/BCIR/*.td` | MLIR | ✅ |
| Verifier **R1–R13** | `-bcir-verify` + `bcir.verify` | MLIR/C++ + Python (oracle ref) | ✅ dual-rail |
| Verifier **R14/R15/R16** (CIM dispatch / DVFS clock / alloc placement) | `-bcir-verify` + the `-bcir-lower-to-llvm` checkpoint + oracle | MLIR/C++ + Python | ✅ **first-class `-bcir-verify` laws** (dual-rail with `verify.{verify_cim,verify_dvfs,verify_allocator}`) |
| K_BCIR **cost model** (`_cost`, candidates, stride/tier) | `-bcir-cost-model` (C++23) | MLIR/C++ | ✅ **ported** (bit-exact: vec16 7808, gather 528384, tile 126976) |
| K_BCIR **fusion / CSE / deforestation** | `-bcir-cost-model` | MLIR/C++ | ✅ **ported** (7808 / 5888 / 5100) |
| K_BCIR **min-plus shortest path** (`optimize`) | `-bcir-plan` | MLIR/C++ | ✅ **ported** (per-target, hot/cool Θ) |
| K_BCIR **overlap (max,+)** | `-bcir-overlap` | MLIR/C++ | ✅ **ported** (makespan/gain) |
| K_BCIR **RCSP / Pareto** (per-claim + plan-level) | `-bcir-rcsp`, `-bcir-rcsp-plan` | MLIR/C++ | ✅ **ported** (9472, {16,8}@17280) |
| GEM classify / batch / schedule / lower | `-bcir-*` passes | MLIR/C++ | ✅ |
| GEM **hydrate** (plan → StreamPack **bytes**) | **C** `runtime/c/bcir_encode.c` + Python `abi.streampack_abi.encode` | **C** (the encoder) | ✅ **ported** — `bcir_sp_reencode` is byte-identical to the Python encoder (v1 + v2) |
| GEM **deterministic executor** (decode → drive kernels) | **C** `runtime/c/bcir_exec.c` + Python `gem.execute` | **C** (hot path) | ✅ **ported** — Python↔C dispatch-order + telemetry parity + libFuzzer |
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
- **Verifier R1–R16** run on both rails, negative-tested per law, plus a verifier
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
   too (defense in depth). Verifier dual-rail symmetry is complete — every law R1–R16 is
   checkable by `-bcir-verify` alone.
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
3. **A C++ `-bcir-verify` law-for-law differential** against the oracle verifier (the
   remaining toolchain-rail step of the verifier differential).

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
3. **The `precision="compensated"` C-kernel emission variant** — the precision module
   models the compensated Q8 reduction; the C-kernel emitter does not yet produce it
   (a numerical-accuracy lowering gap).

### 5.3 New deterministic features (not ports) — for C++/MLIR

1. **Multi-claim bundle (joint) optimization** — the genuine combinatorial joint step
   over claim bundles, where the (min,+) formulation should start paying for itself
   beyond the pairwise coupling (shared-input fusion, deforestation, CSE, overlap) that
   ships today. Must bound compile time and emit search certificates.
2. **Proof-carrying optimization records** — replayable decision records + rewrite /
   lowering certificates + `bcir-explain` / `bcir-replay` / `bcir-reduce` (R13 is the
   foundation).
3. **Compositional semantics** — functions/calls, control flow, dynamic shapes,
   alias/effect modeling, numerical contracts (the path past straight-line array
   kernels).

### 5.4 Measured real-silicon calibration (DEFERRED — the top differentiator)

The software path is merged and certified on host; the one thing no architecture
substitutes for is a *measured* (not synthetic) replan win on a bare-metal rig with
`intel_pstate=passive` + a userspace governor + RAPL exposed (the exact rig is in
`HARDWARE_VALIDATION.md`). This converts "optimal-w.r.t.-a-model" into evidence and is
the single most valuable next result once a rig is available.

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
5. **The `precision="compensated"` C-kernel** (§5.2.3) + its accuracy-contract verifier
   law — *the next step.* The precision module models the compensated Q8 reduction; the
   C-kernel emitter and a dual-rail accuracy law are the remaining lowering gap.
6. **A C++ `-bcir-verify` law-for-law differential** (§5.1.3) — fault-inject each law on
   the toolchain rail and confirm `-bcir-verify` flags it (the oracle side already does).
7. **Multi-claim bundle (joint) optimization** (§5.3.1) — the first genuinely *new*
   optimizer capability, where the combinatorial (min,+) formulation earns its keep
   beyond the pairwise coupling that ships today (bound compile time; emit search
   certificates).
8. **Proof-carrying optimization records** (§5.3.2) — `bcir-explain` / `bcir-replay` /
   `bcir-reduce` over R13's provenance foundation.
9. **(When a rig is available)** the measured real-silicon replan win (§5.4) — the top
   differentiator.

---

## 7. Release ladder (reconciled)

✅ done · ◑ in progress · ☐ next

- **0.2 — reproducible compiler** (✅ effectively complete): 5 C++ GEM passes, multi-version
  LLVM matrix, R13 provenance manifest, generated differential parity, the widened
  corpus, the full optimizer-core C++ port, named pipelines, the six-target matrix,
  initial + C fuzzing. *Remaining polish:* doc-classification/link CI.
- **0.3 — measured adaptive compiler** (◑): real-hardware CT4 evidence (§5.4) + durable
  telemetry (schema registry, backpressure, a live broker in CI behind a fake producer)
  + compile-time/peak-memory regression budgets.
- **0.4 — proof-carrying** (☐): replay records + certificates + `bcir-explain`/`replay`/`reduce` (§5.3.2).
- **1.0** (☐): stable language/ABI policy; no known Python↔C++ divergence (generated +
  fuzzed); ≥2 real hardware targets with measured evidence; R1–R16 dual-rail symmetry
  (§5.1.1); one external frontend; published benchmark methodology; upgrade tests; a
  clear native-backend decision (the gate, §5.5).

---

## 8. Risk register

- **Substrate/intelligence inversion** — a rich learned/categorical stack over a backend
  that can't yet codegen on real silicon and tables that aren't yet measured. *Mitigation:*
  §5.4 (real-silicon calibration) is the top priority; the quarantine keeps the learned
  side off the deterministic path until measured.
- **Multi-rail divergence** ("the law trails the oracle") — *largely mitigated:* the
  whole optimizer core + R1–R16 are dual-rail and cross-checked by the generated
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
compiled `bcir-opt` on LLVM 18/19 with R1–R16; the learning/intelligence organs; the
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
