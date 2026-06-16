# BCIR Master Roadmap — MLIR / C / C++ lowering (current-state reconciliation)

> **Status: current implementation guide.** Supersedes the lowering sections of the
> two pasted planning notes (`BCIR_next_phase` = the original ODS/build-layer bootstrap;
> `BCIR_Roadmap` = the Phase A–H post-roadmap audit). Both were written against an
> **older** tree (136/303 tests, the 5 GEM passes *unimplemented*, "R13 is the next
> law"). This document reconciles them against the repo as it stands and gives the
> definitive MLIR/C/C++ placement, a lowering+testing audit, and the next steps.
> Pairs with `BCIR_STRATEGY_AND_ROADMAP.md` (strategy), `REPO_CURRENT_STATE_AUDIT.md`
> (snapshot), and `BCIR_LOWERING_PLAN.md` (the reformulated MLIR/C/C++ lowering plan +
> the ordered port-from-the-oracle build steps).

## 0. Current state at a glance (measured)

| Fact | Pasted-roadmap assumption | **Now** |
|---|---|---|
| Oracle conformance tests | 136 / 303 | **509** (incl. the generated differential + verifier + fuzz campaigns) |
| 5 GEM C++ passes (classify/select/batch/schedule/lower) | *unimplemented* | **all implemented** in `mlir/lib/BCIRPasses.cpp` |
| Verifier laws | R1–R12, "R13 next" | **R1–R16** (R1–R13 dual-rail; R14 CIM-dispatch, R15 DVFS-clock, R16 allocator-tier as MLIR laws + oracle gates) |
| LLVM version policy | loose, single-version | **multi-version matrix LLVM 18 + 19, both gating** |
| Perf vs Clang | none | **measured** (`docs/CLANG_COMPARISON.md`: match on dense, 1.3–14× on irregular memory) |
| Calibration loop | seeded constants | **closed on host** (microbench → `FrozenCalibrator` → R13 replan) + real-signal wiring (`bcir.silicon`) |

## 1. Reconciliation of the quoted six-item roadmap

| # | Quoted item | Status | Evidence / gap |
|---|---|---|---|
| 1 | Implement the 5 GEM passes in C++, cross-checked vs pinned scores | ✅ **DONE** | `BCIRPasses.cpp` classify/select/batch/schedule/lower; `-bcir-select-realization` recomputes min-plus and reproduces **7808/9472**; `gem_passes{,_neg}.mlir` |
| 2 | Symmetric cross-validation; extend parity beyond `vector_add` | ✅ **DONE** | Multi-version matrix ✅; **generated, adversarial Python↔MLIR differential** (`bcir.kbcir.differential`: `gen_module` + independent `law_select` + `shrink` + `run_campaign`) proves oracle↔law agreement over thousands of random modules + the six targets; `mlir/test/passes/gem_corpus.mlir` recomputes the widened corpus under `bcir-opt` |
| 3 | Widen the op surface (reductions, real matmul, scan/histogram) | ✅ **DONE** | `reduce.gather` ✅; **real corpus** shipped (`examples.{matmul_tiled, scan, multi_histogram}` = `examples.CORPUS`): blocked matmul with register-resident K-accumulation, a multi-stage scan pipeline, a map/reduce multi-claim histogram — MLIR parity across all six targets (`test_differential.py`) |
| 4 | Close the CT4 loop live (real HW counters, broker, measured replan) | ◑ **PARTIAL** | Loop closed on host (`calibloop`, `FrozenCalibrator`, `Broker`, microbench, `bcir.silicon`); **gap:** real PMU counters + RAPL need a bare-metal rig (documented in `HARDWARE_VALIDATION.md`); no live broker in CI |
| 5 | Multi-claim joint optimization | ◑ **PARTIAL** | Pairwise coupling shipped: shared-input fusion, producer→consumer **deforestation**, **CSE**, (max,+) overlap; **gap:** true combinatorial *bundle* joint optimization |
| 6 | Native backend — only if warranted | ⛔ **DEFERRED (correct)** | Decision gate stands; LLVM front-half is the chosen path; `bcir.target.lower_contract` structures either choice |

**Risk register (from `BCIR_Roadmap`) — current status:** *validation realism* — largely **mitigated** (Clang comparison + multi-version matrix + 468 tests). *Multi-rail divergence / "law trails the oracle"* — **mitigated for the GEM pipeline** (5 passes + R1–R16 in C++). *Substrate/intelligence inversion* — **partially mitigated** (calibration closed on host, silicon wiring, honest hardware-limit doc); the learned organs remain modeled until a bare-metal rig closes them. *Doc drift, fuzzing, packaging, ASan/UBSan, generated schema* — **still open** (the `BCIR_Roadmap` Phase A/B items remain the right next work).

**Net:** the quoted roadmap is now **~3 done, 2 partial, 1 deferred** — items #2 (symmetric cross-validation) and #3 (widen the op surface) closed by the generated differential harness + the real corpus; #1 was already done; the project has *added* beyond it (R14–R16, the adaptive smart layer, CSE/deforestation, silicon wiring, the Clang evidence rail). The forward plan is no longer "build the 5 passes" or "make parity generated" but **"close calibration on real hardware, port the deterministic optimizer core (RCSP/Pareto/fusion) to C++, and fuzz the trust boundaries."**

## 2. The MLIR / C / C++ placement map (where every part goes)

The port boundary is BCIR's own **L0–L3 / two-truth line**: deterministic integer/Q-fixed machinery is C++/MLIR (and the C runtime); graded, float, train-time machinery stays Python and emits frozen Q8 the deterministic rail consumes. This is unchanged and correct.

| Subsystem | Today (impl) | **Target home** | Status |
|---|---|---|---|
| Semantic model (registry/claim/phase/resource) | Python `bcir.model` + ODS ops | **MLIR/C++** (ODS is the law) | ODS ✅; C++ accessors used by passes ✅ |
| Verifier **R1–R13** | Python `bcir.verify` + `-bcir-verify` | **MLIR/C++** (law) + Python (oracle ref) | dual-rail ✅ |
| Verifier **R14–R16** (CIM / DVFS clock / alloc tier) | `-bcir-lower-to-llvm` + oracle gates (`gem.cim`, `gem.dvfs`, `kbcir.allocator`) + `verify.{verify_cim,verify_dvfs,verify_allocator,verify_smart_lowering}` | **MLIR/C++** (law) + Python (oracle ref) | MLIR ✅; oracle-gate ✅; **`bcir.verify` fns ✅ (dual-rail)** |
| K_BCIR selection (min-plus scalarization) | Python `realize` + `-bcir-select-realization` | **MLIR/C++** | selection scoring ✅ (reproduces 7808/9472) |
| K_BCIR **RCSP / Pareto** | Python (`rcsp`) + **`-bcir-rcsp`** (`BCIRPasses.cpp`, C++23: budget label-DP argmin + Pareto front) | **MLIR/C++** | **ported ✅** (reproduces 9472 + the size-2 front; `rcsp.mlir`) |
| K_BCIR **overlap / fusion / CSE** | Python (`overlap`, `realize.fused_candidates`) | **MLIR/C++** (deterministic) | **oracle-only** — the next C++ port (needs the cost model on the MLIR rail to recompute base costs) |
| GEM classify / batch / schedule / lower | `-bcir-classify-lanes/-batch/-schedule/-lower-to-llvm` | **MLIR/C++** | ✅ |
| GEM hydrate (plan → StreamPack) | Python `gem.streampack.hydrate` | **MLIR/C++** | partial (lower-to-llvm consumes segments; full hydrate op pending) |
| GEM deterministic executor | Python `gem.execute` | **C++/C** (hot path) | oracle-only |
| StreamPack ABI codec | **C** `runtime/c/` + Python encoder | **C** (frozen ABI) | ✅ (CRC-gated, parity-tested) |
| Portable kernel emission (C23) | Python `lower.c_kernel` | **C++** (or stays a thin emitter) | emits C; Clang compiles — the kernel is the *output*, the emitter can be C++ later |
| Lowering contracts (`target.lower_contract`, `isa.*`, `packet.*`) | ODS ops | **MLIR/C++** (law) | ODS ✅; consumed by lowering |
| Telemetry **ring** (zero-copy) | Python `telemetry` + **C** producer (`memory_model.emit_ring_header_c`) | **C** (shared mmap) | ✅ C↔Python bridge |
| Real-signal probes / DVFS actuation | Python `bcir.silicon`, `gem.dvfs.actuate` | **C/C++** (runtime) | read ✅; actuation gated (needs bare-metal) |
| **Stays Python (graded, offline, L2/L3):** `bayescal`, `softdp`, `moegate` *training*, `microbench`, `regret` ledger, `calibrate` SGD | Python | **Python** (emit frozen Q8 + generation tags) | by design — porting them would violate the quarantine |
| Enriched operad / memory-module fixpoints / two-truth | Python | **Python** unless load-bearing for plan-time caching | research-side tooling |

**One-line rule:** *anything deterministic and integer on the decision/execution path → MLIR-C++ (law) or C (runtime); anything float/learned/train-time → Python that freezes to Q8.*

## 3. Lowering audit (what lowers where today, and the gaps)

**Implemented lowering paths (all Python-emitted today, compiled by the resident toolchain):**
- **Portable C23 kernel** (`lower.c_kernel`, R12-attested) — the primary path; matches Clang on dense, wins on irregular memory.
- **LLVM IR** AOT (clang) / JIT (lli), **WASM** (node), **stackify** (JVM/CIL/WASM bytecode), **per-target llc** descriptors (x86-64, AArch64, RISC-V, NVPTX, eBPF, best-effort SPIR-V).
- **MLIR** `-convert-bcir-to-llvm` (LLVM-dialect lowering) + the GEM pipeline `→ lower-to-llvm` (R12/R14/R15/R16 contract checks).

**Lowering gaps (prioritized):**
1. **`precision="compensated"` C-kernel variant** — the precision module models compensated reduction; the *emitter* doesn't yet produce it. (Small, high-signal.)
2. **Width-aware codegen on the MLIR rail** — the oracle emits floor-vs-cap (go-fast/throttle); the MLIR `lower-to-llvm` records width but doesn't emit the C. (The C emission is Python-only.)
3. **Real tiled-matmul / scan lowering** — the tile path is a skeleton; no blocked-matmul or scan kernel emitter.
4. **GPU-C gather variants** — NVPTX/SPIR-V descriptors exist; no gather/scatter kernel emitter per target.
5. **One target end-to-end** — every machine-code path still shells out to clang/llc/lli; no BCIR-native object emission (correctly gated).

## 4. Testing audit (what's covered, and the gaps)

**Covered:** 468 oracle conformance tests; the MLIR rail (tblgen, IRDL round-trip, `bcir-opt` build, ODS examples, R1–R16 pass tests incl. `gem_passes{,_neg}`) on **LLVM 18 + 19**; StreamPack C-decode parity; the **measured Clang comparison** (`test_clang_compare`, gated); per-target parity pins for saxpy/histogram; pinned 7808/9472 both rails.

**Testing gaps (the `BCIR_Roadmap` Phase B items):**
1. ✅ **Generated differential Python↔MLIR testing** — `bcir.kbcir.differential`: a structured/adversarial `gen_module` runs both rails (oracle shortest-path vs. independent `law_select` per-claim argmin) and diffs the selected realization / per-claim + total score / budget feasibility / schedule order, **shrinking** mismatches to a minimal witness (`shrink`). `run_campaign` sweeps the six targets × Θ × policy; `test_differential.py` asserts zero divergence over ≥1500 modules; `mlir/test/passes/gem_corpus.mlir` recomputes the widened corpus under real `bcir-opt`. *(This turned "two rails that must agree" into a proof.)*
2. ✅ **Property / metamorphic tests** — shipped in `test_differential.py`: ID-renaming preserves the score; a tighter budget never raises the capped dimension (and feasibility holds); emit→read (`plan_view`→`law_select`) is lossless; the committed corpus equals a fresh emission (drift gate).
3. ◑ **Fuzzing of trust boundaries** — ✅ a generator-seeded Python fuzz (`bcir.kbcir.fuzz`) covers the StreamPack codec, the ROP/MAP/ETL front-ends, the calibration JSON, and the MLIR emitter (valid round-trip + graceful malformed rejection). **Remaining:** libFuzzer + ASan/UBSan for the C/C++ decoders (the StreamPack C runtime, the MLIR parser) on the toolchain rail.
4. ✅ **R14–R16 in the Python verifier** — `verify.{verify_cim,verify_dvfs,verify_allocator,verify_smart_lowering}` mirror the `-bcir-lower-to-llvm` laws (dual-rail symmetry), negative-tested per law; plus a **verifier differential** (`gen_illegal_module` + `run_verifier_campaign`) that fault-injects each law and confirms the verifier catches it.
5. ☐ **Compile-time / peak-memory regression budgets**; reproducibility checks that rebuild archived plans from frozen inputs.

## 5. The updated master roadmap (reconciled)

The `BCIR_Roadmap` Phase A–H structure and the 0.2→1.0 release ladder remain the right spine; below is the **reconciled, current-state** version. ✅ done · ◑ in progress · ☐ next.

### Near-term — finish "BCIR 0.2: reproducible compiler" (≈90% done)
- ✅ 5 C++ GEM passes; ✅ multi-version LLVM matrix; ✅ compilation/provenance manifest (R13); ✅ measured baseline vs Clang.
- ✅ **Generated differential Python↔MLIR parity** (Phase B.1) — `bcir.kbcir.differential` (generator + independent law rail + diff + shrink) + `lower.mlir.to_mlir` emitter; was the top correctness lever, now landed.
- ✅ **Widen the corpus**: real tiled matmul + scan + multi-claim histogram (`examples.CORPUS`), with MLIR parity across the six targets (`gem_corpus.mlir`) — closes quoted #2, #3.
- ☐ **Named pass pipelines** (`bcir-plan`, `bcir-hydrate`, `bcir-lower-llvm`, `bcir-aot`, `bcir-audit`) with declared input/output levels + verifier checkpoints (stop relying on ad-hoc ordering). *(Now the highest-value MLIR-rail lever: the `to_mlir` emitter gives the named-pipeline inputs for free.)*
- ☐ **Doc-classification + link CI** (Normative / Current / Historical / Research) + retired-path checker.
- ✅ **R14–R16 as `bcir.verify` functions** (dual-rail symmetry) — `verify.{verify_cim,verify_dvfs,verify_allocator}`.
- ✅ **Verifier differential** — `gen_illegal_module` + `run_verifier_campaign` fault-inject each law and confirm the verifier catches it (the C++ `-bcir-verify` law-for-law diff is the remaining toolchain-rail step).
- ◑ **Fuzz the trust boundaries** — `bcir.kbcir.fuzz` (Python, generator-seeded); C/C++ libFuzzer + ASan/UBSan remain.

### Mid-term — "BCIR 0.3: measured adaptive compiler" (the headline)
- ◑ **Close CT4 on real hardware** (quoted #4): the *software path* now lands end-to-end — `bcir.silicon` reads real PMU + **RAPL energy** + **on-die thermal**, `kbcir.calibloop.measured_replan` builds measured telemetry, trains+freezes a `LinearCalibrator`, replans, and certifies the win (`MeasuredReplanCertificate`, provenance-tagged real-vs-synthetic; CLI `bcir.run --silicon`). It degrades honestly in a sandbox (synthetic, win 0) and lights up the identical path on a bare-metal rig. **Remaining:** run it on a rig with `intel_pstate=passive` + RAPL exposed and publish one *measured* (not synthetic) replan win. *This is still the single most valuable next result — it converts "optimal-w.r.t.-a-model" into evidence.*
- ☐ Durable telemetry (schema registry, backpressure, auth); live broker in CI behind a fake producer (the unit exists).
- ☐ Fuzzing + ASan/UBSan jobs; compile-time/memory regression budgets.

### Mid/long-term
- ☐ **Multi-claim joint (bundle) optimization** (quoted #5) — where the (min,+) formulation should start paying for itself; bound compile time + emit search certificates.
- ◑ **Port RCSP/Pareto/overlap/fusion/CSE to C++** (the deterministic optimizer core) — ✅ **RCSP + Pareto** ported (`-bcir-rcsp`, C++23; reproduces 9472 + the size-2 Pareto front, cross-checks the corpus). **Remaining:** the **cost model** on the MLIR rail (compute base costs from claim + capability), which then enables a C++ **overlap (max,+)** and **fusion/CSE** recomputation (today those trust the emitter-baked path costs). Completes "law catches the oracle." *(Unblocked: the MLIR toolchain is built + gating in CI on LLVM 18+19.)*
- ☐ **Proof-carrying optimization records** (Phase D): R13 is in; add replayable decision records + rewrite/lowering certificates + `bcir-explain`/`bcir-replay`/`bcir-reduce`.
- ☐ **Compositional semantics** (Phase F): functions/calls, control flow, dynamic shapes, alias/effect modeling, numerical contracts.
- ☐ **Native backend** (quoted #6) — only behind the documented decision gate; if taken, a one-target experiment (eBPF or x86-64 scalar) with stop criteria.

### Release criteria (reconciled)
- **0.2** (reproducible compiler): + differential parity, named pipelines, wider corpus, doc cleanup, initial fuzzing. *(GEM passes + matrix + manifest already done.)*
- **0.3** (measured adaptive): real-hardware CT4 evidence + durable telemetry + perf regression tracking.
- **0.4** (proof-carrying): replay records + certificates + `bcir-explain`/`replay`/`reduce`.
- **1.0**: stable language/ABI policy, no known Python↔C++ divergence (generated+fuzzed), ≥2 real hardware targets with measured evidence, R1–R16 dual-rail symmetry, one external frontend, published benchmark methodology, upgrade tests, a clear native-backend decision.

## 6. Bottom line

The pasted roadmaps remain a sound *spine*, but their headline task (the 5 C++ GEM passes) and their "next law" (R13) are **done**, validation realism is **measured**, and as of this phase two of the three former center-of-gravity items are also done: Python↔C++/MLIR equivalence is now **generated and adversarial** (`bcir.kbcir.differential` — thousands of random modules diffed across the six targets, shrunk on mismatch, recomputed by real `bcir-opt` on the committed corpus), and the corpus is **widened** past toy kernels (real tiled matmul / scan / multi-claim histogram with per-target MLIR parity). The center of gravity now narrows to **three**: (1) **close calibration on real silicon** (the one differentiator no amount of architecture substitutes for — now the single most valuable next result), (2) **port the deterministic optimizer core to C++/MLIR** (RCSP/Pareto/overlap/fusion/CSE — the remaining "law catches the oracle" gap, for which the generated harness is the readymade conformance net), and (3) **fuzz the trust boundaries** (parsers, decoders, MLIR — seeded by the new `gen_module`). The MLIR/C/C++ split is settled by the two-truth line — deterministic core to C++/MLIR + C runtime, graded organs stay Python and freeze to Q8.
