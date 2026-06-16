# BCIR Repository Current State Audit

> Audited 2026-06-14 against the `bcir/` (oracle) + `mlir/` (law) tree, after
> Phases 13–26 (the learning/intelligence organs), the oracle optimization pass,
> and the MLIR-native GEM pipeline passes. The normative status lives in
> [`BCIR_LANGREF.md`](BCIR_LANGREF.md) §16; the forward plan in
> [`BCIR_STRATEGY_AND_ROADMAP.md`](BCIR_STRATEGY_AND_ROADMAP.md); this file is the
> honest snapshot. Earlier revisions described the retired C++ `ir/` skeleton
> (removed 2026-06-07) and the pre-Phase-13 tree (audited 2026-06-12).

## Snapshot

- Two trees implement BCIR in lockstep ([`PARITY.md`](PARITY.md)):
  - **`bcir/`** — the executable conformance oracle (pure Python, no third-party
    deps), ~11.8K LOC: model, K_BCIR optimizer (min-plus + RCSP/Pareto +
    (max,+) overlap), GEM hydration/scheduling/execution, ROP/MAP front-ends, M5
    ETL, telemetry/calibration, StreamPack ABI, the R1–R13 verifier, lowering
    (clang AOT / lli JIT / WASM / stackify / per-target llc / **portable C23
    kernel**), **and** the
    Phase 13–26 organs: calibration (microbench + Bayesian/conformal), policy
    portfolio + replay gate, MoE gate, search accelerator, soft optimizer, regret
    ledger, provenance manifest, e-graph + memory-module fixpoints, the two-truth
    quarantine, modular mapping functions, the enriched-operad memory
    interface, and the closed calibration loop (`calibloop`: measure → freeze →
    replan → certified win). Suite: `python -m bcir.tests.run_all` (**361 checks**).
  - **`mlir/`** — the law: the ODS/TableGen dialect family (~80 ops), the compiled
    `bcir-opt` with `-bcir-verify` (R1–R13), `-bcir-promote-lanes`,
    `-convert-bcir-to-llvm`, and the **GEM pipeline passes** (`-bcir-classify-lanes
    / -select-realization / -batch / -schedule / -lower-to-llvm`), plus the IRDL
    projection for stock `mlir-opt`. Validated in CI on a multi-version matrix —
    LLVM 18 and 19, both gating (`mlir-rail-validate`).
- **`runtime/c/`** — the freestanding (no-libc) C StreamPack runtime for the
  frozen ABI v1, with a Python-encode ↔ C-decode parity gate.

## Confirmed strengths

1. The oracle runs the whole correspondence chain end to end, deterministically
   (integer/Q-fixed), with worked-example parity pinned (`vector_add` AVX-512
   cool Θ → vec16, score **7808**; under a 700 thermal/power cap → vec8, **9472**).
2. Verifier laws **R1–R13** run on both rails and are negative-tested per law:
   `bcir/verify` and the MLIR `-bcir-verify` pass.
3. The **GEM pipeline is now MLIR-native and cross-checked against the oracle**:
   `-bcir-select-realization` recomputes the min-plus score from `cost · weights`
   and reproduces 7808/9472; the other four stages classify/batch/schedule/lower
   with positive (FileCheck) and negative (`-verify-diagnostics`) tests.
4. **Hot/cold separation is verified and locked** (`bcir/tests/test_hot_cold.py`):
   the executor and ABI codec import no learned organ or planner; no
   planning→execution→telemetry runtime recursion.
5. The StreamPack ABI v1 is frozen, CRC-gated, and decoded by a freestanding C
   runtime; cross-language parity is CI-gated.
6. CI gates every push: the oracle suite, the C runtime, the LLVM-training
   validators, and the full MLIR rail (tblgen, IRDL round-trip, `bcir-opt` build,
   ODS corpus, pass tests including the GEM pipeline).
7. **First measured win on real silicon.** The evidence rail (`bcir.bench`) shows
   BCIR's gather-avoidance (picking the direct realization over GGG) is **~6–7×
   faster** than the gather form (random indices) — the bare-metal-calibrated
   `gather_penalty` realized. And budget feasibility (`rcsp.feasible`,
   `api.build_artifact(budget=…)`) is a **correctness** win: BCIR emits the
   feasible vec8 where the naive max-width vec16 violates a 700 thermal/power cap.
   The library façade (`bcir.api`) packages a plan as a deployable, R12-attested
   artifact (AOT or driver-embedded).

## Confirmed limitations

1. **No BCIR-native instruction selection** (by design — the strategy is to emit
   C/LLVM and reuse the resident backend). A portable **C23 kernel backend**
   (`lower.c_kernel`) now emits restrict-qualified, bounds-safe, width-driven C
   from the selected StreamPack (library-first, self-checking, R12-verified), and
   LLVM/llc/lli/wasm remain the other machine-code paths. Register allocation and
   linking are still the resident toolchain's job; `bcir.target.lower_contract` is
   the seam. **Remaining:** GPU-C dialect variants and one target end-to-end.
2. **The calibration loop is fully closed.** Bare-metal cost constants
   (`runtime/c/bcir_microbench.c`, `calibrate_native`, `--calibrate --native`,
   real cache latency: gather_penalty ≈ 5), R13-certified replan (`kbcir.calibloop`),
   **a trained calibrator** (`calibrate.train_calibrator` → `FrozenCalibrator`, an
   online model frozen to deterministic Q8) and **a live broker**
   (`telemetry.Broker`, pub/sub fan-out). The frozen calibrator drives the loop end
   to end. *(Production hardening — a real Kafka deployment — remains operational.)*
3. **The example corpus is widening.** Beyond vector_add: `saxpy_strided`
   (strided gather-avoidance, ~1.4× measured), `gather_reduce` (reduction
   gather-avoidance, ~16×), `fused_chain` (multi-claim overlap + the fusion
   discount), `scan_chain` (a dependency chain that serializes), with per-target
   parity pinned for saxpy/histogram. Real tiled matmul / scan codegen and
   joint multi-claim optimization remain future work.
4. **Intelligence ahead of substrate.** Phases 13–26 added a rich learned/
   categorical optimization stack over a backend that cannot yet codegen and
   tables that are not yet measured; the ROI is unproven until §1–2 close.
5. **Multi-version LLVM matrix (LLVM 18 + 19, both gating).** The
   `mlir-rail-validate` CI job is a parametric matrix and **both LLVM 18 and 19
   now gate** (the forward-compat sweep is done). The Symbol-container ops
   (`registry` / `kbcir.plan` / `gem.stream_pack` / `parse.grammar` /
   `fsm.machine` / `binary.format`) carry the `SymbolTable` trait, so LLVM 19's
   stricter "symbol's parent must have the SymbolTable trait" verifier is
   satisfied; the trait is a no-op under the lax LLVM 18, so the same ODS builds
   and validates clean on both. No remaining LLVM 19 blocker.

## Recommended next milestones (see the roadmap for detail)

1. **Calibration loop** ◑ — closed + certified on host (`kbcir.calibloop`, R13).
   Remaining: a *trained* calibrator + live broker, and bare-metal numbers from
   the C runtime. *Top priority for the remaining half.*
2. **Widen the GEM passes + corpus**: multi-claim batching/fusion and real
   durations; reductions, tiled matmul, scan; per-target parity beyond
   `vector_add`.
3. **C backend** ✔ — portable C23 kernels (`lower.c_kernel`, R12) + the library
   façade (`bcir.api`). The cost-model win is now **measured** (gather avoidance
   ~6–7×; budget feasibility a correctness win). Next: GPU-C gather variants and
   multi-claim fusion.
4. **Driver/runtime integration** of the rehydrating planner (the StreamPack as
   the hot, Θ-replanned artifact).

## Changelog

- 2026-06-07: Reorganized into `bcir/` (oracle) + `mlir/` (law); retired the
  legacy C++ `ir/` tree (`docs/BCIR_Repo_Structure.md`).
- 2026-06-12: Rewrote against the post-reorg tree; verifier R1–R12 completed on
  both rails.
- 2026-06-14: Refreshed for Phases 13–26 (learning/intelligence organs), R13, the
  oracle optimization pass (recursive-planning overhead removed; hot/cold locked),
  and the MLIR-native GEM pipeline passes cross-checked against the oracle. Added
  `docs/BCIR_STRATEGY_AND_ROADMAP.md`. Closed the calibration loop
  (`kbcir.calibloop`, R13) and added the portable **C23 kernel backend**
  (`lower.c_kernel`, R12: `verify.verify_c_lowering`).
- 2026-06-15: MLIR-side R7 reduction-write parity (+ `gather_reduce_ct1.mlir` and
  the reduction test pair); the strided gather-avoidance (`saxpy_strided`,
  ~1.4×); `scan_chain` (serialization) + the fusion discount + per-target parity;
  the trained calibrator (`FrozenCalibrator`) + live `Broker` (Kafka bridge); and
  the multi-version LLVM CI matrix (LLVM 18 + 19, both gating — the `SymbolTable`
  forward-compat sweep on the six container ops landed, so 19 is green not
  informational; the training MLIR grader now grades against the matrix's
  `LLVM_SUFFIX` major instead of a hard-pinned 18, and the 18-calibrated corpus
  grades clean on 19).
- 2026-06-15: **Performance audit** (strategy doc §6). Found the simplest-process
  tax was fixed *import* overhead — planning a kernel eagerly loaded the whole
  Phase-13..26 research stack + GEM executor. Made `bcir.kbcir` / `bcir.lower` /
  `bcir.gem` import lazily (PEP 562; public API unchanged): `bcir.api` cold import
  −33%, `bcir.kbcir` −49%. Added structural perf guards (`bcir/tests/test_perf.py`)
  so the heavy stack stays unloaded on the plan→emit path. Documented the
  elementwise loop-form finding (bandwidth-bound, measured-neutral; the width cap
  is a load-bearing thermal throttle) and corrected the `bench.py` narrative; the
  width-aware C codegen + R12 refinement is a tracked follow-up.
- 2026-06-15: **Width-aware C lowering + R12 refinement** (strategy §6.2). The C
  backend used to cap at the selected width *unconditionally* — even at the full
  hardware lane, which let the planner override the compiler's isel. Now the width
  is a *floor* at the full lane (idiomatic loop, `emit_kernel_c(hw_width=…)`) and a
  *ceiling* when sub-maximal (a hard cap that honors a `Theta.hot` thermal
  throttle). R12 (`verify_c_lowering`) refined to match: no sub-lane cap on a
  full-lane kernel, a mandatory cap on a throttled one. Threaded `h.vector_width`
  through api/bench/CLI/self-check. Rigorous re-measurement (separate-process,
  alternated, median-of-N) showed the prior “~12% blocked penalty” was a
  measurement artifact — loop form is measured-neutral — so this lands as a
  correctness/semantics fix, not a speedup. Dual-rail-safe (MLIR R12 checks
  StreamPack lane-segment preservation, not C structure; segment width unchanged).
  +5 tests (372 total).
- 2026-06-15: **Adaptive "smart" layer** (strategy §7) — 8 intent-aware
  capabilities, each deterministic, opt-in (never on the default plan/emit/import
  path), and gains-only (tested no-op fallback): RL allocator / smart malloc
  (`kbcir.allocator`), compute-in-memory dispatch (`gem.cim` + `LaneSegment.dispatch`),
  persistent e-graph with telemetry pivot (`kbcir.egraph.ResidentEGraph`), JIT shape
  specialist (`lower.specialist`), active uncertainty-gated telemetry
  (`kbcir.sensing`), zero-copy telemetry ring (`telemetry.TelemetryRing`), fuzzy/
  continuous MoE routing (`kbcir.moegate.route_fuzzy`/`harden` + `FrozenGate.distribution`),
  and phase-aware DVFS (`gem.dvfs`). New organs are lazy (the `test_perf` guard
  asserts they stay unloaded on the simple path — `bcir.api` cold import unchanged).
  MLIR-law parity for these is future work. +48 tests (420 total).
- 2026-06-15: **Measured wiring to real signals + first MLIR parity law for the
  smart layer.** New `bcir.silicon` probes read the real machine (read-only,
  honest): `/sys` cache topology (real L1/L2/L3 tier map for the allocator),
  cpufreq table + nominal (DVFS anchor; actuation gated on a `userspace` governor +
  privilege — reported, never faked), and `getrusage`/timers feeding the telemetry
  ring (the guest exposes no hardware PMU — `perf_event_open` ENOENT — so OS
  counters are used and that fact is reported). **Re-validated on this Xeon:** the
  zero-copy ring is **~31× faster** than JSON serialization; cache-resident access
  is **~166× lower latency** than DRAM (pointer-chase), justifying hot→SRAM. **MLIR
  parity:** the `bcir.gem.lane_segment` op gains an append-only `dispatch` attr and
  a new law **R14** (`-bcir-lower-to-llvm`: `dispatch = "pim"` legal only on a
  `reduce.*` op), mirroring `gem.cim`; built + validated on LLVM **18 and 19**, with
  positive+negative `.mlir` cases. DVFS/allocator MLIR laws follow the same pattern
  (tracked). +9 oracle tests (429 total) + the R14 MLIR law (positive+negative).
- 2026-06-15: **Privileged/bare-metal paths (attempt + degrade) + R15/R16 MLIR
  parity.** `bcir.silicon.read_hw_counters` opens real hardware PMU counters
  (cycles/instructions/cache-misses via `perf_event_open`, user-space) and feeds the
  ring with them when present; this guest exposes no PMU (ENOENT) so it degrades to
  OS counters — reported, not faked. `gem.dvfs.actuate` **attempts** to set the real
  CPU clock (`scaling_setspeed`, read back via `scaling_cur_freq`) and returns a
  dry-run `ActuationResult` naming the missing capability when there is no
  `userspace` governor / privilege (the sandbox case) — a safe no-op. **MLIR
  parity:** `bcir.gem.lane_segment.clock_q8` (append-only) + **R15**
  (`-bcir-lower-to-llvm`: clock ∈ [64,512]; a `pim` memory-bound segment must not
  overclock) mirroring `gem.dvfs`; and `bcir.resource.placement` (append-only
  `BCIR_MemTier`) + **R16** (an L1 placement ≤ 64 KiB, L2 ≤ 4 MiB; static
  `product(shape)*4`) mirroring `kbcir.allocator`. Built + validated on LLVM **18
  and 19** (positive + negative `.mlir`). New `docs/HARDWARE_VALIDATION.md` states
  the sandbox limits and the exact bare-metal rig (PMU + `intel_pstate=passive`
  userspace governor + root + RAPL) needed to measure the DVFS power-savings claim —
  which we do **not** assert until measured. +5 oracle tests (434 total).
- 2026-06-15: **Second pass over the adaptive layer — audited 8 proposed
  refinements, shipped the ones that hold gains, excluded the ones that don't.**
  Verdicts: (1) Persistent EGraph + pivot — **already DONE** (`ResidentEGraph`), no
  change. (2) Uncertainty-gated sensing — **shipped the delta**:
  `accel.FrozenRanker.confidence` (top-2 z-margin) + `sensing.sense_by_ranker`
  (a-priori gating: instrument only columns the ranker can't resolve). (3)
  Continuous routing — **excluded**: a soft distribution already exists
  (`moegate.route_fuzzy`); an execution-layer blend is redundant compute (recomputes
  the answer N×) — exploration value only, **no steady-state gain**, not shipped. (4)
  Predictive allocator — **shipped** `allocator.pool_plan`/`live_intervals`
  (liveness interval-partitioning: disjoint-lifetime tensors share an arena ⇒
  peak ≤ naive; gains-only modeled footprint win). (5) CIM/PIM partitioning —
  **shipped** `c_kernel.optimize_spatial` + `is_pim_target` (a `pim` ISA-feature
  target binds reductions to memory, modeled transport-saved; reuses the R14 law;
  real PIM emitter is next-phase). (6) JIT specialist — **measured, no gain**:
  specialist-vs-generic = generic/spec ≈ 1.0 on bandwidth-bound elementwise (same
  finding as the loop-form audit), so it is **not** wired as a perf path — kept only
  as a correctness-preserving option (`test_specialist_is_correctness_preserving_only`).
  (7) Zero-copy ring — **shipped the C side**: `memory_model.emit_ring_header_c`
  (atomic release-store producer via `hazard_to_ordering`) + `telemetry.parse_shared_ring`;
  measured C-writes/Python-reads the same mmap, no syscall/serialization. (8)
  Phase-aware DVFS — **shipped** `gem.schedule.schedule_power_rail` (per-Slot clock
  over the placed timeline; energy figure modeled — no RAPL in-sandbox). All
  deterministic, opt-in, off the simple path (test_perf guard). +13 tests (447
  total). No new MLIR laws needed (these are planning/runtime passes; the PIM
  binding is covered by R14).
- 2026-06-16: **Mined two prior-project (BDI) research notes — MPAT + AEDACI — for
  precision/accuracy ideas; shipped the deterministic, integer subset, skipped the
  bloat.** New `kbcir.precision` (opt-in, off the default plan path — pinned scores
  unchanged): the Q8-ULP error unit (`ulp_distance`); integer interval error bounds
  (`Interval`, `accuracy_bound`, `reduction_error_bound`) that give the `accuracy`
  cost dim a real producer + `meets_tolerance` (a checkable accuracy contract);
  a **compensated Q8 reduction** (`compensated_reduce_q8`, residual-carry MAC) that
  is bit-identical to the int64-exact result vs the naive accumulator's `count`-ULP
  drift — the measured numerical win; and stability diagnostics (`cancellation`,
  `condition_milli`) emitted as two-truth `Graded` signals (inform, never legislate).
  **Skipped as bloat/wrong-substrate:** the ECC/coding-theory catalogue, ML decoders
  and the novel-algorithm zoo, arbitrary-precision bignum, decimal, CORDIC,
  root-finder/optimizer/quadrature libraries, affine arithmetic, the Newton-Raphson
  condition search, and the runtime hot-patch/kernel-daemon machinery. +10 tests
  (457 total). Next-step roadmap (the MLIR/C/C++ lowering pass): a `precision=
  "compensated"` C-kernel variant + the accuracy-contract verifier law on both rails
  (strategy §8).
- 2026-06-16: **Final BDI-notes mining pass for K_BCIR/GEM (5 docs) — verdict +
  one shipped gain.** Screened all five uploads against the two engines with a hard
  filter (deterministic integer, IR-level/plan-time, gains-only). Result: **four
  yield nothing** — `paradigms_and_concepts` (philosophy/survey), the `BDI SRS`
  (requirements prose), the `AI-Trainer` (a float toy-VM + symbolic curriculum), and
  `PrimeDivisor_Tools` (symbolic-AI; its divisor-lattice tiling idea tests **neutral**
  — the ceil cost model already prefers wider-with-remainder, so divisor-alignment
  gives no gain). `execution_and_compiler_research` is ~95% runtime/backend; its
  "strength reduction" is exactly what BCIR delegates to LLVM. BCIR already has the
  rigorous versions (min-plus, RCSP, Pareto, (max,+) overlap, EFT/HEFT, affinity,
  bandwidth-knee). **The one real gap it named** — *deforestation / producer→consumer
  fusion* — is now shipped: `realize.fused_candidates` bakes a memory discount into a
  consumer claim that reads an operand a prior same-phase claim produced (the
  intermediate never round-trips), applied **uniformly** across optimize / RCSP /
  overlap / accel / softdp so `makespan ≤ serial` and the unbounded-RCSP == optimize
  invariants hold. Measured **~12% lower plan score** on a producer→consumer chain,
  no width churn, pinned 7808/9472 intact. Oracle-only (no `mlir/` change; the MLIR
  select reproduces 7808 for single-claim vector_add unchanged). +3 tests (460 total).
