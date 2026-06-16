# BCIR Repository Current State Audit

> Audited 2026-06-16 against the `bcir/` (oracle) + `mlir/` (law) tree, after the
> **full deterministic optimizer-core C++ port**, the six-target capability matrix, the
> C23 `_BitInt`/`#embed` kernels + ETL-binary C decoder, and the native-object decision
> gate. The normative status lives in [`BCIR_LANGREF.md`](BCIR_LANGREF.md) §16; the
> forward plan in [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) (the single,
> consolidated roadmap); this file is the honest snapshot + changelog. Earlier revisions
> described the retired C++ `ir/` skeleton (removed 2026-06-07) and the pre-Phase-13 tree
> (audited 2026-06-12).

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
    replan → certified win). Suite: `python -m bcir.tests.run_all` (**540 checks**).
  - **`mlir/`** — the law: the ODS/TableGen dialect family (~80 ops), the compiled
    `bcir-opt` with `-bcir-verify` (R1–R16), `-bcir-promote-lanes`,
    `-convert-bcir-to-llvm`, the **GEM pipeline passes** (`-bcir-classify-lanes
    / -select-realization / -batch / -schedule / -lower-to-llvm`), the **full
    deterministic optimizer core in C++23** (`-bcir-cost-model` cost+fusion/CSE →
    `-bcir-plan` coupled min-plus → `-bcir-overlap` (max,+) → `-bcir-rcsp` /
    `-bcir-rcsp-plan` constrained search), a `bcir.kbcir.theta` context op, named pass
    pipelines (`bcir-audit`/`-optimize`/`-hydrate`/`-lower-llvm`/`-aot`), and the IRDL
    projection for stock `mlir-opt`. Validated in CI on a multi-version matrix — LLVM 18
    and 19, both gating (`mlir-rail-validate`).
- **`runtime/c/`** — the freestanding (no-libc) C runtime: the StreamPack ABI v1 decoder
  (Python-encode ↔ C-decode parity + libFuzzer), the ETL binary-record decoder
  (`bcir_binrec.c`, parity + libFuzzer), the C23 `#embed` frozen Q8 tier table, and the
  microbench. The C23 kernel emitter adds exact-width `_BitInt(N)` Q-fixed lanes.

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

1. **No BCIR-native instruction selection** (by design — emit C/LLVM and reuse the
   resident backend; the explicit decision gate is `BCIR_NATIVE_OBJECT_GATE.md`). A
   portable **C23 kernel backend** (`lower.c_kernel`) emits restrict-qualified,
   bounds-safe, width-driven C (now with exact-width `_BitInt(N)` Q-fixed lanes) from
   the selected StreamPack; LLVM/llc/lli/wasm are the other machine-code paths. **One
   target end-to-end is closed** (`codegen.codegen_object_c`: emitted C → resident clang
   → real eBPF / x86-64 ELF object, ELF-verified) — without hand-rolled isel. Register
   allocation/linking stay the resident toolchain's job; `bcir.target.lower_contract` is
   the seam. **Remaining:** GPU-C dialect gather variants.
2. **The calibration loop is fully closed.** Bare-metal cost constants
   (`runtime/c/bcir_microbench.c`, `calibrate_native`, `--calibrate --native`,
   real cache latency: gather_penalty ≈ 5), R13-certified replan (`kbcir.calibloop`),
   **a trained calibrator** (`calibrate.train_calibrator` → `FrozenCalibrator`, an
   online model frozen to deterministic Q8) and **a live broker**
   (`telemetry.Broker`, pub/sub fan-out). The frozen calibrator drives the loop end
   to end. *(Production hardening — a real Kafka deployment — remains operational.)*
3. **The example corpus is real, not toy.** Beyond vector_add: `saxpy_strided`
   (strided gather-avoidance, ~1.4× measured), `gather_reduce` (reduction
   gather-avoidance, ~16×), `fused_chain` (multi-claim overlap + the fusion
   discount), `scan_chain` (a dependency chain that serializes), **and the widened
   `examples.CORPUS`** — `matmul_tiled` (real blocked matmul, register-resident
   K-accumulation → deforestation), `scan` (multi-stage prefix pipeline),
   `multi_histogram` (map/reduce multi-claim gather). Python↔MLIR parity is now
   **generated and adversarial** (`bcir.kbcir.differential`), with per-target parity
   across the six TARGETS for the whole corpus (`mlir/test/passes/gem_corpus.mlir`).
   Joint multi-claim *bundle* optimization remains future work.
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

## Recommended next milestones (see [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) §5–6 for detail)

The deterministic optimizer core (cost model, fusion/CSE, min-plus plan, (max,+)
overlap, per-claim + plan-level RCSP) is **fully ported to C++/MLIR** and bit-exact vs
the oracle; the six-target capability matrix, C23 `_BitInt`/`#embed`, the ETL-binary C
decoder fuzz, and the native-object gate all landed. The next build steps:

1. **Verifier dual-rail symmetry** — lift **R14 (CIM dispatch) + R15 (DVFS clock)** out
   of `-bcir-lower-to-llvm` into first-class `-bcir-verify` checks (R16 already moved),
   and give the **`verification` cost axis** a real producer (today modeled as 0).
2. **The C StreamPack executor** (`gem/execute.py` → `runtime/c/`) — decode the pack and
   drive the emitted kernels in phase order with per-phase telemetry, so the StreamPack
   is a no-Python hot artifact a driver runs end-to-end; plus the **C encoder** for a
   full C round-trip.
3. **Measured real-silicon calibration** — the software path (`kbcir.calibloop`,
   `bcir.silicon`) is closed + certified on host; the one deferred result is a *measured*
   (not synthetic) replan win on a bare-metal rig (the rig spec is in
   `HARDWARE_VALIDATION.md`). *The single most valuable next result.*
4. **Multi-claim bundle (joint) optimization** — the first genuinely new optimizer
   capability, where the combinatorial (min,+) formulation earns its keep beyond the
   pairwise coupling that ships today.

## Changelog

- 2026-06-16: **Doc consolidation -- one master roadmap.** Reviewed all 15 `docs/`
  files; consolidated the strategy/blueprint/plan notes into a single, current
  [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) (positioning + measured state + the
  two-truth MLIR/C/C++ placement map + what's done + the forward roadmap + next build
  steps) and **removed** the now-redundant `BCIR_STRATEGY_AND_ROADMAP.md`,
  `BCIR_LOWERING_PLAN.md`, `BCIR_BLUEPRINT.md`, `BCIR_Codex_Blueprint.md`,
  `BCIR_Full_LLVM_Build_Blueprint.md`, and `BCIR_LLVM_IR.md` (the last three were
  work-orders for the retired C++ `ir/` skeleton). Kept the normative/evidence/governance
  references (`BCIR_LANGREF`, `BCIR_STREAMPACK_ABI`, `PARITY`, `BCIR_NATIVE_OBJECT_GATE`,
  `HARDWARE_VALIDATION`, `CLANG_COMPARISON`, `BCIR_Repo_Structure`). Re-pointed the
  references in README / mlir-README / bench / test docstrings. The roadmap names the
  remaining ports: Oracle→MLIR (R14/R15 into `-bcir-verify`, the verify-cost dimension);
  Oracle→C (the StreamPack executor + encoder, the compensated-reduction kernel); and the
  new deterministic features (multi-claim bundle optimization, proof-carrying records).
- 2026-06-16: **More C23 where it pays + the native-object decision gate** (PR #224).
  (1) `_BitInt(N)` exact-width Q-fixed lane kernels (`lower.c_kernel.emit_qfixed_kernel_c`)
  -- a `_BitInt(N)` lane + `_BitInt(2N)` product accumulator, where a standard int
  promotes and a 12/20/24-bit lane has no type at all, with a preprocessor-selected C11
  fallback so the same source is bit-identical under `-std=c23`/`-std=c11`. (2) `#embed`
  frozen Q8 tables (`bcir.abi.q8_tables` -> `runtime/c/{q8_tiers.bin,bcir_q8_tables.h}`,
  nested `__has_embed` guard + fallback array, drift-gated). (3) An ETL binary-record C
  decoder (`runtime/c/bcir_binrec.c`, the `etl.binary` twin) with a libFuzzer + ASan/UBSan
  harness and a Python<->C decode-parity gate; `kbcir.fuzz` gains the `etl.binary`
  boundary. (4) The native-object decision gate (`docs/BCIR_NATIVE_OBJECT_GATE.md`, GO
  G1-G4 / STOP S1-S4) + the warranted slice: `codegen.codegen_object_c` compiles the
  emitted kernel to a real eBPF (EM_BPF) / x86-64 (EM_X86_64) object via the resident
  compiler. +19 tests (540 total).
- 2026-06-16: **The six-target capability matrix** (PR #223). All six TARGETS are now
  emitted with their own `bcir.target.capability` seeds and cross-checked on the MLIR/C++
  rail (`mlir/test/passes/target_matrix.mlir`, generated by
  `differential.emit_target_matrix` / `--emit-matrix`, drift-gated): `-bcir-plan`/
  `-bcir-overlap`/`-bcir-rcsp-plan` recompute the oracle's per-target plan / makespan+gain
  / constrained optimum **from the capability alone** -- vector_add avx512/sve/rvv 16->7808,
  avx2 8->9472, neon 4->12800, ptx 32->6976; the GPU's coalesced gather halves histogram
  (266240 vs 528384). The emitter now writes a full descriptor (`warp` + `mem_channels`
  added). The six-target parity is proven on both rails, not just Python. +11 tests.
- 2026-06-16: **Post-optimizer lowering batch -- named pipelines, the Theta context op,
  C-runtime hardening.** (1) `registerBCIRPipelines` adds named, verifier-checkpointed
  pipelines: `bcir-audit` (verify -> cost/plan/overlap), `bcir-optimize` (claims+H ->
  coupled plan), `bcir-hydrate` (plan -> StreamPack), `bcir-lower-llvm`, `bcir-aot`
  (verify -> hydrate -> LLVM). (2) New `bcir.kbcir.theta` op carries the runtime state
  into the IR; `-bcir-plan`/`-bcir-overlap` now apply the multiplicative thermal
  coupling, so the C++ plan matches the oracle under hot Theta (matmul hot 1159168,
  `theta_hot.mlir`) -- the cool-regime restriction is lifted; the cool corpus is
  unchanged (theta 0). (3) The StreamPack C decoder (a trust boundary) gains `restrict`
  + `[[nodiscard]]` + a frozen-ABI `static_assert` -- which caught the header struct
  being 60 bytes vs the declared 64 (reserved[22]->[26]); builds clean under C11 + C23
  and is fuzzed under libFuzzer + ASan/UBSan (`runtime/c/fuzz_streampack.c`,
  `tools/c/fuzz_streampack.sh`, 500k runs in CI) with a sanitizer smoke over a real
  pack + byte mutations. +1 test (drift gate; 510 total). Deferred: real-silicon
  calibration (needs a rig).
- 2026-06-16: **Optimizer-core C++ port COMPLETE -- step 5, plan-level RCSP.** New
  `-bcir-rcsp-plan` (`BCIRRcspPlanPass.cpp`) ports `rcsp.optimize_constrained`: the
  accumulated-budget label DP over the fused candidate columns (labels carry score +
  per-tracked-dim totals; dominance pruning + infeasible-extension cuts). A plan-wide
  cap bounds the plan's accumulated thermal/power -- it narrows one claim where a
  per-claim cap cannot: two vec16 claims (thermal 2176) under thermal<=2000 ->
  {16,8} @ 17280, <=1500 -> {8,8} @ 18944, matching optimize_constrained.
  `mlir/test/passes/rcsp_plan.mlir`. **With this the whole deterministic optimizer
  core is on the MLIR rail (C++23):** cost+fusion (`-bcir-cost-model`) -> coupled
  shortest path (`-bcir-plan`) -> overlap (`-bcir-overlap`) -> per-claim + plan-level
  constrained search (`-bcir-rcsp`, `-bcir-rcsp-plan`), all bit-exact vs the oracle.
  Next (BCIR_LOWERING_PLAN.md): named pass pipelines, a Theta context op, C-runtime
  hardening.
- 2026-06-16: **Optimizer-core C++ port, step 4 -- overlap (max,+) M(pi,Theta).** New
  `-bcir-overlap` (`BCIROverlapPass.cpp`) ports `gem/overlap.py`'s
  `price_scheduled`/`_makespan`: over the coupled plan it does the wave assignment by
  conflict, round-robin affinity bins, per-bin re-coupling against the in-bin
  predecessor, max over bins/tail, series over phases. Reproduces the oracle's
  scheduled price bit-for-bit: matmul makespan 253952 / gain 761856 (4 tile chains
  fan out over 8 domains), scan & histogram gain 0, the shared-input chain gain 5888.
  The emitter now also emits `affinity_domains`; `gem_corpus.mlir` regenerated.
  `mlir/test/passes/overlap.mlir` + a corpus cross-check (check_passes.sh). Next:
  step 5, plan-level multi-claim RCSP -- the last optimizer-core piece.
- 2026-06-16: **Optimizer-core C++ port, step 3 -- the layered min-plus shortest path
  (the full optimize() in C++).** New `-bcir-plan` (`BCIRPlanPass.cpp`) runs the coupled
  tropical shortest path over the fused candidate columns (shared cost/fusion logic now
  in `BCIRCostModel.h`), each edge coupling `_context_factor`'s path-based shared-input
  fusion. It reproduces the oracle's `optimize` bit-for-bit on every module: 7808
  (vector_add), 13696 (shared-input chain, `plan.mlir`), and the corpus -- matmul
  1015808 / scan 101888 / histogram 1595520. The emitter (`to_mlir`) now emits a
  registry + capability so the law plans from first principles; passes are scoped per
  bcir.module. The per-claim argmin `-bcir-select-realization` is now subsumed by the
  coupled `-bcir-plan` for multi-claim. Next: step 4 overlap (max,+), step 5
  plan-level RCSP. See `BCIR_LOWERING_PLAN.md`.
- 2026-06-16: **Optimizer-core C++ port, step 2 -- fusion / deforestation / CSE.**
  `-bcir-cost-model` now processes claims in (phase, declared) order with
  value-numbering + a produced-rid set and applies the two intra-phase redundancy
  credits -- producer->consumer deforestation (x0.75 memory) and CSE (compute zeroed,
  copy-priced memory) -- matching the oracle's `fused_candidates` bit-for-bit
  (7808 / 5888 / 5100) and annotating `kbcir.cm_fusion`.
  `mlir/test/passes/cost_model_fusion.mlir`. Next: step 3, the layered min-plus
  shortest path with `_context_factor` (the full `optimize` in C++).
- 2026-06-16: **The keystone C++ port -- `-bcir-cost-model` (the K_BCIR cost algebra).**
  `mlir/lib/passes/BCIRCostModel.cpp` recomputes each claim's candidate set + 12-d cost
  vectors from `bcir.claim` + `bcir.target.capability` (a faithful C++23 port of
  `cost.py::_cost` / `realize.candidates_for` / `_stride_penalty`, constexpr tier table +
  seeded constants read off the capability, which gained
  mem_unit/base_overhead/thermal_density/power_density/per_op_heat/elem_bytes defaulted
  to the CPU seeds). Reproduces the oracle bit-for-bit -- vec16 @ 7808 (compute 64,
  memory 3840), gather @ 528384, tile @ 126976 -- from the claim graph alone, so the
  law no longer trusts emitter-baked path costs. `mlir/test/passes/cost_model.mlir` +
  a cross-check on `full_vec_add_ct1.mlir`, gated by `check_passes.sh`. Next:
  fusion/CSE (step 2) then the layered min-plus shortest path (step 3) -- see
  `BCIR_LOWERING_PLAN.md`.
- 2026-06-16: **De-monolithed the C++ pass library + the reformulated lowering plan.**
  The 1.5k-line `mlir/lib/BCIRPasses.cpp` is split into one TU per pass group under
  `mlir/lib/passes/` (`BCIRVerifyPass`, `BCIRPromotePass`, `BCIRConvertToLLVM`,
  `BCIRGEMPasses`, `BCIRSelectPass`, `BCIRRcspPass`) sharing `BCIRPassSupport.h`;
  `BCIRPasses.cpp` is now registration-only (factory-callback `registerPass`). Builds
  clean at C++23, all passes/ODS/IRDL validate, every flag still registered -- a far
  better CI module than a single script. New `docs/BCIR_LOWERING_PLAN.md` reanalyzes
  the oracle vs the MLIR/C/C++ rails, sets the two-truth placement, the C23/C++23/26
  modernization map, and the ordered port plan -- the immediate next build step is
  `-bcir-cost-model` (the K_BCIR cost algebra on the MLIR rail, the keystone that lets
  the law recompute costs instead of trusting emitter-baked path costs).
- 2026-06-16: **MLIR toolchain in place + the optimizer core starts porting to C++23.**
  The dev toolchain (mlir-18-tools / libmlir-18-dev / llvm-18-dev) builds `bcir-opt`
  locally; every pass validates, including the generated `gem_corpus.mlir` (the C++
  `-bcir-select-realization` recomputes the oracle's per-claim scores for the widened
  corpus -- the prior Python work confirmed against the real law). The dialect now
  builds at **C++23** (renamed the `bcir.opt.*` `$requires` attr -> `$require`, a C++20+
  keyword that blocked the standard bump). First optimizer-core port: **`-bcir-rcsp`**
  (`BCIRPasses.cpp`) ports `kbcir.rcsp` -- the budget-feasible label-DP argmin + the
  Pareto front over (score, thermal, power); reproduces 9472 under the 700 cap and the
  size-2 {vec16, vec8} front (`mlir/test/passes/rcsp.mlir` + a `gem_corpus` cross-check,
  gated by `check_passes.sh`). Remaining C++ ports: the cost model on the MLIR rail
  (enables overlap (max,+) + fusion/CSE recomputation). PMU/RAPL/DVFS real-silicon
  calibration is explicitly deferred ("do later").
- 2026-06-16: **Next-steps phase 2 -- real-silicon calibration path, R14-R16
  verifier + verifier differential, trust-boundary fuzzing, the overlap conformance
  net.** (1) `bcir.silicon` reads real RAPL package energy + on-die thermal and
  `kbcir.calibloop.measured_replan` (`MeasuredReplanCertificate`, CLI
  `bcir.run --silicon`) closes CT4's software path -- measured telemetry trains a
  frozen `LinearCalibrator`, replans, certifies the win, provenance-tagged
  real-vs-synthetic (honest degrade in a sandbox; lights up on a bare-metal rig).
  (2) `verify.{verify_cim,verify_dvfs,verify_allocator,verify_smart_lowering}` add
  R14-R16 to the Python verifier (dual-rail with `-bcir-lower-to-llvm`), plus a
  verifier differential (`gen_illegal_module` + `run_verifier_campaign`). (3)
  `kbcir.fuzz` fuzzes the trust boundaries seeded by `gen_module`. (4)
  `kbcir.differential.check_overlap` nets the (max,+) scheduled-price law for the
  pending C++ optimizer-core port. +26 tests (509 total). The C++ port of
  RCSP/Pareto/overlap/fusion/CSE and a measured replan on real silicon remain the
  toolchain/hardware-gated next steps.
- 2026-06-16: **Generated, adversarial Python↔MLIR differential testing + the
  widened corpus.** `bcir.kbcir.differential` turns the parity contract into a
  proof: a structured/adversarial `gen_module`, an independent `law_select`
  (per-claim min-plus argmin, mirroring `-bcir-select-realization`), a `check_module`
  diff (selection / per-claim+total score / RCSP budget feasibility / schedule
  order), a `shrink`er, and `run_campaign` over the six targets × Θ × policy.
  `lower.mlir.to_mlir` emits the GEM-pipeline IR from any oracle plan (the
  Python→law bridge), frozen for the corpus in `mlir/test/passes/gem_corpus.mlir`
  (drift-gated; recomputed by real `bcir-opt` when present). The corpus is widened
  past toy kernels: `examples.{matmul_tiled, scan, multi_histogram}` (real blocked
  matmul / multi-stage scan / map-reduce histogram), with per-target parity pinned
  across all six TARGETS. Closes quoted roadmap #2 (symmetric cross-validation) and
  #3 (widen the op surface). +15 tests (483 total).
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
- 2026-06-16: **Full optimizer-completeness audit (10 classic data-flow opts) + CSE
  shipped.** Audited K_BCIR/GEM for every classic optimization the cost model should
  capture but doesn't (the producer→consumer-fusion miss suggested others). Verdict:
  **CSE / duplicate-claim elimination was MISSING and is now fixed** — the egraph
  already detected the liked-pair (`module_exprs`/`shared_blocks`) but it never
  reached the plan cost. `realize.fused_candidates` now value-numbers each claim
  `(op, operand-versions)`; a repeat within a phase is priced as a *copy* (compute
  zeroed, memory scaled to the copy fraction), and a write between duplicates bumps
  the operand version and soundly withdraws the credit. ~15% cheaper on a duplicate,
  applied uniformly across all five rails, 7808/9472 + makespan≤serial intact, +4
  tests (464 total). The rest, assessed and **not** shipped (honest): **DCE** —
  blocked (no `Resource.is_output` flag, so "written but never read" is the program's
  result, e.g. C in vector_add); **in-place/WAR** — no gain in the traffic-based
  memory cost (`x=x+y` streams the same 3 operands as `z=x+y`; the saving is
  allocation, already covered by `allocator.pool_plan` liveness); **residency /
  multi-hop shared-input** — order-dependent, can't be priced uniformly without
  breaking makespan≤serial / overlap_gain==0 (or it over-credits past cache
  eviction); **reduction-tree depth** — doesn't fit the atomic-claim granularity
  (intra-claim critical path isn't modeled); **algebraic-identity / const-fold /
  remat** — N/A or speculative on array kernels. Oracle-only (no `mlir/`; the MLIR
  select reproduces 7808 for single-claim vector_add unchanged).
- 2026-06-16: **Fair BCIR-vs-Clang comparative analysis** (`bcir/clang_compare.py`,
  `docs/CLANG_COMPARISON.md`). Held the compiler constant (Clang 18) and compared
  BCIR's planned realization vs the naive one, separate binaries, alternated,
  median-of-N. Verdict (Xeon @ 2.80 GHz): **MATCH** on simple dense kernels
  (elementwise 0.98x stream / 1.00x L1 -- BCIR's go-fast loop regresses nothing);
  **WIN** wherever BCIR exploits program intent Clang's backend lacks -- gather
  avoidance **6.0x**, reduction order **14.1x**, strided-vs-gather **1.33x**; plus
  budget feasibility (a correctness win). **LOSE** only where a planning layer is
  expected to: it cannot out-codegen the backend it delegates to (MATCH is the
  ceiling on pure compute) and carries a one-time ~58 ms Python import (per-kernel
  plan+emit is 0.31 ms, ~1% of the 34 ms compile that happens anyway). +4 gated
  tests (468 total). Confirms the design contract: a cost-governed access-pattern /
  feasibility planner on top of LLVM, not a faster LLVM.
- 2026-06-16: **Master lowering roadmap reconciliation** (`docs/BCIR_MASTER_ROADMAP.md`).
  Reconciled the two pasted planning notes (the original ODS/build-layer bootstrap and
  the Phase A-H post-roadmap audit) against the current tree: their headline task (the
  5 C++ GEM passes) and "next law" (R13) are **done**, validation realism is now
  **measured** (Clang comparison + LLVM 18/19 matrix + 468 tests), so the quoted
  six-item roadmap stands at ~1 done / 4 partial / 1 deferred. The doc gives the
  definitive **MLIR / C / C++ placement map** (deterministic core -> C++/MLIR + C
  runtime; graded organs stay Python and freeze to Q8 -- the two-truth line), a
  lowering audit (what emits where + gaps: compensated-C variant, real matmul/scan,
  GPU-C gather, one target end-to-end), a testing audit (gaps: generated differential
  Python<->MLIR parity, property/metamorphic, fuzzing, R14-R16 as bcir.verify fns), and
  the reconciled 0.2->1.0 ladder. Center of gravity now: generated adversarial
  cross-rail equivalence, closing calibration on real silicon, and widening the corpus.
