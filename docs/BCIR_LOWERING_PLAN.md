# BCIR Lowering Plan — MLIR / C++ / C (reformulated)

> **Status: current implementation guide for the lowering rail.** Written after the
> C++ pass library was de-monolithed (`mlir/lib/passes/*.cpp`), the build moved to
> **C++23**, and the first optimizer-core piece (**RCSP/Pareto**) was ported to C++.
> Pairs with `BCIR_MASTER_ROADMAP.md` (program-wide) and `PARITY.md` (the oracle↔law
> cross-map). This document reanalyzes what lives where and gives the ordered C/C++
> build plan from the Python oracle.

## 0. The line that decides where code goes

BCIR's **two-truth quarantine** is the placement rule, and it is not negotiable:

> **Deterministic + integer/Q-fixed on the decision/execution path → C++/MLIR (law)
> or C (runtime). Graded / float / train-time → Python that *freezes* to Q8.**

So the optimizer's *search and cost algebra* (integer, deterministic) belong in
C++/MLIR; the hot *execution + ABI* belong in C; the *learned organs* (bayescal,
softdp, moegate, calibrate SGD, regret) stay in Python and emit frozen Q8 artifacts
the deterministic rail consumes. Porting a learned organ to C++ would violate the
quarantine; porting the cost algebra to C++ *completes* it.

## 1. Where the lowering lives today (measured)

| Layer | Today | Home | State |
|---|---|---|---|
| Dialect / ODS (the law's vocabulary) | `mlir/include/BCIR/*.td` | MLIR | ✅ |
| Verifier **R1–R16** | `mlir/lib/passes/BCIRVerifyPass.cpp` + `bcir.verify` (Python oracle ref) | MLIR/C++ + Python | ✅ dual-rail |
| Lane opt-law (GGG→UX) | `BCIRPromotePass.cpp` | MLIR/C++ | ✅ |
| compute/barrier → LLVM | `BCIRConvertToLLVM.cpp` | MLIR/C++ | ✅ (partial dialect) |
| GEM pipeline (classify/batch/schedule/lower, R12/R14–R16) | `BCIRGEMPasses.cpp` | MLIR/C++ | ✅ |
| K_BCIR **selection** (min-plus argmin over *declared* path costs) | `BCIRSelectPass.cpp` | MLIR/C++ | ✅ (trusts emitted costs) |
| K_BCIR **RCSP / Pareto** (budget label-DP + front) | `BCIRRcspPass.cpp` | MLIR/C++ | ✅ **ported** (reproduces 9472 + size-2 front) |
| **Cost model** (`_cost`, `candidates_for`, stride/tier penalties) | `bcir/kbcir/cost.py` + `realize.py` | → **MLIR/C++** | ☐ **Python-only** (the keystone gap) |
| **Fusion / deforestation / CSE** (`fused_candidates`, `_context_factor`) | `bcir/kbcir/realize.py` | → **MLIR/C++** | ☐ Python-only (needs the cost model first) |
| **Overlap (max,+)** scheduled price | `bcir/gem/overlap.py` | **`-bcir-overlap`** (`BCIROverlapPass.cpp`) | ✅ **ported** — reproduces `price_scheduled` (matmul makespan 253952 / gain 761856, corpus) |
| Min-plus **shortest path** over the layered DAG | `bcir/kbcir/semiring.py` + `realize.optimize` | **`-bcir-plan`** (`BCIRPlanPass.cpp`) | ✅ **ported** — reproduces the oracle's coupled `optimize` (7808 / corpus 1015808·101888·1595520) |
| Python→MLIR **emitter** (the bridge) | `bcir/lower/mlir.py` | Python (stays) | ✅ — but its job *shrinks* as the cost model moves to C++ |
| Portable **C23 kernel** emission | `bcir/lower/c_kernel.py` | C output (emitter may become C++) | ✅ emits C23 |
| **StreamPack ABI** codec + runtime | `runtime/c/` + Python encoder | **C** (frozen ABI) | ✅ CRC-gated parity |
| Telemetry ring (zero-copy) | `runtime/c` producer + Python reader | **C** | ✅ |
| Learned organs (bayescal/softdp/moegate/calibrate/regret) | Python | **Python** (freeze to Q8) | by design |

**Net:** the *law's structure* (verify, GEM pipeline, selection, RCSP/Pareto) is on
the MLIR rail; the *cost algebra that feeds it* is still Python. The select/RCSP
passes therefore trust the **emitter-baked** path costs. Closing that — computing
costs in C++ from the claim graph + target capability — is the next keystone.

## 2. The reformulated split (target homes)

- **MLIR/C++ (plan-time law):** dialect + verifier (R1–R16), the **whole** K_BCIR
  optimizer core — cost model, candidate enumeration, fusion/CSE/deforestation,
  min-plus shortest path, RCSP/Pareto, overlap (max,+), selection — and the lowering
  contracts. Goal: the MLIR rail recomputes the oracle's plan *from first principles*
  (claim graph + `bcir.target.capability`), not from emitted numbers.
- **C (run-time):** the StreamPack ABI codec, the deterministic executor hot path,
  the telemetry ring, the emitted compute kernels (C23). No allocation churn, no GC,
  arena/`mmap` lifetimes, SIMD-friendly layouts.
- **Python (train-time / graded):** the learned organs and the offline calibration
  SGD, each freezing to a generation-tagged Q8 artifact; plus the conformance
  **oracle** (`bcir/`) and the **generators** (`differential.gen_module`, `fuzz`)
  that keep the C++/C rails honest. The Python→MLIR emitter remains the test bridge.

## 3. Modernization: where C23 / C++23 / C++26 actually pay

Not novelty for its own sake — only where determinism, memory, or speed improves:

- **C++23 in the passes (now the build standard):**
  - `constexpr` **cost tables** — the seeded target constants (mem_unit, base_overhead,
    thermal/power density, tier Q8 factors) become `constexpr std::array` tables,
    folded at compile time and trivially deterministic across hosts.
  - `std::span` / `std::ranges` for cost-vector and candidate iteration (no copies,
    clearer than index loops); `std::mdspan` (C++23) for tile geometry in the matmul
    cost path.
  - `std::expected<Plan, Diagnostic>` for pass-internal cost/plan computation
    (explicit error channel alongside MLIR's `LogicalResult`).
  - structured bindings + designated initializers throughout (already used).
- **C++26 (where the toolchain allows):** gated — clang-18/gcc-13 `-std=c++2c` is
  experimental and the stock MLIR 18 headers don't compile clean under it, so C++23
  is the **gating** standard and the code stays `c++2c`-ready. Adopt C++26 niceties
  (e.g. `std::submdspan`, reflection when it lands) only behind a feature check.
- **C23 in the runtime + emitted kernels:** `constexpr` objects, `[[attributes]]`,
  `typeof`, `_BitInt(N)` for exact fixed-width lanes, and `#embed` for frozen Q8
  tables baked into the C runtime. The C-kernel emitter already targets C23; widen it
  to emit `restrict` + `_BitInt` + `[[assume]]` where the contract proves it safe.
- **Memory/speed where it matters:** the deterministic hot path (executor, StreamPack
  decode, cost evaluation in a search loop) is exactly where C/C++ beat the Python
  oracle — arena allocation, cache-friendly SoA cost vectors, branch-free min-plus.

## 4. The ordered build plan (from the oracle)

Each step is gated by the generated differential harness (the ready-made conformance
net) + FileCheck, and builds on the now-working LLVM 18/19 toolchain.

1. ✅ **`-bcir-cost-model` — port the K_BCIR cost algebra to C++ (the keystone). DONE.**
   `mlir/lib/passes/BCIRCostModel.cpp` computes each claim's candidate set + 12-d cost
   vectors from `bcir.claim` + `bcir.target.capability` (a faithful port of
   `cost.py::_cost` / `realize.candidates_for` / `_stride_penalty`, with a `constexpr`
   memory-tier table and the seeded constants read off the capability — extended with
   `mem_unit`/`base_overhead`/`thermal_density`/`power_density`/`per_op_heat`/`elem_bytes`,
   all defaulted to the CPU seeds). Annotates `kbcir.cm_candidates`/`cm_min_cost`/
   `cm_min_score`; reproduces the oracle bit-for-bit across op-classes — vec16 @ **7808**
   (compute 64, memory 3840), gather @ **528384**, tile @ **126976** — from the claim
   graph alone. `mlir/test/passes/cost_model.mlir` + a cross-check on
   `full_vec_add_ct1.mlir`. **The law no longer depends on emitter-baked path costs.**
2. ✅ **Fusion / deforestation / CSE in C++. DONE.** `-bcir-cost-model` now processes
   claims in (phase, declared) order with value-numbering + a produced-rid set,
   applying the two dependency-based redundancy credits (producer→consumer
   **deforestation** ×0.75 memory; **CSE** = compute zeroed + copy-priced memory),
   matching the oracle's `fused_candidates` bit-for-bit (7808 / 5888 / 5100) and
   annotating `kbcir.cm_fusion`. `mlir/test/passes/cost_model_fusion.mlir`. (The
   *path-based* shared-input half of `_context_factor` lands with step 3.)
3. ✅ **Min-plus shortest path over the layered DAG in C++. DONE.** `-bcir-plan`
   (`BCIRPlanPass.cpp`) runs the coupled tropical shortest path over the fused
   candidate columns, each edge coupling `_context_factor`'s path-based shared-input
   fusion (a wide candidate whose wide predecessor shares a read → ×0.75 memory). It
   reproduces the oracle's `optimize` bit-for-bit on *every* module: 7808 on
   `vector_add`, 13696 on the shared-input chain (`plan.mlir`), and the corpus —
   matmul **1015808**, scan **101888**, histogram **1595520** (the emitter now emits a
   registry + capability so the law plans from first principles; `gem_corpus.mlir`).
4. ✅ **Overlap (max,+) in C++. DONE.** `-bcir-overlap` (`BCIROverlapPass.cpp`) ports
   `gem/overlap.py`'s `price_scheduled`/`_makespan`: wave assignment by conflict, round-
   robin affinity bins, per-bin re-coupling against the in-bin predecessor, max over
   bins/tail, series over phases. Reproduces the oracle: matmul makespan 253952 / gain
   761856, scan & histogram gain 0, the shared-input chain gain 5888
   (`mlir/test/passes/overlap.mlir` + a corpus cross-check).
5. ✅ **Plan-level multi-claim RCSP. DONE.** `-bcir-rcsp-plan` (`BCIRRcspPlanPass.cpp`)
   ports `rcsp.optimize_constrained`: the accumulated-budget label DP over the fused
   candidate columns (labels carry score + per-tracked-dim totals; dominated labels
   pruned, infeasible extensions cut). A plan-wide cap bounds the *plan's* thermal/
   power, so it narrows just one claim where a per-claim cap cannot — reproduces the
   oracle: two vec16 claims (thermal 2176) under thermal≤2000 → {16, 8} @ 17280,
   under ≤1500 → {8, 8} @ 18944 (`mlir/test/passes/rcsp_plan.mlir`).

**The deterministic optimizer core is now fully on the MLIR rail (C++23):** cost
algebra + fusion (`-bcir-cost-model`) → coupled shortest path (`-bcir-plan` =
`optimize`) → (max,+) overlap (`-bcir-overlap`) → per-claim + plan-level constrained
search (`-bcir-rcsp`, `-bcir-rcsp-plan`), each bit-exact against the oracle and
gated by `check_passes.sh` + the differential harness.
6. **C runtime hardening (C23) + fuzz.** libFuzzer + ASan/UBSan on the StreamPack C
   decoder and the MLIR parser; `_BitInt`/`#embed` in the runtime.
7. **Deferred ("do later"):** PMU/RAPL/DVFS real-silicon calibration — the software
   path is merged; it needs a bare-metal rig to publish a measured replan win.

## 5. The immediate next build step

**The optimizer-core port (steps 1–5) is complete** — the entire deterministic
K_BCIR optimizer (cost algebra, fusion/CSE, the coupled min-plus shortest path, the
(max,+) overlap, and per-claim + plan-level constrained search) now runs on the MLIR
rail in C++23, each bit-exact against the Python oracle. The remaining lowering work,
in priority order:

1. **Named pass pipelines** (`bcir-plan`, `bcir-hydrate`, `bcir-lower-llvm`,
   `bcir-aot`, `bcir-audit`) wiring the now-complete passes into declared
   input/output-level pipelines with verifier checkpoints — stop relying on ad-hoc
   pass ordering.
2. **The cost model from a Θ context op** — generalize `_context_factor`'s thermal
   coupling (today the cool regime) by carrying Θ in the IR, so the C++ plan matches
   the oracle under hot/mem-bound Θ too.
3. **C runtime hardening (C23)** — `_BitInt`/`#embed`/`restrict` in the StreamPack C
   runtime + emitted kernels; libFuzzer + ASan/UBSan on the C/MLIR decoders.
4. **Deferred ("do later"):** PMU/RAPL/DVFS real-silicon calibration — the software
   path is merged; it needs a bare-metal rig to publish a measured replan win.

---

### (historical) Step 3 — the layered min-plus shortest path in C++. Build a
`-bcir-plan` pass that, over the fused candidate columns (one per claim, from the
cost model), runs `semiring.dag_shortest_path`: SOURCE → per-claim candidate nodes →
SINK, edge cost = `cand.cost.couple(_context_factor(prev, cand)).dot(w)`, where the
*path-based* `_context_factor` adds the shared-input fusion discount (prev wide + cand
wide + shared reads → ×0.75 memory). It emits the full plan (per-claim selected width
+ total score), making the C++ selection match the oracle's `optimize` for **all**
modules, not just the coupling-free per-claim argmin `-bcir-select-realization` does
today. Cross-check: it reproduces 7808 on `vector_add` and the corpus plan scores.
(Note: `_context_factor`'s *thermal* coupling needs Θ, which isn't in the IR — the
cool regime, valid for the whole corpus/curated set, is the first target; a `Θ`
context op generalizes it. Then steps 4–5: overlap (max,+) and plan-level RCSP.)
