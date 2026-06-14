# BCIR Strategy & Roadmap

> Living strategy note. Pairs with the honest snapshot in
> [`REPO_CURRENT_STATE_AUDIT.md`](REPO_CURRENT_STATE_AUDIT.md) and the normative
> status in [`BCIR_LANGREF.md`](BCIR_LANGREF.md) §16 (milestone map). Last
> revised 2026-06-14 (Phases 13–26 + the oracle optimization pass + the
> MLIR-native GEM pipeline passes).

## 1. Positioning (the decision)

**BCIR is a cost-governed planning + verification layer that sits *above* LLVM
and *inside* drivers/runtimes — not a from-scratch replacement for Clang.**

Replacing Clang the C/C++ *frontend* is out of scope and not the point: BCIR has
no C/C++ source frontend (it ingests claims via ROP/MAP and the M5 event
transduction layer, not C++). What BCIR adds is a level LLVM does not model:

- **Cost as a first-class IR object** — a 12-d cost vector with a tropical
  (min,+) / RCSP optimizer selecting realization paths.
- **Θ-feasibility** — live machine state (thermal/power/contention) changes
  *legality*, not just cost: a hot machine makes wide SIMD infeasible.
- **A principled, enforced ML boundary** — the L0–L3 learning-placement law and
  the two-truth quarantine (graded confidence may inform, never legislate).
- **Provenance + reproducibility as obligations** — the manifest spine, memory
  fixpoints, and the R1–R13 verifier.

These are runtime/planning concerns. The highest-ROI shapes for BCIR are therefore:

1. **A cost-governed scheduling/placement + verification layer above LLVM**
   (emit LLVM IR or C; let LLVM do isel/regalloc). Lowest risk, plays to every
   strength, and makes "no native codegen" a non-issue by design.
2. **The planning brain of an AI-accelerator / heterogeneous-SoC runtime or
   driver** — the GEM StreamPack is already designed as the hot artifact
   *rehydrated on Θ/generation mismatch*: a driver-resident specializer.
3. **A "principled ML in compilers" research vehicle** — the L0–L3 quarantine
   and the classical/graded two-truth separation are genuinely novel.

"Forcing a refactor of Clang/LLVM" is a research result, not a milestone: it
means demonstrating, on silicon, that cost-as-IR + Θ-feasibility + frozen-learned
planning beats LLVM's cost models on workloads LLVM handles poorly (irregular
memory, multi-target placement, power/thermal-capped kernels). Reachable in a
**niche first**, far sooner than any general claim.

## 2. Where we are (2026-06-14)

- **The spine** (registry/claims/phases/lanes, K_BCIR optimizer, GEM hydration +
  deterministic executor, StreamPack ABI + freestanding C runtime, R1–R13
  verifier) — runnable and pinned (`vector_add` AVX-512 cool Θ → vec16, **7808**).
- **The intelligence organs** (Phases 13–26): physics-anchored + Bayesian/conformal
  calibration, policy portfolio + replay gate, MoE GNN gate, propose-verify
  accelerator, soft/differentiable optimizer, MDL regret ledger, provenance
  manifest, e-graph + memory-module fixpoints (`a = Lim(Res(U))`), the two-truth
  quarantine, modular mapping functions, and the enriched-operad memory interface.
- **Hot/cold separation** — verified and locked (`bcir/tests/test_hot_cold.py`):
  the executor and ABI codec carry decisions, never models; no
  planning→execution→telemetry runtime recursion (the one real recursion,
  provenance re-planning, was removed in the optimization pass).
- **The MLIR rail** — the ODS dialect family (~80 ops), `bcir-opt` with
  `-bcir-verify` (R1–R13), `-bcir-promote-lanes`, `-convert-bcir-to-llvm`, and
  now the **GEM pipeline passes** `-bcir-classify-lanes / -select-realization /
  -batch / -schedule / -lower-to-llvm`, cross-checked against the oracle
  (`-bcir-select-realization` recomputes the min-plus score and reproduces
  **7808** cool / **9472** under the thermal cap).

The honest asymmetry (see the audit): the optimizer/intelligence stack is rich,
while **native code generation is absent** (machine code exits through
clang/llc/lli) and **cost constants are seeded, not measured**. The strategy is
to close that gap, not add more intelligence.

## 3. Roadmap

### Near term (substrate + evidence — the priority)
1. **Close the calibration loop on real hardware.** Wire `kbcir.microbench` →
   frozen Q8 tables → CI, with a trained calibrator and a real measurement run,
   so selected plans are optimal w.r.t. silicon, not a model. Add a measured
   replan win to the worked-example matrix.
2. **C++/MLIR GEM passes against the oracle.** ✔ (this milestone) — the five
   declared GEM passes are implemented MLIR-native and cross-checked against the
   oracle's pinned constants. Next: widen them past the single-claim example
   (multi-claim batching/fusion, real durations).
3. **Widen the corpus** — reductions, real tiled matmul, scan; per-target
   worked-example parity beyond `vector_add`.

### Mid term (one target, end to end)
4. **A first-class C backend** (kernel lingua franca) — emit clean C from the
   selected StreamPack so the resident toolchain compiles it; the natural artifact
   for a driver-resident JIT. (See §4.)
5. **One `bcir.target.lower_contract` end to end** — pick a niche where BCIR's
   cost model beats LLVM's (gather/scatter-heavy or power-capped) and show a
   measured win.

### Long term (the research result)
6. **Driver/runtime integration** — BCIR as the rehydrating planner inside an
   accelerator runtime (Θ-driven replan, generation-tagged repack).
7. **BCIR-native instruction selection** behind `lower_contract`, only where a
   target's economics demand it (do not chase general isel).

## 4. C and C++ implementation plan

**The port boundary is already drawn — it is the L0–L3 / two-truth line**
(`bcir/tests/test_hot_cold.py` is its executable contract):

- **Port to C++ (deterministic, classical, hot-path-adjacent).** The model
  (registry/claims/phases), the R1–R13 verifier (the structural subset is already
  `-bcir-verify`), the K_BCIR optimizer (min-plus + RCSP), GEM hydration + the
  deterministic executor, the StreamPack ABI codec (the C runtime exists), and
  the lowering contracts. The MLIR rail is the vehicle; the GEM pipeline passes
  are the first installment.
- **Keep in Python (offline, graded, L2/L3).** `bayescal`, `softdp`, MoE-gate
  training, `microbench`, the regret ledger. By §13 these are float and
  non-deterministic across architectures; they stay offline and emit **frozen
  Q8, generation-tagged** artifacts the C++ side consumes. Porting them would
  violate the quarantine.
- **Enriched operad / memory interface** — C++ only if/when it becomes
  load-bearing for plan-time caching/retrieval; otherwise Python-side tooling.

### The C backend (for discussion)
- **Keep a C backend as a first-class lowering target.** C is the universal
  kernel language (CUDA C / HIP / OpenCL C / ISPC / plain CPU C / eBPF-restricted
  C). Emitting C (a) sidesteps native isel while remaining a real deliverable,
  (b) is the natural artifact for a driver-resident JIT (emit → compile with the
  resident toolchain), and (c) gives portability LLVM-IR alone does not. The
  codegen layer already has a C-source fallback target; promote it.
- **Should drivers be compilers?** Yes — and that is where BCIR belongs. The
  StreamPack-rehydration design *is* a driver-resident specializer; the product
  is the planning + verification brain inside a runtime (like a GPU shader
  compiler or an accelerator runtime), emitting C/LLVM for the resident backend.
- **Native isel is deferred**, not abandoned: pursue it only for a specific
  target whose economics beat "emit C/LLVM + reuse the resident backend."

## 5. Risks (carry-forward)
1. **Substrate/intelligence inversion** — learned planning over a backend that
   can't codegen and tables that aren't measured. Mitigation: §3 near-term.
2. **Multi-rail divergence** — oracle ↔ MLIR ↔ (future C++). Mitigation: every
   C++ stage cross-checks the oracle's pinned constants in CI (the GEM passes set
   the pattern).
3. **Validation realism** — green ≠ competitive. Mitigation: a real-workload
   corpus + a measured comparison vs `clang -O3` on niche kernels.
4. **Complexity / bus factor** — theory ahead of load-bearing utility.
   Mitigation: prioritize substrate + measurement; let new intelligence land only
   behind a measured win.
