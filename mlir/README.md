# `mlir/` — the BCIR dialect family (the IR law)

Source of truth for BCIR syntax/types/attributes/operations, authored in
TableGen/ODS per [`../docs/BCIR_LANGREF.md`](../docs/BCIR_LANGREF.md), plus a pure
**IRDL projection**. The Python package under [`../bcir/`](../bcir/) is the
executable conformance oracle that must agree with these definitions
([`../docs/PARITY.md`](../docs/PARITY.md)).

> **Validated on LLVM 18** (`mlir-opt`/`mlir-tblgen`/`bcir-opt` 18.1.3), gated in
> CI (job `mlir-rail-validate`):
> - **ODS rail** — every TableGen generator (decls **and** defs) passes for the
>   whole `.td` family: `tools/wsl/tblgen_check.sh`.
> - **IRDL rail** — the projection loads into stock `mlir-opt` and the
>   generic-syntax corpus in `test/irdl/` round-trips: `tools/irdl/check_corpus.sh`.
> - **Compiled `bcir-opt` (LangRef M3)** — `lib/BCIRDialect.cpp` + `tools/bcir-opt.cpp`
>   build the real dialect (`tools/wsl/build_mlir.sh`); the *pretty* ODS corpus in
>   `examples/` parses/verifies + FileCheck-round-trips through it
>   (`tools/wsl/check_ods_examples.sh`).
> - **Named pipelines (verifier-checkpointed)** — `registerBCIRPipelines` wires the
>   passes into declared input/output-level pipelines: `bcir-audit` (verify ->
>   cost/plan/overlap), `bcir-optimize` (claims+H -> coupled plan), `bcir-hydrate`
>   (plan -> StreamPack), `bcir-lower-llvm`, and `bcir-aot` (verify -> hydrate ->
>   LLVM). A `bcir.kbcir.theta` op carries the runtime state so `-bcir-plan`/`-overlap`
>   apply the hot-Theta thermal coupling (`test/passes/theta_hot.mlir`).
> - **Modular pass library (C++23)** — `bcir-opt` is a real compiler, not a parser.
>   The passes are one translation unit per group under `lib/passes/`
>   (`BCIRVerifyPass`, `BCIRPromotePass`, `BCIRConvertToLLVM`, `BCIRGEMPasses`,
>   `BCIRSelectPass`, `BCIRRcspPass`), sharing `lib/passes/BCIRPassSupport.h`;
>   `lib/BCIRPasses.cpp` is registration-only. `-bcir-verify` (R1–R16),
>   `-bcir-promote-lanes` (GGG→UX), `-convert-bcir-to-llvm` (compute/barrier → LLVM).
>   Tests in `test/passes/`, gated by `tools/wsl/check_passes.sh`. The C/C++/MLIR
>   placement map + roadmap is `docs/BCIR_MASTER_ROADMAP.md`.
> - **Cost model (optimizer core, C++23)** — `-bcir-cost-model` (`lib/passes/BCIRCostModel.cpp`)
>   ports `cost.py`: recomputes each claim's candidate cost vectors from
>   `bcir.claim` + `bcir.target.capability` (constexpr tier table + seeded constants),
>   so the law stops trusting emitter-baked path costs. Reproduces vec16 @ 7808 /
>   gather @ 528384 / tile @ 126976 from the claim alone (`test/passes/cost_model.mlir`).
> - **Plan / min-plus shortest path (optimizer core, C++23)** — `-bcir-plan`
>   (`lib/passes/BCIRPlanPass.cpp`) is the full `realize.optimize` in C++: the coupled
>   tropical shortest path over the fused candidate columns (shared cost+fusion logic in
>   `lib/passes/BCIRCostModel.h`). Reproduces the oracle's coupled score on every module —
>   7808, the shared-input chain 13696 (`test/passes/plan.mlir`), and the corpus
>   1015808 / 101888 / 1595520 (`gem_corpus.mlir`).
> - **Overlap / scheduled price (optimizer core, C++23)** — `-bcir-overlap`
>   (`lib/passes/BCIROverlapPass.cpp`) ports `gem/overlap.py`: the (max,+) wave makespan
>   M(pi,Theta) over the coupled plan. Reproduces the oracle — matmul makespan 253952 /
>   gain 761856, the shared-input chain gain 5888 (`test/passes/overlap.mlir`).
> - **RCSP plan-level (optimizer core, C++23)** — `-bcir-rcsp-plan`
>   (`lib/passes/BCIRRcspPlanPass.cpp`) ports `rcsp.optimize_constrained`: the
>   accumulated-budget label DP over the plan. A plan-wide cap narrows one claim where a
>   per-claim cap cannot — thermal≤2000 → {16,8} @ 17280 (`test/passes/rcsp_plan.mlir`).
>   **With it the whole deterministic optimizer core is on the MLIR rail** (cost+fusion →
>   shortest path → overlap → constrained search; see `docs/BCIR_MASTER_ROADMAP.md`).
> - **RCSP / Pareto (optimizer core, C++23)** — `-bcir-rcsp` (`lib/passes/BCIRRcspPass.cpp`) ports
>   `bcir/kbcir/rcsp.py`: the budget-feasible min-plus label-DP argmin + the Pareto front
>   over (score, thermal, power). Reproduces the oracle's 9472 under the 700 cap and the
>   {vec16, vec8} front (size 2); `test/passes/rcsp.mlir` + a cross-check on `gem_corpus.mlir`.
> - **GEM pipeline + generated parity** — `-bcir-classify-lanes / -select-realization /
>   -batch / -schedule / -lower-to-llvm` recompute the oracle's plan; `test/passes/
>   gem_passes.mlir` (curated `vector_add` @ 7808/9472) and `test/passes/gem_corpus.mlir`
>   (the widened corpus — real matmul/scan/histogram — **generated** by the oracle via
>   `bcir.lower.mlir.to_mlir`; regenerate with `python -m bcir.kbcir.differential
>   --emit-corpus`) both FileCheck the law's recomputed min-plus scores.

## Dual rail

```
Track A (ODS):   include/BCIR/*.td + passes/*.td + CMakeLists.txt -> bcir-opt
                 full dialect, pretty syntax, R1-R12 verifier, optimizer.
Track B (IRDL):  irdl/bcir.irdl.mlir -> stock mlir-opt --irdl-file=...
                 pure-data structural projection, generic syntax, portability proof.
```

## Dialect family (one `bcir` namespace during bootstrap; distinct layers of law)

| File | Layer | Defines |
|---|---|---|
| `include/BCIR/BCIRDialect.td` | — | dialect + base Op/Type/Attr classes |
| `include/BCIR/BCIRAttrs.td` | — | Lane/StrideClass/Domain/Hazard/Verify/Bounds/Layout/Policy/Semiring/CostClass/MemTier/Access enums; Precision; 12-d CostVector |
| `include/BCIR/BCIRTypes.td` | — | `!bcir.resource`, `!bcir.path`, `!bcir.stream` |
| `include/BCIR/BCIRInterfaces.td` | — | `KBCIRCostOpInterface` |
| `include/BCIR/BCIRCoreOps.td` | BCIR-0..2 | module/registry/resource/phase/claim/index_range/load/store/compute/barrier |
| `include/BCIR/BCIRTargetOps.td` | H | `target.capability` |
| `include/BCIR/BCIRMemOps.td` | CT1 | `mem.tier` / `mem.ham` / `mem.cxl_swap` |
| `include/BCIR/BCIRKBCIROps.td` | BCIR-3 | `kbcir.policy/path/plan/select` |
| `include/BCIR/BCIRGEMOps.td` | BCIR-4 | `gem.stream_pack/lane_segment/prefetch/block` |
| `include/BCIR/BCIRTraceOps.td` | cross | `trace.note` |
| `include/BCIR/BCIRVerifyOps.td` | M1 | `verify.*` (R1–R12 as IR) |
| `include/BCIR/BCIROptOps.td` | M2 | `opt.*` (rewrite/layout/mem laws as IR) |
| `include/BCIR/BCIRLoweringContractOps.td` | M3 | `isa.*` / `packet.*` / `target.lower_contract` |
| `include/BCIR/BCIREventOps.td` | M5 | `event.stream/kind/emit/consume` |
| `include/BCIR/BCIRTransducerOps.td` | M5 | `fsm.machine/state/transition/stack/capture/reduce` |
| `include/BCIR/BCIRParseOps.td` | M5 | `parse.grammar/token/rule/lower_to_fsm` |
| `include/BCIR/BCIRBinaryFormatOps.td` | M5 | `binary.format/field/record/decode` |
| `include/BCIR/BCIRAsyncOps.td` | Phase 8 | `async.fork`/`async.await` (`!bcir.token`) |
| `include/BCIR/BCIROps.td` | — | umbrella (op-gen entry point) |
| `passes/GEMPasses.td` | — | `-bcir-classify-lanes`, `-bcir-select-realization`, `-bcir-batch`, `-bcir-schedule`, `-bcir-lower-to-llvm` |

The canonical pretty IR is `examples/full_vec_add_ct1.mlir`; the IRDL smoke is
`test/irdl/00_smoke_generic.mlir`.
