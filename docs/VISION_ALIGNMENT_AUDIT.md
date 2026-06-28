# BCIR — Vision-Alignment Audit (2026-06-28)

> **Purpose.** A dated, evidence-backed honest-state snapshot mapping the
> **"C-as-Macro-Assembly / Registry-Oriented + IR-owns-everything + bare-metal-AI"**
> architectural vision onto what is *actually built* in the repository today. It is a
> companion to [`REPO_CURRENT_STATE_AUDIT.md`](REPO_CURRENT_STATE_AUDIT.md) (the dated
> changelog) and feeds the forward plan in
> [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) §5 and
> [`BCIR_ML_AI_INTEGRATION_ROADMAP.md`](BCIR_ML_AI_INTEGRATION_ROADMAP.md) §2/§6.
>
> Method: the vision was decomposed into seven testable pillars; each was independently
> audited against the source (file:line) and, where possible, exercised live (the C
> subset was run end-to-end on the MMIO register-map fixture). Verdicts are deliberately
> conservative — a pillar is only **ACHIEVED** when there is running, tested evidence.

---

## The thesis being audited

> Strip C down to its essence — a type-safe **"Macro Assembly" / Registry-Oriented**
> layer that maps to the physical silicon — while the **IR takes full ownership of the
> math, memory layouts, and execution streams**. C serves two hardware purposes: a
> **Registry Definition Layer** (structs/pointers map registers, MMIO, DMA boundaries)
> and a **Macro Target** (the IR lowers its dataflow graph into heavily-unrolled, flat,
> transparent C). On top of this, push **AI-driven optimization** down to the
> macro-assembly layer (cache/bank prediction, layout pivoting, autonomous fusion),
> lift legacy math libraries through the IR, and run **AI inference + a large part of
> training on bare-metal C, bypassing C++** — handing off to C++ only for dynamic-graph
> and distributed-orchestration complexity.

---

## Scorecard

| # | Pillar | Verdict | One-line state |
|---|--------|---------|----------------|
| 1 | **C = Registry Definition Layer** (registers / MMIO / DMA) | 🟢 ACHIEVED (core) · 🟡 boundaries | MMIO/volatile/bitfield/barriered-hazard done & Clang-gated; DMA-boundary + device-isolation modeling missing |
| 2 | **C = Macro Target** (flat unrolled C; IR owns math/layout/scheduling) | 🟢 ACHIEVED (core) | Emitter has zero scheduling logic; one flat statement per claim; specialist-kernel unrolling exists |
| 3a | "Layer-1 AI" cache-line / bank-conflict prediction | 🔴 MISSING | `memory` is a static 12-d cost formula, not a learned prediction |
| 3b | AI SoA ↔ AoS layout pivot before emit | 🔴 MISSING | `layout="soa"` is hard-coded; never transformed |
| 3c | Autonomous matmul+activation fusion via tropical min-plus | 🟡 PARTIAL | elementwise deforestation/CSE priced tropically; no matmul+activation fusion |
| 4a | Tropical rewriting of **kernel arithmetic** into semirings | ⚪ BY DESIGN NOT DONE | tropical algebra is the **cost optimizer**, not a kernel-arithmetic rewrite (correct & deliberate) |
| 4b | Lift legacy C math libraries **into** the IR | 🟡 PARTIAL | FFI call-out (`c.call.libm:`) only — BLAS + libm wrapped; LAPACK/FFTW/GSL/SLEEF not yet |
| 4c | Graph linearization into async strided data streams | 🟢 ACHIEVED | StreamPack: strided blocks, fork/await tokens, pipelined phases, channel dispatch |
| 4d | Q8↔float32↔Q8 bridge certified by R17 | 🟢 ACHIEVED | dual-rail R17 law (oracle + MLIR); compensated reduction bit-exact to int64 |
| 5a | Baked-weights inference kernel in C | 🟡 PARTIAL | `#embed` Q8 tables + `gem.matmul` plan exist; no end-to-end baked-weights kernel emitted |
| 5b | Forward/backward training kernels on bare metal | 🟡 PARTIAL | autodiff is a complete **Python-oracle-only** organ; not lowered to MLIR/C; no training loop (B4) |
| 5c | Tensor ops as first-class claims | 🟡 PARTIAL | `gem.matmul` done; no conv/attention/activation ops |
| 5d | C++ hand-off boundary (dynamic graphs / MPI / NCCL) | 🔴 MISSING | single-node by design; no distributed contract defined |

Legend: 🟢 achieved · 🟡 partial · 🔴 missing · ⚪ deliberately-not-done (vision clarification).

---

## Pillar-by-pillar evidence

### Pillar 1 — C as the Registry Definition Layer · 🟢 core / 🟡 boundaries

**Achieved.** `Domain.MMIO` (`bcir/model/lanes.py`) + per-resource domain (`bcir/model/graph.py`);
MMIO load/store emission `*(volatile uint32_t *)((volatile char *)regs + off)`
(`bcir/frontends/cfront/emit.py`); barriered hazard discipline on MMIO
(`hazard=="barriered"`, tested in `bcir/tests/test_cfront.py`); bitfield mask/shift
(`c.bf.get`/`c.bf.set`); real UART/DMA register fixtures (`runtime/c/cfront_regmap.c`,
`cfront_driver*.c`) ingested end-to-end through both rails, Clang-behaviour-equivalent.

**Gaps (boundaries).** No explicit **DMA-transfer-boundary** modeling (no `Domain.DMA`,
no transfer-size/alignment/atomic-unit claim — a DMA block currently lowers as scalar
load/store). No **device-isolation** annotation (two registers of the same device are
unrelated resources; cannot express "all UART registers serialize; GPU and CPU may
overlap"). Multi-register bitfields unsupported. `Domain.NVM` declared but unused.

### Pillar 2 — C as the Macro Target · 🟢 core

**Achieved & demonstrated live.** The MMIO fixture compiled through
`python -m bcir.frontends.cfront` emits flat, one-statement-per-claim C
(`regs->control = word` → `*(volatile uint32_t *)((volatile char *)regs + 4) = word;`;
`cfg.baud` → `(t >> 3) & 31u`) carrying an R1–R18 attestation + R13 provenance digest +
plan score. The emitter (`bcir/frontends/cfront/emit.py`) contains **no scheduling
logic** — phase/wave order is owned entirely by the IR (`bcir/gem/execute.py`,
`schedule.py`, `async_tokens.py`). Unrolled specialist kernels with compile-time trip
counts + explicit remainders exist (`bcir/lower/specialist.py`).

**Gaps (transparency / formalization).** Bounds checks emit as an opaque
`BCIR_CHK(...)` call rather than inline guards (less "transparent assembly"). There is
no normative **flatness law** (e.g. "every claim lowers to ≤N flat statements; no loops
in the IR") and no `emit_style="flat"` mode that forbids C loops. No DMA-descriptor
specialist (a DMA transfer should be one descriptor, not unrolled scalars). The
"IR owns ordering; C is a generated textual artifact" contract is implicit, not written
into the LangRef.

### Pillar 3 — AI-driven optimization at the macro-assembly layer · 🔴 / 🟡

**This is the largest gap between vision and implementation.**

- **3a cache/bank prediction — MISSING.** The `memory` cost dimension
  (`bcir/kbcir/cost.py`) is a **static** Q8 formula over bandwidth/latency factors +
  stride penalty; there is no model that *predicts* cache-line utilization or memory-bank
  conflicts from graph shape. The allocator heat-scorer (`bcir/kbcir/allocator.py`) is a
  frozen linear model that picks a memory *tier* (L1/L2/L3/DRAM/HBM), not a cache
  predictor.
- **3b SoA↔AoS pivot — MISSING.** `Resource.layout` defaults to `"soa"`
  (`bcir/model/graph.py`) and is never analyzed or transformed; the MLIR lowering
  hard-codes `#bcir.layout<soa>`.
- **3c autonomous fusion — PARTIAL.** Deforestation + CSE fusion exists and *is* priced
  by the tropical min-plus optimizer (`bcir/kbcir/realize.py`), but only for adjacent
  **elementwise** claims sharing an operand. There is no **matmul+activation** fusion and
  no graph-shape-driven fusion synthesis.

**Learned organs that DO exist (but are off the default path).** RL allocator placer,
GNN MoE router (`moegate`), Bayesian calibrator (`bayescal`), soft-DP posterior
(`softdp`), regret sensor (`sensing`/`regret`), e-graph (`egraph`/`memory`) — all in
`bcir/kbcir/`, all freeze to Q8, all **opt-in** (the only learned artifact on the default
path is the hand-set allocator placer). They optimize *plan selection*, not memory layout
or fusion — i.e. none of them is the "Layer-1 AI" the vision describes.

### Pillar 4 — tropical / lifting / linearization / precision

- **4a kernel-arithmetic tropical rewrite — BY DESIGN NOT DONE (vision clarification).**
  BCIR's tropical (min-plus) + (max-plus) algebra is the **compilation cost optimizer**
  (`bcir/kbcir/semiring.py`, `matmul.py`): min-plus is shortest-*cost*-path over the
  candidate-realization DAG; max-plus is the roofline bottleneck within a candidate. The
  user kernel's own arithmetic stays standard ring arithmetic (`matmul` is `+=`/`*`;
  `quantize.scaled_dot` is `*`/`+`). Rewriting kernel arithmetic into min-plus would
  change computed results, so this is correctly **not** done — the vision phrasing
  ("addition becomes minimum… matrix ops become path-finding networks") describes the
  planner, not the kernel.
- **4b library lifting — PARTIAL (FFI, not lift).** The `c.call.libm:` edge wraps a
  trusted external kernel at link time (B5 `cblas_sgemm` with a portable reference
  fallback, `bcir/lower/c_kernel.py::emit_blas_gemm_c`); BCIR owns the *calling* side.
  This is honest "integrate, don't reinvent," but it is **not** a source-level lift of
  library code into the claim graph. Wrapped today: **BLAS** (gemm) + **libm** (sqrt/pow/
  fabs/…/complex). **Not yet wrapped: LAPACK, FFTW, GSL, SLEEF** — the Area-B work.
- **4c linearization — ACHIEVED.** The StreamPack (`docs/BCIR_STREAMPACK_ABI.md`,
  `bcir/gem/streampack.py`) is a portable linearized artifact: strided blocks, `!bcir.token`
  fork/await async DAG, double-buffered pipelined phases (v2), heterogeneous channel
  dispatch. *Gap:* the freestanding C executor runs v1 serial phases; the v2 pipelined
  executor in `runtime/c/` is pending.
- **4d Q8↔f32↔Q8 R17 bridge — ACHIEVED.** R17 is a first-class dual-rail law
  (`bcir/verify` + `mlir/lib/passes/BCIRVerifyPass.cpp`); per-group Q8 quantization
  (`bcir/kbcir/quantize.py`), static per-claim ULP error bounds
  (`bcir/kbcir/precision.py`), compensated reduction bit-exact to int64.

### Pillar 5 — bare-metal AI inference/training + the C++ boundary · 🟡 / 🔴

- **5a baked-weights inference — PARTIAL.** Frozen Q8 tables bake via C23 `#embed`
  (`bcir/abi/q8_tables.py`, fixtures); `gem.matmul` plans + lowers (MLIR
  `BCIRLowerGemMatmulPass.cpp`). But no end-to-end function is emitted that bakes a
  specific model's weights as `static const …[]` and fuses them into a single-pass kernel.
- **5b forward/backward training — PARTIAL.** `bcir/kbcir/autodiff.py` is a complete,
  correct, content-addressed reverse-mode autodiff organ (forward eval, reverse/symbolic
  gradients, Hessians, finite-difference-gated) — but it is **Python-oracle-only, kept
  cold, not lowered to MLIR/C**, so gradients cannot run on the deployed rail. The
  hybrid tropical-structure + gradient-tune training loop (B4) is not built.
- **5c tensor ops as claims — PARTIAL.** `gem.matmul` is a first-class planned claim with
  K_BCIR tile/loop search; there are no `gem.conv` / `gem.attention` / `gem.activation`
  ops.
- **5d C++ hand-off boundary — MISSING.** No MPI/NCCL/distributed/dynamic-graph
  machinery anywhere; the design is single-node, and the contract for "too big for C —
  hand off to C++" is undefined.

---

## Prioritized remaining-work backlog

Ordered by leverage toward the vision, holding the two-truth quarantine +
prototype-then-port discipline. Each item is PR-sized and parity-gated.

**Immediate (the queued Area-B slices — they advance Pillar 4b + the calling-side half of
Pillar 3):**
1. **B1 — `bcir-cc --emit-c` automatic link-flag emission** (`-lblas`/`-lfftw3`/… emitted
   from the `c.call.libm:` edges a unit uses) so the FFI wrap is linkable end-to-end.
2. **B2 — wrap a new C math library** (FFTW *or* LAPACK *or* GSL *or* SLEEF) through
   `c.call.libm:`, B5-style, with the R17 Q8↔f32↔Q8 bridge at the seam.
3. **B3 — calling-side tuning** (layout / tiling / prefetch / channel selection) around a
   wrapped kernel — the first concrete step toward Pillar-3 layout intelligence.
4. **Area B breadth — ATLAS / GSL / FFTW / OpenBLAS-LAPACK / SLEEF** wrapped through the
   same edge.

**Pillar-3 intelligence (the largest vision gap):**
5. A **memory-layout cost producer** + a SoA↔AoS selection step the cost model can price
   (Pillar 3b), prototyped in the oracle, ported to the law.
6. A **cache/bank cost signal** trained from microbench measurements, frozen to Q8 and fed
   to the cost model as a new dimension (Pillar 3a).
7. **matmul+activation fusion** — add fusible `gem.activation` ops + a fused lowering, let
   the bundle/deforestation optimizer price it (Pillar 3c, 5c).

**Pillar-5 bare-metal AI:**
8. **Lower B3 autodiff to a backward-pass kernel** (MLIR/C), parity-gated to the oracle —
   unlocks training on the deployed rail.
9. **Baked-weights inference kernel emitter** — `static const` weights + fused single-pass
   loop for a frozen model (Pillar 5a).
10. Define the **C↔C++ hand-off boundary** doc-first (dynamic-graph + distributed), even
    before building it, so the single-node limit is explicit (Pillar 5d).

**Pillar-1/2 boundaries & formalization:**
11. `Domain.DMA` + DMA-descriptor specialist + device-isolation annotation (Pillar 1).
12. A normative **flatness/ownership law** in the LangRef + an optional inline-bounds
    `emit_style="flat"` (Pillar 2).

---

## Bottom line

The **foundation the vision rests on is real and demonstrable**: C *is* a thin,
type-safe, registry-oriented macro-assembly target; the IR *does* own scheduling and the
math; the linearized StreamPack and the certified Q8 precision bridge exist. The
**unbuilt half is the intelligence and the ML payload**: the macro-assembly-layer
"Layer-1 AI" (cache/bank prediction, layout pivoting, autonomous fusion), the breadth of
library integration, and the end-to-end bare-metal inference/training pipeline. The
queued Area-B work is the correct next increment, and items 5–10 above are the path from
"a verifiable C macro-assembly substrate" to "an AI-optimizing compiler that runs
inference and training on bare metal."
