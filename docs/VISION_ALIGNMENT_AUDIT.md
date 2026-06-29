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
| 3a | "Layer-1 AI" cache-line / bank-conflict prediction | 🟢 BUILT (dual-rail) | frozen-Q8 contention predictor (line-waste + bank-conflict) on the CONTENTION axis, informs-only; oracle `cache_predict.py` + MLIR `-bcir-cache-contention` |
| 3b | AI SoA ↔ AoS layout pivot before emit | 🟢 BUILT (dual-rail) | cost-priced SoA↔AoS selection (stride-penalty), address-map-invariant; oracle `layout.py` + MLIR `-bcir-layout-pivot` |
| 3c | Autonomous matmul+activation fusion via tropical min-plus | 🟢 BUILT (dual-rail) | sole-consumer epilogue fusion priced by the deforestation discount; oracle `fusion.py` + MLIR `-bcir-fuse-matmul-activation` |
| 4a | Tropical rewriting of **kernel arithmetic** into semirings | ⚪ BY DESIGN NOT DONE | tropical algebra is the **cost optimizer**, not a kernel-arithmetic rewrite (correct & deliberate) |
| 4b | Lift legacy C math libraries **into** the IR | 🟡 ADVANCED | FFI call-out (`c.call.libm:`) + auto link-flags (B1) + a new FFTW wrap (B2) + calling-side tuning (B3); LAPACK/GSL/SLEEF breadth = remaining Area-B |
| 4c | Graph linearization into async strided data streams | 🟢 ACHIEVED | StreamPack: strided blocks, fork/await tokens, pipelined phases, channel dispatch |
| 4d | Q8↔float32↔Q8 bridge certified by R17 | 🟢 ACHIEVED | dual-rail R17 law (oracle + MLIR); compensated reduction bit-exact to int64 |
| 5a | Baked-weights inference kernel in C | 🟢 BUILT | `emit_inference_kernel_c`: `static const` weights (`#embed`/literal) fused single-pass, R17-bounded; reference-verified bit-exact (relu) |
| 5b | Forward/backward training kernels on bare metal | 🟢 BUILT | `emit_autodiff_kernel_c` lowers the autodiff DAG to a forward+backward C kernel + SGD step; gradients match oracle + finite-difference |
| 5c | Tensor ops as first-class claims | 🟢 BUILT (dual-rail) | `gem.matmul`/`activation`/`conv`/`attention` — all oracle + MLIR law ops, reference-verified |
| 5d | C++ hand-off boundary (dynamic graphs / MPI / NCCL) | 🟡 SCAFFOLDED | boundary contract defined + a compilable, round-trip-tested single-node seam; dynamic/distributed backends are marked stubs |

Legend: 🟢 achieved/built · 🟡 partial/advanced · 🔴 missing · ⚪ deliberately-not-done (vision clarification).

> ### Build update (2026-06-28, post-gap-program)
> The 🔴/🟡 pillars above (except 4a, which is correct as-is) were **built and merged** in a
> 21-PR program this cycle (oracle prototype → MLIR law, each parity-gated + CI-green):
> **3a** cache/bank contention (G4), **3b** SoA↔AoS pivot (G3), **3c** matmul+activation fusion
> (G2), **5a** baked-weights inference (G5), **5b** forward/backward training kernel (G6),
> **5c** the `gem.activation`/`conv`/`attention` ops (G1/G7) — all **also ported to the MLIR
> law rail** (`gem.activation`/`fused_matmul_activation`/`conv`/`attention`/`layout_pivot`/
> `contention`), and **4b** advanced via B1/B2/B3. Conformance **956 → 1235**. The per-pillar
> sections below retain the original audit prose for provenance; the verdicts above are current.
> Remaining: **4b breadth** (LAPACK/GSL/SLEEF), **5d** distributed/dynamic backends (need
> multi-node hardware), and the Pillar-1/2 boundary items (DMA/device-isolation, flatness law).

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

**Telemetry signal-provider registry (T1) — BUILT (cost-side only).** The graded L2/L3
measurement seam now has a vendor-neutral PAPI-component registry
([`SIGNAL_REGISTRY.md`](SIGNAL_REGISTRY.md), `bcir/signal_registry.py`): typed providers
wrapping `bcir/silicon.py` (thermal/die-temp → `thermal`, RAPL energy → `power`, cpufreq →
`compute`, L1/L2/L3 cache → `memory`, PMU capability → `compute`) plus honest-unavailable
gap providers (GPU/BMC power, mem-bandwidth, fabric, throttle, reliability) so the namespace
is complete and a future NVML/amd-smi/Redfish backend just fills them in. Provenance/units
live on a `MetricDefinition` (Redfish split); a `register()` plugin seam mirrors the channel
plugin; `registry_for_channel` maps `energy_source`→power provider. It is strictly on the
*cost/optimization* side: it returns only `Reading`/`None` (never a verdict/Diagnostic),
surfaces a graded 0..100 signal to feed `theta`, and touches neither `bcir/verify` nor the
cost-vector DIMS — the two-truth quarantine, applied to measurement.

**UART telemetry frame (T2) — BUILT (cost-side only).** The embedded telemetry tap now has a
framed, CRC-sealed, resync-able transport ([`TELEMETRY_FRAME_ABI.md`](TELEMETRY_FRAME_ABI.md),
`bcir/telemetry_frame.py`): a producer drains `TelemetryRing` and emits self-delimiting frames
(`"BTLM"` magic | version | seq | timestamp | the ring's 56-byte `<7q>` records | CRC-32) over a
byte egress; the host decoder reuses the RT3 gate (`sanitize_events`/`TelemetryIntegrity`),
resyncs on magic, and bounds a corrupt byte to one frame. It is dual-rail (a freestanding C twin
`runtime/c/bcir_telemetry_frame.{c,h}` pinned byte-identical to the Python reference, CRC reused
from `bcir_runtime.c`), mirroring the StreamPack discipline. Egress-over-UART (`out` → `uart_send`)
is a documented adapter, not a hardware dependency. Strictly cost-side: a frame carries graded
L2/L3 data, never a verdict/`Diagnostic`, and touches neither `bcir/verify` nor the cost DIMS.

**Derived metrics + plan-cost sensitivity (T3) — BUILT (cost-side only).** The telemetry pipeline
now has the SPICE `.MEASURE`/`.SENS` analogy (`bcir/kbcir/telemetry_metrics.py`,
[`TELEMETRY_PIPELINE_RESEARCH.md`](TELEMETRY_PIPELINE_RESEARCH.md) §6): edge-computed
figures-of-merit (`derive_field_metrics`/`derive_all` → per-field max/min/avg/rms/count with
documented float-free round-half-up `avg` and an inspection-only `rms`; `measure_trig_targ` for the
`TRIG…TARG` crossing span) over RT3-sanitized batches, plus a `signal_sensitivity` ranking that
**finite-differences the existing `realize.optimize`** over a perturbed `Theta` pressure (mapping a
T1-registry `cost_dim` → a Theta channel: thermal/power/memory/contention) and ranks signals by
`|Δscore|` so `sampling_budget` can steer a fixed budget toward the high-impact signals (thermal
ranks top on a tile-heavy `matmul_tiled`). Two-truth: it perturbs COST/Theta only, reuses (never
reimplements) the cost model, emits no `Diagnostic`, and the module still `verify()`s clean before
and after — it never reads or alters the legality path.

**Telemetry export adapters (T4) — BUILT (cost-side only).** The telemetry pipeline now has its
external export boundary (`bcir/telemetry_export.py`,
[`TELEMETRY_PIPELINE_RESEARCH.md`](TELEMETRY_PIPELINE_RESEARCH.md) §6): read-only egress of the T1
registry readings + T3 metrics in the two industry-standard shapes plus an out-of-band read,
stdlib-only (`json`/`re`). **Prometheus/OpenMetrics** (PULL, `to_prometheus`/`scrape`): `# HELP`/`#
TYPE`/`bcir_<name>{channel,cost_dim,provenance,unit} <value>` lines — a monotonic-counter source
(RAPL `energy_uj`, `cycles`) → `counter`, a pressure/level/capacity → `gauge`; an unavailable signal
emits no value but a `bcir_signal_up{...} 0` (the honest real/unavailable split); names sanitized to
the Prometheus charset; deterministic. **OTLP** (PUSH, `to_otlp`/`otlp_to_json`/`export_push` + a
frozen `OtlpMetric`): the OTLP-JSON `resourceMetrics`/`scopeMetrics` data model with kind +
temporality + monotonicity declared EXPLICITLY per metric (counters = sum+cumulative+monotonic;
gauges = gauge+unspecified+non-monotonic) and the channel as the `Resource`. **Redfish** (out-of-band
PULL): `to_redfish_metric_report` + `metric_definitions` (the 4-split — units/provenance on the
`MetricDefinition`) + `parse_redfish_metric_report` (parse a BMC report back into `Reading`s, tolerant
of foreign fields). The `TelemetryIntegrity` witness exports as `up`/`accepted`/`rejected`/`blind` so
suppression stays observable. Honest depth: no live collector/server/BMC — these produce/parse the
exposition bytes+JSON (the testable contract); HTTP/gRPC/protobuf transport is a documented unbuilt
adapter. Two-truth: read-only egress of the graded signal — no `Diagnostic`, touches neither
`bcir/verify` nor the cost DIMS, never a verdict.

**Telemetry pipeline (T1–T4) — COMPLETE (cost-side only; never a verdict).** With T4 the
telemetry/monitoring pipeline is built end-to-end: **T1** vendor-neutral signal-provider registry →
**T2** framed CRC-sealed UART transport (dual-rail with a C twin) → **T3** derived metrics + plan-cost
sensitivity → **T4** Prometheus/OTLP/Redfish export adapters. Every stage is pure-Python (T2 also a
freestanding C twin), strictly on the *cost/optimization* side of the two-truth quarantine: telemetry
may inform `theta` / cost calibration but is never — and can never become — an R-law legality verdict;
the decision path (`bcir/verify`, R1–R21) reads no telemetry.

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
- **5b forward/backward training — BUILT (kernels + loss head).** `bcir/kbcir/autodiff.py`
  is a complete, correct, content-addressed reverse-mode autodiff organ; `emit_autodiff_kernel_c`
  lowers it to a forward+backward C kernel + SGD step (gradients match oracle + finite-difference).
  The **loss head is now built too** (`bcir/kbcir/losses.py`, M1): MSE, softmax cross-entropy,
  BCE-with-logits, hinge, so a full training step (forward → loss → backward → param grads) runs
  end-to-end. It follows the **autodiff closure property** — MSE is built into the `Tape` as
  closed-set nodes (the existing `grad`/`emit_autodiff_kernel_c` handle it for free); the
  transcendental losses (softmax-CE, BCE) keep their `log`/`exp` forward value off the
  re-differentiable path and instead provide the closed-form `grad_logits`
  (`softmax−onehot` / `sigmoid−target`, reusing the G1 activation references) that SEEDS the model
  backward. Unlocks logistic regression + multiclass classification. The hybrid tropical-structure +
  gradient-tune training loop (B4) remains the next step.
- **5c tensor ops as claims — PARTIAL.** `gem.matmul` is a first-class planned claim with
  K_BCIR tile/loop search; there are no `gem.conv` / `gem.attention` / `gem.activation`
  ops.
- **5d C++ hand-off boundary — SCAFFOLDED.** The boundary is now defined doc-first
  ([`CPP_HANDOFF_BOUNDARY.md`](CPP_HANDOFF_BOUNDARY.md)) and backed by a compilable,
  round-trip-tested seam ([`runtime/cpp/`](../runtime/cpp), gated by
  [`tools/cpp/check_handoff.sh`](../tools/cpp/check_handoff.sh), wired into the c-runtime
  suite). The contract: the C/IR rail produces a frozen StreamPack; a C++ `Orchestrator`
  consumes it, decides placement/topology, and re-enters the existing single-node C kernels
  per shard — it may schedule/shard/retry/replicate but may NEVER alter the artifact's
  semantics or become an R-law verdict (the two-truth quarantine extends across the seam).
  **Honest depth:** the `SingleNodeOrchestrator` reference is REAL (it round-trips an
  artifact through the existing C decoder and its dispatch order == the direct C/IR path);
  the dynamic-graph (RL node spawning / mixed-length token graphs) and distributed
  (MPI/NCCL multi-node) backends are documented STUBS behind the same interface — they fail
  loudly and add no real MPI/NCCL dependency (that needs multi-node hardware we don't have).

---

## Prioritized remaining-work backlog

Ordered by leverage toward the vision, holding the two-truth quarantine +
prototype-then-port discipline. Each item is PR-sized and parity-gated.

**Immediate (the queued Area-B slices — they advance Pillar 4b + the calling-side half of
Pillar 3):**
1. ✅ **B1 — `bcir-cc --emit-c` automatic link-flag emission** — DONE (`-lm`/`-lcblas`/
   `-lfftw3` derived from the `c.call.libm:` edges, dual-rail).
2. ✅ **B2 — wrap a new C math library** — DONE (FFTW 1-D FFT via `c.call.libm:`, R17 bridge,
   portable DFT fallback).
3. ✅ **B3 — calling-side tuning** — DONE (cost-priced major-order / tile / prefetch / channel).
4. **Area-B breadth — ATLAS / GSL / OpenBLAS-LAPACK / SLEEF** wrapped through the same edge —
   **remaining** (the active frontier).
4b. ✅ **SYCL/SPIR-V backend channel + differential oracle** (heterogeneous interop — "measure a
    heterogeneous backend before committing to a resident driver") — **BUILT**: a modeled SPIR-V GPU
    channel ([`channels/sycl.channel.json`](../channels/sycl.channel.json)) the planner prices + routes
    like any other, plus a toolchain-gated SAXPY `parallel_for` differential oracle
    ([`bcir/kbcir/sycl_saxpy.py`](../bcir/kbcir/sycl_saxpy.py) + `emit_sycl_saxpy_c`, gated by
    [`tools/cpp/check_sycl.sh`](../tools/cpp/check_sycl.sh)) that verifies a device reproduces BCIR's
    own reference (portable scalar fallback does the real work; the `-fsycl` device path self-skips on
    CI). SYCL is a compiler MODE, **not** a `c.call.libm:` link edge (no `linkflags.py` rule); its
    dynamic scheduler is held off the legality path. See [`SYCL_INTEROP.md`](SYCL_INTEROP.md). No
    `mlir/` changes, no new R-laws. **S2 (resident dispatch):** the channel now has a host-side
    **dispatcher** ([`bcir/lower/sycl_dispatch.py`](../bcir/lower/sycl_dispatch.py) `SyclDispatcher` +
    `build_execute_kernels`) so a module `orchestrate` places onto a tower including `sycl_spirv` is
    **run end-to-end** through `gem.execute`, the sycl-placed claim dispatched through the emitted kernel
    (portable fallback on CI, `-fsycl` device path gated), round-trip-verified vs the reference and gated
    by `#sycl-dispatch` in [`tools/cpp/check_sycl.sh`](../tools/cpp/check_sycl.sh); the SPIR-V codegen
    identity is reachable via the `spirv64` target (best-effort). Above the G8 boundary, never a verdict.
    **S3 (all three declared capabilities):** the channel's `{data_parallel, matmul, reduce}` caps are now
    each differentially verified + dispatchable — `reduce` (`sycl::reduction`,
    [`bcir/kbcir/sycl_reduce.py`](../bcir/kbcir/sycl_reduce.py) + `emit_sycl_reduce_c`, `#sycl-reduce`) and
    `matmul` (a 2-D `parallel_for` reusing the B1/B5 `matmul_reference`, `emit_sycl_matmul_c`,
    `#sycl-matmul`) join SAXPY, with `SyclDispatcher.run_reduce` / `.run_matmul` executing them
    ([`bcir/tests/test_sycl_reduce_matmul.py`](../bcir/tests/test_sycl_reduce_matmul.py)). The reduce device
    path honestly notes its tree-reorder tolerance; the portable fallback sums sequentially and matches the
    reference exactly. Still a compiler MODE, no `linkflags.py` rule, no `mlir/` changes, no new R-laws.

**Pillar-3 intelligence:**
5. ✅ **SoA↔AoS layout pivot** (Pillar 3b) — DONE: oracle `layout.py` + MLIR `-bcir-layout-pivot`.
6. ✅ **cache/bank cost signal** (Pillar 3a) — DONE: frozen-Q8 `cache_predict.py` + MLIR
   `-bcir-cache-contention`, informs-only.
7. ✅ **matmul+activation fusion** (Pillar 3c, 5c) — DONE: `fusion.py` + MLIR
   `-bcir-fuse-matmul-activation`, deforestation-priced.

**Pillar-5 bare-metal AI:**
8. ✅ **Lower the autodiff oracle to a backward-pass kernel** (Pillar 5b) — DONE:
   `emit_autodiff_kernel_c` + SGD; gradients match oracle + finite-difference.
9. ✅ **Baked-weights inference kernel emitter** (Pillar 5a) — DONE: `emit_inference_kernel_c`.
10. ✅ **Define the C↔C++ hand-off boundary** doc-first (dynamic-graph + distributed) so the
    single-node limit is explicit (Pillar 5d). **DONE** —
    [`CPP_HANDOFF_BOUNDARY.md`](CPP_HANDOFF_BOUNDARY.md) + a compilable, round-trip-tested
    single-node seam ([`runtime/cpp/`](../runtime/cpp), gated by
    [`tools/cpp/check_handoff.sh`](../tools/cpp/check_handoff.sh)); the dynamic/distributed
    backends remain marked stubs (no real MPI/NCCL). *Follow-up:* build a real dynamic-graph
    freeze loop + an MPI/NCCL backend when multi-node hardware is available.

**Pillar-1/2 boundaries & formalization:**
11. `Domain.DMA` + DMA-descriptor specialist + device-isolation annotation (Pillar 1).
12. A normative **flatness/ownership law** in the LangRef + an optional inline-bounds
    `emit_style="flat"` (Pillar 2).

---

## Bottom line

The **foundation the vision rests on is real and demonstrable**: C *is* a thin,
type-safe, registry-oriented macro-assembly target; the IR *does* own scheduling and the
math; the linearized StreamPack and the certified Q8 precision bridge exist.

As of the 2026-06-28 build update, the **intelligence and ML-payload half is now built too**
(items 5–10 above, all merged): the macro-assembly-layer "Layer-1 AI" (cache/bank contention
prediction, SoA↔AoS layout pivoting, autonomous matmul+activation fusion), the tensor-op
vocabulary (`gem.activation`/`conv`/`attention`), and the end-to-end bare-metal
inference (`emit_inference_kernel_c`) + training (`emit_autodiff_kernel_c`) kernels — each
prototyped in the oracle and **ported to the MLIR law rail**, parity-gated. BCIR has moved
from "a verifiable C macro-assembly substrate" to "an AI-optimizing compiler that runs
inference and training on bare metal," with the learned/predicted signals held off the
deterministic legality path by the two-truth quarantine.

**What genuinely remains** is breadth and the hardware-gated frontier, not core capability:
**4b** library breadth (LAPACK/GSL/SLEEF through the existing edge), **5d** the real
dynamic-graph + MPI/NCCL distributed backends (a contract + single-node seam exist; the
multi-node backends need cluster hardware), and the **Pillar-1/2 boundary** items
(DMA/device-isolation domains, a normative flatness law).
