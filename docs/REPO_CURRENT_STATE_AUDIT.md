# BCIR Repository Current State Audit

> Audited 2026-07-01 against the `bcir/` (oracle) + `mlir/` (law) + `runtime/c` (C rail)
> tree — after the vision-alignment gap-closure program (tensor ops, bare-metal
> inference/training kernels), the R19–R21 law promotion, the telemetry T1–T4 pipeline,
> the ML Tier-1 trio + breadth slices (M1–M3, E1–E7), and the asm/driver arc
> (ASM1–ASM3b, `bcir.asm`/`bcir.portio`/`bcir.volatile_*`/`bcir.creg_*`/`bcir.msr_*`).
> The normative status lives in [`BCIR_LANGREF.md`](BCIR_LANGREF.md); the forward plan in
> [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) (the single, consolidated roadmap);
> live counts in the generated [`STATUS.md`](STATUS.md); this file is the honest
> snapshot. The dated changelog that used to live here is summarized in
> [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md) (full detail in git history).
> Earlier revisions described the retired C++ `ir/` skeleton (removed 2026-06-07) and
> the pre-Phase-13 tree (audited 2026-06-12).

## Snapshot

- Two trees implement BCIR in lockstep ([`PARITY.md`](PARITY.md)):
  - **`bcir/`** — the executable conformance oracle (pure Python, no third-party
    deps): model, K_BCIR optimizer (min-plus + RCSP/Pareto + (max,+) overlap +
    soft-temperature + branch-and-bound rails), GEM hydration/scheduling/execution,
    ROP/MAP front-ends + the **cfront C frontend** (full preprocessor, 5-target ABI
    matrix, atomics/fences/inline-asm/port-I/O edges, VLAs, `_BitInt`, `_Complex`,
    variadics), M5 ETL, telemetry/calibration (T1–T4: signal registry, UART frame ABI,
    derived metrics, OTLP/Prometheus/Redfish export), StreamPack ABI (v1 frozen,
    v2/v3 append-only), the **R1–R21** verifier, lowering (clang AOT / lli JIT / WASM /
    stackify / per-target llc / portable C23 kernels / Area-B library wraps), the
    Phase 13–26 learned organs (calibration, portfolio + replay gate, MoE gate, search
    accelerator, soft optimizer, regret ledger, provenance manifest, e-graph +
    memory-module fixpoints, two-truth quarantine), and the ML substrate (autodiff with
    a machine-proven closed primitive set, losses, optimizers, training loop, OLS/PCA,
    transformer block, recurrent cells, classical-ML predict, unsupervised + pipeline).
    Suite: `python -m bcir.tests.run_all` (live count + coverage in
    [`STATUS.md`](STATUS.md), generated from the tree — see that file rather than a
    hard-coded number here, which is what the 580/615/631 drift came from).
  - **`mlir/`** — the law: the ODS/TableGen dialect family (op count in
    [`STATUS.md`](STATUS.md)), the compiled `bcir-opt` with `-bcir-verify`
    (**R1–R21**), the full deterministic optimizer core in C++23 (`-bcir-cost-model`
    cost+fusion/CSE → `-bcir-plan` coupled min-plus → `-bcir-overlap` (max,+) →
    `-bcir-rcsp`/`-bcir-rcsp-plan` constrained search, plus the bundle/compose/
    schedule/async/power/allocator passes and the gem tensor-op cost + lowering
    passes), the asm-edge law ops lowering to `llvm.inline_asm`
    (assemble-smoke-tested through `llc`), named pass pipelines
    (`bcir-audit`/`-optimize`/`-hydrate`/`-lower-llvm`/`-aot`), and the IRDL
    projection for stock `mlir-opt`. Validated in CI on the latest LLVM/MLIR
    release — LLVM 22, gating (`mlir-rail-validate`).
- **`runtime/c/`** — the production C rail (component count in
  [`STATUS.md`](STATUS.md)): the freestanding (no-libc) StreamPack decoder/encoder/
  executor + hydrate + scalar planner, the ETL binary-record decoder, the UART
  telemetry-frame codec, the C23 `#embed` frozen Q8 tier table, the bounds-quarantine
  runtime — and the **plug-in C compiler** (`bcir_cpp.c` preprocessor → `bcir_cfront.c`
  frontend → `bcir_verify.c` → `bcir_plan`/`bcir_hydrate`/`bcir_exec`, driven by the
  cc-like `bcir-cc`), Python↔C parity-gated per stage (structural digest + emitted-C
  Clang equivalence) over the shared `cfront_*.c` fixture corpus, with
  libFuzzer+ASan/UBSan on every trust boundary. `runtime/cpp/` is the small C↔C++
  hand-off seam (single-node orchestrator real; dynamic/distributed backends honest
  stubs — [`CPP_HANDOFF_BOUNDARY.md`](CPP_HANDOFF_BOUNDARY.md)).

## Confirmed strengths

1. The oracle runs the whole correspondence chain end to end, deterministically
   (integer/Q-fixed), with worked-example parity pinned (`vector_add` AVX-512
   cool Θ → vec16, score **7808**; under a 700 thermal/power cap → vec8, **9472**).
2. Verifier laws **R1–R21** run on both rails and are negative-tested per law
   (`bcir/verify` and the MLIR `-bcir-verify` pass; coverage table generated in
   [`STATUS.md`](STATUS.md)). R19/R20 (timing) and R21 (lifetime) ride optional claim
   metadata and are vacuous over the scalar/C subset — the non-disturbance invariant.
3. The **entire deterministic optimizer core is MLIR-native and cross-checked against
   the oracle** — bit-exact scores across the six-target capability matrix, gated by a
   generated adversarial Python↔MLIR differential rather than curated pins.
4. **Hot/cold separation is verified and locked** (`bcir/tests/test_hot_cold.py`):
   the executor and ABI codec import no learned organ or planner; no
   planning→execution→telemetry runtime recursion.
5. The StreamPack ABI v1 is frozen, CRC-gated, and decoded by a freestanding C
   runtime; cross-language parity is CI-gated, and CRC-valid-but-corrupt packs are
   rejected with specific error codes (seven adversarial corruption classes).
6. CI gates every push: the oracle suite (x86-64 + native arm64), the C runtime
   (incl. the sanitizer sweep and 500k-run libFuzzer campaigns), the LLVM-training
   validators, the full MLIR rail (tblgen, IRDL round-trip, `bcir-opt` build, ODS
   corpus, pass tests), and docs governance (generated status drift, links,
   retired paths, import quarantine).
7. **First measured wins on real silicon.** The evidence rail (`bcir.bench`,
   [`CLANG_COMPARISON.md`](CLANG_COMPARISON.md)) shows gather-avoidance ~6–7× (up to
   ~16× on reductions), strided ~1.3–1.4×, a match band on dense kernels, and budget
   feasibility as a **correctness** win (the feasible vec8 under a thermal cap where
   naive vec16 violates it). The library façade (`bcir.api`) packages a plan as a
   deployable, R12-attested artifact.
8. The **C compiler rail is a freestanding driver-subset C23 compiler candidate**:
   real register-map/UART/DMA driver fixtures ingest end-to-end (`C → bcir_cpp →
   bcir_cfront → verify → plan → hydrate → exec`, no Python), with Clang-grade
   diagnostics, a `--fallback` route-to-LLVM contract, and a differential fuzzer that
   has flushed ~21 real frontend bugs into permanent gates.

## Confirmed limitations

1. **No BCIR-native instruction selection** (by design — emit C/LLVM and reuse the
   resident backend; the explicit decision gate is
   [`BCIR_NATIVE_OBJECT_GATE.md`](BCIR_NATIVE_OBJECT_GATE.md), the feasibility study
   [`BCIR_NATIVE_BACKEND_FEASIBILITY.md`](BCIR_NATIVE_BACKEND_FEASIBILITY.md)).
   The portable C23 kernel backend, LLVM/llc/lli/WASM, and the resident-compiler
   object path (ELF-verified for x86-64/aarch64/eBPF) are the machine-code paths.
2. **The one genuinely deferred result is a *measured* bare-metal replan win.** The
   software path is closed and push-button (`tools/silicon/measure_replan.sh` prints a
   rig-ready verdict; CI exercises degrade mode); it lights up the moment a host with
   PMU + RAPL + a userspace cpufreq governor runs the runbook
   ([`HARDWARE_VALIDATION.md`](HARDWARE_VALIDATION.md)).
3. **Modeled vs measured is explicit.** The `fpga_systolic`/`nvme_stream`/`hbm_pim`
   channels are modeled (no resident driver yet — Phase D); CIM/PIM offload and DVFS
   energy numbers are models, not measured Joules; DVFS actuation reports exactly why
   it cannot actuate in a sandbox.
4. **Intelligence ahead of substrate** remains the top structural risk: a rich
   learned/ML stack over cost tables that are calibrated on host but not yet validated
   on a rig; the quarantine keeps it off the deterministic path until measured.
5. The C compiler rail is a **subset** compiler: unsupported constructs route honestly
   to `--fallback` (LLVM); the road to a hosted C23 replacement is the ordinary
   hard-compiler work (master roadmap §5.9–§5.10). `_Decimal*` is blocked on a
   reference compiler that can compile it. `runtime/cpp/` dynamic-graph and
   distributed orchestration are stubs pending multi-node hardware.

## Recommended next milestones (see [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md) §5–6 for detail)

1. **Measured real-silicon calibration** (§5.4) — the single most valuable next
   result; everything is staged for a rig.
2. **The freestanding-C23-driver release 0.3b** (§5.14 Phase 3) — multi-file driver
   projects through `bcir-cc` (compile-database, dependency output, real
   UAPI/CMSIS/PCIe/NVMe/ACPI fixtures), on the now-first-class R1–R21 law table.
3. **The driver ladder** ([`BCIR_DRIVER_KERNEL_ROADMAP.md`](BCIR_DRIVER_KERNEL_ROADMAP.md))
   — promote the polled UART into a channel-backed driver (D2.1), then climb
   interrupts → timers → table parsers → PCI enumeration.
4. **Area-B / ML breadth** ([`BCIR_ML_AI_INTEGRATION_ROADMAP.md`](BCIR_ML_AI_INTEGRATION_ROADMAP.md))
   — calling-side tuning around the wrapped kernels, the C1 streaming slice, and the
   Phase-C data/memory organs.

## Changelog

The dated, per-landing changelog that used to live in this file (2026-06-07 →
2026-06-25, ~90 entries) is condensed in
[`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md); the full entries remain available
in this file's git history (`git log -p -- docs/REPO_CURRENT_STATE_AUDIT.md`).
