# BCIR Repository Current State Audit

> Audited 2026-07-15 against the `bcir/` (oracle) + `mlir/` (law) + `runtime/c` (C rail)
> tree — after the vision-alignment gap-closure program (tensor ops, bare-metal
> inference/training kernels), the R19–R21 law promotion, the telemetry T1–T4 pipeline,
> the ML Tier-1 trio + breadth slices (M1–M3, E1–E7), and the asm/driver arc
> (ASM1–ASM3b, `bcir.asm`/`bcir.portio`/`bcir.volatile_*`/`bcir.creg_*`/`bcir.msr_*`).
> The normative status lives in [`BCIR_LANGREF.md`](BCIR_LANGREF.md); overall portfolio
> planning lives in [`BCIR_MASTER_ROADMAP.md`](BCIR_MASTER_ROADMAP.md), while
> [`BCIR_DRIVER_KERNEL_ROADMAP.md`](BCIR_DRIVER_KERNEL_ROADMAP.md) is the canonical
> execution sequence for driver/kernel work; live counts are in generated
> [`STATUS.md`](STATUS.md); this file is the honest
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
    variadics), M5 ETL, telemetry/calibration (T1–T4: stable signal IDs and explicit metric
    semantics, strict UART frame ABI with sequence evidence,
    derived metrics, and OTLP/Prometheus-text/Redfish serialization adapters), StreamPack ABI
    (v1 frozen, v2/v3 append-only), the **R1–R23** verifier, lowering (the single-claim
    elementwise LLVM AOT/JIT/WASM subset /
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
    (**R1–R23**), the full deterministic optimizer core in C++23 (`-bcir-cost-model`
    cost+fusion/CSE → `-bcir-plan` coupled min-plus → `-bcir-overlap` (max,+) →
    `-bcir-rcsp`/`-bcir-rcsp-plan` constrained search, plus the bundle/compose/
    schedule/async/power/allocator passes and the gem tensor-op cost + lowering
    passes), typed x86 long-mode entry/descriptor/segment/ordinary-interrupt edges lowering
    to LLVM module/inline assembly (assemble-smoke-tested through `llc` and real-object
    disassembly), named pass pipelines
    (`bcir-audit`/`-optimize`/`-hydrate`/`-lower-llvm`; `bcir-aot` is partial AOT
    preparation that leaves unsupported BCIR/GEM operations in mixed-dialect IR), and the IRDL
    projection for stock `mlir-opt`. Validated in CI on the latest LLVM/MLIR
    release — LLVM 22, gating (`mlir-rail-validate`).
- **`runtime/c/`** — the production C rail (component count in
  [`STATUS.md`](STATUS.md)): the freestanding (no-libc) StreamPack decoder/encoder/
  executor + hydrate + scalar planner, direct append-only RuntimeChannel v1 hooks and
  loopback, the ETL binary-record decoder, the UART telemetry-frame codec (strict
  flags/CRC/exact decode), the fixed ordinary-x86 interrupt-frame contract, the C23
  `#embed` frozen Q8 tier table, the bounds-quarantine
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
2. Verifier laws **R1–R23** run on the law rail and have a negative fixture per law
   (`bcir/verify` and the MLIR `-bcir-verify` pass; static negative-fixture inventory generated in
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
   rejected with specific error codes by a deterministic adversarial corpus, including
   CRC-valid undeclared trailing body bytes and invalid pipeline/dispatch semantics.
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
   register-map/UART/DMA *compiler fixtures* ingest end-to-end (`C → bcir_cpp →
   bcir_cfront → verify → plan → hydrate → exec`, no Python), with Clang-grade
   diagnostics, a `--fallback` route-to-LLVM contract, and a differential fuzzer that
   has flushed ~21 real frontend bugs into permanent gates. These fixtures validate
   compilation; they are not resident device drivers.
9. The **first machine/HAL operator slices are code-backed**: `bcir-pack dis/hexdump`
   validate before displaying exact codec-derived record spans, and `bcir-registry`
   show/getp/setp advances `data_gen` so stale packs fail R11. Both remain in-process
   baselines, not hardware tools.

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
6. **There is no resident UART driver or production driver telemetry ABI.** The UART register
   header/polling source is a compiler fixture; the channel-backed driver, UART simulator,
   IRQ service, and U0–U9 program remain planned. Telemetry has a registry, codecs, metrics,
   and deterministic serialization, but no UART egress, HTTP/Prometheus host, OTLP transport,
   Redfish/BMC client, or live provider transport. BTLM v1 lacks source/session/generation/
   clock identity, and shared-ring v1 is only a quiescent snapshot (no tail, per-slot publish,
   loss/backpressure, or peer-death protocol). Stable signal definitions are Python-only until
   one fixed-width C table is generated.
7. **The x86 asm edge is not a reset/exception subsystem.** `bcir.entry` assumes long
   mode. The ordinary trampoline exposes a fixed 176-byte C frame and refuses #DB, NMI,
   #DF, #MC, and AMD #VC; reset-mode transition, paranoid/IST nesting, SMAP/CET/IBT,
   CR3/PTI and speculation policy, extended-state policy, CFI/unwind, and hardware/QEMU
   execution remain open.

## Recommended next milestones

1. **Pre-driver telemetry ABI v0**
   ([`BCIR_DRIVER_KERNEL_ROADMAP.md`](BCIR_DRIVER_KERNEL_ROADMAP.md) §4.3/§7.1) — generate
   the fixed-width Python/C signal table and ID-range policy, then differentially gate a
   source/session/generation/clock-aware envelope and live SPSC ring. Revise these
   experimental contracts from traces; do not alter frozen BTLM/ring v1 bytes.
2. **UART D0–D3** — turn the compiler fixture into the first direct RuntimeChannel
   resident driver, prove simulator/direct lifecycle and telemetry behavior, then add the
   Linux-hosted adapter. This is the first evidence source for the future UAPI.
3. **virtio-console and virtio-blk D0–D3** — prove queue/DMA, cancellation, reset,
   saturation, and direct/Linux parity. Freeze UAPI v1 only after UART and virtio-blk
   agree across MMIO/event and queue/DMA classes.
4. **Measured calibration and certified priors** — run the existing rig-ready silicon
   gate where hardware permits, then use driver telemetry/replay evidence to promote
   device-specific immutable Q8 artifacts. Local ARM and board evidence remains
   hardware-gated, not emulated without an explicit bounded CI job.

## Changelog

The dated, per-landing changelog that used to live in this file (2026-06-07 →
2026-06-25, ~90 entries) is condensed in
[`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md); the full entries remain available
in this file's git history (`git log -p -- docs/REPO_CURRENT_STATE_AUDIT.md`).
