# BCIR repository structure

> Current for package version `0.2.0` and the 2026-07-22 tree. This document is an
> ownership map, not a generated inventory or a migration log. Static counts belong in
> [`STATUS.md`](STATUS.md); historical reorganizations belong in
> [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md).

BCIR is one system with three independently testable implementation rails:

1. `bcir/` is the executable Python conformance oracle.
2. `mlir/` is the ODS/TableGen/C++ law rail and partial lowering pipeline.
3. `runtime/` contains production C and bounded C++ runtime/compiler surfaces.

The rails share contracts through checked files and differential tests, not through
source inclusion. This separation keeps agreement meaningful.

## 1. Top-level ownership

| Path | Owner and purpose | Primary validation |
|---|---|---|
| `bcir/` | Dependency-free oracle plus an import-quarantined optional hosted-model adapter; the default package path remains third-party-free | `python -m bcir.tests.run_all`; hosted job separately |
| `mlir/` | Normative dialect families, verifier and optimizer passes, IRDL projection, examples, and pass fixtures | `tools/wsl/check_passes.sh`, `tools/irdl/check_corpus.sh` |
| `runtime/c/` | Freestanding runtime plus hosted C compiler/model tools and direct RuntimeChannel bindings | `tools/c/check_runtime.sh` and sanitizer/fuzz gates |
| `runtime/cpp/` | Narrow C++ orchestration boundary; it does not own legality, planning, or learned policy | `tools/cpp/check_handoff.sh` |
| `channels/` | Channel/profile inputs and their documentation; modeled channels are labeled as such | oracle and C channel tests |
| `tools/` | Build, validation, documentation, model-gate, performance, and hardware scripts | called by CI and local runbooks |
| `training/` | Standalone teaching, evaluation, and model-training corpus, one folder per subject (`llvm/` first); never a build dependency of BCIR | per-subject curriculum/autograder gates |
| `docs/` | Normative references, current state, execution roadmaps, history, and scoped research | generated-status, link, and retired-path checks |
| `.github/workflows/` | Required host, oracle, C, MLIR, training, and documentation CI jobs | GitHub Actions |

The retired C++ `ir/` prototype is not part of the current tree. Its useful semantics
were absorbed by `bcir/` and `mlir/`; its chronology is preserved in development
history and git.

## 2. Oracle package (`bcir/`)

| Package | Responsibility |
|---|---|
| `model/` | Registry-first resources, claims, phases, lane/domain/hazard types |
| `verify/` | Executable R-law reference and plan/pack/lowering checks |
| `kbcir/` | Cost vectors, min-plus/RCSP/(max,+) planning, certified learned organs, calibration, AD, quantization, bounded hardware-RL, adaptive-transformer, byte-native BLT/MambaByte, sequence-interface/tokenizer-adaptation, causal-series, and constructive-growth contracts, exact static memory planning, verified HAM residency/routing, context shards, the dual-memory oracle, and the cold explicit native-AI bridge |
| `gem/` | Hydration, StreamPack construction, scheduling, overlap, execution, event/DMA/device contracts |
| `frontends/` | ROP, MAP, Python C-front oracle, and model manifest/tokenizer/header-only assessment rails |
| `etl/` | Text and binary event-to-claim transduction |
| `abi/` | StreamPack, BCAB multi-backend bundles, and other byte-level host contracts |
| `lower/` | Portable C, the single-claim elementwise LLVM AOT/JIT subset, WASM, stack-machine, library, SYCL, and model lowering helpers |
| `codegen/` | Resident-toolchain object/assembly paths and target validation |
| `hosted/models/` | Opt-in PyTorch Llama training, pickle-free exact-resume checkpoints, strict Safetensors export, and train-to-C gating; never imported by the oracle path |
| `hosted/training/` | Dependency-free corpus/BPE/provider/stage/ledger/hardware-policy specs plus lazily imported PyTorch SFT, reward, DPO, PPO, reasoning, embedding, MLP, GRU, encoder, adaptive-language/multi-patch, byte-latent/selective-SSM, cross-tokenizer/continued-BPE/growth, and GNN/Transformer hardware-policy references |
| `tests/` | Explicit test registry and named quick, C-runtime, silicon-degrade, and thorough tiers |

`bcir.performance_audit` is an opt-in, dependency-free audit coordinator. It exercises
representative graph/GEM/StreamPack/memory/telemetry/ML organs and publishes deterministic
result hashes plus informative host timings. It is not imported by normal planning or
execution entry points; see [`PERFORMANCE_AUDIT.md`](PERFORMANCE_AUDIT.md).

`bcir-model-assess` is the payload-free model planning entry point. Its model directory parser
reads only `config.json` and validated Safetensors headers, while caller-authored hardware and
workload artifacts drive exact capacity accounting. It can publish a canonical cost report,
content-addressed execution plan, and verified StreamPack; it never publishes weights. Requested
bounded CPU measurement uses the native C twin by default, with an explicit Python-oracle mode.
The lowering also retains exact RID→bank bindings. `kbcir.static_memory` turns those bindings
and phase lifetimes into an independently verified aligned address plan; the hosted hardware
policy may rank the finite candidates but cannot replace assessment, verification, or measured
promotion evidence.

`bcir-ham-plan` is the payload-free semantic-memory entry point. A strict `HAMWorkload` and
`HardwareEnvelope` produce independently replayed route/capacity/generation actions, an ordinary
verified BCIR module, and a channel-tagged StreamPack. `kbcir.context_shard` owns attested
model/Q8/plan references plus quiescent rollback-capable activation; `kbcir.optimization_memory`
is the bounded exact-Q15/hard-fact oracle for future ANN/property-graph adapters. None of these
modules configures GDS, P2PDMA, CXL, NVMe, or controller firmware.

Top-level modules such as `telemetry_frame.py`, `telemetry_intake.py`, `telemetry_export.py`,
`signal_registry.py`, `signal_table.py`, and `channels.py` own host-side protocol/reference
behavior. They must remain consistent with the corresponding fixed-width C contracts where one
exists.

## 3. Law rail (`mlir/`)

| Path | Responsibility |
|---|---|
| `include/BCIR/*.td` | ODS operations, types, attributes, interfaces, and pass declarations |
| `lib/` | Dialect registration, R1–R25 verification, K_BCIR/GEM passes, and partial conversions |
| `tools/bcir-opt.cpp` | Registered command-line law/pipeline driver |
| `test/` | Positive, negative-diagnostic, pass, assembly, and lowering fixtures |
| `examples/` | Canonical readable modules and worked parity anchors |
| `irdl/` | Structural portability projection for stock `mlir-opt`; it does not carry semantic R-laws |

`bcir-aot` is partial AOT preparation and may leave mixed BCIR/GEM/LLVM dialect IR.
The Python LLVM path supports exactly one elementwise add/sub/mul compute claim and
rejects additional executable claims. Neither path claims arbitrary-graph native code.

## 4. C and C++ runtime classes

[`C_MEMORY_DISCIPLINE.md`](languages/C_MEMORY_DISCIPLINE.md) defines three C classes:

- **Freestanding core:** no heap or libc dependency; caller-owned buffers, capacities,
  fixed-width types, deterministic errors, and idempotent cleanup where applicable.
- **Hosted compiler/model tools:** explicit ownership, checked growth, allocator
  injection, complete init/destroy contracts, and fail-every-allocation tests.
- **Driver adapters:** opaque generation-tagged handles and byte offsets across ABI or
  process boundaries; never shared raw pointers.

`runtime/cpp/` is an orchestration layer above those contracts. The ownership boundary
is documented in [`CPP_HANDOFF_BOUNDARY.md`](languages/CPP_HANDOFF_BOUNDARY.md).

The hosted model data plane in `runtime/c/` includes the validating BCIRQ8 loader,
standalone GQA decoder, Q8/Q4 conversion and projection kernels, exact Q15 top-k, and a
fixed-storage benchmark. Python parity and placement rules are documented in
[`BCIR_PYTHON_NATIVE_BOUNDARY_AUDIT.md`](machine-learning/BCIR_PYTHON_NATIVE_BOUNDARY_AUDIT.md).
The kernels have a C ABI usable from C++; this does not expand `runtime/cpp/` ownership.

## 5. Contract ownership

| Contract | Normative prose | Executable/reference owners |
|---|---|---|
| BCIR semantics and R-laws | [`BCIR_LANGREF.md`](BCIR_LANGREF.md) | `bcir/model`, `bcir/verify`, `mlir/` |
| Oracle ↔ law/twin agreement | [`PARITY.md`](PARITY.md) | differential tests and C-front parity gates |
| StreamPack v1–v4 | [`BCIR_STREAMPACK_ABI.md`](kernel/BCIR_STREAMPACK_ABI.md) | `bcir/abi/streampack_abi.py`, `runtime/c/bcir_streampack.h` |
| ExecutionPlanV1 (the plan as bytes) | [`BCIR_EXECUTION_PLAN_ABI.md`](kernel/BCIR_EXECUTION_PLAN_ABI.md) | `bcir/gem/execution_plan.py`, `bcir/abi/execution_plan_abi.py`, `runtime/c/bcir_execution_plan.h`, `bcir/asn1/execution_plan.py`, BCAB kind 25 |
| The native planner's input and realization records (BKPI/BKPR, version zero) | [`BCIR_PLANNER_ABI.md`](kernel/BCIR_PLANNER_ABI.md) | `bcir/kbcir/realize.py` (the compact planner; `realize_reference.py` is the pre-G17 planner, read only by the parity gate), `bcir/abi/planner_abi.py`, `runtime/c/bcir_kplan.h` (harness `test_kplan.c`, fuzz `fuzz_kplan.c`), `tools/c/check_planner.py`; no BCAB kind |
| The declared alias facts (G9: the elementwise kernel's aliasing, element type, volatility and ordering, carried to LLVM) | [`BCIR_ALIAS_FACTS.md`](kernel/BCIR_ALIAS_FACTS.md) | `bcir/lower/alias_facts.py` (`kernel_facts`, `hazard_refusal`, `bound_harness`), the emitters in `bcir/lower/llvm.py`, `c_kernel.py`, `specialist.py`, `wasm.py`, R12's reader `bcir/verify/alias.py`, `tools/perf/check_alias.py`; not an ABI -- no new bytes |
| The StreamPack encoder's compiled record layouts (SP-ENC: the reference codec, faster, with the same bytes and refusals) | [`BCIR_STREAMPACK_ABI.md`](kernel/BCIR_STREAMPACK_ABI.md) | `bcir/abi/streampack_abi.py` (the record functions and their layouts), `bcir/tests/encode_fixtures.py` (`encode_reference`, the parity grader), `tools/perf/check_encode.py`; not an ABI change -- no byte moved |
| The cfront escape analysis, indirect-call narrowing and effect footprint (G10: one points-to analysis behind `CompileResult.commute`, both rails) | [`BCIR_ESCAPE_ANALYSIS.md`](kernel/BCIR_ESCAPE_ANALYSIS.md) | `bcir/frontends/cfront/escape.py` (`analyze`, the reports), the twin's `esc_*` in `runtime/c/bcir_cfront.c` (`bcir_cfront_effects`, `bcir_cfront_escape`), `bcir/tests/escape_fixtures.py` (generator, witness, grader), `tools/perf/check_escape.py`; not an ABI -- two report formats, byte-identical on both rails |
| Data movement as a first-class transformation (G8: the joint movement/compute planner, Semantic Swap, MV1–MV11, ExecutionPlan v3) | [`BCIR_DATA_MOVEMENT.md`](kernel/BCIR_DATA_MOVEMENT.md) | `bcir/kbcir/movement.py` (`MovementSpec`, `transform`, `plan_movement`, `execution_plan_of`, `verify_movement`), `verify_execution_plan(movement=)`, the v3 wire in `bcir/abi/execution_plan_abi.py` and `runtime/c/bcir_runtime.c`, `bcir/tests/movement_fixtures.py`, `tools/perf/check_movement.py` |
| `volatile` through both cfront rails (CF-VOL: one device-region model, R3's pass, exact-width emit) | [`PARITY.md`](PARITY.md) | `bcir/frontends/cfront/{ctype_model,lower,emit}.py`, the twin's `ty_mmio` / `mark_access` / `order_device_claims` / `vol_*` in `runtime/c/bcir_cfront.c`, `bcir/tests/volatile_fixtures.py` (corpus, judges, grader), `tools/perf/check_volatile.py` |
| The delta chain (G18: the chain advanced by declared deltas) | [`BCIR_DELTA_CHAIN.md`](kernel/BCIR_DELTA_CHAIN.md) | `bcir/kbcir/delta.py` (`Delta`, `apply_delta`, `IncrementalOffer`, `IncrementalPlan`), `bcir/gem/delta_pack.py`, `bcir/verify/delta.py`, `bcir/gem/delta_chain.py`, `tools/perf/check_delta.py`; not an ABI -- no new bytes |
| ControlRecordV1 (the control plane as bytes) | [`BCIR_CONTROL_PLANE_ABI.md`](kernel/BCIR_CONTROL_PLANE_ABI.md) | `bcir/gem/control.py`, `bcir/abi/control_abi.py`, `runtime/c/bcir_control_plane.h` (+ `bcir_sha256.h`), `bcir/asn1/control_plane.py`; no BCAB kind |
| Artifact Bundle v1 | [`BCIR_ARTIFACT_BUNDLE_ABI.md`](kernel/BCIR_ARTIFACT_BUNDLE_ABI.md) | Python codec/tool/builder, additive ASN.1 DER/COER projection, allocation-free C reader, C++ view, and MLIR metadata ops |
| BCIRQ8 v1 | [`BCIR_LANGREF.md`](BCIR_LANGREF.md#16-bcirq8-v1-decoder-artifact-contract) §16 | Python artifact reader/writer and portable C loader |
| Telemetry frame and registry | [`TELEMETRY_FRAME_ABI.md`](kernel/TELEMETRY_FRAME_ABI.md), [`SIGNAL_REGISTRY.md`](kernel/SIGNAL_REGISTRY.md) | Python codec/registry and fixed C frame codec |
| TelemetryEnvelopeV0, the generated signal table and the intake (version zero) | [`TELEMETRY_ENVELOPE_ABI.md`](kernel/TELEMETRY_ENVELOPE_ABI.md) | `bcir/abi/telemetry_envelope.py`, `bcir/signal_table.py` (generates `runtime/c/bcir_signal_table.h`), `bcir/telemetry_intake.py`, `runtime/c/bcir_telemetry_envelope.h` |
| The live SPSC ring (version zero) | [`BCIR_LIVE_RING_ABI.md`](kernel/BCIR_LIVE_RING_ABI.md) | `bcir/gem/ring.py`, `runtime/c/bcir_ring.h` (harness `test_ring.c`, fuzz `fuzz_ring.c`), `tools/c/check_ring.py`; no BCAB kind |
| The data-plane hand-off: the pack table, the per-step freeze and the Stage 3 exit flow | [`BCIR_DATA_PLANE_HANDOFF.md`](kernel/BCIR_DATA_PLANE_HANDOFF.md), [`CPP_HANDOFF_BOUNDARY.md`](languages/CPP_HANDOFF_BOUNDARY.md) | `bcir/gem/handoff.py`, `runtime/c/bcir_handoff.h`, `bcir_hydrate_generations` in `runtime/c/bcir_hydrate.h`, `runtime/cpp/bcir_handoff.hpp` (harnesses `test_handoff.c` / `test_handoff.cpp`, the shared flow `test_stage3.h`, fuzz `fuzz_handoff.c`), `tools/c/check_handoff.py`; no BCAB kind |
| The manifest-of-shards (BSHM, version zero) | [`BCIR_SHARD_MANIFEST_ABI.md`](kernel/BCIR_SHARD_MANIFEST_ABI.md) | `bcir/abi/shard_manifest.py`, `runtime/c/bcir_shard_manifest.h`; no BCAB kind |
| RuntimeChannel and future UAPI | [`BCIR_DRIVER_KERNEL_ROADMAP.md`](kernel/BCIR_DRIVER_KERNEL_ROADMAP.md) | direct C hook table today; Linux/native adapters later |
| HAM routes and context shards | [`BCIR_HAM_MEMORY_FABRIC.md`](kernel/BCIR_HAM_MEMORY_FABRIC.md) | `bcir/kbcir/ham.py`, `context_shard.py`, `optimization_memory.py`, and deterministic tests |
| Hosted allocation | [`C_MEMORY_DISCIPLINE.md`](languages/C_MEMORY_DISCIPLINE.md) | hosted allocator implementation and fault-injection tests |

When prose, generated inventory, and implementation disagree, resolve them in this
order: normative wire/language contract, executable parity tests, then descriptive
roadmaps. Never “fix” a wire format by silently changing one implementation.

## 6. Documentation taxonomy

The root of `docs/` is intentionally small and cross-cutting:

| Root document | Role |
|---|---|
| `BCIR_LANGREF.md` | Normative language, laws, IR levels, and BCIRQ8 contract |
| `BCIR_MASTER_ROADMAP.md` | Dependency-ordered portfolio execution plan only |
| `REPO_CURRENT_STATE_AUDIT.md` | Dated, source-backed snapshot of what exists and does not |
| `STATUS.md` | Generated static inventory; never hand-edit |
| `PARITY.md` | Cross-rail correspondence contract |
| `DEVELOPMENT_HISTORY.md` | Merged chronology and retired-roadmap closure ledger |
| `ONBOARDING_DEEP_DIVE.md` | Guided orientation and reading order |
| `VISION_ALIGNMENT_AUDIT.md` | Dated thesis-versus-evidence assessment |
| `BCIR_MACHINE_CODE_HAL_ISA_AUDIT.md` | Cross-cutting MC1–MC15 backend/HAL gap register |
| `BCIR_NATIVE_OBJECT_GATE.md` | GO/STOP decision for any native instruction selector |
| `RELEASE_NOTES_0.3b.md` | Unreleased draft; not a current-version declaration |

Subdirectories have one clear subject:

- `docs/kernel/` — drivers, kernel, StreamPack/Artifact Bundle, RuntimeChannel, telemetry, signals,
  hardware validation, heterogeneous channels, and SYCL interoperability.
- `docs/machine-learning/` — ML/model architecture, training/inference, language
  placement, third-party model provenance, and product-integration research.
- `docs/languages/` — C-front usage, C memory discipline, C++ handoff, and future
  language-frontend/backend plans.
- `docs/research/` — comparative or feasibility studies whose accepted decisions are
  linked from canonical roadmaps.

Research files do not become normative by location. Once a study is resolved, migrate
its decision/open work to the owning contract or roadmap and retain only useful
historical evidence.

## 7. Build and validation entry points

```bash
# Fast dependency-free oracle tier
python -m bcir.tests.run_all --tier quick -j 2

# Full local oracle/toolchain tier, with bounded concurrency
python -m bcir.tests.run_all --tier thorough -j 2

# Production C and C++ boundaries
bash tools/c/check_runtime.sh
bash tools/cpp/check_handoff.sh

# Optional pinned hosted-model CPU gate (one thread in CI)
python tools/models/test_hosted_model_lab.py --output-dir build/hosted-model-gate
python tools/models/test_training_pipeline.py --output-dir build/training-pipeline-gate
python tools/models/test_adaptive_transformers.py --output-dir build/adaptive-transformer-gate
python tools/models/test_byte_latent_models.py --output-dir build/byte-native-model-gate

# MLIR/IRDL rails when the coherent LLVM toolset is installed
bash tools/wsl/check_passes.sh
bash tools/irdl/check_corpus.sh

# Documentation governance
python tools/docs/gen_status.py --check
python tools/docs/check_links.py
git diff --check

# Bounded cross-organ performance/correctness evidence (no timing floor in shared CI)
python tools/perf/run_tmsao_audit.py --repeats 3
```

Tool-dependent cases report explicit skips when the required compiler, LLVM toolset,
hardware counter, or architecture is unavailable. CI supplies the required host matrix;
local development must stay bounded and must not emulate unsupported hardware in an
uncontrolled loop.

## 8. Change-placement rules

1. Put semantic behavior in the oracle first, then implement the appropriate law or
   production twin and add a differential regression.
2. Put stable byte layouts in their ABI document and both language implementations in
   the same change.
3. Put historical landing detail in development history, not the master roadmap.
4. Put generated counts only in `STATUS.md` through `tools/docs/gen_status.py`.
5. Put platform adapters around transport-neutral contracts; do not introduce Linux
   IPC or hosted allocation into the freestanding core.
6. Put future language roadmaps under `docs/languages/`, driver/kernel contracts under
   `docs/kernel/`, and unresolved comparative studies under `docs/research/`.
