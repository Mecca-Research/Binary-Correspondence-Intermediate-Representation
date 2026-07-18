# BCIR repository structure

> Current for package version `0.2.0` and the post-PR-640 tree. This document is an
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
| `llvm-training/` | Standalone LLVM/MLIR teaching and evaluation corpus; never a build dependency of BCIR | its own curriculum/autograder gates |
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
| `kbcir/` | Cost vectors, min-plus/RCSP/(max,+) planning, certified learned organs, calibration, AD, quantization, model/training helpers |
| `gem/` | Hydration, StreamPack construction, scheduling, overlap, execution, event/DMA/device contracts |
| `frontends/` | ROP, MAP, and Python C-front oracle |
| `etl/` | Text and binary event-to-claim transduction |
| `abi/` | StreamPack and other byte-level host contracts |
| `lower/` | Portable C, the single-claim elementwise LLVM AOT/JIT subset, WASM, stack-machine, library, SYCL, and model lowering helpers |
| `codegen/` | Resident-toolchain object/assembly paths and target validation |
| `hosted/models/` | Opt-in PyTorch Llama training, pickle-free exact-resume checkpoints, strict Safetensors export, and train-to-C gating; never imported by the oracle path |
| `hosted/training/` | Dependency-free corpus/BPE/provider/stage/ledger contracts plus lazily imported PyTorch SFT, reward, DPO, PPO, reasoning, embedding, MLP, GRU, and encoder references |
| `tests/` | Explicit test registry and named quick, C-runtime, silicon-degrade, and thorough tiers |

Top-level modules such as `telemetry_frame.py`, `telemetry_export.py`,
`signal_registry.py`, and `channels.py` own host-side protocol/reference behavior. They
must remain consistent with the corresponding fixed-width C contracts where one exists.

## 3. Law rail (`mlir/`)

| Path | Responsibility |
|---|---|
| `include/BCIR/*.td` | ODS operations, types, attributes, interfaces, and pass declarations |
| `lib/` | Dialect registration, R1–R23 verification, K_BCIR/GEM passes, and partial conversions |
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

## 5. Contract ownership

| Contract | Normative prose | Executable/reference owners |
|---|---|---|
| BCIR semantics and R-laws | [`BCIR_LANGREF.md`](BCIR_LANGREF.md) | `bcir/model`, `bcir/verify`, `mlir/` |
| Oracle ↔ law/twin agreement | [`PARITY.md`](PARITY.md) | differential tests and C-front parity gates |
| StreamPack v1–v3 | [`BCIR_STREAMPACK_ABI.md`](kernel/BCIR_STREAMPACK_ABI.md) | `bcir/abi/streampack_abi.py`, `runtime/c/bcir_streampack.h` |
| BCIRQ8 v1 | [`BCIR_LANGREF.md`](BCIR_LANGREF.md#16-bcirq8-v1-decoder-artifact-contract) §16 | Python artifact reader/writer and portable C loader |
| Telemetry frame and registry | [`TELEMETRY_FRAME_ABI.md`](kernel/TELEMETRY_FRAME_ABI.md), [`SIGNAL_REGISTRY.md`](kernel/SIGNAL_REGISTRY.md) | Python codec/registry and fixed C frame codec |
| RuntimeChannel and future UAPI | [`BCIR_DRIVER_KERNEL_ROADMAP.md`](kernel/BCIR_DRIVER_KERNEL_ROADMAP.md) | direct C hook table today; Linux/native adapters later |
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

- `docs/kernel/` — drivers, kernel, StreamPack, RuntimeChannel, telemetry, signals,
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

# MLIR/IRDL rails when the coherent LLVM toolset is installed
bash tools/wsl/check_passes.sh
bash tools/irdl/check_corpus.sh

# Documentation governance
python tools/docs/gen_status.py --check
python tools/docs/check_links.py
git diff --check
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
