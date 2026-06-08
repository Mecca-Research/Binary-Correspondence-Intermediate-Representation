# BCIR Repository Structure (decision record)

This document records how the repository is organized and how the IRDL, MLIR,
and LLVM sections stay separate. It is the steering reference for the reorg that
introduced `ir/`.

## Problem this structure solves

Before the reorg, three terms each meant several different things:

- **"dialect"** named a C++ hand-written surface parser (`dialect/`) that is
  *not* an MLIR dialect — colliding with the real MLIR dialect we are building.
- **"LLVM"** was spread across a textual emitter (`runtime/src/llvm_emit.cpp`),
  an ABI-substrate string (`runtime/src/bcir_llvm_ir.cpp`), a hand-written `.ll`
  seed (`runtime/llvm/*.ll`), and training examples — with no single owner.
- The **training corpus** (`llvm-training/`, ~470 files) was interleaved with
  the IR even though it is not part of the IR.

Lane/opcode/claim definitions were duplicated in four places (C++ model, surface
parser, `.ll` seed metadata, training docs) and had already drifted
(`BcirClaimV2` metadata vs. C++ `BcirClaimV1`).

## Top-level separation

- `ir/` **is** the IR.
- `llvm-training/` is a separate training corpus (see `AGENTS.md`). The IR has no
  build dependency on it.
- `tools/`, `tests/`, `docs/`, `cmake/` are shared project infrastructure.

## The IR pipeline and section ownership

```
bcir.surface(text) ─► bcir.core(typed graph) ─┬─► ir/irdl  (pure-IR dialect definition; no compile)
                                               └─► ir/mlir  (compiled dialect + conversion) ─► mlir.llvm ─► llvm ir
                                               ir/llvm     (textual emitter + ABI substrate)
                                               ir/runtime  (GEM execution engine)
```

| Section      | Builds            | In default build?                  |
|--------------|-------------------|------------------------------------|
| `ir/surface` | `bcir-surface`    | yes                                |
| `ir/core`    | `bcir-core`       | yes                                |
| `ir/irdl`    | round-trip test   | tests only; skips without mlir-opt |
| `ir/mlir`    | `bcir-mlir`       | no — `-DBCIR_ENABLE_MLIR=ON`       |
| `ir/llvm`    | `bcir-llvm`       | yes                                |
| `ir/runtime` | `gem-runtime`     | yes                                |

## IRDL vs MLIR vs LLVM — what each builds, and why they are separate

### IRDL (`ir/irdl/`) — pure IR, no compilation
- A declarative `*.irdl.mlir` definition loaded at runtime by
  `mlir-opt --irdl-file=`. No TableGen/ODS, no C++, no lowering.
- Owns the **structural contract**: types, ops, attributes, and the closed
  enums, projected from `ir/core/include/bcir/bcir_ir.hpp`. Becomes the single
  source of truth for structure, replacing the previous four-way duplication.
- Encodes only structural constraints (typing, attribute presence, enum
  membership, scalar ranges). Cross-op/semantic invariants stay in the C++
  surface verifier.

### MLIR (`ir/mlir/`) — compiled dialect + conversion
- ODS/TableGen op definitions, generated C++ op classes, a registered verifier,
  custom types/attributes, and conversion patterns to the LLVM dialect.
- This **is** compilation; needs MLIR/LLVM dev libraries; opt-in via
  `BCIR_ENABLE_MLIR` (OFF by default) so default CI needs no MLIR toolchain.
- Implements the same dialect that `ir/irdl/` describes. Structure/validation
  lives in `irdl/`; executable ops/verifier/lowering live in `mlir/`.

### LLVM (`ir/llvm/`) — legal IR emission
- C++ textual LLVM emitter, ROP→LLVM lowering tables, and the ABI substrate.
- Lowering target is legal LLVM only (`load/store/atomicrmw/cmpxchg/fence/call`);
  atomics are never rewritten into load/op/store pseudo-atomics.

## How separation is enforced

- **Directory boundaries**: each section owns its sources, headers, and
  `CMakeLists.txt`; cross-section use is via explicit `target_link_libraries`.
- **Build options**: `BCIR_ENABLE_MLIR` (compiled MLIR, OFF) and
  `BCIR_ENABLE_MLIR_TOOL_TESTS` (IRDL round-trip, ON but auto-skips without
  `mlir-opt`) keep optional toolchains off the default critical path.
- **No training coupling**: nothing under `ir/` references `llvm-training/`.

## Migration notes (this reorg)

- `dialect/` → `ir/surface/`; header `bcir/dialect.hpp` → `bcir/surface.hpp`;
  target `bcir-dialect` → `bcir-surface`. Public C++ symbols (`parse_dialect`,
  `verify_rop`, `tokenize_dialect`, `dialect_component_banner`) were kept stable.
- `runtime/src/*` split into `ir/core/` (builder), `ir/llvm/` (lowering/emit),
  and `ir/runtime/` (engine); target `bcir-lowering` → `bcir-llvm`.
- `include/bcir/*` headers moved next to their sections (logical include path
  `bcir/<name>.hpp` preserved).
- `docs_BCIR_LLVM_IR.md` → `docs/BCIR_LLVM_IR.md`.
- **Removed**: the hand-written `runtime/llvm/*.ll` seed, its `validate_*.sh`
  scripts, the `BCIR_Phase4_Assembler_and_Blob_Pipeline.md` doc, and the
  corresponding CI steps. Rationale: basic, duplicated the canonical model, and
  carried schema drift. The forward LLVM path is `ir/llvm/` + `ir/mlir/`.

## Build matrix (current — post-fold)

> Note: the section-ownership table and migration notes **above** describe the
> earlier PR #153 restructure (the `ir/` C++ tree, `cmake`/`ctest`,
> `BCIR_ENABLE_MLIR`). That tree has since been retired — see "One tree" below.
> The current build/validate entry points are:

```bash
# bcir/ -- the oracle (no third-party deps; CI job oracle-and-training):
python3 -m bcir.tests.run_all

# mlir/ -- the dialect law (needs libmlir-NN-dev + llvm-NN-dev; CI job mlir-rail-validate):
bash tools/wsl/tblgen_check.sh        # ODS generators
bash tools/wsl/build_mlir.sh          # build bcir-opt (LangRef M3)
bash tools/wsl/check_ods_examples.sh  # pretty ODS corpus via bcir-opt
bash tools/irdl/check_corpus.sh       # IRDL projection on stock mlir-opt
```

## One tree (the `ir/` fold is complete)

The repo is now a **single BCIR realization**:

- **`bcir/` (oracle) + `mlir/` (law).** A runnable Python conformance oracle that
  realizes the full `K_BCIR(G|H,Θ)` optimizer, GEM hydration + execution, LLVM
  lowering (clang AOT + lli JIT), the R1–R12 verifier subset, the M5 event
  transduction layer, the ROP/MAP front-ends, and the CT1–CT5 tracks — paired with
  the MLIR dialect family as the law: the TableGen/ODS ops, the **compiled
  `bcir-opt`** that parses/verifies the pretty corpus, and the IRDL portability
  projection (`docs/BCIR_LANGREF.md`, `docs/PARITY.md`).

**The legacy C++ skeleton (`ir/`) has been retired.** The earlier
surface-parser/verifier + textual-emitter + threaded-runtime milestone is gone:
its semantics are subsumed by `bcir/` (parsing → `bcir/etl` + `bcir/frontends`;
the typed model → `bcir/model` + `bcir/kbcir`; lowering → `bcir/lower`; the
deterministic GEM executor → `bcir/gem/execute.py`) and the real C++ dialect is
now `mlir/` (`bcir-opt`). Removed in the fold: `ir/`, the C++ CLI
(`tools/bcir-tools`) + `tools/bcir-as`, the C++ `tests/` (ctest), the
`cmake/BCIRConfig` package export, and the top-level `CMakeLists.txt`.

**Fold history (done):**
1. Declared the canonical stack; marked `ir/` legacy.
2. Ported the deterministic phase-sliced GEM executor into `bcir/gem/execute.py`
   with parity tests.
3. Built the compiled `bcir-opt` (LangRef M3) so the pretty ODS corpus validates,
   then retired `ir/surface` + `ir/llvm` (and the rest of `ir/`).
4. Collapsed to a single tree; rewired CI to `oracle-and-training` +
   `mlir-rail-validate`.

The two trees stay independently buildable: the oracle needs only `python3`; the
law needs `libmlir-NN-dev` + `llvm-NN-dev` (`tools/wsl/build_mlir.sh`).
