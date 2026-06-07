# BCIR Repository Current State Audit

## Snapshot
- Repository is a C++17 modular skeleton with `dialect/`, `runtime/`, `tools/`, `include/`, and `tests/`.
- Build and test flow is CMake + CTest.
- BCIR docs and blueprint are present and consistently define:
  - BCIR as canonical source IR
  - ROP/MAP as surface/lowering forms
  - LLVM IR as legal emission target

## Confirmed strengths
1. Surface parser + AST + verifier exist for core BCIR surface constructs.
2. MAP atomic operations are preserved across MAP lowering (not rewritten into pseudo-atomic load-op-store sequences).
3. 64-byte claim schema (`BcirClaimV1`) exists with compile-time size assertion.
4. Runtime includes phase/dependency execution and deterministic scheduling mode.
5. LLVM emission lives in the `ir/llvm/` section (textual emitter + ABI
   substrate). The earlier hand-written `runtime/llvm/*.ll` seed was removed in
   the 2026-06-07 reorg (see note below).

## Confirmed limitations
1. `bcir_ir.hpp` remains a minimal model and is not yet full canonical core graph model.
2. Parser is BCIR surface parser, not LLVM grammar parser.
3. Epoch/phase semantics are only partially represented in end-to-end APIs.
4. ROP stream remains mostly string-heavy and not yet a full schedule graph over typed BCIR core.
5. LLVM lowering table is intent/dispatch-oriented; no full emitter pipeline yet.
6. LLVM toolchain validation (`llvm-as`, `opt`, `lli`) is not yet integrated in CMake tests.

## Recommended implementation boundary (near-term)
Implement next milestone in **pure textual LLVM IR backend mode**:
- core graph model expansion
- registry descriptors
- epoch/phase verifier
- hazard verifier
- schedule builder
- textual LLVM IR emitter
- LLVM tool validation tests (optional, tool-detected)

Defer full MLIR dialect and BDI-K autotuning until semantics stabilize.

- 2026-05-26: Added pure textual LLVM backend milestone scaffolding (core graph builder, registry/epoch/hazard verifiers, deterministic schedule, textual LLVM emitter, and pipeline test).
- 2026-06-07: Reorganized the IR into `ir/{surface,core,irdl,mlir,llvm,runtime}`
  and fenced off `llvm-training/` as a separate corpus. Renamed the C++
  `dialect/` (a surface parser, not an MLIR dialect) to `ir/surface/`. Added
  scaffolds for the IRDL projection (pure IR, no compilation) and the compiled
  MLIR dialect (opt-in). Removed the basic hand-written `runtime/llvm/*.ll` seed,
  its `validate_*.sh` scripts, and the Phase-4 assembler doc (they duplicated the
  canonical `ir/core` model and carried `BcirClaimV1`/`BcirClaimV2` schema drift).
  See `docs/BCIR_Repo_Structure.md`.
