# BCIR Design and LLVM Lowering Report (Reimplementation Plan)

## Executive Summary
BCIR should be implemented as a **registry-first, provenance-carrying IR** with a strict multi-level lowering path:

`bcir.surface -> bcir.core -> bcir.rop -> mlir.llvm -> llvm ir`

This repository already has a parser, AST, verifier passes, ROP stream projection, and GEM runtime controls, but it does not yet build an MLIR dialect or emit a concrete LLVM IR module. This document formalizes BCIR semantics and defines a practical backend path compatible with current C++ components.

## Current Repository Baseline
- Parser/AST/verifier support for `ld/st`, `bin`, `lane`, `phase`, `barrier`, macro expansion, and MAP ops.
- ROP linearization and opcode-lowering registry with backend extension points.
- GEM runtime with deterministic ordering options, retries/rollback hooks, and telemetry.

## Architecture Recommendation

### 1) Layering
1. **BCIR Surface**: text format near current grammar for tooling and fixtures.
2. **BCIR Core**: SSA + explicit effects/provenance (primary optimization layer).
3. **BCIR ROP**: linearized schedule/correspondence layer.
4. **LLVM ABI Form**: legal LLVM IR using standard instructions + metadata + runtime hooks.

### 2) Why MLIR first-class dialect
LLVM IR itself is fixed and should not be treated as a custom instruction language. BCIR-specific semantics should be represented via:
- MLIR dialect ops/types/attrs + conversion patterns,
- legal LLVM ops (`load/store/atomicrmw/cmpxchg/fence/call`),
- metadata (`!dbg`, `!tbaa`, `!alias.scope`, `!bcir.*`).

## BCIR Core Semantic Model

### Node/edge model
- Node fields: `id`, `opcode`, `lane`, `epoch`, `phase`, `registry`, `offset`, `operands`, `cost`, `metadata`.
- Edge kinds: `DataFlow`, `ControlFlow`, `HazardOrder`, `PhaseOrder`, `CostFlow`.

### Lane semantics
- `U`: uniform/broadcast contiguous access.
- `UX`: aligned contiguous stream/vector lane.
- `T`: tiled block lane.
- `GGG`: gather/scatter lane.
- `A`: atomic lane.
- `H`: hazard-contract lane.

### Phase/epoch rule
Phases are **epoch-scoped**. `phase` can reset only when `epoch` increments.

### Hazard contracts
Hazard kinds: `RAW`, `WAR`, `WAW` with explicit contracts:
- `promise_no_hazard`
- `requires_ordering`

## LLVM ABI Lowering Contract
BCIR lowers to legal LLVM IR only:
- `bcir.load/store` -> `load/store` (or stable runtime wrappers)
- `bcir.atomic.*` -> `atomicrmw/cmpxchg`
- `bcir.barrier` -> `fence` or target runtime barrier
- `bcir.phase` -> runtime phase hooks + metadata
- lane ops -> vector ops/intrinsics/runtime hooks depending on static knowledge

## K_BDI Integration Points
Use runtime-conditional objective:
`K_BDI(G|H,Theta) = min_pi C_H(pi,Theta)`

BCIR additions:
- per-node cost tuple: `(e_switch, t_latency, m_move, b_pressure, r_recompile, s_sync, phi_thermal)`
- module-level theta snapshot + validity region
- cost-flow edges to enable cross-layer optimization reasoning

## 64-byte Claim Schema (Implementation-safe)
`BcirClaimV1` remains 64 bytes and cacheline-aligned with explicit:
- opcode/lane/epoch/phase/flags
- stride bytes
- `rdRids[4]`, `wrRids[4]`
- `hazardDomain`, `immediates[2]`, `costHint`

## Immediate C++ Changes Required
1. Extend graph model with lane/epoch/opcode/cost metadata.
2. Add explicit edge-kind representation.
3. Add registry descriptor capacity/lifetime/alias semantics.
4. Add epoch-aware phase verifier.
5. Add H-lane hazard verifier.
6. Preserve MAP atomic ops until atomic lowering stage.
7. Add LLVM-ABI emission layer (no pseudo-instruction-only dispatch).

## Migration Plan
- **A**: Data model + parser compatibility.
- **B**: Verifier upgrades (warning mode).
- **C**: Strict epoch/phase and registry checks.
- **D**: Real MLIR->LLVM lowering + target legalization.
- **E**: K_BDI-guided scheduling and recompilation thresholding.
