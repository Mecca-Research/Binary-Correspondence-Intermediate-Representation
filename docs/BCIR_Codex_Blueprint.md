# BCIR Codex Master Blueprint (Version #4 Spine)

This blueprint is the implementation work-order for rebuilding BCIR around the canonical pipeline:

`bcir.surface -> bcir.core -> bcir.rop -> mlir.llvm -> llvm ir`

North-star rule:

- **BCIR is the canonical source IR**.
- **ROP/MAP are surface/lowering dialects**.
- **LLVM IR is the legal emission target**.

---

## Build Task Matrix (End-to-End)

## Stage 1 — Documentation Consolidation

### Task 1.1: Create authoritative blueprint docs
- Add/maintain `docs/BCIR_Codex_Blueprint.md` as the steering spec.
- Ensure it states Version #4 as architecture spine.
- Merge strongest retained requirements:
  - atomics-preserved contract,
  - LLVM ABI substrate hooks/tests,
  - richer BCIR graph/registry/theta/claim model,
  - migration plan.

### Task 1.2: Align project-level docs
- Update `README.md` to describe BCIR as canonical IR and summarize the 5-layer lowering path.
- Link to:
  - `docs/BCIR_Codex_Blueprint.md`
  - `docs/BCIR_LLVM_IR.md`
  - `ir/core/include/bcir/bcir_ir.hpp`

### Task 1.3: Publish semantic invariants
- Document lane semantics (`U, UX, T, GGG, A, H`).
- Document epoch/phase legality:
  - non-decreasing `(epoch, phase)`
  - phase reset allowed only on epoch increment.
- Document hazard contract model and verifier acceptance/rejection rules.

---

## Stage 2 — Core C++ Model (`bcir.core`)

### Task 2.1: Expand core IR header
Update `ir/core/include/bcir/bcir_ir.hpp` with:
- `BcirLane`
- `BcirEdgeKind`
- `BcirHazardKind`
- `BcirContractMode`
- `BcirLifetime`
- `BcirRegistryDescriptor`
- `BcirThetaHeader`
- `BcirCostTuple`
- `BcirNode`
- `BcirEdge`
- `BcirGraph`
- `BcirClaimV1`

### Task 2.2: Keep binary claim contract stable
- Preserve `static_assert(sizeof(BcirClaimV1) == 64)`.
- Keep field layout explicit: opcode/lane/phase/epoch/flags/stride/rdRids/wrRids/hazard/immediates/costHint.

### Task 2.3: Add utility helpers
- Lane stringification/parsing helpers.
- Contract/hazard enum stringification.
- Registry descriptor sanity checks.

### Task 2.4: Add graph construction helpers
- Node/edge insertion utilities.
- Edge-kind constrained constructors.
- Optional metadata/provenance map helpers.

---

## Stage 3 — Verifier Expansion

### Task 3.1: Keep API compatibility
- Keep `verify_rop(const ModuleNode&)` stable.
- Extend internals to run additional checks and emit structured diagnostics.

### Task 3.2: Implement pass sequence
Current implemented verifier passes (as of this repo state):
1. `phase_monotonicity_and_annotations`
2. `lane_consistency_per_rid`
3. `offset_alignment_legality_by_lane`
4. `barrier_placement_and_hazard_contract`
5. `concurrent_registry_access_by_lane_and_atomic_constraints`

Target pass roadmap (planned):
- `verify_registry_descriptors`
- `verify_type_and_lane_compatibility`
- `verify_epoch_phase_legality`
- `verify_hazard_contracts`
- `verify_registry_bounds_and_alignment`
- `verify_atomic_legality`
- `verify_macro_hygiene`
- `verify_rop_roundtrip`
- `verify_llvm_lowering_contract`

### Task 3.3: Structured diagnostics
Ensure each failure maps to:
- pass name,
- severity,
- message,
- optional node/rid reference.

### Task 3.4: Required rejection cases
- Unknown lane or invalid lane index (`>63`).
- Invalid UX/T typing in binary op.
- Illegal phase reset without epoch increment.
- Same-phase WAW hazards without atomic/barrier/disjoint proof.
- Atomic op on non-atomic-capable registry.
- Out-of-bounds or misaligned access.
- Macro parameter collisions.

---

## Stage 4 — MAP/ROP Normalization

### Task 4.1: Macro hygiene
- Keep hygienic macro expansion in `expand_macros`.
- Ensure generated temporaries never collide with user symbols.

### Task 4.2: MAP lowering policy
- Lower non-atomic MAP load/store to canonical core load/store.
- **Preserve MAP atomic ops** until true atomic lowering stage.

### Task 4.3: Atomic safety rule (non-negotiable)
- Never transform `map_atomic_*` into:
  - barrier + load + binop + store (+ barrier).

### Task 4.4: ROP role clarification
- Keep ROP as schedule/correspondence stream.
- Do not treat ROP as canonical source IR.

---

## Stage 5 — LLVM ABI Substrate

### Task 5.1: Add/maintain ABI reference emitter
Provide (or retain) `bcir_reference_llvm_abi_module()` that emits legal LLVM textual substrate.

### Task 5.2: Required ABI hooks
Ensure module contains symbols for:
- `bcir.rt.phase.enter`
- `bcir.rt.phase.leave`
- `bcir.rt.barrier`
- `bcir.rt.prov.note`
- `bcir.rt.load.*`
- `bcir.rt.store.*`
- `bcir.rt.atomic.add.*`
- `bcir.rt.atomic.sub.*`
- `bcir.rt.atomic.xor.*`

### Task 5.3: Legal IR guarantee
- Emit standard LLVM instructions and calls only.
- Use opaque-pointer style (`ptr`) in emitted text.
- Attach `!bcir.*` metadata hooks as available.

### Task 5.4: Atomic lowering legality
- Lower atomic add/sub/xor to `atomicrmw` forms.
- Lower CAS forms to `cmpxchg`.
- Lower barriers to `fence` or ABI barrier hooks.

---

## Stage 6 — Scheduler & Performance Layout

### Task 6.1: Phase-indexed execution views
Add schedule structures:
- `executionOrder`
- `phaseBegin`
- `phaseEnd`

### Task 6.2: Deterministic ordering
- Sort by `(epoch, phase, lane, opcode, hazardDomain)` in deterministic mode.

### Task 6.3: SoA-friendly internal view
Introduce an internal `BcirCoreSoA` projection for runtime scheduler/cache locality.

### Task 6.4: Phase execution performance rules
- Never scan all nodes per phase.
- Execute via `[phaseBegin[p], phaseEnd[p])` slices.
- Batch UX/T/GGG lane runs where legal.

---

## Stage 7 — Test Plan and CI Tasks

### Task 7.1: Unit tests (data model)
- `BcirClaimV1` size + packing tests.
- Enum parsing/stringification tests.

### Task 7.2: Parser tests
- Valid/invalid lane tokens.
- Surface syntax retention for ROP/MAP constructs.

### Task 7.3: Verifier tests
- Epoch-phase legality resets.
- Hazard RAW/WAR/WAW legality.
- Atomic-capability constraints.
- Bounds and alignment failures.

### Task 7.4: Lowering tests
- MAP atomic preservation tests.
- LLVM ABI module string tests for:
  - `atomicrmw`
  - `cmpxchg`
  - `fence`
  - required runtime hook symbols.

### Task 7.5: Roundtrip tests
- `surface -> core -> rop -> core` metadata/effect consistency.

### Task 7.6: Build-and-test pipeline commands
Run after each stage:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

---

## Guardrails (Do Not Regress)

- Do **not** invent non-standard LLVM instructions.
- Do **not** rewrite atomics into load/op/store pseudo-atomics.
- Do **not** break existing public APIs without compatibility notes/tests.
- Do **not** treat ROP/MAP as canonical BCIR source.

---

## Deliverable Definition of Done

BCIR is complete for this blueprint when:
1. BCIR Core is the semantic source of truth.
2. MAP atomics are preserved until true atomic lowering and verified in tests.
3. LLVM ABI substrate emits legal IR-only constructs.
4. Verifier passes cover lane/phase/epoch/hazard/atomic/bounds invariants.
5. Scheduler supports deterministic phase-sliced execution structures.
6. Full build + test passes in CI.
