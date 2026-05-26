# BCIR Full LLVM Build Blueprint (Execution Plan)

This plan turns BCIR from parser/dispatch skeleton into a concrete pure-LLVM backend while preserving legal IR constraints.

## Priority ordering
P0 Fix core model  
P1 Add registry descriptors  
P2 Add epoch/phase verifier  
P3 Add precise hazard verifier  
P4 Build schedule view  
P5 Implement textual LLVM emitter  
P6 Validate with `llvm-as`/`opt`  
P7 Fix GEM scheduler ready-node handling  
P8 Add runtime ABI definitions  
P9 Add metadata/provenance/cost emission  
P10 Add T/GGG specialization hooks  
P11 Optional MLIR dialect  
P12 Optional BDI-K autotuning

## Detailed stages

### Stage 0 — Freeze baseline
- Run and record baseline build/test.
- Keep existing semantics unchanged.
- Maintain `docs/REPO_CURRENT_STATE_AUDIT.md`.

### Stage 1 — Expand `bcir.core`
- Expand `include/bcir/bcir_ir.hpp` with typed opcode/lifetime/address-space/type/registry/theta/graph models.
- Keep `BcirClaimV1` 64-byte invariant.

### Stage 2 — Surface AST -> Core graph builder
- Add `include/bcir/core_builder.hpp` and `runtime/src/core_builder.cpp`.
- Convert parsed AST into typed `BcirGraph`.
- Emit DataFlow/ControlFlow/HazardOrder/PhaseOrder edges.

### Stage 3 — Epoch/phase legality
- Add epoch-aware phase syntax/normalization path.
- Verifier transition rule: `(epoch,phase)` monotonic; phase reset only on epoch increment.

### Stage 4 — Registry verification
- Verify descriptor presence, capacity bounds, alignment, address-space validity, alias group legality, and atomic capability constraints.

### Stage 5 — Hazard verification
- Implement RAW/WAR/WAW contracts with atomic/barrier exceptions and alias-group awareness.

### Stage 6 — Schedule view
- Add `BcirSchedule` with `executionOrder`, `phaseBegin`, `phaseEnd`.
- Deterministic sort key: epoch, phase, lane, opcode, hazardDomain, nodeId.

### Stage 7 — Textual LLVM emitter
- Add `include/bcir/llvm_emit.hpp` and `runtime/src/llvm_emit.cpp`.
- Emit legal LLVM IR only (`load/store/add/...`, `atomicrmw`, `cmpxchg`, `fence`, runtime calls).

### Stage 8 — Metadata schema
- Emit module metadata (`!bcir.module`, `!bcir.pipeline`, `!bcir.lanes`).
- Emit per-instruction BCIR metadata where available (`!bcir.node`, `!bcir.rid`, `!bcir.phase`, etc.).

### Stage 9 — LLVM toolchain validation tests
- Add CMake option `BCIR_ENABLE_LLVM_TOOL_TESTS`.
- Detect `llvm-as`, `opt`, optional `lli`.
- Add emitter artifact tests for vector-add, atomic-add, CAS, phase/epoch, hazard metadata.

### Stage 10 — Runtime ABI definitions
- Keep ABI declarations in reference module.
- Implement C++ runtime counterparts for linkable execution mode.

### Stage 11 — GEM scheduler correction
- Ensure future-phase ready nodes are preserved.
- Enqueue same-phase successors as soon as indegree reaches zero.

### Stage 12 — Performance hardening
- Intern hot strings, introduce numeric RID path, preallocate arrays, phase slices, and SoA projections for scheduler hot loops.

## Non-regression rules
- Never invent non-standard LLVM instructions.
- Never rewrite atomics into load/op/store pseudo-atomics.
- Keep public APIs stable unless migration notes/tests accompany changes.
