# BCIR LLVM Phase 4: Assembler, Translator, Blob Pack, and Feedback Loop

Phase 4 adds a tiny BCIR source frontend that emits LLVM IR `.ll` artifacts.

- Semantic owner remains `runtime/llvm/*.ll`.
- Assembler emits claims/resources/batches/phase-range globals.
- Blob schema/verification, telemetry, and rehydrate policy are LLVM-visible.

## Pipeline
1. `tools/bcir-as/bcir-as examples/vector_add.bcir -o build/vector_add.generated.ll`
2. link with runtime phase modules
3. `llvm-as`
4. `opt -passes=verify`

## Scope constraints
- No C++ runtime migration.
- No MLIR or full language frontend.
- BCIRClaim tiny grammar only.
