# bcir-as (Phase 4 seed)

Minimal BCIR assembler for Phase 4.

## Usage

```bash
tools/bcir-as/bcir-as examples/vector_add.bcir -o build/vector_add.generated.ll
```

Input scope: `MODULE`, `HEADER`, `REGISTRY RES/RECORD`, `PHASE`, `CLAIM`, `rd/wr` contracts.
Output: LLVM IR `.ll` globals for claims/resources/batches/phase ranges, aligned for BCIR runtime linking.
