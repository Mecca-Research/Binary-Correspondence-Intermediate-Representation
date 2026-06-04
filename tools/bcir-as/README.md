# bcir-as (Phase 4 seed)

Minimal BCIR assembler for Phase 4.

## Usage

```bash
tools/bcir-as/bcir-as examples/vector_add.bcir -o build/vector_add.generated.ll
```

Input scope: `MODULE`, `HEADER`, `REGISTRY RES/RECORD`, `PHASE`, `CLAIM`, `rd/wr` contracts.
Output: LLVM IR `.ll` globals for claims/resources/batches/phase ranges, aligned for BCIR runtime linking.

## Validation prerequisite

`runtime/llvm/validate_phase4.sh` treats `tools/bcir-as/bcir-as` as a required generated-IR producer and runs a Python syntax preflight before invoking it. Keep `tools/bcir-as/bcir-as` as the executable symlink to `tools/bcir-as/bcir-as.py`; if this tool is missing, not executable, or syntactically invalid, Phase 4 validation fails before attempting to generate `build/vector_add.generated.ll`.
