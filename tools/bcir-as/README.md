# bcir-as (minimal BCIR assembler)

Minimal BCIR assembler: reads a tiny BCIR source grammar and emits LLVM IR
`.ll` globals for claims/resources/batches/phase ranges.

## Usage

```bash
tools/bcir-as/bcir-as ir/surface/examples/vector_add.bcir -o build/vector_add.generated.ll
```

Input scope: `MODULE`, `HEADER`, `REGISTRY RES/RECORD`, `PHASE`, `CLAIM`, `rd/wr` contracts.
Output: LLVM IR `.ll` globals for claims/resources/batches/phase ranges.

> Note: the hand-written `runtime/llvm/*.ll` seed and its `validate_phase4.sh`
> harness (which previously exercised this tool) were removed in the 2026-06-07
> reorg. `bcir-as` is retained as a standalone surface→LLVM-IR assembler; the
> forward LLVM path is `ir/llvm/` plus the compiled `ir/mlir/` conversion. See
> `docs/BCIR_Repo_Structure.md`.
