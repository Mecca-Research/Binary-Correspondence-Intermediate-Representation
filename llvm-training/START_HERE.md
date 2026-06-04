# START HERE: Agent Consumption Protocol

This folder is a compact context/reference pack for coding agents. It is not a
fine-tuning corpus and should not be treated as training data for model updates.

## Preferred path for the expanded corpus

1. Read [`INDEX.md`](INDEX.md) first to choose the correct lookup axis.
2. Open [`RECIPES.md`](RECIPES.md) and select the row closest to the current
   task; prefer the advanced rows for BCIR lowering, MLIR integration,
   backend/JIT diagnostics, binary-analysis evidence, or repair fixtures.
3. Read the recipe's **Read first** file before opening examples.
4. Use [`CURRICULUM.md`](CURRICULUM.md) only when the task needs a fuller
   learning path rather than a focused recipe.
5. Before writing or verifying artifacts, review [`EXAMPLES.md`](EXAMPLES.md)
   and [`SEMVER.md`](SEMVER.md) so `.ll`, `.invalid.ll.txt`, `.mlir`, CSV/data,
   and generated BCIR mapping outputs are handled by the correct script.
6. Before writing LLVM IR, review [`08-pitfalls/README.md`](08-pitfalls/README.md)
   and any relevant pitfall pages.
7. Use `10-grammar/llvm-ir.tm` only for syntax-shape questions; use LLVM
   LangRef for canonical semantics.
8. After reading a selected path, use [`EVAL.md`](EVAL.md) as the final
   self-check.

## Practical protocol

- Load only the files relevant to the current task after reading the index and
  selected recipe row.
- Prefer targeted pitfall checks before generating, repairing, or reviewing LLVM
  IR.
- Treat examples here as guidance and context; verify semantic claims against
  LLVM LangRef when correctness matters.
- Run or review `tools/verify-examples.sh` when working with standalone `.ll`
  examples or generated BCIR mapping outputs.
- Run or review `tools/verify-invalid-fixtures.sh` for intentionally broken
  `.invalid.ll.txt` repair inputs.
- Run or review `tools/verify-mlir-examples.sh` only when MLIR tools are present;
  otherwise treat `.mlir` files as review artifacts.
- Respect the broken `.ll.txt` sentinel for intentionally non-compilable IR
  examples.
- Use [`bcir-mapping/README.md`](bcir-mapping/README.md) for BCIR lowering,
  runtime-call boundaries, and diagnostic metadata preservation.
- Use [`14-mlir-bridge/01-what-is-mlir.md`](14-mlir-bridge/01-what-is-mlir.md)
  and the MLIR rows in [`RECIPES.md`](RECIPES.md) when the source
  representation starts as MLIR or should remain structured before LLVM
  lowering.
- Use [`12-backend-jit/`](12-backend-jit/) when target lowering, ORC/LLJIT,
  MC emission, relocations, or TableGen diagnostics are relevant.
- Use [`15-binary-analysis/README.md`](15-binary-analysis/README.md) when
  dynamic traces, side channels, optimized binaries, or BCSA evidence are
  relevant.
