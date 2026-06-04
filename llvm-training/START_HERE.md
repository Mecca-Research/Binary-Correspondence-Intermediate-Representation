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

- Load only the files relevant to the current task after reading the index.
- Prefer targeted pitfall checks before generating or editing LLVM IR.
- Treat examples here as guidance and context; verify semantic claims against LLVM LangRef when correctness matters.

## MLIR lowering tasks

When a task starts above LLVM IR or involves BCIR dialect lowering, read [`14-mlir-bridge/README.md`](14-mlir-bridge/README.md) before editing examples.
