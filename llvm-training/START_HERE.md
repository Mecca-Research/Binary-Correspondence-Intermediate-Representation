# START HERE: Agent Consumption Protocol

This folder is a compact context/reference pack for coding agents. It is not a fine-tuning corpus and should not be treated as training data for model updates.

## Required reading order

1. Read [`INDEX.md`](INDEX.md) first.
2. Use [`RECIPES.md`](RECIPES.md) as the task-oriented lookup.
3. Use [`CURRICULUM.md`](CURRICULUM.md) for learning paths.
4. Before writing or verifying examples, review [`EXAMPLES.md`](EXAMPLES.md) and [`SEMVER.md`](SEMVER.md).
5. Before writing LLVM IR, review [`08-pitfalls/README.md`](08-pitfalls/README.md) and any relevant pitfall pages.
6. Use `10-grammar/llvm-ir.tm` only for syntax questions; use LLVM LangRef for canonical semantics.
7. After reading a selected path, use [`EVAL.md`](EVAL.md) as the final self-check.

## Practical protocol

- Load only the files relevant to the current task after reading the index.
- Prefer targeted pitfall checks before generating or editing LLVM IR.
- Treat examples here as guidance and context; verify semantic claims against LLVM LangRef when correctness matters.
- Run or review `tools/verify-examples.sh` when working with standalone examples.
- Respect the broken `.ll.txt` sentinel for intentionally non-compilable IR examples.
- Use [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md) for optimized-binary analysis.
- Use [`14-mlir-bridge/README.md`](14-mlir-bridge/README.md) when a task starts above LLVM IR or needs BCIR dialect lowering.
- Use [`15-binary-analysis/README.md`](15-binary-analysis/README.md) when dynamic traces, side channels, or BCSA are relevant.
