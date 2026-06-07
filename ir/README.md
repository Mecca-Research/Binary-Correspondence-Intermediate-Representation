# `ir/` — legacy C++ IR skeleton (superseded)

> **Status: LEGACY.** This is the earlier C++ milestone (surface parser/verifier
> + textual LLVM emitter + a threaded GEM runtime). The **canonical** BCIR system
> is now the IR-first stack: the runnable oracle in [`../bcir/`](../bcir/) + the
> MLIR/IRDL law in [`../mlir/`](../mlir/). New work goes there. See
> [`../docs/BCIR_Repo_Structure.md`](../docs/BCIR_Repo_Structure.md) for the
> staged, non-destructive fold plan.

This tree is retained because it still owns the installable CMake targets
(`bcir-surface`, `bcir-core`, `bcir-llvm`, `gem-runtime`) and the C++ `ctest`
suite, and because deleting it is irreversible in this environment. It is not
deleted in the current PR.

## Fold progress

| Step | What | State |
|---|---|---|
| 1 | Declare canonical stack; mark `ir/` legacy | done |
| 2 | Port unique `ir/` semantics into `bcir/` | **in progress** — the deterministic phase-sliced GEM executor (`ir/runtime/`) is ported to `bcir/gem/execute.py` with parity tests |
| 3 | Retire `ir/surface` + `ir/llvm` (subsumed by `bcir/etl` + `bcir/lower` + the `mlir/` law) | pending |
| 4 | Collapse to a single tree once parity is proven | pending |

## Sections (as-built)

`surface/` (`bcir-surface`) · `core/` (`bcir-core`) · `irdl/` + `mlir/`
(scaffolds) · `llvm/` (`bcir-llvm`) · `runtime/` (`gem-runtime`). These remain
buildable via the top-level CMake (`cmake -S . -B build && cmake --build build`).
