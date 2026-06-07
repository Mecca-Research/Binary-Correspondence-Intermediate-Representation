# Repository map for agents

This repository contains **two separate things**. Do not conflate them.

## 1. `ir/` — the BCIR intermediate representation (the project)

This is the actual IR. It is organized by pipeline role:

| Path          | What it is |
|---------------|------------|
| `ir/surface/` | text frontend: tokenizer + parser + ROP/MAP verifier (`bcir-surface`) |
| `ir/core/`    | canonical typed graph model + surface→core builder (`bcir-core`) — **the source of truth** (`ir/core/include/bcir/bcir_ir.hpp`) |
| `ir/irdl/`    | declarative dialect projection — **pure IR, no compilation** (loaded by `mlir-opt --irdl-file`) |
| `ir/mlir/`    | compiled MLIR dialect + conversion to the LLVM dialect (opt-in: `-DBCIR_ENABLE_MLIR=ON`) |
| `ir/llvm/`    | legal LLVM IR emission + ABI substrate (`bcir-llvm`) |
| `ir/runtime/` | GEM execution engine (`gem-runtime`) |

`tools/` and `tests/` are top-level consumers of these libraries.

## 2. `llvm-training/` — an LLVM/MLIR training corpus (NOT the IR)

`llvm-training/` is agent-readable teaching material: chaptered lessons,
checked `.ll`/`.mlir` examples, exercises, and verify scripts. It exists to
train agents that work on the IR. **It is not part of the BCIR IR and the IR
does not depend on it.** Changes to the IR must not require changes here, and
vice versa. Its conventions live in `CONTRIBUTING.md`.

## Where to work

- Building/changing the IR → work under `ir/` (plus `tools/`, `tests/`).
- Writing/curating training material → work under `llvm-training/`.
- Architecture and the IRDL/MLIR/LLVM section boundaries →
  [`docs/BCIR_Repo_Structure.md`](docs/BCIR_Repo_Structure.md).
