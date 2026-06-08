# Repository map for agents

This repository contains **two separate things**. Do not conflate them.

## 1. The BCIR IR (the project) — `bcir/` + `mlir/`

The IR is realized as two cooperating trees:

| Path     | What it is |
|----------|------------|
| `bcir/`  | the **executable conformance oracle** (Python, dependency-free, runnable today): the K_BCIR optimizer, GEM hydration/execution, M5 event transduction, ROP/MAP front-ends, LLVM lowering (AOT clang + JIT lli), the data-DNA telemetry loop, and the R1–R12 verifier subset |
| `mlir/`  | the **IR law**: the TableGen/ODS dialect family, the **compiled `bcir-opt`** (`lib/BCIRDialect.cpp` + `tools/bcir-opt.cpp`), and the pure-data **IRDL projection** |

The Python package is the runnable conformance reference for the MLIR dialect law
(see [`docs/PARITY.md`](docs/PARITY.md)); they must agree. `tools/` holds the
validation scripts (tblgen / IRDL round-trip / build + check `bcir-opt`).

> History: an earlier C++ skeleton (`ir/surface`, `ir/core`, `ir/llvm`,
> `ir/runtime`) was the first milestone. It has been **retired** — its semantics
> are subsumed by the `bcir/` oracle and the compiled `mlir/` dialect. The repo is
> now a single tree (see `docs/BCIR_Repo_Structure.md`).

## 2. `llvm-training/` — an LLVM/MLIR training corpus (NOT the IR)

Agent-readable teaching material: chaptered lessons, checked `.ll`/`.mlir`
examples, exercises, and verify scripts. It exists to train agents that work on
the IR. **It is not part of the BCIR IR and the IR does not depend on it.**
Changes to the IR must not require changes here, and vice versa. Its conventions
live in `CONTRIBUTING.md`.

## Where to work

- Changing the IR semantics/optimizer → `bcir/` (with parity in `mlir/`).
- Changing the IR law (ops/types/attrs) → `mlir/` (validate with `tools/`).
- Writing/curating training material → `llvm-training/`.
- Architecture → [`docs/BCIR_Repo_Structure.md`](docs/BCIR_Repo_Structure.md).
