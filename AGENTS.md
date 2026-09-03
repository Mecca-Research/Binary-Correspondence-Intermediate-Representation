# Repository map for agents

This repository contains **two separate things**. Do not conflate them.

## 1. The BCIR IR (the project) — `bcir/` + `mlir/`

The IR is realized as two cooperating trees:

| Path     | What it is |
|----------|------------|
| `bcir/`  | the **executable conformance oracle** (Python, dependency-free, runnable today): the K_BCIR optimizer, GEM hydration/execution, M5 event transduction, ROP/MAP front-ends, LLVM lowering (AOT clang + JIT lli), the data-DNA telemetry loop, and the R1–R25 verifier reference |
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

## Non-negotiable pre-PR validation

Do not commit, push, or open/update a PR merely because the default local Python
suite passes. Read [`.github/workflows/ci.yml`](.github/workflows/ci.yml), map the
affected jobs, and run the relevant checks available on the development host.
Use GitHub Actions for unavailable operating systems, Python versions, and native
architectures. A pass on a different environment is not equivalent evidence, and
resource-constrained workstations must not emulate ARM to manufacture it.

During iteration, focused tests are appropriate. Before a commit or PR update:

On the local development workstation, serialize heavy gates and cap parallel work
at two workers. Use `-j 2` for the Python runner,
`CMAKE_BUILD_PARALLEL_LEVEL=2` for builds, and
`-DLLVM_TRAINING_LIT_JOBS=2` for the LLVM-training aggregate. Do not run the
Python, C sanitizer/fuzzer, model, LLVM-training, or MLIR gates concurrently.

1. Run focused regressions and the bounded complete quick oracle locally. Run
   broader local gates only when the host and resource budget support them.
2. Delegate the full Python/differential/fuzz, C runtime/model, LLVM training,
   MLIR, Windows, and native ARM matrix to GitHub Actions as appropriate.
3. Treat hardware-only validation that neither the local host nor CI provides as
   an explicit, documented skip. Never hide it or replace it with unsafe emulation.
4. Confirm tests leave tracked files unchanged and run `git diff --check`.
5. After publishing, wait for the full Actions run and fix every failure before
   handing off the PR as complete.

The command inventory and tool-gated rules live in
[`CONTRIBUTING.md`](CONTRIBUTING.md#required-pre-pr-ci-equivalence-gate).
