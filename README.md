# Binary-Correspondence-Intermediate-Representation

LLVM and MLIR project for BCIR with a CMake-based build, install/export rules,
and C++ implementation.

> **Two separate things live in this repo.** `ir/` **is** the BCIR intermediate
> representation. `llvm-training/` is an LLVM/MLIR training corpus for agents and
> is **not** part of the IR. See [`AGENTS.md`](AGENTS.md) and
> [`docs/BCIR_Repo_Structure.md`](docs/BCIR_Repo_Structure.md).

## Top-level layout

```
.
├── ir/                  the BCIR IR itself
│   ├── surface/         bcir-surface: tokenizer + parser + ROP/MAP verifier
│   ├── core/            bcir-core: canonical typed graph model + surface→core builder
│   ├── irdl/            pure-IR dialect projection (no compilation; mlir-opt round-trip)
│   ├── mlir/            compiled MLIR dialect + conversion (opt-in: -DBCIR_ENABLE_MLIR=ON)
│   ├── llvm/            bcir-llvm: legal LLVM IR emission + ABI substrate
│   └── runtime/         gem-runtime: GEM execution engine
├── tools/               bcir-tools CLI and helper assemblers (consume the IR)
├── tests/               cross-cutting CTest integration tests
├── docs/                blueprints, design notes, and the repo-structure decision
└── llvm-training/       SEPARATE agent-readable LLVM/MLIR curriculum (not the IR)
    ├── 00-foundations/ … 18-mlir-lowering-to-llvm/   chaptered lessons + examples
    ├── bcir-mapping/    BCIR-concept → lowered LLVM IR examples
    ├── exercises/       numbered prompt/solution exercises
    └── reference/       instruction quickref, intrinsics, glossary
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contributor conventions, training
example naming rules, and verification scripts to run before opening a change.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Install

```bash
cmake --install build --prefix /tmp/bcir-install
```

This exports CMake package files under `lib/cmake/BCIR`.

## Concurrency and determinism controls

`GemCreateOptions` exposes multithreaded execution controls:

- `workerThreads`: number of worker threads (0 defaults to one worker).
- `deterministicOrdering`: forces deterministic node dispatch order within each phase
  (nodes are scheduled by ascending node id).
- `phaseWaitTimeoutMs`: optional deadlock/livelock safeguard timeout for phase
  completion barriers (`0` disables timeout).

Dialect verification includes explicit concurrent registry/atomic checks via the
`concurrent_registry_access_by_lane_and_atomic_constraints` pass:

- MAP operations touching the same RID in one phase/epoch must not race across
  different lanes unless the accesses are atomic-only.
- Atomic and non-atomic MAP accesses to the same RID must be separated by a
  phase transition or barrier.
- MAP lane directives are constrained to `lane0..lane63`.

## BCIR v1 formalization artifacts

- `docs/BCIR_LLVM_IR.md` — formal BCIR graph spec, resolved lane/hazard/phase semantics,
  LLVM textual dialect mapping, K_BDI integration points, and migration plan.
- `ir/core/include/bcir/bcir_ir.hpp` — C++ data model for BCIR nodes/edges/cost tuples and a
  fixed 64-byte `BcirClaimV1` binary schema compatible with cache-line scheduling.

## BCIR Codex blueprint

The master implementation work-order is documented in:

- `docs/BCIR_Codex_Blueprint.md`

This blueprint makes **BCIR** the canonical source IR and defines the staged build tasks
for the full path: `bcir.surface -> bcir.core -> bcir.rop -> mlir.llvm -> llvm ir`.
