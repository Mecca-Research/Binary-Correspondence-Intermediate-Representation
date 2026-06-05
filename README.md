# Binary-Correspondence-Intermediate-Representation

LLVM and MLIR project skeleton for BCIR with a CMake-based build, install/export rules, and C++ implementation.

## Top-level layout

```
.
├── dialect/             bcir-dialect target (ROP/MAP IR + parser/printer + verifier)
├── runtime/             bcir-lowering, gem-runtime, and LLVM runtime artifacts
├── tools/               bcir-tools and helper CLIs
├── include/             public headers installed for consumers
├── tests/               bcir-tests plus CTest integration
├── docs/                implementation blueprints and design notes
└── llvm-training/       agent-readable LLVM IR curriculum and reference
    ├── 00-foundations/  IR basics, SSA, IR vs assembly/other IRs
    ├── 01-syntax/       modules, functions, basic blocks, instruction format
    ├── 02-types/        primitive, composite, opaque, and pointer types
    ├── 03-constants/    integer, floating-point, string, global/local constants
    ├── 04-memory/       alloca, load/store, globals, address spaces
    ├── 05-control-flow/ branches, switch, indirectbr
    ├── 06-metadata/     metadata tags, debug info, profile/loop metadata
    ├── 07-optimization/ pass model, analyses, transforms, optimization levels
    ├── 08-pitfalls/     verifier and real-world IR failure modes
    ├── 09-vectorization/ auto-vectorization and vector IR quick references
    ├── 10-grammar/      Textmapper grammar and syntax notes
    ├── 11-concurrency/  atomics, orderings, fences, volatile vs atomic
    ├── 12-backend-jit/  codegen pipeline, TableGen, ORC/LLJIT
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

- `docs_BCIR_LLVM_IR.md` — formal BCIR graph spec, resolved lane/hazard/phase semantics,
  LLVM textual dialect mapping, K_BDI integration points, and migration plan.
- `include/bcir/bcir_ir.hpp` — C++ data model for BCIR nodes/edges/cost tuples and a
  fixed 64-byte `BcirClaimV1` binary schema compatible with cache-line scheduling.

## BCIR Codex blueprint

The master implementation work-order is documented in:

- `docs/BCIR_Codex_Blueprint.md`

This blueprint makes **BCIR** the canonical source IR and defines the staged build tasks
for the full path: `bcir.surface -> bcir.core -> bcir.rop -> mlir.llvm -> llvm ir`.
