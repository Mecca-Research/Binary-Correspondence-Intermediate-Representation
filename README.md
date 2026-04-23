# Binary-Correspondence-Intermediate-Representation

C++ project skeleton for BCIR with a CMake-based build and install/export rules.

## Top-level layout

- `dialect/` — `bcir-dialect` target (ROP/MAP IR + parser/printer + verifier)
- `runtime/` — `bcir-lowering` and `gem-runtime` targets
- `tools/` — `bcir-tools` CLI target
- `include/` — public headers installed for consumers
- `tests/` — `bcir-tests` plus CTest integration

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
