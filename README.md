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
