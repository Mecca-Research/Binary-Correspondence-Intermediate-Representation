# BCIR build roadmap: from the shell build to CMake, CTest and the BCIR Make file

*Roadmap (owns the future). Landing notes go to [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md);
counts of tests live in generated [`STATUS.md`](STATUS.md). BUILD-0 (this document) and
BUILD-1 landed together; the state table in §10 is what BUILD-1's own runs measured.*

## 1. Why: the shell build, measured

The C and C++ rails have been built by shell since PR #153. The scripts under `tools/` were
measured for this roadmap on 2026-10-04:

| What | Measured |
|---|---|
| Shell scripts under `tools/` | 27 scripts, 8,600 lines |
| `tools/c/check_runtime.sh` alone | 4,543 lines, 116 gate sections, 223 compiler invocations |
| Distinct source lists spelled by hand | the gate's link arrays (`ctl_sources`, `ring_sources`, `handoff_sources`, `kplan_sources`, `CFRONT_SRCS`, `LOOP_SRCS`), `fuzz_streampack.sh`'s 19 `add_target` rows, `check_handoff.sh`'s `seam_c`/`seam_cpp`, `check_memory_discipline.sh`'s two arrays, and the Python harnesses' copies (`native_bench._SOURCES`, `control_fixtures.C_SOURCES`, `ring_fixtures.C_SOURCES`, `handoff_fixtures.C_UNITS`/`CPP_UNITS`, `planner_fixtures.C_UNITS`, the model gates' five-file list, and a dozen single-harness tests) |
| Freestanding proof | 14 units compiled `-ffreestanding -nostdlib` by the gate, each in its own section, at C11 and C23 |
| Compilers and standards | GCC 13 and Clang 18/23, `-std=c23` with a per-section fallback to `c2x` or `c11`, C++17 for the seam |

Three things made this a monolith rather than a build system:

1. **The same list in many places.** A unit linked by a gate and by a harness is spelled in both,
   and the two drift (PR #719 shipped a gate and a harness that linked different sources; the
   `test_the_gate_and_the_harness_link_the_same_sources` tests were the per-pair patch).
2. **Policy spelled per invocation.** `-Wall -Wextra`, the standard, the fallback, `-O2`, the
   include path and the sanitizer flags are written 223 times in one file; a policy change is a
   223-site edit, and each site can disagree with the rest (laws.md L12).
3. **No graph.** The gate is a sequence; nothing knows that `test_ring` depends on three units
   and not on the other 35, so nothing can rebuild only what changed, run two independent
   sections at once under the two-worker cap, or say what a change to `bcir_runtime.c` affects.

What shell is *good* at here stays shell (§7): provisioning a toolset, driving CI, one-off
utilities, and the gates' own content until each section has been migrated with a proof.

## 2. Principles

1. **One manifest.** [`runtime/manifest.json`](../runtime/manifest.json) is the only list of what
   the rails are made of. CMake reads it directly (`string(JSON)`, no Python at configure);
   `tools/build/manifest.py --check` holds every gate, harness, class inventory and preset to it,
   and `bcir/tests/test_build_manifest.py` injects a violation of each rule and watches it fire.
   Nothing is listed in a `CMakeLists.txt`: a source belongs to a manifest unit or it is not built.
2. **The gates' recipes are the law until a section migrates.** A CMake target may be built
   differently from the gate only when a parity gate proves the outcome identical. BUILD-1's proof is
   `tools/build/build_parity.py`: the CMake-built `bcir-cc` against the gate's one-command recipe,
   every mode over every fixture, stdout + stderr + status + pack bytes identical.
3. **Building is a proof.** The freestanding object checks (every `freestanding_core` unit at C11
   and at C23 with `-ffreestanding -nostdlib`) are part of `ALL`, so a hosted dependency that
   creeps into the core fails `cmake --build`, not a later script.
4. **Nothing optional is required.** The rails are dependency-free C and C++ by law. MLIR, the
   numeric libraries and Python are *found and recorded* (`bcir-deps.json`), never demanded, and
   the MLIR law is built only when asked (`BCIR_BUILD_MLIR=ON`, the `mlir` preset) -- a package the
   host happens to carry on `PATH` must not decide what the default preset builds.
5. **One memory class per library.** A library's `class` is one of
   [`runtime/c/MEMORY_CLASSIFICATION.txt`](../runtime/c/MEMORY_CLASSIFICATION.txt)'s three, every
   unit in it carries that class there, and the checker refuses a mixed library. The link graph now
   shows the memory discipline: a freestanding library never links a hosted one.
6. **The two-worker law is in the presets.** Every build preset says `jobs: 2`, every test preset
   `execution.jobs: 2`, every heavy CTest entry declares `PROCESSORS 2` so two never overlap, and
   the checker refuses a preset that says otherwise (AGENTS.md).
7. **Fail closed.** An absent compiler in the CI job that exists to use it is a failure; an empty
   fixture corpus is `INVALID`, not a pass; a fuzz entry with zero runs is refused at configure
   (laws.md L1, L2).
8. **Read the rails, never mirror them.** The checker reads the gates' compile lines and the
   Python modules' literal groups out of their own text; a third list would drift like the first two.

## 3. Target layout (BUILD-1)

```
CMakeLists.txt                 project BCIR (C, CXX), options, presets' home
CMakePresets.json              default / gcc / clang / asan / ubsan / tsan / fuzzer / mlir
cmake/BCIRCompilerFlags.cmake  C23 (c2x where needed), C++17 -Wpedantic, -Wall -Wextra [-Werror],
                               -O2 without NDEBUG, BCIR_SANITIZE, the freestanding object check
cmake/BCIRDependencies.cmake   Python, Threads, MLIR/LLVM, FFTW3F/LAPACKE/GSL/SLEEF/libcerf -> bcir-deps.json
cmake/BCIRManifest.cmake       the manifest reader, closures, link-name and option mapping
cmake/BCIRTests.cmake          CTest registration (labels, PROCESSORS 2, the gates as entries)
runtime/manifest.json          libraries / tools / harnesses / fuzzers / seam_libraries / seam_tests
runtime/c/CMakeLists.txt       targets from the manifest; nothing listed by hand
runtime/cpp/CMakeLists.txt     the seam's libraries and tests over the C libraries built as C
mlir/CMakeLists.txt            unchanged; add_subdirectory()'d when BCIR_BUILD_MLIR is on
```

| Manifest kind | Units | Rule |
|---|---|---|
| `libraries` | 10 static libraries over the 38 `bcir_*.c` units that are not tool mains: `bcir_base` (runtime, SHA-256), `bcir_streampack`, `bcir_plane` (control plane, rings, hand-off, planner), `bcir_artifact`, `bcir_asn1`, `bcir_cfront` (the twin, the preprocessor, the verifier, diagnostics), `bcir_quarantine`, `bcir_models`, `bcir_channel`, `bcir_driver` | class-homogeneous; `libraries` is the dependency edge; `link` names `m`/`pthread`; `options` carries the gates' per-unit flags (`gcc:`/`clang:` scoped) |
| `tools` | `bcir-cc`, `bcir_asn1_bench`, `bcir_microbench`, `bcir_ai_microbench`, `bcir-llama` | one `hosted_tool` main each |
| `harnesses` | the 32 `test_*.c` mains | built in `ALL` (the link proof); the gates run them with their own argv until BUILD-2 |
| `fuzzers` | the 19 `fuzz_*.c` targets | Clang only; each compiles its library closure into itself with `-fsanitize=fuzzer,address,undefined` so the code under test is instrumented; `max_len` equals the gate's `-max_len` |
| `seam_libraries` / `seam_tests` | `bcir_seam`, `bcir_jer_index`; `test_orchestrator`, `test_handoff_cpp`, `test_artifact_bundle_cpp`, `test_jer_index`, `test_jer_simd` | C++17 over the C libraries; the seam's unit lists equal `check_handoff.sh`'s and `handoff_fixtures.py`'s |
| `freestanding_checks` | the 26 `freestanding_core` units | OBJECT libraries at C11 and C23, `-ffreestanding -nostdlib -O0`; a superset of the 14 the gate compiles |

The build tree is `build/cmake-<preset>/`: libraries and tools under `runtime/c/`, harnesses
under `harnesses/`, fuzz targets under `fuzzers/`, the dependency index at `bcir-deps.json`
(schema `bcir-deps.v1`: compilers, system, one `{name, found, detail}` row per dependency).

## 4. CTest mapping

| Label | Entries (BUILD-1) | Owner on CI |
|---|---|---|
| `build` | `build-manifest` (the checker), `build-parity` (the `bcir-cc` parity gate) | `cmake-build` (gcc + clang) |
| `python` | `python-quick` (the oracle's quick tier, `-j 2`) | the oracle jobs |
| `shell` + `c` / `cpp` / `fuzz` | the shell gates as entries under the configured compilers: `c-runtime`, `c-memory-discipline`, `cpp-handoff`; on Clang `cfront-sanitize`, `fuzz-streampack` | the C-rails jobs |
| `fuzz` + `c` | `fuzz-<target>` for the 19 libFuzzer targets, `-runs=BCIR_FUZZ_RUNS` (bounded, > 0) | `cmake-build` (clang cell, the `fuzzer` preset) |
| `mlir` | `mlir-passes`, `mlir-ods-examples`, `mlir-bytecode`, `mlir-irdl-corpus` with `BCIR_OPT` pointing at the tree's `bcir-opt` | `mlir-rail-validate` |
| `docs` | `docs-status`, `docs-links` | `docs-governance` |

`ctest --preset build` is the fast local check; `ctest --preset all` runs everything registered
with two workers and the heavy entries serialized by `PROCESSORS 2`. A gate wrapped as an entry
is still the gate -- same script, same output -- so the wrapping changes nothing about what is
judged, only who schedules it.

## 5. Dependencies and submodules

- **Required: a C compiler and a C++ compiler.** GCC or Clang on Linux (x86-64 and aarch64, the
  CI matrix) and macOS; MSVC configures but is not claimed (§6). Python is required only by the
  CTest entries that run the oracle or the checker.
- **Found and recorded, never required:** Threads; an MLIR/LLVM package (`MLIR_DIR`; the
  `tools/local/setup_mlir.sh` toolset, or apt's); FFTW3F, LAPACKE, GSL, SLEEF and libcerf -- the
  libraries the twin's `--emit-link-flags` rules name, whose presence decides which E-series
  fallbacks a host can judge natively. Each is one row of `bcir-deps.json`; BUILD-4 makes the
  Python harnesses read that index instead of probing on their own (laws.md L14).
- **Submodules: none, by policy.** Third-party code is consumed as an installed package, never
  vendored; the law rail's LLVM/MLIR comes from conda-forge or apt at one coherent major. If a
  submodule is ever needed it is pinned by SHA, read-only, and `FetchContent`/network access
  never happens at configure on the PR path (a configure must be reproducible offline).
- **Index, not probe.** A harness that needs to know whether FFTW is present reads
  `bcir-deps.json`; the configure is the one place that asks, so two harnesses cannot answer
  differently.

## 6. Platform and compiler policy

| Policy | Setting |
|---|---|
| C standard | `C_STANDARD 23`, `C_EXTENSIONS OFF`; CMake spells it `-std=c2x` on GCC 13 (the gate's own fallback) |
| C++ standard | `CXX_STANDARD 17`, `-Wpedantic` for the seam; `mlir/` keeps its C++23/26 |
| Warnings | `-Wall -Wextra -Werror` (`BCIR_WERROR=OFF` to downgrade); the twin's known GCC families (`-Wmisleading-indentation`, `-Wstringop-truncation`, `-Wformat-truncation`, `-Wmaybe-uninitialized`) are scoped in the manifest, so the gates' "no `-Werror` for the twin" became "every other warning is still an error" |
| Optimization | Release is `-O2` **without** `NDEBUG`: the verifier's asserts compile in, as in every gate |
| Sanitizers | `BCIR_SANITIZE=address,undefined` / `undefined` / `thread` on every target (`asan`/`ubsan`/`tsan` presets), `-fno-omit-frame-pointer -g` |
| Fuzzers | Clang only (`libFuzzer`); GCC configures with `BCIR_BUILD_FUZZERS=OFF` and says so |
| `-march` | never set; the rails are portable C and the measured rigs (`tools/silicon`) choose their own |
| MSVC | `/W4 /WX`, best effort, a configure-time warning that it is not claimed |

## 7. What stays in shell

- **Provisioning:** `tools/local/setup_mlir.sh` (the coherent LLVM/MLIR toolset), `tools/wsl/*`
  (the WSL/apt route), the model checkpoint fetchers.
- **CI/CD glue:** `.github/workflows/*` and the scripts they call in sequence.
- **Ad-hoc utilities and interfaces:** `tools/c/streampack_corrupt.py`-style one-offs, the
  sanitizer sweeps' orchestration, the campaign drivers under `tools/security/`.
- **The gates' content** until each section has migrated (BUILD-2): the migration is section by
  section with a byte-identity proof per section, never a rewrite.

## 8. The BCIR Make file and the smart task runner (design; BUILD-6..8)

The build is itself a graph problem BCIR already knows how to state. The design, not landed:

- **A task is a claim over file resources.** A BCIRfile declares targets whose inputs and outputs
  are files (content-addressed by digest), whose command is a tool identity from the registry
  (R1: registry-first -- a compiler is named by its recorded identity, never by whatever `cc`
  resolves to), and whose declared reads/writes are checked against the command's observed
  footprint (the effect-footprint machinery `bcir-cc --emit-effects` already computes for C).
- **The DAG is a phase graph.** Targets are phases; edges are the file claims; the verifier's
  anti-cycle and ownership laws apply unchanged. Two targets that write the same resource without
  an edge are an error at plan time, not a race at run time.
- **Staleness is a generation tag.** Every artifact carries the digest of its inputs and its tool
  identity (R11's per-resource generation vectors); a target is rebuilt when a tag differs, and
  never otherwise -- no timestamps. Artifacts are immutable within a generation (invariant 6 of the
  engineering method), promoted at quiescent boundaries, rolled back as a whole.
- **Scheduling is the existing schedulers.** The wave/token/EFT schedulers over the task DAG under
  the two-worker cap, with the measured durations of earlier runs as the cost model's priors
  (a `wall` metric: indicative, never gating).
- **The runner lands in the oracle first** (`bcir/make/`: the BCIRfile grammar, the DAG model, the
  dry-run planner, `bcir-make` as a module entry point), then its C twin, then the parity gate
  between them, as every rail here has landed.
- **The first programs run as BCIR Make files:** the C rails' own build (the manifest *is* the
  first BCIRfile's input), then the gate sections as tasks whose outputs are the gates' recorded
  verdict files.
- **Make-related managers** that fit the module: the artifact cache (content-addressed, generation
  tagged), the dependency pinning manager (the `bcir-deps.json` index with recorded tool
  identities), and the task telemetry (one envelope per task run, through the existing ring).

What this is not: a general build language. It is a build *law* for this repository's own
rails, judged like everything else here by the oracle, the twin and their parity.

## 9. The ladder

| Slice | Deliverable | Gate |
|---|---|---|
| **BUILD-0** | this roadmap with the measured inventory | linked from the README, the repo-structure map and AGENTS.md |
| **BUILD-1** (landed) | top-level CMake, the four modules, `runtime/manifest.json`, the presets, CTest registration, `tools/build/manifest.py`, `tools/build/build_parity.py`, the `cmake-build` CI job | the checker clean and every injected violation a finding; build parity over every fixture and mode on GCC and Clang with 0 rows differing; every CTest label runs; the MLIR law builds under the top-level project |
| **BUILD-2** | `check_runtime.sh`'s sections as CTest entries, one group of sections per PR, each harness's argv registered once | per section: the CTest entry's outputs byte-identical to the shell section's on both compilers; the section deleted from the script only after that proof |
| **BUILD-3** | install and export: `install(TARGETS …)`, `BCIRConfig.cmake`, versioned headers; `find_package(BCIR)` from an installed tree | an out-of-tree consumer builds `test_runtime` against the installed package on both compilers |
| **BUILD-4** | the dependency index consumed by the Python harnesses (`native_bench`, the link-flag rules, the model gates) instead of per-harness probing | a harness and the configure cannot disagree about a dependency: the harness reads the index, and a test injects a missing row |
| **BUILD-5** | the MLIR rail under the top-level project in CI (`mlir-rail-validate` uses the `mlir` preset; `bcir-opt`'s lit suite as CTest) | the rail's own gates green from the top-level tree on LLVM 22 and 23 |
| **BUILD-6** | the BCIRfile grammar and the DAG model in the oracle, with its laws (claims, anti-cycle, footprint agreement, generation tags), `bcir-make --dry-run` | negative fixtures per law; the C rails' build expressed as a BCIRfile planning the same targets the manifest lists |
| **BUILD-7** | the smart task runner executing the gates: content-addressed reuse, the two-worker scheduler, task telemetry through the ring | a no-change second run executes zero tasks; a one-unit change executes exactly its dependents; the gates' verdicts byte-identical to their shell runs |
| **BUILD-8** | the C twin of `bcir-make`, its parity with the oracle, the artifact cache and the pinning manager; the migrated shell sections retired | twin/oracle parity over a generated corpus of BCIRfiles; `check_runtime.sh` reduced to what §7 keeps |

## 10. State (BUILD-1's own runs, 2026-10-04)

| Measure | Result |
|---|---|
| Manifest | 10 libraries (38 units), 5 tools, 32 harnesses, 19 fuzz targets, 2 seam libraries, 5 seam tests, 26 freestanding units at C11 and C23 |
| `tools/build/manifest.py --check` | ok; 22 injected violations each a finding |
| Build parity, GCC 13.3 (CMake `-std=c2x` vs the recipe's `c11` fallback) | 240 fixtures × 8 modes = 1,920 rows, 0 differ |
| Build parity, Clang 23.1.2 | 1,920 rows, 0 differ |
| `cmake --preset gcc` + build | 13 s wall at two workers |
| `fuzzer` preset (Clang 23) | 19 targets built; `ctest -L fuzz` 20/20 at 1,000 runs each |
| `mlir` preset (Clang 23, MLIR 23.1.2) | `bcir-opt` built under the top-level project; `ctest -L mlir` 4/4 |
| `ctest -L build` (gcc) | 2/2 |
| `clang` preset (system Clang 18.1.3) | 0 warnings; `ctest -L build` 2/2; `fuzzer` preset `ctest -L fuzz` 20/20 |
| The shell gates as CTest entries (gcc tree) | `cpp-handoff` and `c-memory-discipline` pass; the latter only after the gate's strict compile took its own compatibility warnings (it had passed under clang and failed under gcc: laws.md L12, found by the wrap) |

The next slice is BUILD-2; its first group is the sections that already have a harness built here
(`test_runtime`, `test_sha256`, `test_exec`, `test_encode`, `test_telemetry_frame`) because their
argv is a fixture path and their verdict is their exit status.
