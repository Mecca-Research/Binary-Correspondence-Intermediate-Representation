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
CMakePresets.json              default / gcc / clang / clang23 / asan / ubsan / tsan / fuzzer / mlir
cmake/BCIRCompilerFlags.cmake  C23 (c2x where needed), C++17 -Wpedantic, -Wall -Wextra [-Werror],
                               -O2 without NDEBUG, BCIR_SANITIZE, the freestanding object check
cmake/BCIRDependencies.cmake   Python, Threads, MLIR/LLVM, TSan, FFTW3F/LAPACKE/GSL/SLEEF/libcerf -> bcir-deps.json
cmake/BCIRManifest.cmake       the manifest reader, closures, link-name and option mapping
cmake/BCIRTests.cmake          CTest registration (labels, PROCESSORS 2, the gates as entries)
cmake/BCIRInstall.cmake        install and export (BUILD-3): the manifest's libraries and tools, the
                               public headers and bcir_version.h, the BCIR package (BCIRConfig.cmake.in)
tools/c/sections/*.sh          the gate sections that migrated (BUILD-2): one script each, run by the
                               gate over its own binaries and by CTest over the manifest's harnesses,
                               variants and tools (a script that compiles takes the compiler as CC;
                               a compiler-only one takes no binary and judges the oracle's emit)
tools/build/mutate.py          the one applier of the manifest's fault injections (gate and CMake)
tools/build/emit_kernel.py     the one writer of the manifest's kernels (gate, CMake and the section-
                               parity recipe), with a depfile of the oracle modules each one imported
tools/build/sanitizer.py       the one predicate for "this sanitizer builds and runs here" (gate,
                               configure, and through the configure the section-parity gate)
tools/build/install_consumer.py the install gate: installs a configured build into a scratch prefix and
                               builds tools/build/consumer against it, knowing nothing of the tree
runtime/manifest.json          libraries / tools / harnesses / fuzzers / seam_libraries / seam_tests,
                               plus sections (BUILD-2), variants (the harness rebuilds they judge),
                               kernels (programs the Python oracle emits, their drivers in runtime/c/kernels)
                               and delegated (the gates check_runtime.sh calls, one CTest entry each)
runtime/c/CMakeLists.txt       targets from the manifest; nothing listed by hand
runtime/cpp/CMakeLists.txt     the seam's libraries and tests over the C libraries built as C
mlir/CMakeLists.txt            unchanged; add_subdirectory()'d when BCIR_BUILD_MLIR is on
```

| Manifest kind | Units | Rule |
|---|---|---|
| `libraries` | 10 static libraries over the 38 `bcir_*.c` units that are not tool mains: `bcir_base` (runtime, SHA-256), `bcir_streampack`, `bcir_plane` (control plane, rings, hand-off, planner), `bcir_artifact`, `bcir_asn1`, `bcir_cfront` (the twin, the preprocessor, the verifier, diagnostics), `bcir_quarantine`, `bcir_models`, `bcir_channel`, `bcir_driver` | class-homogeneous; `libraries` is the dependency edge; `link` names `m`/`pthread`; `options` carries the gates' per-unit flags (`gcc:`/`clang:` scoped) |
| `tools` | `bcir-cc`, `bcir_asn1_bench`, `bcir_microbench`, `bcir_ai_microbench`, `bcir-llama` | one `hosted_tool` main each |
| `harnesses` | the 35 `test_*.c` mains | built in `ALL` (the link proof); the sections run them over the CMake-built binaries as the gate runs its own |
| `variants` | 26 harness rebuilds the migrated sections judge: `-O0`/`-O3` of `test_ring`, `test_handoff`, `test_kplan` and of the six ASN.1 twins' harnesses (`test_per`, `test_xer`, `test_jer`, `test_emit`, `test_per_plan`, `test_oer`); one fault-injected mutant each of `test_control_plane`, `test_ring`, `test_handoff`, `test_kplan`; the C11 builds of `test_x86_interrupt` and `test_q8_tables`; and the ring's two ThreadSanitizer builds (`test_ring_tsan`, and `test_ring_plain`, whose relaxed atomic stores are made plain) | each compiles its harness's whole library closure with the variant's options appended (so `-O0` reaches the twin, not only the harness main), in its `standard` when that is not C23, with its `sanitizer` on every object and the link; a mutant's source is generated at build time by `tools/build/mutate.py` -- the applier the gate uses, from the manifest's one spelling of the fault, applied only when its anchor occurs exactly once -- and is built without the warning policy, as the gate builds it. A sanitizer variant is built only where `tools/build/sanitizer.py` finds the runtime working (the dependency index's `TSAN` row) and no conflicting `BCIR_SANITIZE` is set; otherwise its section is reported as not registered, with the reason, and `BCIR_REQUIRE_TSAN=ON` turns that into a configure failure |
| `kernels` | 7 programs the Python oracle emits (`bcir.lower.c_kernel`): `kernel_ols`, `kernel_pca`, `kernel_layernorm`, `kernel_lstm`, `kernel_svm`, `kernel_tree`, `kernel_kmeans` | each is one unit, the emitter's text with its driver (`runtime/c/kernels/<main>`) appended, written by `tools/build/emit_kernel.py` for the gate and the build alike; CMake writes it at build time with a depfile naming every oracle module the emitter imported, so an edit to one of them re-emits it and no other edit does; built in C11 under the warning policy; a host without Python records each as not built, with the reason |
| `delegated` | 7 gates `tools/c/check_runtime.sh` calls, each building and judging binaries of its own: `c-memory-discipline`, `c-streampack-semantic`, `c-target-abi`, `cpp-jer-index`, `cpp-jer-simd`, `cpp-handoff`, `cpp-sycl` | each is a CTest entry of its own under its labels; the gate skips them when the CTest `c-runtime` entry sets `BCIR_SKIP_DELEGATED_GATES=1`, so `ctest` runs each once; M16 holds the gate's guarded calls, the manifest and `cmake/BCIRTests.cmake` in step, and every other script the gate calls is a section or the cfront sanitizer under its own switch |
| `fuzzers` | the 19 `fuzz_*.c` targets | Clang only; each compiles its library closure into itself with `-fsanitize=fuzzer,address,undefined` so the code under test is instrumented; `max_len` equals the gate's `-max_len` |
| `seam_libraries` / `seam_tests` | `bcir_seam`, `bcir_jer_index`; `test_orchestrator`, `test_handoff_cpp`, `test_artifact_bundle_cpp`, `test_jer_index`, `test_jer_simd` | C++17 over the C libraries; the seam's unit lists equal `check_handoff.sh`'s and `handoff_fixtures.py`'s |
| `freestanding_checks` | the 26 `freestanding_core` units | OBJECT libraries at C11 and C23, `-ffreestanding -nostdlib -O0`; a superset of the 14 the gate compiles |

The build tree is `build/cmake-<preset>/`: libraries and tools under `runtime/c/`, harnesses
under `harnesses/`, fuzz targets under `fuzzers/`, the dependency index at `bcir-deps.json`
(schema `bcir-deps.v1`: compilers, system, one `{name, found, detail}` row per dependency).

## 4. CTest mapping

| Label | Entries (BUILD-1) | Owner on CI |
|---|---|---|
| `build` | `build-manifest` (the checker), `build-deps-index` (the dependency index), `build-parity` (the `bcir-cc` parity gate), `build-section-parity` (every migrated section byte-identical over the gate-built and the CMake-built harnesses), `build-install` (the install gate: an out-of-tree consumer of the installed package) | `cmake-build` (gcc + clang) |
| `section` + `c` | `c-section-<name>` for every `sections` entry of the manifest: the gate section's own script over the harnesses built here, or, for a compiler-only section, over none, under the configured compiler | `cmake-build` |
| `python` | `python-quick` (the oracle's quick tier, `-j 2`) | the oracle jobs |
| `shell` + `c` / `cpp` / `fuzz` | the shell gates as entries under the configured compilers: `c-runtime`, which skips the gates it delegates to, and one entry per manifest `delegated` gate (`c-memory-discipline`, `c-streampack-semantic`, `c-target-abi`, `cpp-jer-index`, `cpp-jer-simd`, `cpp-handoff`, `cpp-sycl`); on Clang `cfront-sanitize`, `fuzz-streampack` | the C-rails jobs |
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
| **BUILD-2** (landed: groups 1 to 3g) | `check_runtime.sh`'s sections as CTest entries, one group of sections per PR: each section's text moves into `tools/c/sections/<name>.sh`, the gate compiles its harnesses and calls the script, the manifest's `sections` registers script + harnesses, CMake runs it as `c-section-<name>`. Group 1: `runtime`, `artifact_bundle`, `executor`, `encoder`, `execution_plan`, `telemetry_frame`. Group 2: `control_plane`, `ring`, `handoff`, `kplan`, whose optimisation and fault-injection binaries are manifest `variants`. Group 3a: `x86_interrupt`, `q8_tables` (C11 variants) and `ring_tsan`, the ring's ThreadSanitizer leg (sanitizer variants). Group 3b: `cfront`, `cfront_abi`, `cfront_loop`, `channel`, the sections that drive the C twin and its Python parity. Group 3c: `bcir_cc`, `emitlink`, `recover`, `writeguard`, the first `bcir-cc` sections, over the manifest tool and a compiler named as CC. Group 3d: `diag`, `fallback`, `r21policy`, `project`, `link`, `effects`, `linkflags`, the `bcir-cc` contracts against the Python oracle, and `cexpr`, the constant folds, once both rails' emits spelled a switch's labels in C11 (CF-CASELABEL). Group 3e: the five link-flag rules (`linkflags_fftw`, `_lapack`, `_gsl`, `_sleef`, `_cerf`) over one harness, `test_link_flag_rules`, and the seven E-series sections (`ols`, `pca`, `layernorm`, `lstm`, `svm`, `tree`, `kmeans`) over the manifest's `kernels`. Group 3f: the 52 sections of the "emit == Clang" family, `scale` through `stmtexpr`, each driving `bcir-cc` over a unit and, where it compiles what the tool emits beside the source and a driver, taking CC. Group 3g: the ASN.1 twins (`per`, `xer`, `jer`, `asn1_emit`, `per_plan`, `oer` over their `-O0`/`-O3` variants; `asn1bench`; `asn1fast`), the two probe programs as harnesses (`atomicring`, `extentassert`), the oracle's asm emits as compiler-only sections (`inlineasm`, `portio`, `barrier`), and the gates `check_runtime.sh` calls as the manifest's `delegated` entries, one CTest entry each | per section: `build-section-parity` holds the script's stdout, stderr and status byte-identical over the gate-built and the CMake-built harnesses on both compilers, and requires a PASS line; the gate's compile lines go only after that proof has run on CI for the group |
| **BUILD-3** (landed) | install and export: `install(TARGETS …)`, `BCIRConfig.cmake`, versioned headers; `find_package(BCIR)` from an installed tree | an out-of-tree consumer builds `test_runtime` against the installed package on both compilers |
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
| BUILD-2 group 1: `ctest -L section` (gcc) | 6/6, 1.5 s; `build-section-parity` 6 sections identical under both builds, 11 PASS lines, 12 s |
| BUILD-2 group 2: `ctest -L section` (gcc) | 10/10, 16 s (the planner's corpus 14 s); `build-section-parity` 10 sections identical, the ring's one declared varying value masked; each mutant byte-identical to the gate's former `sed` output |
| BUILD-2 group 3a: `ctest -L section` (gcc, `BCIR_REQUIRE_TSAN=ON`) | 13/13, 15 s (`ring_tsan` 5.4 s); `build-section-parity` 13 sections identical; the plain-store mutant byte-identical to the gate's former `sed` output; an AddressSanitizer preset and a compiler without TSan each leave `ring_tsan` unregistered with the reason, and fail the configure under `BCIR_REQUIRE_TSAN=ON` |
| BUILD-2 group 3b: `ctest -L section` (gcc, `BCIR_REQUIRE_TSAN=ON`) | 17/17, 16 s; `build-section-parity` 17 sections identical under both builds, the four new ones with 122, 6, 26 and 1 PASS lines; each new section fails on a harness that corrupts one of its answers |
| BUILD-2 group 3c: `ctest -L section` (gcc, `BCIR_REQUIRE_TSAN=ON`) | 21/21, 23 s; `build-section-parity` 21 sections identical, the four `bcir-cc` sections compiling what the tool emits with the tree's compiler on both runs; 195 s with each recipe built once (249 s building `bcir-cc` per section); each new section fails on a `bcir-cc` that corrupts one of its answers |
| BUILD-2 group 3d: `build-section-parity` over the seven new sections (gcc, then clang) | 7 sections identical under both builds with each compiler (22, 6, 10, 5, 6, 2 and 5 PASS lines); each fails on a binary that corrupts one of its answers |
| BUILD-2 group 3f: the 52 new sections (gcc, then clang) | `build-section-parity` holds every one identical under both builds with each compiler: no section prints a value its binaries do not decide, and no diagnostic names a temporary path |
| BUILD-2 group 3g: the 13 new sections and the 7 delegated gates (gcc, clang 18, clang 23) | `ctest -L section` 106/106 on each tree; `build-section-parity` holds all 106 identical, the three compiler-only sections over two runs; the 7 delegated entries pass on each tree, and the `c-runtime` entry skips each by name; the gate prints 399 PASS lines on clang 18 and 388 on clang 23, 0 FAIL, and its 1,633 lines are 879 |
| BUILD-3: the install gate (gcc, clang 18, clang 23) | `build-install` passes on each tree: 12 libraries, 5 tools, 52 public headers and the package files installed and nothing else; the consumer builds every header alone and links `test_runtime` against `BCIR::*` alone, and the `runtime` section prints the same over both builds. A public header left out of the install, and a version file that accepts every request, each fail it |
| BUILD-2 group 3e: the twelve new sections (gcc) | `ctest -L section` runs them 12/12; `build-section-parity` holds them identical, the kernels' recipes written by the same writer; a no-op build re-emits no kernel, an edit to an oracle module the emitters import re-emits all seven, an edit to one they do not import re-emits none, an edit to a driver re-emits its kernel alone |
| BUILD-2 group 3d's `cexpr` (gcc, clang 18, clang 23) | the emit builds under `-std=c11 -pedantic-errors` with each compiler, and the section prints its PASS line; the `bcir-cc` before CF-CASELABEL fails the emitted build under all three, GCC included, which had accepted the C23 form silently |
| `clang` preset (system Clang 18.1.3) | 0 warnings; `ctest -L build` 2/2; `fuzzer` preset `ctest -L fuzz` 20/20 |
| The shell gates as CTest entries (gcc tree) | `cpp-handoff` and `c-memory-discipline` pass; the latter only after the gate's strict compile took its own compatibility warnings (it had passed under clang and failed under gcc: laws.md L12, found by the wrap) |

BUILD-2 continues group by group. Group 1 (landed) took the six sections whose argv is a fixture
path and whose verdict is a PASS line. Group 2 (landed) took the four sections that compile second
binaries of their own (`control_plane`'s mutant; `ring`'s, `handoff`'s and `kplan`'s `-O0`/`-O3`
sweeps and mutants): their scripts take those binaries as further arguments, the manifest's
`variants` let CMake build them, and the gate generates its mutants with the same applier. Group 3a
(landed) took the two-standard sections (`x86_interrupt`, `q8_tables`, over C11 variants) and the
ring's ThreadSanitizer leg (`ring_tsan`, over sanitizer variants: the gate builds the two TSan
binaries -- the plain-store mutant through the applier -- and the script runs their stress). Whether
TSan works on a host is one predicate, `tools/build/sanitizer.py`, asked by the gate and the
configure; the configure records what it could not build, and the section-parity gate reports those
sections by name instead of comparing them. Group 3b (landed) took the four sections that drive
the C twin and its Python parity over harnesses the manifest already builds: `cfront` (the twin's
summary and structural digest against the oracle's, fixture by fixture), `cfront_abi` (the
`--target` data-model matrix), `cfront_loop` (the compile-to-execute loop, no Python) and
`channel` (the routing decision). They need no variants; their scripts run the oracle beside the
harness, as `runtime` does. Group 3c (landed) took the first four `bcir-cc` sections: the
driver's own (`bcir_cc`), then `emitlink`, `recover` and `writeguard`, which compile what the tool
emits against the bounds-quarantine runtime. Two things were new for a section:

- **A section binary may be a manifest tool.** The CTest entry passes the CMake-built `bcir-cc`,
  and the section-parity gate builds the gate's recipe from the tool's closure.
- **A script that compiles takes the compiler as CC.** The gate passes its own, the CTest entry
  the configured C compiler, and the section-parity gate the same one to both runs.

Because a section script now compiles runtime units, M9 reads the section scripts as it reads
the gates. `#atomicring` and `#extentassert`, between them in the gate, stayed there until group
3g: they compiled and ran the runtime header's contract with no binary to take. Group 3d
(landed) took seven `bcir-cc` contracts that compare the twin with the Python oracle. They are the
diagnostic renderer (`diag`, over the `test_diag` harness), the fallback decision, the R21 policy,
project mode, linking (`link`, which compiles what the tool emits and so takes CC), the effect
and escape reports, and the derived link flags. `#cexpr` followed once the emitter was fixed.
Both rails' `--emit-c` had put a declaration right after a `case` label, a C23-only form that GCC
accepted silently and Clang warned about. The warning named the section's temp file, which
differs between two runs, so the section-parity gate would have reported the section as
unstable. Each label now ends in a null statement (CF-CASELABEL), and the section builds the
emit with `-pedantic-errors`, so GCC refuses the old form too. Group 3e (landed) took the
sections whose programs the gate wrote out itself. The five link-flag rules had been five probe
files written from heredocs; they are now one harness, `test_link_flag_rules`, whose argument picks
a rule's table, built once by the gate and by CMake. The seven E-series sections compile C the
Python oracle emits, a driver `main` appended. A program that exists only once the oracle has run
needed a manifest kind of its own: `kernels` names each one's emitter, arguments, driver and
libraries, and one writer, `tools/build/emit_kernel.py`, writes the unit for the gate, for CMake
(at build time, with a depfile of the oracle modules the emitter imported) and for the
section-parity recipe. M15 holds the entries to the oracle and the drivers to the entries, and the
writer asks the same predicate. Group 3f (landed) took the "emit == Clang" family: 52 sections
that run `bcir-cc` over a unit and, all but two of them, compile what it emits beside the source
and a driver, then run both. They needed nothing new: each is group 3c's shape, `bcir-cc` and,
where it compiles, CC. Group 3g (landed) took the rest, in four shapes:

- **The ASN.1 twins.** `per`, `xer`, `jer`, `asn1_emit`, `per_plan` and `oer` run their twin at
  `-O0` and at `-O3` and require the same answers, so each takes two manifest variants; `asn1bench`
  takes the `bcir_asn1_bench` tool and `asn1fast` its harness. The gate's sweeps used to SKIP when
  an `-O0` or `-O3` build failed (one said nothing at all), while the CMake build of the same
  variants fails: they fail in the gate too now (laws.md L12, L21).
- **The probe programs as harnesses.** `#atomicring`'s heredoc is `runtime/c/test_oob_counter.c`.
  `#extentassert` is `runtime/c/test_extent_assert.c`, a correct extent the build compiles; its
  section compiles the tampered extent under CC and requires the assertion's own diagnostic. The
  gate had compiled the tampered unit with `-std=c23` alone, which GCC 13 refuses as an option, so
  under GCC the check passed without reaching the assertion.
- **Compiler-only sections.** `#inlineasm`, `#portio` and `#barrier` judge what the Python oracle
  emits for a C source, compiled, assembled or run under CC and the gcc and clang on `PATH`: there
  is no binary to take. A section may name none only when it says `compiler_only` and its script
  runs `${CC}` (M13); the section-parity gate runs it twice and holds the two runs identical.
- **The delegated gates.** The seven scripts `check_runtime.sh` calls that build and judge
  binaries of their own are the manifest's `delegated` entries. CMake registers each as a CTest
  entry, and the gate skips them under `BCIR_SKIP_DELEGATED_GATES=1`, which the `c-runtime` entry
  sets, so `ctest` runs each once; anyone running the gate directly still gets every one. M16
  holds the guarded calls, the manifest and the registration in step, and refuses a script the
  gate calls that is neither a section nor a delegated gate (the cfront sanitizer keeps its own
  switch).

BUILD-2 is complete. `check_runtime.sh` now holds its compile lines (the gate's own recipes, which
the section-parity gate holds to the CMake build), the freestanding compiles (whose CMake twin is
`freestanding_checks`, M8), and the calls: every check it makes runs as a CTest entry too.
BUILD-8 retires the compile lines once BCIR Make builds what the sections take.
