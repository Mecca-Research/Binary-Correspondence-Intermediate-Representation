# LLVM Training Tools

This directory contains repository-maintenance scripts for the `llvm-training/`
corpus. They are intentionally small shell scripts so CI and local agents can run
the same checks without a build-system dependency.

## Scripts

| Script | Purpose | Required tools |
| --- | --- | --- |
| `verify-examples.sh` | Builds the known-good standalone `.ll` manifest from chapter-local `examples/` directories, assembles each file with `llvm-as`, and runs `opt -passes=verify`. It also checks the broken `.ll.txt` sentinel so intentionally invalid examples do not drift into the manifest. | `llvm-as`, `opt` |
| `smoke-lli.sh` | Runs only curated examples that have a safe no-argument entry point under `lli`. Most training snippets are library-style IR and should stay out of this list. | `lli` |
| `smoke-llc.sh` | Lowers curated examples with `llc` to catch target-codegen regressions without treating every IR snippet as a runnable program. It prints intentional exclusions from `smoke-llc-skip.txt` before running the curated allowlist. | `llc` |
| `verify-exercises.sh` | Assembles every checked-in `llvm-training/exercises/*.solution.ll` file and runs `opt -passes=verify` so reference answers stay valid standalone LLVM IR. | `llvm-as`, `opt` |
| `verify-invalid-fixtures.sh` | Discovers known invalid `.invalid.ll.txt` fixtures plus the broken-example sentinel and asserts each remains rejected by either `llvm-as` or `opt -passes=verify`. | `llvm-as`, `opt` |
| `verify-opt-diff.sh` | Runs curated `opt -S -passes=...` pipelines over chapter examples and diffs normalized output against checked-in golden `.after-<pass>.ll` fixtures. Set `UPDATE_OPT_DIFF=1` to refresh intentional pass-output changes. | `opt`, `diff` |
| `verify-opaque-pointers.sh` | Scans modern `*/examples/*.ll` fixtures for legacy typed-pointer syntax and allows migration-only or intentionally invalid `.ll.txt` fixtures to keep typed-pointer demonstrations explicit. | POSIX shell utilities, `awk` |
| `verify-manifest.sh` | Compares discovered standalone `*/examples/*.ll` files against the table in `llvm-training/examples/README.md` so new or removed examples do not silently drift from the manifest. | POSIX shell utilities |
| `verify-csv-schema.sh` | Validates checked-in `15-binary-analysis/examples/*.csv` fixtures for registered schema-family column counts, non-empty headers, consistent non-empty data rows, and at least one data row. The parser handles single-line CSV records with quoted commas. | POSIX shell utilities, `awk` |
| `verify-mlir-examples.sh` | Validates `*/examples/*.mlir` syntax with `mlir-opt --allow-unregistered-dialect` when `mlir-opt` is installed, and reports a clean skip otherwise. | Optional `mlir-opt` |
| `verify-bcir-mapping.sh` | Validates BCIR mapping fixtures under `bcir-mapping/examples/`: current source-like `.bcir.txt` claim fragments are checked for required markers and lowered `.ll` companions, and real `.bcir` sources are assembled with `bcir-as`, compared to sibling `.generated.ll` files, verified, and refreshed with `UPDATE_BCIR_MAPPING=1`. | POSIX shell utilities; optional `llvm-as`, `opt` for `.bcir.txt` lowered companions; `tools/bcir-as/bcir-as`, `llvm-as`, `opt` when `.bcir` fixtures exist |
| `smoke-bolt.sh` | Builds the BOLT layout demo fixture, records baseline symbol/disassembly text, and exits with a clean skip when `llvm-bolt` is not installed. A full profile-driven rewrite still requires host support for `perf2bolt`/`perf`; see the walkthrough. | Optional `llvm-bolt`; `clang` and `llvm-objdump` when BOLT is present |
| `demo-mem2reg.sh` | Demonstrates `mem2reg` on the checked-in diamond example, first verifying the fixture and then printing the promoted SSA form to stdout. | `opt` |
| `demo-o2.sh` | Runs `default<O2>` on the O2 pipeline inspection fixture, writes the optimized IR to a temporary file, prints it, and optionally smoke-checks the result with `llc` when available. | `opt`; optional `llc` |
| `demo-vectorize.sh` | Shows loop-vectorization remarks from `clang` on the C fixture, then forces a visible loop-vectorizer experiment over the checked-in IR and prints the transformed IR. | `opt`, `clang` |
| `demo-debug-pipeline.sh` | Captures `-debug-pass-manager` output for `default<O2>` into a temporary log, then prints the pass schedule for inspection. | `opt` |


## Advanced-content verification map

Use these script groups when advanced examples or reference paths change:

| Content family | Scripts to run |
| --- | --- |
| Advanced IR, intrinsics, attributes, poison/freeze, fast math | `verify-examples.sh`, `verify-exercises.sh`, `verify-invalid-fixtures.sh`, `verify-opaque-pointers.sh` |
| Optimization before/after examples and pass-pipeline lessons | `verify-opt-diff.sh`, `verify-examples.sh` |
| MLIR bridge examples | `verify-mlir-examples.sh`, then `verify-examples.sh` for lowered `.ll` companions |
| BCIR mapping/source-like fragments | `verify-bcir-mapping.sh`, `verify-examples.sh`, `verify-manifest.sh` |
| Binary-analysis CSV evidence | `verify-csv-schema.sh` |

## CMake batch targets

The training-only `llvm-training/CMakeLists.txt` exposes first-class custom targets for the
training corpus. Configure the project once, then run the targets from the build
directory with `cmake --build`:

```bash
cmake -S llvm-training -B build/llvm-training
cmake --build build/llvm-training --target llvm-training-verify-examples
cmake --build build/llvm-training --target llvm-training-smoke-llc
cmake --build build/llvm-training --target llvm-training-smoke-lli
cmake --build build/llvm-training --target llvm-training-verify-exercises
cmake --build build/llvm-training --target llvm-training-verify-invalid-fixtures
cmake --build build/llvm-training --target llvm-training-verify-adversarial-fixtures
cmake --build build/llvm-training --target llvm-training-verify-opt-diff
cmake --build build/llvm-training --target llvm-training-verify-opaque-pointers
cmake --build build/llvm-training --target llvm-training-lit
cmake --build build/llvm-training --target llvm-training-verify-manifest
cmake --build build/llvm-training --target llvm-training-verify-csv-schema
cmake --build build/llvm-training --target llvm-training-verify-mlir-examples
cmake --build build/llvm-training --target llvm-training-verify-bcir-mapping
cmake --build build/llvm-training --target llvm-training-check
```

The initial lit suite lives in `llvm-training/tests/` and delegates to the same
shell scripts for smoke coverage. Lit marks the opt-diff test with
`REQUIRES: opt`, so hosts without LLVM's optimizer report an unsupported test
instead of a failure.

CMake targets that declare hard external dependencies check for those tools
before running. If the host image does not provide the required tools, the target
prints the same kind of clean skip message used by CI and exits successfully.
Targets whose scripts contain their own fixture-aware skips, such as MLIR and
BCIR mapping validation, always invoke the script so it can decide whether the
current repository state requires the optional toolchain. Running most shell
scripts directly remains fail-closed: missing required tools produce an error so
local maintainers notice incomplete toolchains.

## Golden opt-diff and opaque-pointer checks

`verify-opt-diff.sh` currently protects selected Chapter 7 optimization
examples. Each registered input has a sibling golden file with a pass suffix,
for example `opt-diff-instcombine.after-instcombine.ll` or
`opt-diff-loop-rotate.after-loop-rotate.ll`. The comparison normalizes volatile
`ModuleID` banners and synthesized datalayout lines, then uses `diff -u` for
human-readable drift reports. Refresh expected changes with:

```bash
UPDATE_OPT_DIFF=1 ./llvm-training/tools/verify-opt-diff.sh
```

`verify-opaque-pointers.sh` treats checked-in `.ll` files as modern opaque
pointer examples. Typed-pointer migration material should remain in explicit
`.ll.txt` fixtures such as `02-types/examples/typed-pointer-before.ll.txt`,
which makes legacy syntax discoverable without letting it back into runnable IR.

## Skip-list rationale

`verify-examples.sh` is a positive manifest rather than a global `*.ll*` sweep:

- files ending in `.ll.txt` are reserved for intentionally invalid parser,
  verifier, or migration examples;
- files with `invalid` in the filename are also reserved for expected-failure
  examples;
- only files below an `examples/` directory are considered standalone training
  modules.

CI runs `smoke-llc.sh` immediately after the standalone example verifier when
`llc` is available, guaranteeing that curated runnable-through-codegen training
IR continues to lower successfully. Minimal images without `llc` report an
explicit skip instead of failing the whole workflow.

### `smoke-llc.sh` policy

`smoke-llc.sh` intentionally uses a positive allowlist for examples that should
emit portable assembly on a default `llc` invocation. Keep that allowlist in the
script so reviewers can see exactly which examples are required to lower.

Intentional exclusions live in `smoke-llc-skip.txt`; do not add shell arrays or
inline skip rules to the script. Each non-comment line in the policy file has
this tab-separated format:

```text
<path-or-glob-under-llvm-training><TAB><reason>
```

Use paths relative to `llvm-training/`, and use globs only when an entire family
of teaching fixtures has the same rationale. Reasons should be short human
phrases because the smoke script prints them directly as:

```text
[skip] llvm-training/<path-or-glob> ... <reason>
```

`smoke-bolt.sh` is also a positive, guarded check: it validates the documented
BOLT fixture only when the host has `llvm-bolt`, and otherwise reports an
intentional skip so minimal CI images are not forced to install BOLT packages.

The sentinel `../examples/broken-example.ll.txt` is deliberately malformed.
`verify-examples.sh` still checks that the sentinel stays out of the known-good
manifest. `verify-invalid-fixtures.sh` performs the broader expected-failure
sweep: every `.invalid.ll.txt` fixture, plus the sentinel, must remain rejected
by `llvm-as` or by the verifier pass if assembly succeeds. A fixture that is
intentionally semantic-only, such as poison-prone IR accepted by LLVM
verification, must include `; verify-invalid-fixtures: semantic-only` so the
script can distinguish it from an accidentally valid parser/verifier fixture.

`verify-manifest.sh` is intentionally separate from IR verification. It compares
the checked-in Markdown table in `../examples/README.md` to the discovered set of
standalone `*/examples/*.ll` files using the same inclusion policy as
`verify-examples.sh`.

`verify-csv-schema.sh` is intentionally lightweight and fixture-scoped. It uses
checked-in CSV files under `../15-binary-analysis/examples/`, maps each known
schema family by filename to its expected column count, supports quoted commas in
single-line CSV records, and fails if headers or data rows disappear.

`verify-mlir-examples.sh` is optional-toolchain-friendly. It exits successfully
with an explicit skip when `mlir-opt` is unavailable, but when MLIR is installed
it parses all chapter-local `.mlir` examples with unregistered dialects allowed
so dialect sketches still receive syntax coverage.

`verify-bcir-mapping.sh` validates both source-like `.bcir.txt` claim fragments
and real `.bcir` assembler fixtures under `bcir-mapping/examples/`. The
`.bcir.txt` fragments are not assembler inputs, so the checker applies
fixture-format checks instead: files must stay non-empty, retain their required
BCIR markers or operation keywords, and keep their expected lowered `.ll`
companions. When `llvm-as` and `opt` are available, those lowered companions are
also assembled and verified so the checked text fragments cannot drift away from
valid LLVM IR examples.

If a review adds real `.bcir` sources under `bcir-mapping/examples/`, each
source must have a sibling `<name>.generated.ll` expected output unless the
maintainer is intentionally refreshing outputs with:

```bash
UPDATE_BCIR_MAPPING=1 ./llvm-training/tools/verify-bcir-mapping.sh
```

## Adding a script

When adding a new script:

1. keep it executable and runnable from the repository root;
2. document required external tools and any intentional skips in this README;
3. print the command being demonstrated before executing it;
4. write demo output either to stdout or to a clearly named file under `${TMPDIR:-/tmp}`;
5. make it fail closed when a required fixture disappears;
6. wire it into `.github/workflows/ci.yml` when it guards repository health.

## Advanced chapter integration gates

`verify-manifest.sh` now checks two synchronized inventories: every standalone
`*/examples/*.ll` module and every file below an `examples/` directory. The
second inventory forces MLIR, MIR-shaped text, Markdown/C++ sketches,
source-like BCIR files, invalid fixtures, target-only artifacts, JIT-only
artifacts, and analysis-only notes to receive an explicit classification in
`../examples/README.md`.

`run-if-tools.sh` is the optional-tool wrapper used by the training-only CMake
project and selected CI steps. It recognizes both unversioned LLVM tools and
version-suffixed binaries such as `llvm-as-20`; when a declared tool family is
absent it prints a clean skip instead of running a verifier with incomplete
prerequisites.

Configure all training targets independently from the BCIR IR build:

```bash
cmake -S llvm-training -B build/llvm-training
cmake --build build/llvm-training --target llvm-training-check
```

The aggregate target includes deterministic repository gates and optional LLVM,
MLIR, and lit-backed gates. Optional tools skip cleanly; a present toolchain that
rejects a checked fixture still fails the target.
