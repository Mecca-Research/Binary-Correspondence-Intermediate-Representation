# LLVM Training Tools

This directory contains repository-maintenance scripts for the `training/llvm/`
corpus. They are intentionally small shell scripts so CI and local agents can run
the same checks without a build-system dependency.

## What is here and what is shared

Grading and dataset export are **not** LLVM-specific, and since Phase 0.7 they
are not implemented here. They live in [`../../tools/`](../../tools):

| Shared module | Owns |
| --- | --- |
| [`grading.py`](../../tools/grading.py) | answer confinement, attempt-tree policy, points, structural and rubric checks, the report shape, the CLI |
| [`dataset_export.py`](../../tools/dataset_export.py) | split manifests, checksums, prompt redaction, artifact roles, deterministic JSONL |
| [`subject_profile.py`](../../tools/subject_profile.py) | the vocabulary a subject uses to describe itself to those two |
| [`safe_process.py`](../../tools/safe_process.py) | bounded, environment-controlled subprocess execution |

What remains LLVM's is [`llvm_profile.py`](llvm_profile.py): five answer kinds,
the `llvm-as` / `opt` / `mlir-opt` checks, the version-suffix search, the
solution-path redaction, and the `lli` harness. `grade-exercises.py` and
`export-exercise-dataset.py` are the two-import entry points that hand that
profile to a shared kernel.

The boundary is gated, not just described:
[`verify_grading.py`](../../tools/verify_grading.py) fails if a subject
identifier appears in the shared rail's code, if any of those predicates gains a
second definition anywhere under `training/`, or if a synthetic non-LLVM subject
cannot be graded end to end.

## Scripts

| Script | Purpose | Required tools |
| --- | --- | --- |
| `generate-exercise-variants.py` | Produces a small fixed-seed set of typed prompt/reference pairs, grades every reference through the attempt-grader engine, rejects unsafe/trivial/duplicate IR, and records split lineage plus artifact hashes. | `llvm-as`, `opt`, `lli` |
| `grade-exercises.py` | LLVM entry point to the shared grader: grades stable-ID attempt directories or a single answer with deterministic partial credit; `--self-test` grades registered references. JSON output records every pass/fail/skip, toolchain version, raw score, executed-check score, and confidence. | Python 3; optional LLVM/MLIR tools declared per exercise |
| `llvm_profile.py` | The subject profile the shared rails consume: answer kinds and extensions, the `llvm-as`/`opt`/`mlir-opt` checks, tool discovery, solution redaction, and the `lli` harness. Not a CLI. | Python 3 |
| `grade-exercises.sh` | Repository-root shell entry point for `grade-exercises.py`, preserving arguments and exit status. | Bash, Python 3 |
| `export-exercise-dataset.py` | LLVM entry point to the shared exporter: deterministic JSON Lines for `train`, `validation`, `test`, or `all`; solution content is omitted by default and must remain omitted for model-visible held-out evaluation. Supplies the LLVM/MLIR version assumptions each record carries. | Python 3 |
| `verify-dataset-export.py` | Validates split/leakage assignments and schema, regenerates solution-free and trusted solution-bearing exports, checks hashes and paths, and proves byte-for-byte determinism. | Python 3 |
| `run-eval.py` | Prepares solution-free prompt/context bundles, invokes a provider-neutral local or fixture adapter, grades attempts, and emits aggregate/reproducibility reports. | Python 3; exercise-declared grading tools |
| `verify-examples.sh` | Builds the known-good standalone `.ll` manifest from chapter-local `examples/` directories, assembles each file with `llvm-as`, and runs `opt -passes=verify`. It also checks the broken `.ll.txt` sentinel so intentionally invalid examples do not drift into the manifest. | `llvm-as`, `opt` |
| `smoke-lli.sh` | Runs only curated examples that have a safe no-argument entry point under `lli`. Most training snippets are library-style IR and should stay out of this list. | `lli` |
| `smoke-llc.sh` | Lowers curated examples with `llc` to catch target-codegen regressions without treating every IR snippet as a runnable program. It prints intentional exclusions from `smoke-llc-skip.txt` before running the curated allowlist. | `llc` |
| `verify-exercises.sh` | Assembles every checked-in `training/llvm/exercises/*.solution.ll` file and runs `opt -passes=verify` so reference answers stay valid standalone LLVM IR. | `llvm-as`, `opt` |
| `verify-invalid-fixtures.sh` | Discovers known invalid `.invalid.ll.txt` fixtures plus the broken-example sentinel and asserts each remains rejected by either `llvm-as` or `opt -passes=verify`. | `llvm-as`, `opt` |
| `verify-opt-diff.sh` | Runs curated `opt -S -passes=...` pipelines over chapter examples and diffs normalized output against checked-in golden `.after-<pass>.ll` fixtures. Set `UPDATE_OPT_DIFF=1` to refresh intentional pass-output changes. | `opt`, `diff` |
| `verify-opaque-pointers.sh` | Scans modern `*/examples/*.ll` fixtures for legacy typed-pointer syntax and allows migration-only or intentionally invalid `.ll.txt` fixtures to keep typed-pointer demonstrations explicit. | POSIX shell utilities, `awk` |
| `verify-manifest.sh` | Compares discovered standalone `*/examples/*.ll` files against the table in `training/llvm/examples/README.md` so new or removed examples do not silently drift from the manifest. | POSIX shell utilities |
| `verify-csv-schema.sh` | Validates checked-in `15-binary-analysis/examples/*.csv` fixtures for registered schema-family column counts, non-empty headers, consistent non-empty data rows, and at least one data row. The parser handles single-line CSV records with quoted commas. | POSIX shell utilities, `awk` |
| `generate-binary-analysis-fixtures.py` | Builds manifest-declared x86-64 assembly fixtures, normalizes ELF symbols, instruction classes, basic blocks, direct call edges, and section summaries, writes provenance, or fails on deterministic drift with `--check`. Missing Clang skips unless `--require-tools` is set. | Python 3, Clang for generation |
| `verify-binary-analysis-evidence.py` | Checks that every Chapter 15 CSV has exactly one manifest classification and validates deterministic fixture, CSV, target, toolchain-provenance, and static-evidence-family contracts. | Python 3 |
| `verify-mlir-examples.sh` | Grades the MLIR registry through Tiers 0–4: explicit unregistered syntax sketches, registered verification, declared pipelines, LLVM translation, assembly, and verifier checks. | Optional `mlir-opt`, `mlir-translate`, `llvm-as`, `opt` (required with `--require-tools`) |
| `verify-bcir-mapping.sh` | Validates BCIR mapping fixtures under `bcir-mapping/examples/`: current source-like `.bcir.txt` claim fragments are checked for required markers and lowered `.ll` companions, and real `.bcir` sources are assembled with `bcir-as`, compared to sibling `.generated.ll` files, verified, and refreshed with `UPDATE_BCIR_MAPPING=1`. | POSIX shell utilities; optional `llvm-as`, `opt` for `.bcir.txt` lowered companions; `tools/bcir-as/bcir-as`, `llvm-as`, `opt` when `.bcir` fixtures exist |
| `smoke-bolt.sh` | Builds the BOLT layout demo fixture, records baseline symbol/disassembly text, and exits with a clean skip when `llvm-bolt` is not installed. A full profile-driven rewrite still requires host support for `perf2bolt`/`perf`; see the walkthrough. | Optional `llvm-bolt`; `clang` and `llvm-objdump` when BOLT is present |
| `demo-mem2reg.sh` | Demonstrates `mem2reg` on the checked-in diamond example, first verifying the fixture and then printing the promoted SSA form to stdout. | `opt` |
| `demo-o2.sh` | Runs `default<O2>` on the O2 pipeline inspection fixture, writes the optimized IR to a temporary file, prints it, and optionally smoke-checks the result with `llc` when available. | `opt`; optional `llc` |
| `demo-vectorize.sh` | Shows loop-vectorization remarks from `clang` on the C fixture, then forces a visible loop-vectorizer experiment over the checked-in IR and prints the transformed IR. | `opt`, `clang` |
| `demo-debug-pipeline.sh` | Captures `-debug-pass-manager` output for `default<O2>` into a temporary log, then prints the pass schedule for inspection. | `opt` |
| `build-pass-plugin.sh` | Builds the out-of-tree New PM plugin in `17-new-pass-manager/examples/pass-plugin/` and runs nine assertions through `opt`, including the negative cases (the checker fires on a violating module, `<strict>` exits nonzero, and a stock `loop-unroll-full` breaks the 1:1 contract). Skips cleanly, printing the reason, when the LLVM development headers are absent or when `opt` and `llvm-config` report different LLVM majors. Honours `PASS_PLUGIN_BUILD_DIR` and `PASS_PLUGIN_JOBS`. | `cmake`, a C++17 compiler, LLVM **development** headers, `opt`, `llvm-config` |
| `verify-langref-delta.py` It also holds the WHOLE LLVM 23 surface to account, not just the items that moved, and each of the three surfaces gets the answer its size deserves. **Instructions** get no disposition table, because they do not merit one: 67 opcodes is the language, so the check simply requires every one to be written somewhere in the corpus as code, and it is at 67 of 67. **Attributes** get one disposition each in `reference/langref-attribute-dispositions.json`, recording the clang recipe actually run to produce it (or that none could). **Intrinsics** get classes, in `reference/langref-intrinsic-dispositions.json`: 524 is too many to answer one at a time, and most do not merit an individual answer — `llvm.vp.*` is ninety operations that are the same operation ninety times. The assignment is per intrinsic rather than per `llvm.<family>.*` prefix, because LLVM's prefixes are not reliably semantic (`llvm.get.*` spans the FP environment, the stack and vector shape), and one reason covering three unrelated things is how a disposition becomes a rubber stamp. | Snapshots each tracked LLVM major's language surface — instructions, attributes and target-independent intrinsics — from that toolchain's own generated definitions (`Instruction.def`, `Attributes.td`, `IntrinsicEnums.inc`), then requires every item that moved between them to be either taught by a chapter or declared out of scope with a reason. Also regenerates the counts chapters 23 and 09 display, and checks the graduation table names intrinsics that really exist. `--emit-surface N` rewrites a snapshot; `--require-surface N` fails rather than skips when LLVM N's headers are absent (each CI job passes it for the major it installs); `--update` rewrites the generated blocks. | none to run; `llvm-N-dev` headers for the drift check |
| `verify-mlir-infrastructure.py` | Runs chapter 24's MLIR framework claims against a real `mlir-opt`: the custom/generic syntax round-trip, that an unregistered dialect needs `--allow-unregistered-dialect`, that a function-scoped pass named at module level fails while a non-existent anchor is silently accepted, that a cancelling `unrealized_conversion_cast` pair folds while a lone one survives, and that bytecode carries its magic and round-trips. `--require-tools` fails instead of skipping. | `mlir-opt` (optional without `--require-tools`) |
| `verify-mlir-coverage.py` | Reads every MLIR dialect and operation out of the installed MLIR two independent ways (`mlir-tblgen --gen-op-doc` and the `getOperationName()` accessor in the generated headers) and requires the second to contain the first, then requires every dialect to carry a disposition: a chapter that names one of its real operations in code, a slice that will close the gap, or a written reason it is out of scope. `--emit-surface` writes the snapshot, `--update` rewrites the chapter's table, `--require-tools` fails instead of skipping the drift check. The disposition half needs no toolchain. | `mlir-opt`, `mlir-tblgen` (optional without `--require-tools`) |
| `verify-frontend-lowering.py` | Compiles the checked-in `20-clang-frontend/examples/` sources and asserts the chapter's lowering claims structurally across x86-64 SysV and AArch64 AAPCS: aggregate layout, packed alignment, bit-field erasure, short-circuit branching, `volatile`, mangling, vtable dispatch, template linkage, and per-target argument classification. `--update` refreshes the normalized `.ll` snapshots; `--require-tools` fails instead of skipping. Normalization strips parameter attributes newer than the corpus's **LLVM 18** baseline (SEMVER.md; it moved from 15 in Phase 1.7), and each snapshot is then assembled by the **oldest** `llvm-as` at or above that baseline — suffixed or not, since a name is not a version — because a snapshot checked only against the newest assembler is one nobody has checked. `--require-baseline` turns "no assembler at the declared major" from a silent pass into a failure, and is passed by the CI job that installs that toolchain; without it a host with only a newer assembler says nothing rather than claiming a baseline it did not test. | `clang`, `clang++` (optional without `--require-tools`); an `llvm-as` at the baseline major for `--require-baseline` |
| `verify-mlir-rail-references.py` | Re-checks every file, CMake idiom, and MLIR API that chapters `18/08` and `18/09` cite in this repository's production law rail, resolves their `../../` links, and enforces that the pre-LLVM-23 spellings (`builder.create<`, `applyPatternsAndFoldGreedily`) stay absent. Reads sources only — no MLIR toolchain required. | Python 3 |
| `analyze-benchmark-samples.py` | Grades a baseline/candidate sample pair into a verdict or a refusal: measures the rig's resolution by comparing the baseline against itself, reports dispersion/drift/outlier diagnostics, then applies Mann-Whitney U (tie- and continuity-corrected), Cliff's delta, a Hodges-Lehmann shift, and a seeded bootstrap interval on the median ratio. `--gate` exits nonzero on a regression and refuses to gate a `wall`-class row at all. Standard library only, deterministic. | Python 3 |
| `verify-benchmark-analysis.py` | Self-test for the above: statistical kernels against hand-computable values, plus a pinned expected verdict for every fixture in `21-performance-methodology/examples/` — including the refusals — and a check that no fixture goes ungraded. | Python 3 |
| `probe-hardware-counters.py` | Opens each hardware event with `perf_event_open` and records what the kernel said, alongside the paranoid level, the PMU device list, and a virtualization hint. An unreadable counter is recorded as `null`, never `0`. Exits 0 whether or not counters exist — "no counters here" is a successful measurement of the host — and `--require-counters` inverts that for a rig that must have them. | Python 3, Linux |


## Advanced-content verification map

Use these script groups when advanced examples or reference paths change:

| Content family | Scripts to run |
| --- | --- |
| Advanced IR, intrinsics, attributes, poison/freeze, fast math | `verify-examples.sh`, `verify-exercises.sh`, `verify-invalid-fixtures.sh`, `verify-opaque-pointers.sh` |
| Optimization before/after examples and pass-pipeline lessons | `verify-opt-diff.sh`, `verify-examples.sh` |
| MLIR bridge examples | `verify-mlir-examples.sh`, then `verify-examples.sh` for lowered `.ll` companions |
| BCIR mapping/source-like fragments | `verify-bcir-mapping.sh`, `verify-examples.sh`, `verify-manifest.sh` |
| Binary-analysis CSV evidence | `verify-csv-schema.sh`, `verify-binary-analysis-evidence.py`, `generate-binary-analysis-fixtures.py --check` |
| Out-of-tree pass plugin source or fixtures | `build-pass-plugin.sh`, then `verify-examples.sh` and `verify-manifest.sh` for the `.ll` fixtures |
| Clang frontend chapters or their example sources | `verify-frontend-lowering.py`, then `verify-examples.sh` and `verify-manifest.sh` for the regenerated snapshots |
| Production MLIR chapters, or any change to `mlir/` they cite | `verify-mlir-rail-references.py`, then `verify-mlir-examples.sh` for the registry |
| Performance-methodology chapters, fixtures, or analysis | `verify-benchmark-analysis.py`, then `verify-manifest.sh` for the sample files |

## CMake batch targets

The training-only `training/llvm/CMakeLists.txt` exposes first-class custom targets for the
training corpus. Configure the project once, then run the targets from the build
directory with `cmake --build`:

```bash
cmake -S training/llvm -B build/training/llvm
cmake --build build/training/llvm --target training-llvm-verify-examples
cmake --build build/training/llvm --target training-llvm-autograder-self-test
cmake --build build/training/llvm --target training-llvm-verify-dataset-export
cmake --build build/training/llvm --target training-llvm-smoke-llc
cmake --build build/training/llvm --target training-llvm-smoke-lli
cmake --build build/training/llvm --target training-llvm-verify-exercises
cmake --build build/training/llvm --target training-llvm-verify-invalid-fixtures
cmake --build build/training/llvm --target training-llvm-verify-adversarial-fixtures
cmake --build build/training/llvm --target training-llvm-verify-opt-diff
cmake --build build/training/llvm --target training-llvm-verify-opaque-pointers
cmake --build build/training/llvm --target training-llvm-lit
cmake --build build/training/llvm --target training-llvm-verify-manifest
cmake --build build/training/llvm --target training-llvm-verify-csv-schema
cmake --build build/training/llvm --target training-llvm-verify-binary-analysis-evidence
cmake --build build/training/llvm --target training-llvm-check-binary-analysis-fixtures
cmake --build build/training/llvm --target training-llvm-verify-mlir-examples
cmake --build build/training/llvm --target training-llvm-verify-bcir-mapping
cmake --build build/training/llvm --target training-llvm-check
```

The initial lit suite lives in `training/llvm/tests/` and delegates to the same
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
UPDATE_OPT_DIFF=1 ./training/llvm/tools/verify-opt-diff.sh
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
<path-or-glob-under-training/llvm><TAB><reason>
```

Use paths relative to `training/llvm/`, and use globs only when an entire family
of teaching fixtures has the same rationale. Reasons should be short human
phrases because the smoke script prints them directly as:

```text
[skip] training/llvm/<path-or-glob> ... <reason>
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
verification, must include the exact marker line
`; training-llvm-invalid-kind: semantic-only` (the string the script greps for)
so it can be distinguished from an accidentally valid parser/verifier fixture.

`verify-manifest.sh` is intentionally separate from IR verification. It compares
the checked-in Markdown table in `../examples/README.md` to the discovered set of
standalone `*/examples/*.ll` files using the same inclusion policy as
`verify-examples.sh`.

`verify-csv-schema.sh` is intentionally lightweight and fixture-scoped. It uses
checked-in CSV files under `../15-binary-analysis/examples/`, maps each known
schema family by filename to its expected column count, supports quoted commas in
single-line CSV records, and fails if headers or data rows disappear.

`verify-binary-analysis-evidence.py` applies the provenance layer above those
shape checks. It requires every Chapter 15 CSV, including schematic examples and
generated fixture evidence, to have exactly one entry in
`../15-binary-analysis/evidence-manifest.json`. Deterministic entries must name a
source, target, build, collection command, checked CSV, and provenance JSON;
provenance records target metadata and toolchain versions.

`generate-binary-analysis-fixtures.py --write` refreshes normalized static
evidence. `--check` rebuilds in a temporary directory and fails on deterministic
CSV or stable-provenance drift. Wall-clock values and hardware counters stay
`host-sensitive` and outside this golden diff. Add `--require-tools` in CI so a
missing Clang is a failure rather than a reduced-coverage local skip.

`verify-mlir-examples.sh` is registry-driven and optional-toolchain-friendly.
Without a complete MLIR/LLVM toolchain it exits successfully but reports reduced
coverage (Tier 0 only), never a full MLIR pass. With tools present it keeps
unregistered BCIR sketches at Tier 1, parses registered examples without the
unregistered escape hatch, executes declared conversion pipelines, checks
illegal-op/type elimination and structural requirements, translates supported
LLVM-dialect results, then runs matching-major `llvm-as` and
`opt -passes=verify`. The MLIR-specific CI rail passes `--require-tools` so a
missing tool is a failure there.

That flag once proved less than it looked like it proved. It established that
`mlir-opt` was on `PATH` at a matching major — not that a single claim was
checked. Emptying every `checks` object in the registry left the gate printing
`MLIR tier grading passed` with an identical tier census, because a check list
that is empty, absent or misspelt produces no errors at all and so is
indistinguishable from one that passed. Two rules close that, and they are
deliberately at different levels:

- **A tier that claims a conversion must claim something about its output.** A
  Tier 3 or 4 entry needs at least one of `require_lowered`, `forbid_lowered`,
  `require_llvm_ir` or `required_runtime_calls` (or an `expected_failure` naming
  the operations that must remain); otherwise the pipeline's exit status is the
  whole test, and a pass that emitted an empty module passes it. Tiers 0–2 claim
  no conversion, so there the tier *is* the claim and an empty `checks` is
  honest. This is a property of the manifest, so it is checked on every host,
  toolchain or no toolchain.
- **`--require-tools` counts assertions that actually ran** against text the run
  generated, and refuses to report success over zero. The pass line prints the
  number, so the evidence is in the output rather than in this paragraph.

A key outside the known set is rejected rather than ignored, for the same
reason: `require_lowerd` in a manifest reads exactly like a check that runs.

## Every `--require-X` flag, audited

Finding that one led to auditing all of them, because the flags share a shape: each
one is passed by a CI job that installed a toolchain, and each is meant to turn
"the tool is missing" from a local skip into a failure there. What none of them
automatically does is prove the tool was *used*. Four more had the same hole, and
all four are now closed the same way — by counting the work and refusing zero:

| Flag | What it proved | What it proves now |
| --- | --- | --- |
| `verify-frontend-lowering.py --require-tools` | `clang` was on `PATH` at startup | at least one lowering claim was checked. Every case can take the "target unsupported" branch — that branch is a substring test over clang's stderr, reachable for reasons this gate does not control — and the run would report `PASSED` over zero claims. Its sibling `--require-baseline` already had this floor; the job that passes only `--require-tools` was uncovered |
| `verify-langref-delta.py --require-surface N` | LLVM N's headers were found, and the stored snapshot equalled a freshly read one | both surfaces contain constructs no LLVM omits. The drift check compares a stored surface against a live one read by **the same extractor**, so an extractor that matched nothing wrote an empty snapshot with `--emit-surface` and then compared `[]` to `[]` and passed. `--emit-surface` now refuses to write such a snapshot, and the stored snapshots are checked on every host, toolchain or not |
| `generate-binary-analysis-fixtures.py --check --require-tools` | `find_clang()` returned a path | at least one fixture was built and compared. The build loop runs over manifest entries classified `deterministic`; a manifest with none made every loop iterate zero times while `--check` reported a clean golden diff over nothing |
| `verify-mlir-coverage.py` two-rail check | that `mlir-tblgen`'s operation list was contained in the headers' | that there is a tblgen rail to contain. An empty rail 1 used to stand the check down **and mark it done** — which is the state a broken tblgen extraction produces, so the one check whose job is to catch a broken extraction was disabled by exactly the thing it exists to catch |

Two of these were in gates this corpus had already fixed the other half of, which is
the useful part of the pattern: when a flag turns out to prove less than its name
says, its siblings are worth reading before anything else.

Four more flags live in the shared rail at `training/tools/`, and the sweep covered
them too. `verify_embeddings.py --require-native`, `verify_training_export.py
--require-bcir` and `probe-hardware-counters.py --require-counters` came back sound —
the last one because it correctly fails on a host with no PMU rather than recording an
unreadable counter as zero. The other four did not:

| Flag | What was wrong |
| --- | --- |
| `verify_ml_components.py --require-torch` | The loop it gates runs over `TORCH_GATED`, which is `tuple(c for c in COMPONENTS if c.reach == "torch-gated")`. One misspelt `reach` empties it — and puts that component in no class at all, since the other two filters miss it too. With torch present the check then returns having exercised nothing. `RUNS_HERE` had carried an emptiness floor since it was written; its sibling did not. Both are fixed, and the three classes are now checked to partition the inventory |
| `export_training_examples.py --require-bcir` | A genuine bug rather than a missing floor: `load_tool` re-executed the module on every call, so `main()`'s `native.BackendUnavailable` and the class `export()` raised were **different class objects**. The `except` never matched, the flag's exit path and the unflagged skip beneath it were unreachable for every input, and the failure surfaced as a traceback. `load_tool` now returns the module it already loaded. Separately, an absent distillation directory made the example loop iterate zero times and still write a manifest |
| `search_chunks.py --require-native` | Read only inside the native branch, while `--backend` defaults to `reference` — so the likeliest invocation passed the flag and never imported BCIR, built a kernel or computed a native dot product |
| `embed_chunks.py --require-provider` | Unfalsifiable under the default model: `LexicalHashProvider` is hermetic, with no weights to miss and no package to be absent, so it cannot raise the exception the flag exists to catch. And the flag had no owner — `CORPUS_STANDARD.md` stated that "the CI job that installs a model passes `--require-provider`" when no such job exists, and `README.md` said its refusal path "is checked" when nothing checks it. Both now say so |

The last two are a different failure from the rest and worth naming separately: not a
check that can pass over zero work, but a flag that **cannot fail** on the configuration
it is most likely to be typed with. In a log the two are indistinguishable. Both tools
now refuse that combination as a usage error rather than accepting a guarantee that
could never have been tested.

The canaries deserve a note. They are named constructs (`Ret`, `nounwind`,
`llvm.memcpy`) rather than expected counts, because a count drifts every release and
would need maintaining, while `Ret` leaving LLVM would mean something other than a
broken regex. They were chosen by intersecting the checked-in snapshots rather than
from memory: `Br` looked like an obvious candidate and is absent from LLVM 23, which
splits it into `CondBr` and `UncondBr`.

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
UPDATE_BCIR_MAPPING=1 ./training/llvm/tools/verify-bcir-mapping.sh
```

## Exercise manifest verifier

`verify-exercise-manifests.py` is a dependency-free validator for the
per-exercise JSON files in `../autograder/manifests/`. It checks the checked-in
JSON Schema contract, referenced repository paths, unique IDs, exact point
totals, required-tool/minimum-version consistency, and one-to-one coverage of
numbered graded prompts. It does not execute grading checks, so LLVM and MLIR
tools are not required to run it.

```bash
python3 training/llvm/tools/verify-exercise-manifests.py
```

## Controlled exercise variants

`generate-exercise-variants.py` requires `--seed` and follows
`../dataset/VARIANT_POLICY.md`. The default and maximum reviewed budget is five
accepted records. Output JSON Lines and the optional rejection report are
deterministic; missing oracle tools fail closed. Repeat `--existing` to deduplicate
against previously generated review files. For example:

```bash
python3 training/llvm/tools/generate-exercise-variants.py \
  --seed 20260612 --budget 5 \
  --output /tmp/variants.jsonl --report /tmp/variants-report.json
```

The generator deliberately does not create Markdown review solutions.

## Adding a script

When adding a new script:

1. keep it executable and runnable from the repository root;
2. document required external tools and any intentional skips in this README;
3. print the command being demonstrated before executing it;
4. write demo output either to stdout or to a clearly named file under `${TMPDIR:-/tmp}`;
5. make it fail closed when a required fixture disappears;
6. wire it into `.github/workflows/ci.yml` when it guards repository health;
7. ask whether it is about LLVM at all. If a second subject would want the same
   script, it belongs in `training/tools/` with the subject-specific part in
   `llvm_profile.py` — copying it later is how three copies of `find_tool` came
   to exist, and the gate now refuses the fourth.

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
cmake -S training/llvm -B build/training/llvm
cmake --build build/training/llvm --target training-llvm-check
```

The aggregate target includes deterministic repository gates and optional LLVM,
MLIR, and lit-backed gates. Optional tools skip cleanly; a present toolchain that
rejects a checked fixture still fails the target.

## Provider-neutral evaluation runner

`run-eval.py` materializes exercise prompts/context, invokes a provider-neutral
local command adapter or deterministic fixture adapter, calls the executable
autograder, and writes JSON Lines plus aggregate/reproducibility reports to a
caller-owned output directory. See [`../eval/README.md`](../eval/README.md) for
the adapter contract, modes, resume behavior, metrics, and security boundary.

```bash
python3 training/llvm/tools/run-eval.py run \
  --output-dir /tmp/bcir-eval/reference \
  --exercise 001 \
  --fixture-adapter reference
```

## LTO/BOLT artifact experiments

- `run-lto-matrix.sh` executes the checked-in no-LTO/ThinLTO/FullLTO manifest,
  verifies matching LLVM tool versions, and emits deterministic summaries plus
  an evaluation-oriented JSON report.
- `run-bolt-experiment.sh` preserves the baseline, supplied profile, rewrite
  command, rewritten summary, and tool versions for an optional BOLT leg.
- `smoke-bolt.sh` runs the baseline and reports unavailable BOLT/profile support
  explicitly rather than treating it as a successful rewrite.
