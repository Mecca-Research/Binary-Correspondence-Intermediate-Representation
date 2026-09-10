# ROADMAP — Coverage, Closed Gaps, and Declared Non-goals

This file keeps [`CURRICULUM.md`](CURRICULUM.md) focused on reading paths while
recording what this corpus covers, what it deliberately does not, and why.

Three kinds of entry appear below, and the distinction is the point:

- **Closed** — the material exists *and* something checks it.
- **Blocked** — the work is specified and cannot be done on any host available
  to this project. The obstacle is named.
- **Non-goal** — a deliberate decision not to build it, with the reason.

An item is never "closed" because prose about it was written. A chapter that
states a fact no gate re-checks is prose, and prose goes stale.

## Completed expansion areas

- **BCIR lowering** — claim normalization, graph/GAADMSF lowering, register
  binding, mixed strides, HAM hints, runtime ABI/wrapper calls, and diagnostic
  metadata are covered in [`bcir-mapping/`](bcir-mapping) with checked examples.
- **MLIR integration** — MLIR modules, dialects, operation anatomy, LLVM dialect
  lowering, and BCIR custom-dialect sketches are covered in
  [`14-mlir-bridge/`](14-mlir-bridge).
- **Backend/JIT diagnostics** — codegen stages, TableGen source-vs-generated
  boundaries, ORC/LLJIT ownership, layers, MC emission, relocations, and
  missing-symbol triage are covered in [`12-backend-jit/`](12-backend-jit).
- **Introductory PGO/LTO/BOLT coverage** — the concepts, pipeline boundaries,
  and evidence-review prompts are covered in
  [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md)
  and the binary-analysis material in [`15-binary-analysis/`](15-binary-analysis).
- **Deterministic LTO/BOLT artifact matrix** — a checked manifest, tiny
  cross-translation-unit fixture, matching-version tool discovery, JSON report,
  no-LTO/ThinLTO/FullLTO artifact summaries, and optional profile-driven BOLT
  leg are covered in
  [`07-optimization/10-lto-bolt-experiment-matrix.md`](07-optimization/10-lto-bolt-experiment-matrix.md).
- **Repair and prediction exercises** — exercises include standalone IR
  writing, invalid-fixture repair, pass-output prediction, metadata/attribute
  reviews, BCIR lowering, MLIR reviews, and backend/JIT diagnostics.
- **Example governance** — [`EXAMPLES.md`](EXAMPLES.md) distinguishes
  standalone `.ll`, before/after pass snapshots, `.invalid.ll.txt` fixtures,
  `.mlir` review artifacts, CSV/data artifacts, and generated BCIR mapping
  outputs. Every file under any `examples/` tree must be classified in
  [`examples/README.md`](examples/README.md) or `verify-manifest.sh` fails.
- **Closed-loop grading and dataset export** — the training workflow validates
  declarative exercise manifests, grades reference or external attempts with
  deterministic partial credit, records explicit optional-tool skips, and
  exports stable-ID records across curated train/validation/test splits. The
  export is an evaluation and regression dataset, not a production-scale
  fine-tuning corpus. Held-out bundles omit reference-solution content by
  default.

## Closed gaps

Each of these was a "remaining gap" in an earlier revision of this file. Each is
now covered by material *and* by a check.

### Calls, returns, and comparisons

[`05-control-flow/05-call-and-ret.md`](05-control-flow/05-call-and-ret.md) and
[`05-control-flow/06-comparisons-and-select.md`](05-control-flow/06-comparisons-and-select.md):
calling conventions, ABI parameter attributes, tail-call markers and their
preconditions, multi-value returns, `icmp`/`fcmp` predicates including the
ordered/unordered split, vector masks, and why `select` evaluates both arms and
does not filter poison.

*Checked by:* two standalone examples in the known-good manifest
(`verify-examples.sh`, `verify-opaque-pointers.sh`).

### Custom optimization pass implementation

[`17-new-pass-manager/06-building-an-out-of-tree-pass.md`](17-new-pass-manager/06-building-an-out-of-tree-pass.md)
plus a complete, buildable plugin in
[`17-new-pass-manager/examples/pass-plugin/`](17-new-pass-manager/examples/pass-plugin):
a custom analysis with an `AnalysisKey` and an `invalidate` hook, a module pass
reaching function analyses through the proxy, a function transform with an
honest `PreservedAnalyses`, a printer, and registration through
`llvmGetPassPluginInfo` including a parameterized pass name and an extension
point. Build integration, `lit`/`FileCheck` conventions, and the pass-manager
debugging flags are covered.

*Checked by:* [`tools/build-pass-plugin.sh`](tools/build-pass-plugin.sh) —
builds the plugin and runs nine assertions through `opt`, including the negative
cases (the checker fires on a violating module; `<strict>` exits nonzero; a
stock `loop-unroll-full` breaks the invariant and is caught). Skips cleanly,
with the reason printed, when LLVM development headers are absent or when `opt`
and `llvm-config` disagree on the LLVM major.

### C/C++ frontend internals

[`20-clang-frontend/`](20-clang-frontend): the driver/`-cc1` boundary and
compilation phases, AST node families and the implicit nodes Sema adds, exact C
lowering rules (aggregate layout, packing, bit-field erasure, unions,
short-circuit operators, `volatile`, integer promotion and `nsw`), exact C++
lowering rules (Itanium mangling, vtable dispatch, constructor/destructor
placement, template instantiation linkage, references), ABI and target lowering
across two triples, and the correspondence with this repository's own C-front
rail.

*Checked by:* [`tools/verify-frontend-lowering.py`](tools/verify-frontend-lowering.py)
— compiles the checked-in sources and asserts the chapter's lowering claims
structurally across x86-64 SysV and AArch64 AAPCS. A Clang release that changes
a rule fails the gate rather than leaving the prose wrong.

### Production MLIR pass/dialect implementation

[`18-mlir-lowering-to-llvm/08-production-dialect-and-build-integration.md`](18-mlir-lowering-to-llvm/08-production-dialect-and-build-integration.md)
and
[`18-mlir-lowering-to-llvm/09-production-conversion-pass.md`](18-mlir-lowering-to-llvm/09-production-conversion-pass.md):
the ODS-through-TableGen-to-registered-dialect build graph, generator flags,
non-disturbing attribute design, the three registrations, `lit` conventions,
conversion-pattern anatomy (`adaptor` versus `op`), `TypeConverter`,
`ConversionTarget`, partial versus full conversion, the greedy-rewriter
boundary, and the MLIR 22 → 23 API migration. A complete standalone skeleton is
in [`18-mlir-lowering-to-llvm/examples/production-dialect/`](18-mlir-lowering-to-llvm/examples/production-dialect).

*Checked by:* [`tools/verify-mlir-rail-references.py`](tools/verify-mlir-rail-references.py)
— re-checks every file, CMake idiom, and API the chapters cite in this
repository's production law rail, and enforces that the pre-LLVM-23 spellings
stay absent. Needs no MLIR toolchain. The skeleton's `scale.mlir` is a
registered Tier-1 entry in the MLIR example registry.

*Boundary:* the skeleton is **reviewed reference code**. No corpus gate builds
it, because building it needs an MLIR development install that the corpus does
not assume. The production dialect that CI does build is [`../mlir/`](../mlir),
and that is what the chapters teach from.

### Statistically rigorous performance studies

[`21-performance-methodology/`](21-performance-methodology): workload selection
and the four microbenchmark failure modes, clocks and their quanta, sampling
versus instrumentation bias, repeated trials, run order, warm-up policy, noise
reduction ranked by what it actually buys, and the statistics — Mann-Whitney U
with tie correction, Cliff's delta, Hodges-Lehmann shift, a seeded bootstrap
interval, and the rig's own resolution measured by comparing the baseline
against itself.

*Checked by:* [`tools/verify-benchmark-analysis.py`](tools/verify-benchmark-analysis.py)
— statistical kernels against hand-computable values, plus a pinned expected
verdict for every fixture. The fixture set is chosen by *failure shape*: an
effect below the rig's resolution, a build compared against itself, a drifting
baseline, a significant-but-trivial difference, and a wall-clock row used as a
gate. Three of the analysis tool's five verdicts are refusals.

### Backend target development

[`12-backend-jit/08-target-backend-porting-map.md`](12-backend-jit/08-target-backend-porting-map.md):
the complete file inventory of an LLVM target, the build order with its
checkpoints, the testing layers, a ranked account of where the schedule actually
goes, and the narrower alternatives to check first.

*Boundary:* this is a **map, not a manual** — see the non-goal below.

## Blocked: named obstacle, not effort

### Production-grade hardware-counter collection

*Partially closed.* [`21-performance-methodology/06-hardware-counter-harness.md`](21-performance-methodology/06-hardware-counter-harness.md)
and [`tools/probe-hardware-counters.py`](tools/probe-hardware-counters.py)
deliver the capability probe and the evidence schema: `perf_event_open` per
event, errno-level diagnosis, the PMU device list, the paranoid level, a
virtualization hint, and — the load-bearing part — an unreadable counter
recorded as `null` rather than `0`.

*Still open:* attribution to a workload (event grouping, per-thread and
per-cgroup scoping), multiplexing correction from `time_enabled`/`time_running`,
and the ring-buffer machinery for sampled events.

*Obstacle:* **no PMU on any available host.** `perf_event_open` returns `ENOENT`
for every hardware event regardless of privilege — observed at euid 0 with a
paranoid level of 2 and no `cpu` entry in `/sys/bus/event_source/devices`.
Building a collection harness against that would produce code nobody could test,
in a corpus about evidence. See
[`../docs/BCIR_TARGET_ACCESS.md`](../docs/BCIR_TARGET_ACCESS.md).

### Guaranteed BOLT profiling and rewrite in CI

The optional runner records explicit unsupported results and preserves baseline
evidence, but CI does not require BOLT.

*Obstacle:* no stable matching binary-rewriting and profile environment across
the runner matrix. A required gate that skips on most cells is a gate that
reports green for not running — the failure mode the whole corpus is built
against. It becomes required when the environment is stable, not before.

## Declared non-goals

These are decisions, not omissions. Each names why.

- **A Clang contributor guide.** Writing a Clang plugin, adding an attribute,
  extending the preprocessor, or modifying Sema. [`20-clang-frontend/`](20-clang-frontend)
  covers what a reader needs in order to *read* frontend output and attribute a
  surprise to the right stage; changing Clang is a different project with a
  different audience.
- **A full backend target-porting manual.** The map in
  [`12-backend-jit/08-target-backend-porting-map.md`](12-backend-jit/08-target-backend-porting-map.md)
  is deliberate: porting a target is a multi-engineer-year project whose details
  live in the LLVM tree and in an ISA manual, and this repository has a written
  decision *not* to build one
  ([`../docs/BCIR_NATIVE_OBJECT_GATE.md`](../docs/BCIR_NATIVE_OBJECT_GATE.md)).
  The map is the estimate that decision rests on.
- **An exhaustive target/workload LTO study.** The checked matrix in
  [`07-optimization/10-lto-bolt-experiment-matrix.md`](07-optimization/10-lto-bolt-experiment-matrix.md)
  compares build artifacts, deterministically and tinily, on purpose. A
  target-by-target empirical study across real workloads and toolchain versions
  is a research programme, not a chapter, and
  [`21-performance-methodology/`](21-performance-methodology) now supplies the
  method anyone conducting one would need.
- **A benchmarking framework.** The analysis and the discipline are here; the
  harness is not. Frameworks exist, and the half that most reported speedups get
  wrong is the analysis.
- **A production-scale fine-tuning corpus.** Stated in
  [`README.md`](README.md): this is a context pack. The exercise, autograder,
  and dataset-schema substrate is what an export would build on.

## Adding to this corpus

If you close one of the open items, or add a chapter:

1. Put the material where the numbering says it goes, and add the chapter to its
   parent `README.md` dispatcher.
2. Classify every new file under an `examples/` tree in
   [`examples/README.md`](examples/README.md); `verify-manifest.sh` enforces it
   recursively.
3. **Add a check.** A structural claim gate, a fixture with a pinned verdict, or
   a negative test — and then break the thing it guards and watch it fire. A
   gate that has never failed has not been shown to work.
4. Register the check in [`CMakeLists.txt`](CMakeLists.txt) and in
   `.github/workflows/ci.yml`, and make it skip cleanly — printing the reason —
   when its optional tools are missing.
5. Update this file: move the item, and say what checks it.
