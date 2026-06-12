# ROADMAP — Known Gaps and Out-of-scope Topics

This file keeps [`CURRICULUM.md`](CURRICULUM.md) focused on reading paths while
preserving roadmap notes for future expansion.

## Completed expansion areas

The top-level corpus has been expanded beyond basic IR syntax and now includes
introductory or intermediate coverage for these advanced families:

- **BCIR lowering** — claim normalization, graph/GAADMSF lowering, register
  binding, mixed strides, HAM hints, runtime ABI/wrapper calls, and diagnostic
  metadata are covered in [`bcir-mapping/`](bcir-mapping/) with checked examples.
- **MLIR integration** — MLIR modules, dialects, operation anatomy, LLVM dialect
  lowering, and BCIR custom-dialect sketches are covered in
  [`14-mlir-bridge/`](14-mlir-bridge/).
- **Backend/JIT diagnostics** — codegen stages, TableGen source-vs-generated
  boundaries, ORC/LLJIT ownership, layers, MC emission, relocations, and
  missing-symbol triage are covered in [`12-backend-jit/`](12-backend-jit/).
- **Introductory PGO/LTO/BOLT coverage** — the concepts, pipeline boundaries,
  and evidence-review prompts are covered in
  [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md)
  and the binary-analysis material in [`15-binary-analysis/`](15-binary-analysis/).
- **Deterministic LTO/BOLT artifact matrix** — a checked manifest, tiny
  cross-translation-unit fixture, matching-version tool discovery, JSON report,
  no-LTO/ThinLTO/FullLTO artifact summaries, and optional profile-driven BOLT
  leg are covered in
  [`07-optimization/10-lto-bolt-experiment-matrix.md`](07-optimization/10-lto-bolt-experiment-matrix.md).
- **Repair and prediction exercises** — exercises now include standalone IR
  writing, invalid-fixture repair, pass-output prediction, metadata/attribute
  reviews, BCIR lowering, MLIR reviews, and backend/JIT diagnostics.
- **Example governance** — [`EXAMPLES.md`](EXAMPLES.md) now distinguishes
  standalone `.ll`, before/after pass snapshots, `.invalid.ll.txt` fixtures,
  `.mlir` review artifacts, CSV/data artifacts, and generated BCIR mapping
  outputs.
- **Closed-loop grading and dataset export** — the active training workflow now
  validates 42 declarative exercise manifests, grades reference or external
  attempts with deterministic partial credit, records explicit optional-tool
  skips, and exports 42 stable-ID records across curated train/validation/test
  splits. The export is a small evaluation and regression dataset for grader,
  agent, and curriculum quality checks—not a production-scale fine-tuning
  corpus. Held-out bundles omit reference-solution content by default.

## Remaining gaps

If your task touches these remaining gaps, you'll need external references or a
new chapter before relying on this corpus alone:

- **Custom optimization pass implementation** — pass-manager internals beyond
  introductory `opt` commands and pass-output reasoning.
- **C/C++ frontend internals** — Clang AST, semantic analysis, and exact
  frontend lowering rules.
- **Production MLIR pass/dialect implementation** — TableGen ODS, conversion
  pattern code, pass pipelines, and build integration are only sketched.
- **Statistically rigorous performance studies** — the deterministic LTO/BOLT
  matrix compares build artifacts only. The corpus still does not provide a
  complete workload-selection, profile-collection, repeated-trial, noise-control,
  or significance-analysis methodology for runtime speedup claims.
- **Production-grade hardware-counter harness** — counter examples show schema
  shape and review use, not a deployable cross-platform collection harness.
- **Guaranteed BOLT profiling/rewrite in CI** — the optional runner records
  explicit unsupported results and preserves baseline evidence, but CI does not
  require BOLT until it has a stable matching binary-rewriting and profile
  environment.
- **Exhaustive target/workload LTO study** — the checked matrix is intentionally
  tiny and host-oriented, not a target-by-target empirical study across real
  workloads and toolchain versions.
- **Complete backend target development** — TableGen and codegen chapters are
  diagnostic guides, not a full target-porting manual.
- **Calls / returns / comparisons** — a small dedicated chapter may be worth
  adding if this training set keeps expanding beyond the quick reference.

These are roadmap items; PRs welcome.
