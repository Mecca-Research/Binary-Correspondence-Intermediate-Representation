# ROADMAP — Known Gaps and Out-of-scope Topics

This file keeps [`CURRICULUM.md`](CURRICULUM.md) focused on reading paths while preserving the roadmap notes for future expansion.

## What's intentionally NOT here yet

The curriculum now includes introductory coverage for topics that used to need
out-of-tree notes: microarchitectural review in
[`15-binary-analysis/01-microarchitecture-side-channels.md`](15-binary-analysis/01-microarchitecture-side-channels.md),
dynamic trace schemas and runtime counters in
[`15-binary-analysis/02-dynamic-traces-and-counters.md`](15-binary-analysis/02-dynamic-traces-and-counters.md),
PGO/LTO/BOLT concepts in
[`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md),
and interpretable BCSA triage in
[`15-binary-analysis/03-interpretable-bcsa-features.md`](15-binary-analysis/03-interpretable-bcsa-features.md).

MLIR bridge coverage now includes BCIR dialect sketches, type conversion,
conversion patterns, pass diagnostics, and an end-to-end final `.ll` snapshot in
[`14-mlir-bridge/README.md`](14-mlir-bridge/README.md).

If your task touches these remaining gaps, you'll need external references:

- **Custom optimization pass design** — pass-manager internals beyond the introductory `opt` vectorization commands
- **C/C++ frontend internals** — Clang, AST, lowering rules
- **Statistically complete benchmarking methodology** — the dynamic-analysis chapters define schemas and review loops, not a complete benchmark design, sampling plan, or statistical analysis workflow
- **Production-grade hardware-counter harness** — the counter examples show schema shape and review use, not a deployable cross-platform collection harness
- **Guaranteed BOLT availability in CI** — the PGO/LTO/BOLT chapter explains concepts and expected artifacts, but CI environments may not ship `llvm-bolt` or a compatible binary-rewriting setup
- **Exhaustive FullLTO-vs-ThinLTO empirical matrix** — the LTO material explains the concepts and comparison prompts, not a target-by-target empirical matrix across workloads and toolchain versions
- **Calls / returns / comparisons** — a small dedicated chapter may be worth adding if
  this training set keeps expanding beyond the quick reference
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
- **Binary-analysis evidence** — side-channel review, dynamic traces, hardware
  counters, PGO/LTO/BOLT artifacts, and interpretable BCSA feature schemas are
  covered in [`15-binary-analysis/`](15-binary-analysis/) and
  [`07-optimization/06-pgo-lto-bolt.md`](07-optimization/06-pgo-lto-bolt.md).
- **Repair and prediction exercises** — exercises now include standalone IR
  writing, invalid-fixture repair, pass-output prediction, metadata/attribute
  reviews, BCIR lowering, MLIR reviews, and backend/JIT diagnostics.
- **Example governance** — [`EXAMPLES.md`](EXAMPLES.md) now distinguishes
  standalone `.ll`, before/after pass snapshots, `.invalid.ll.txt` fixtures,
  `.mlir` review artifacts, CSV/data artifacts, and generated BCIR mapping
  outputs.

## Remaining gaps

If your task touches these remaining gaps, you'll need external references or a
new chapter before relying on this corpus alone:

- **Custom optimization pass implementation** — pass-manager internals beyond
  introductory `opt` commands and pass-output reasoning.
- **C/C++ frontend internals** — Clang AST, semantic analysis, and exact
  frontend lowering rules.
- **Production MLIR pass/dialect implementation** — TableGen ODS, conversion
  pattern code, pass pipelines, and build integration are only sketched.
- **Statistically complete benchmarking methodology** — dynamic-analysis
  chapters define schemas and review loops, not a complete sampling or
  significance-analysis workflow.
- **Production-grade hardware-counter harness** — counter examples show schema
  shape and review use, not a deployable cross-platform collection harness.
- **Guaranteed BOLT availability in CI** — PGO/LTO/BOLT material explains
  concepts and expected artifacts, but CI environments may not ship `llvm-bolt`
  or a compatible binary-rewriting setup.
- **Exhaustive FullLTO-vs-ThinLTO empirical matrix** — the LTO material explains
  review prompts, not a target-by-target empirical matrix across workloads and
  toolchain versions.
- **Complete backend target development** — TableGen and codegen chapters are
  diagnostic guides, not a full target-porting manual.
- **Calls / returns / comparisons** — a small dedicated chapter may be worth
  adding if this training set keeps expanding beyond the quick reference.

These are roadmap items; PRs welcome.
