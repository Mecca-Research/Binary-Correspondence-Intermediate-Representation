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

If your task touches these remaining gaps, you'll need external references:

- **Custom optimization pass design** — pass-manager internals beyond the introductory `opt` vectorization commands
- **C/C++ frontend internals** — Clang, AST, lowering rules
- **Statistically complete benchmarking methodology** — the dynamic-analysis chapters define schemas and review loops, not a complete benchmark design, sampling plan, or statistical analysis workflow
- **Production-grade hardware-counter harness** — the counter examples show schema shape and review use, not a deployable cross-platform collection harness
- **Guaranteed BOLT availability in CI** — the PGO/LTO/BOLT chapter explains concepts and expected artifacts, but CI environments may not ship `llvm-bolt` or a compatible binary-rewriting setup
- **Exhaustive FullLTO-vs-ThinLTO empirical matrix** — the LTO material explains the concepts and comparison prompts, not a target-by-target empirical matrix across workloads and toolchain versions
- **Calls / returns / comparisons** — a small dedicated chapter may be worth adding if
  this training set keeps expanding beyond the quick reference

These are roadmap items; PRs welcome.
