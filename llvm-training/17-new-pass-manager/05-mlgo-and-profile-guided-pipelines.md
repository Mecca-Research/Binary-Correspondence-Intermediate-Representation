# MLGO and profile-guided pipelines

Profile-guided optimization and MLGO can improve pass decisions, but they do not
replace IR legality checks. Treat profiles and learned models as policy inputs
that influence profitability, ordering, or thresholds; keep BCIR correctness in
verifier passes and explicit lowering contracts.

## What profiles can influence

Profiles can guide:

- inlining and CGSCC decisions;
- block layout and branch probability assumptions;
- loop unrolling, vectorization, and rotation profitability;
- hot/cold splitting and outlining; and
- custom GAADMSF choices such as whether a HAM prefetch or hardware-aware
  memory transform is profitable.

For BCIR, a hot profile edge does not prove that an operation has HAM metadata,
that a register mapping can be destroyed, or that diagnostic metadata can be
removed. Those are legality questions.

## MLGO in the pass-manager mental model

MLGO integrates with LLVM optimization decisions through model-backed advisors
and pass-pipeline configuration. From a pipeline-design perspective:

1. `PassBuilder` still constructs the pipeline or parses `-passes=...`.
2. Analyses still flow through `ModuleAnalysisManager`, `CGSCCAnalysisManager`,
   `FunctionAnalysisManager`, and `LoopAnalysisManager`.
3. Transforms still return `PreservedAnalyses`.
4. BCIR verifier passes still enforce register correspondence, metadata, and
   lowering-stage invariants.

That means stale-analysis and metadata-preservation bugs remain ordinary pass
bugs even when a model chose the transform threshold.

## BCIR profile-guided checklist

- Keep profile metadata and BCIR diagnostic metadata distinct. Do not let a pass
  that rewrites branch weights erase BCIR diagnostics attached to the same
  instruction or terminator.
- Re-run BCIR invariant verification after profile-driven CFG transforms such as
  block placement, loop rotation, unswitching, or hot/cold splitting.
- If a GAADMSF transform is selected by profile profitability, still require the
  GAADMSF gate: attributes, metadata, or a hardware profile.
- Invalidate BCIR analyses after profile-guided rewrites that clone, delete, or
  merge mapped operations.
- Record when a lowering stage consumes 1:1 register correspondence so later
  profile-guided passes do not try to diagnose a mapping that intentionally no
  longer exists.

## Pitfalls to review

- **Stale analyses:** profile-guided transforms often rewrite CFG and memory
  shape; do not preserve dominator tree, loop info, MemorySSA, or BCIR analyses
  unless audited.
- **Overclaiming preserved analyses:** a pass that only changes metadata can
  still invalidate a BCIR metadata analysis.
- **Optimizer erases diagnostics:** cloning or simplifying hot paths may drop
  diagnostic attachments unless the pass explicitly copies or remaps them.
- **Model confidence vs. legality:** high confidence from MLGO is not permission
  to run a hardware-specific transform without GAADMSF evidence.

## Exercises and prompt templates

Use these as code-review or implementation prompts:

1. **Verifier pass:** implement a function pass named `bcir-verify-regmap` that
   uses a `BcirRegisterMapAnalysis` result to reject duplicate source-register
   mappings and missing `!bcir.reg` attachments on register-bound operations.
2. **HAM-gated transform:** add a transform pass named `bcir-ham-lower` that
   returns `PreservedAnalyses::all()` and emits a remark when HAM metadata is
   absent, but lowers only the operations explicitly marked with HAM metadata.
3. **Stale-analysis review:** review a pass plugin and list every path that
   changes BCIR metadata, memory effects, or CFG shape. For each path, state
   whether `PreservedAnalyses` is correct or overclaims preservation.
