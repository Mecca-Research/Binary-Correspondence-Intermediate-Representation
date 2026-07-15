# Modern LLVM Pass Infrastructure

The new pass manager is the default mental model for modern LLVM optimizer work.
It makes the unit of work explicit (`Module`, CGSCC, `Function`, and `Loop`),
keeps analyses in typed managers, and lets tools such as `opt` build pipelines
from textual names through `PassBuilder`.

For BCIR, the pass manager is also the place where lowering policy becomes
operational: a pipeline can verify BCIR invariants before and after destructive
transforms, preserve register-correspondence metadata only while it remains
valid, and gate hardware-specific rewrites on explicit evidence.

## Key takeaways

- `PassBuilder` wires the modern optimizer together: it registers analyses,
  parses textual pipelines, builds default optimization pipelines, and exposes
  callbacks for plugins and extension points.
- Textual pipelines such as `opt -passes='verify,function(require<domtree>),sccp,loop-rotate,verify'`
  are the preferred command-line interface for assembling reproducible optimizer
  experiments.
- Modern analyses live in separate managers: `ModuleAnalysisManager`,
  `CGSCCAnalysisManager`, `FunctionAnalysisManager`, and `LoopAnalysisManager`.
  Call `crossRegisterProxies` after registering analyses so invalidation can
  flow across IR-unit boundaries.
- Transform passes return `PreservedAnalyses`. Returning too much is a stale
  analysis bug; returning too little is usually slower but correct.
- Out-of-tree plugins enter through `llvmGetPassPluginInfo`, install
  `registerPassBuilderCallbacks`, and usually add both pipeline parsing
  callbacks and extension-point callbacks.
- BCIR pipelines should verify invariants before and after destructive
  transforms, preserve 1:1 register correspondence unless a lowering stage
  explicitly consumes it, and invalidate analyses whenever BCIR metadata,
  memory effects, or CFG shape changes.

## Chapter map

1. [`01-passbuilder-and-pipelines.md`](01-passbuilder-and-pipelines.md) —
   `PassBuilder`, analysis managers, `crossRegisterProxies`, and textual
   `opt -passes=...` pipelines.
2. [`02-custom-passes-and-analyses.md`](02-custom-passes-and-analyses.md) —
   custom transform passes, custom analysis passes, analysis retrieval, and
   `PreservedAnalyses` discipline.
3. [`03-passbuilder-callbacks-and-plugins.md`](03-passbuilder-callbacks-and-plugins.md) —
   pass plugin registration through `llvmGetPassPluginInfo`,
   `registerPassBuilderCallbacks`, pipeline parsing callbacks, and extension
   points.
4. [`04-adaptive-bcir-pipelines.md`](04-adaptive-bcir-pipelines.md) — adaptive
   BCIR and GAADMSF pipelines that use metadata, attributes, and hardware
   profiles without violating lowering contracts.
5. [`05-mlgo-and-profile-guided-pipelines.md`](05-mlgo-and-profile-guided-pipelines.md) —
   MLGO, PGO, and profile-guided pass selection as policy inputs rather than
   substitutes for verifier-enforced legality.

## Expected prerequisites

Read these first if the terms are new:

- [`07-optimization/`](../07-optimization) for pass categories, common
  analysis/transform names, debugging pipelines, optimization levels, PGO/LTO,
  and the existing BCIR optimizer lessons.
- [`bcir-mapping/`](../bcir-mapping) for BCIR register binding, claim lowering,
  HAM/GAADMSF metadata, runtime boundaries, and diagnostic metadata.
- [`exercises/`](../exercises) for repair prompts and custom-pass invariant
  exercises that this chapter extends.

## Examples

- [`examples/bcir-pass-plugin-skeleton.cpp.md`](examples/bcir-pass-plugin-skeleton.cpp.md) —
  documentation-only C++ skeleton for a BCIR verifier analysis/transform plugin.
- [`examples/adaptive-pipeline-sketch.cpp.md`](examples/adaptive-pipeline-sketch.cpp.md) —
  documentation-only sketch for building a pipeline from module attributes and a
  hardware profile.
- [`examples/gaadmsf-pipeline-before.ll`](examples/gaadmsf-pipeline-before.ll) —
  known-good LLVM IR with BCIR/GAADMSF metadata before the walkthrough pipeline.
- [`examples/gaadmsf-pipeline-after.ll`](examples/gaadmsf-pipeline-after.ll) —
  known-good LLVM IR showing a plausible post-pipeline shape with diagnostic
  metadata retained.

## BCIR pass-manager checklist

- Run a BCIR invariant verifier before and after destructive transforms such as
  CFG simplification, loop canonicalization, vectorization, outlining, or custom
  lowering.
- Preserve 1:1 source-register-to-IR-value correspondence unless the current
  lowering stage explicitly consumes that contract and records the replacement
  mapping.
- Invalidate analyses when a pass changes BCIR metadata, memory effects, aliasing
  promises, call attributes, loop structure, dominance, or CFG shape.
- Gate GAADMSF-specific transforms on attributes, metadata, or hardware profiles;
  never infer hardware legality from a pipeline name alone.
- Treat optimizer passes as metadata adversaries: audit cloned, merged, deleted,
  and hoisted instructions for BCIR diagnostic metadata retention.

## Walkthrough command

A minimal modern `opt` walkthrough that asks for dominator-tree availability,
runs SCCP and loop rotation, and verifies both ends is:

```bash
opt -S \
  -passes='verify,function(require<domtree>),sccp,loop-rotate,verify' \
  llvm-training/17-new-pass-manager/examples/gaadmsf-pipeline-before.ll \
  -o /tmp/gaadmsf-pipeline-after.ll
```

In a real BCIR plugin pipeline, insert verifier passes around destructive stages,
for example `bcir-verify,sccp,loop-rotate,bcir-verify`, and register those names
with a pipeline parsing callback.

## Adversarial pass-pipeline checks

The [adversarial exercise track](../exercises/adversarial) provides seeds and
prompt templates for isolating the first pass that loses metadata, operand
bundles, debug provenance, address-space facts, or BCIR 1:1 correspondence.
Record the exact pipeline and test prefixes during reduction rather than keeping
only the final failing module.
