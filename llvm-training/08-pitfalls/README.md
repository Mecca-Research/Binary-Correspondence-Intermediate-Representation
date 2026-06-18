# Pitfalls — Real-World LLVM IR Bugs

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


A checklist of common mistakes when writing or generating LLVM IR.
Most entries come from the sibling **BCIR** project, where each was
caught by `llvm-as` or `llvm-link` and fixed with a referenced commit.

Read these *before* writing IR. They're the cheapest insurance you'll
buy.

## Key takeaways

- Run this checklist before writing or reviewing generated LLVM IR; most issues are cheaper to catch before `llvm-as` or `llvm-link` fails.
- Verifier failures often come from structural mismatches: PHI predecessors, duplicate labels/symbols, invalid immarg operands, or address-space drift.
- Valid IR can still be wrong when debug metadata, volatile operations, atomics, or pass-pipeline assumptions encode stale semantics.
- Use the linked pages as repair patterns, not just bug descriptions.

## Index

| # | File | One-liner |
|---|---|---|
| 01 | [`01-nested-instruction-expressions.md`](01-nested-instruction-expressions.md) | `or i1 (xor i1 %x, true), %y` — invalid; can't nest instructions as expressions |
| 02 | [`02-phi-predecessor-mismatch.md`](02-phi-predecessor-mismatch.md) | "PHI node entries do not match predecessors" |
| 03 | [`03-duplicate-block-labels.md`](03-duplicate-block-labels.md) | Two basic blocks with the same label name in one function |
| 04 | [`04-duplicate-symbols.md`](04-duplicate-symbols.md) | Two modules `define` the same `@symbol` |
| 05 | [`05-type-schema-drift.md`](05-type-schema-drift.md) | `%T = type { i32, i32 }` in module A; `{ i32, i32, i32 }` in module B |
| 06 | [`06-immarg-violation.md`](06-immarg-violation.md) | `call void @llvm.foo(i32 %dynamic)` where `i32` arg is declared `immarg` |
| 07 | [`07-debug-metadata-bloat.md`](07-debug-metadata-bloat.md) | Valid IR, but millions of duplicate `!DILocation` nodes make debug info huge |
| 08 | [`08-stale-debug-locations.md`](08-stale-debug-locations.md) | Rewritten instructions keep misleading old `!dbg` locations |
| 09 | [`09-atomic-ordering-mismatch.md`](09-atomic-ordering-mismatch.md) | Invalid `cmpxchg` failure ordering, or valid atomics with the wrong synchronization contract |
| 10 | [`10-volatile-is-not-atomic.md`](10-volatile-is-not-atomic.md) | `volatile` flag used for thread synchronization instead of atomic acquire/release |
| 11 | [`11-address-space-confusion.md`](11-address-space-confusion.md) | `ptr addrspace(N)` accidentally used as plain `ptr`, or cross-space `bitcast` |
| 12 | [`12-vectorization-blocked-by-aliasing.md`](12-vectorization-blocked-by-aliasing.md) | `loop not vectorized: cannot prove it is safe to reorder memory operations` |
| 13 | [`13-pass-pipeline-ordering-surprise.md`](13-pass-pipeline-ordering-surprise.md) | A pass works alone but fails or disappears inside a different pipeline order |
| 14 | [`14-orc-jit-symbol-resolution.md`](14-orc-jit-symbol-resolution.md) | ORC/LLJIT reports `Symbols not found: [ foo ]` or resolves the wrong symbol |
| 15 | [`15-tablegen-generated-file-confusion.md`](15-tablegen-generated-file-confusion.md) | Generated `*Gen*.inc` file appears missing because it lives in the build tree |
| 16 | [`16-sanitizer-instrumentation.md`](16-sanitizer-instrumentation.md) | Shadow-memory and redzone checks look redundant but are sanitizer instrumentation |

## BCIR instance summary

The **Commit** column separates two kinds of training material. Short hashes (or PR identifiers, when used) point to concrete BCIR provenance in this repository: either the commit that introduced the relevant BCIR LLVM artifact, the commit that fixed the regression, or both when a pitfall spans an original artifact and a later validation hardening change. `Training-only / preventive` marks exemplar pitfalls included to teach LLVM IR failure modes that are relevant to BCIR authors, but for which this repository does not record an affected BCIR `.ll` regression.

| # | Pitfall | Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|---|---|
| 01 | [`01-nested-instruction-expressions.md`](01-nested-instruction-expressions.md) | `runtime/llvm/bcir_claim_verify.ll` | `1f62e86` | `llvm-as runtime/llvm/bcir_claim_verify.ll -o /dev/null` | Split nested boolean expressions into named SSA temporaries. | [`01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md); [`03-constants/04-global-vs-local.md`](../03-constants/04-global-vs-local.md); [`10-grammar/llvm-ir.tm`](../10-grammar/llvm-ir.tm) |
| 02 | [`02-phi-predecessor-mismatch.md`](02-phi-predecessor-mismatch.md) | `runtime/llvm/bcir_batch_executor.ll` | `5754354` | `llvm-as runtime/llvm/bcir_batch_executor.ll -o /dev/null` | Make phi incoming labels match the block's actual CFG predecessors. | [`00-foundations/02-ssa.md`](../00-foundations/02-ssa.md); [`05-control-flow/02-conditional-br.md`](../05-control-flow/02-conditional-br.md); [`05-control-flow/01-unconditional-br.md`](../05-control-flow/01-unconditional-br.md) |
| 03 | [`03-duplicate-block-labels.md`](03-duplicate-block-labels.md) | `runtime/llvm/bcir_claim_verify.ll` | `1f62e86` | `llvm-as runtime/llvm/bcir_claim_verify.ll -o /dev/null` | Delete the duplicate MMIO block trio, or otherwise give copied blocks and SSA values unique names. | [`00-foundations/02-ssa.md`](../00-foundations/02-ssa.md); [`01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md); [`02-phi-predecessor-mismatch.md`](02-phi-predecessor-mismatch.md) |
| 04 | [`04-duplicate-symbols.md`](04-duplicate-symbols.md) | `runtime/llvm/bcir_gem_seed.ll`; `runtime/llvm/bcir_worklist.ll` | `1f62e86` | `llvm-link runtime/llvm/bcir_gem_seed.ll runtime/llvm/bcir_worklist.ll -S -o /dev/null` | Keep one definition of `@execute_worklist` and make the other module declare it. | [`04-memory/03-global-variables.md`](../04-memory/03-global-variables.md); [`05-type-schema-drift.md`](05-type-schema-drift.md) |
| 05 | [`05-type-schema-drift.md`](05-type-schema-drift.md) | `runtime/llvm/bcir_registry_schema.ll`; `runtime/llvm/bcir_blob_schema.ll`; `runtime/llvm/bcir_blob_verify.ll` | `1f62e86` | `llvm-link runtime/llvm/bcir_registry_schema.ll runtime/llvm/bcir_blob_schema.ll runtime/llvm/bcir_blob_verify.ll -S -o /dev/null` | Align all named struct definitions to the canonical BCIR schema. | [`02-types/02-composite-types.md`](../02-types/02-composite-types.md); [`02-types/03-opaque-and-pointer-types.md`](../02-types/03-opaque-and-pointer-types.md); [`04-duplicate-symbols.md`](04-duplicate-symbols.md) |
| 06 | [`06-immarg-violation.md`](06-immarg-violation.md) | `runtime/llvm/bcir_prefetch_profiles.ll` | `1f62e86` | `opt -passes=verify runtime/llvm/bcir_prefetch_profiles.ll -o /dev/null` | Replace SSA operands to `immarg` intrinsic parameters with literal constants. | [`01-syntax/02-instruction-format.md`](../01-syntax/02-instruction-format.md); [`reference/intrinsics.md`](../reference/intrinsics.md); [`reference/instruction-quickref.md`](../reference/instruction-quickref.md) |
| 07 | [`07-debug-metadata-bloat.md`](07-debug-metadata-bloat.md) | Training-only exemplar; no affected BCIR `.ll` file recorded | Training-only / preventive | `opt -passes=verify <bcir-output-with-debug-metadata>.ll -o /dev/null` plus debug-info size checks | Uniquify common debug metadata and avoid emitting duplicate `DILocation` nodes for equivalent locations. | [`06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md); [`06-metadata/02-debug-info.md`](../06-metadata/02-debug-info.md); [`01-syntax/03-comments-metadata.md`](../01-syntax/03-comments-metadata.md) |
| 08 | [`08-stale-debug-locations.md`](08-stale-debug-locations.md) | Training-only exemplar; no affected BCIR `.ll` file recorded | Training-only / preventive | `opt -passes=verify <rewritten-bcir-output>.ll -o /dev/null` (usually passes; inspect `!dbg`) | Drop or recompute debug locations when cloned or moved instructions no longer match the original address/source span. | [`06-metadata/02-debug-info.md`](../06-metadata/02-debug-info.md); [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md); [`07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md) |
| 09 | [`09-atomic-ordering-mismatch.md`](09-atomic-ordering-mismatch.md) | `runtime/llvm/bcir_ops.ll`; `runtime/llvm/bcir_claim_verify.ll` | `f78ba96`; `08a0011` | `opt -passes=verify <bcir-atomics>.ll -o /dev/null` | Use legal `cmpxchg` failure orderings and preserve acquire/release semantics instead of defaulting to `monotonic`. | [`11-concurrency/01-atomic-orderings.md`](../11-concurrency/01-atomic-orderings.md); [`11-concurrency/02-atomic-instructions.md`](../11-concurrency/02-atomic-instructions.md); [`11-concurrency/03-volatile-vs-atomic.md`](../11-concurrency/03-volatile-vs-atomic.md) |
| 10 | [`10-volatile-is-not-atomic.md`](10-volatile-is-not-atomic.md) | `runtime/llvm/bcir_claim_verify.ll`; `runtime/llvm/bcir_ops.ll` | `08a0011`; `f78ba96` | `opt -passes=verify <bcir-volatile-or-atomic>.ll -o /dev/null` (semantic review still required) | Use `volatile` only for observable accesses/MMIO and model inter-thread synchronization with atomic operations and orderings. | [`11-concurrency/03-volatile-vs-atomic.md`](../11-concurrency/03-volatile-vs-atomic.md); [`11-concurrency/01-atomic-orderings.md`](../11-concurrency/01-atomic-orderings.md); [`04-memory/02-load-store.md`](../04-memory/02-load-store.md) |
| 11 | [`11-address-space-confusion.md`](11-address-space-confusion.md) | Training-only exemplar; no affected BCIR `.ll` file recorded | Training-only / preventive | `opt -passes=verify <bcir-address-space-output>.ll -o /dev/null` | Preserve pointer address spaces through loads, stores, GEPs, casts, and helper signatures. | [`04-memory/04-address-spaces.md`](../04-memory/04-address-spaces.md); [`02-types/03-opaque-and-pointer-types.md`](../02-types/03-opaque-and-pointer-types.md); [`04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) |
| 12 | [`12-vectorization-blocked-by-aliasing.md`](12-vectorization-blocked-by-aliasing.md) | Training-only exemplar; no affected BCIR `.ll` file recorded | Training-only / preventive | `opt -passes=loop-vectorize -pass-remarks-missed=loop-vectorize <bcir-loop>.ll -o /dev/null` | Preserve alias, alignment, and loop-shape facts so the vectorizer can prove memory reordering is safe. | [`09-vectorization/README.md`](../09-vectorization/README.md); [`07-optimization/02-common-analysis-passes.md`](../07-optimization/02-common-analysis-passes.md); [`04-memory/02-load-store.md`](../04-memory/02-load-store.md) |
| 13 | [`13-pass-pipeline-ordering-surprise.md`](13-pass-pipeline-ordering-surprise.md) | BCIR validation pipeline scripts: `runtime/llvm/validate_phase3.sh`; `runtime/llvm/validate_phase4.sh` | `08a0011` | `opt -passes=<bcir-analysis-pipeline> <bcir-module>.ll -o /dev/null` | Run BCIR analyses before destructive cleanup passes, and preserve required analyses between dependent passes. | [`07-optimization/01-pass-model.md`](../07-optimization/01-pass-model.md); [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md); [`07-optimization/04-optimization-levels.md`](../07-optimization/04-optimization-levels.md) |
| 14 | [`14-orc-jit-symbol-resolution.md`](14-orc-jit-symbol-resolution.md) | Training-only exemplar; no affected BCIR `.ll` file recorded | Training-only / preventive | `lli -jit-kind=orc-lazy <bcir-jit-module>.ll` or the project ORC harness lookup | Mangle symbols through ORC, install process/runtime symbol generators, and keep JITDylib search order explicit. | [`12-backend-jit/03-orc-jit.md`](../12-backend-jit/03-orc-jit.md); [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md); [`01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) |
| 15 | [`15-tablegen-generated-file-confusion.md`](15-tablegen-generated-file-confusion.md) | Training-only exemplar; no affected BCIR `.ll` file recorded | Training-only / preventive | `llvm-tblgen -I <llvm-include> -I <target-dir> -gen-instr-info <target>.td -o <build>/*GenInstrInfo.inc` | Regenerate TableGen `.inc` files from `.td` inputs in the build tree instead of editing or searching only source-tree outputs. | [`12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md); [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md); [`10-grammar/README.md`](../10-grammar/README.md) |
| 16 | [`16-sanitizer-instrumentation.md`](16-sanitizer-instrumentation.md) | Training-only exemplar; no affected BCIR `.ll` file recorded | Training-only / preventive | `opt -passes=verify <sanitized-bcir-output>.ll -o /dev/null` plus an execution test under the intended sanitizer runtime | Preserve sanitizer-inserted checks, shadow-memory traffic, redzone checks, and `!nosanitize` metadata unless a sanitizer-aware pass proves they are obsolete. | [`04-memory/02-load-store.md`](../04-memory/02-load-store.md); [`06-metadata/01-metadata-basics.md`](../06-metadata/01-metadata-basics.md); [`07-optimization/03-common-transform-passes.md`](../07-optimization/03-common-transform-passes.md) |

## How to use this list

When you write IR:
1. Skim the one-liner column.
2. If any look relevant to what you're writing, open that file.
3. Each pitfall page includes: the exact verifier message, a minimal
   reproducer, the fix pattern, and (where applicable) the BCIR
   commit that fixed the real instance.

When you debug failing IR:
1. Copy the `llvm-as` / `llvm-link` / `opt -passes=verify` error
   message.
2. Grep this directory for distinctive words from the message.
3. Each pitfall page documents the exact text the verifier emits.

## What this list *does not* cover

- **Optimization correctness bugs** (e.g., "my pass is producing
  poison"). Those are in pass-design territory.
- **Backend / codegen ICEs.** Use `llc -mtriple=... -O0` and report
  to LLVM.
- **Linker errors not caused by IR issues** (missing libraries,
  wrong target).

## Pattern recognition

Many of these bugs share a root cause: **a generator that synthesizes
IR algorithmically and doesn't validate against `llvm-as` after every
emit**.

If you write an IR generator (compiler, JIT, DSL frontend):

1. Pipe its output through `llvm-as -o /dev/null` in test.
2. Pipe through `opt -passes=verify -o /dev/null` to catch semantic
   bugs the assembler accepts.
3. Add the BCIR-style validate scripts (`runtime/llvm/validate_*.sh`
   in the parent repo) to CI.

Most of the pitfalls below would have been caught the first time the
generator was wired into CI.

## Adversarial review and fuzzing track

The [adversarial exercise track](../exercises/adversarial/) turns these pitfalls
into classified seeds. It explicitly separates verifier-valid semantic hazards,
expected-invalid inputs, target-specific cases, and metadata-preservation tests.
Use it when a bug can survive `llvm-as`, when reduction might delete the evidence,
or when a target-only example must stay out of portable smoke tests.
