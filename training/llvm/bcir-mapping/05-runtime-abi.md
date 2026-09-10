# Runtime ABI Mapping

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/kernel/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


The runtime ABI is the boundary where BCIR semantic records become concrete LLVM
IR structs, globals, declarations, and calls. Keep this layer boring and stable:
most verifier and linker failures come from ABI drift rather than from complex
IR instructions.

## BCIR-level meaning

- A claim describes one operation with packed control, read/write resources,
  hazard-domain information, and immediates.
- A registry maps resource IDs to runtime resource descriptors.
- A worklist, batch, phase range, or stream pack selects the order in which
  claims execute.
- Blob records describe serialized layouts and views used to move BCIR data
  across process or compiler boundaries.
- Execution context records carry mutable phase, epoch, barrier, and runtime
  state that should not be confused with graph payloads.

## Likely LLVM IR representation

- Use named LLVM struct types for ABI records and keep a single canonical schema
  per module family.
- Use declarations for ABI functions shared across modules and definitions only
  in the module that owns the implementation.
- Use `getelementptr` with constant struct field indexes for schema fields and
  variable indexes for array elements.
- Use globals for static test fixtures and pointer arguments for runtime-owned
  data.
- Keep target triple and datalayout choices explicit when target lowering starts
  to matter; the current seed uses unknown/empty values for portable IR tests.

## Relevant runtime ABI structs/functions

- Schema records: `%bcir.claim`,
  `%bcir.execctx`,
  `%bcir.costvec.q16`,
  `%bcir.res`,
  `%bcir.exe`,
  `%bcir.wl`,
  `%bcir.blob.header`, and
  `%bcir.blob.view`.
- Schedule records: `%bcir.phase.range`,
  `%bcir.batch`,
  `%bcir.layout.profile`,
  `%bcir.prefetch.profile`,
  `%bcir.tile.profile`, and
  `%bcir.stream.pack`.
- Accessors and verifiers: `runtime/llvm/bcir_claim_accessors.ll`,
  `runtime/llvm/bcir_claim_verify.ll`,
  `runtime/llvm/bcir_batch_verify.ll`,
  `runtime/llvm/bcir_blob_verify.ll`, and
  `runtime/llvm/bcir_phase_epoch.ll`.
- Executors: `runtime/llvm/bcir_gem_seed.ll`,
  `runtime/llvm/bcir_worklist.ll`,
  `runtime/llvm/bcir_phase_worklist.ll`,
  and `runtime/llvm/bcir_batch_executor.ll`.
- Existing examples: `runtime/llvm/bcir_master_reference_v2.ll`,
  `runtime/llvm/bcir_examples.ll`,
  `runtime/llvm/bcir_examples_phase3.ll`,
  and `runtime/llvm/bcir_examples_phase4_generated.ll`.

## Verifier risks

- Type schema drift is the primary ABI failure: the same struct name must not
  mean different field layouts in linked modules.
- Duplicate definitions of ABI helpers fail at link time; keep shared helpers as
  declarations except in the owning runtime module.
- Calling conventions, return types, and pointer/address-space types must match
  exactly at every call site.
- Packed bitfield accessors must use legal shifts, masks, truncations, and
  extensions; do not rely on prose field descriptions alone.
- Runtime loops over worklists, batches, or phases need valid PHIs and structured
  exits.

## Optimization risks

- ABI structs are ordinary memory; optimization can reorder accesses unless
  calls, atomics, fences, or alias information impose the required constraints.
- Inlining helper wrappers may expose layout details and make later schema
  changes harder to audit.
- Dead-code elimination can remove advisory hooks if they have no modeled side
  effects.
- ABI-visible globals with weak/linkonce linkage need careful symbol-resolution
  expectations when JITing or linking multiple modules.

## Pitfall links

- [`02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md)
- [`04-duplicate-symbols.md`](../08-pitfalls/04-duplicate-symbols.md)
- [`05-type-schema-drift.md`](../08-pitfalls/05-type-schema-drift.md)
- [`09-atomic-ordering-mismatch.md`](../08-pitfalls/09-atomic-ordering-mismatch.md)
- [`10-volatile-is-not-atomic.md`](../08-pitfalls/10-volatile-is-not-atomic.md)
- [`11-address-space-confusion.md`](../08-pitfalls/11-address-space-confusion.md)
- [`13-pass-pipeline-ordering-surprise.md`](../08-pitfalls/13-pass-pipeline-ordering-surprise.md)
- [`14-orc-jit-symbol-resolution.md`](../08-pitfalls/14-orc-jit-symbol-resolution.md)
