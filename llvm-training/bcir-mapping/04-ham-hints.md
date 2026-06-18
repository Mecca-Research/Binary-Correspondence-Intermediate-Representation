# HAM Hints and Memory Guidance

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


HAM hints are BCIR-side guidance for heterogeneous, hierarchical, or
high-affinity memory behavior. In LLVM IR they should lower to concrete memory
attributes, metadata, prefetch calls, address spaces, phase/barrier calls, or
runtime profile records rather than informal comments.

## BCIR-level meaning

- A hint describes intended memory behavior: locality, prefetch distance, hazard
  domain, lane, memory domain, or access phase.
- Hints may be advisory, but they must not change the correctness contract unless
  the runtime ABI has an explicit verifier and execution rule for them.
- Hazard and memory-domain hints help group graph/claim execution and protect
  ordering-sensitive resources.
- HAM hints can attach to claims, batches, prefetch profiles, or runtime resource
  descriptors.

## Likely LLVM IR representation

- Use `llvm.prefetch` for explicit locality/prefetch guidance.
- Use named structs such as prefetch profiles for ABI-visible scheduling hints.
- Use atomic operations and fences for synchronization; do not encode ordering
  with comments or plain metadata only.
- Use address spaces when the target/runtime really has distinct memory spaces.
- Use metadata only when an LLVM pass or BCIR-specific consumer is prepared to
  interpret it; otherwise retain the canonical information in structs or calls.

## Relevant runtime ABI structs/functions

- `%bcir.prefetch.profile` stores
  prefetch-related schedule information.
- `@bcir.op.prefetch.linear` and
  `@bcir.op.prefetch.strided`
  wrap `llvm.prefetch` for linear and strided access patterns.
- `%bcir.claim` contains hazard-domain
  and control fields used by hint-aware scheduling.
- `%bcir.res` can carry resource
  domain and bounds-like values for memory placement decisions.
- `@bcir.op.barrier`,
  `@bcir.op.atomic.add.i32`, and
  `@bcir.op.cmpxchg.i32` encode real ordering
  behavior when a hint crosses into correctness.
- Existing examples: `runtime/llvm/bcir_prefetch_profiles.ll`,
  `runtime/llvm/bcir_examples_phase3.ll`,
  and [`llvm-training/exercises/014-ham-hint-metadata.prompt.md`](../exercises/014-ham-hint-metadata.prompt.md).

## Verifier risks

- Intrinsics with immediate-only operands require literal constants, not decoded
  runtime hint values.
- Atomic ordering pairs must be legal for the instruction, especially `cmpxchg`
  success and failure orderings.
- Address-space casts must use valid operations for the source and destination
  pointer spaces.
- Metadata must be syntactically well formed and attached only where LLVM allows
  that metadata kind.

## Optimization risks

- Advisory prefetches may be dropped, moved, or ignored by a backend; do not rely
  on them for correctness.
- Plain metadata usually does not prevent reordering; use calls, atomics, fences,
  or memory effects when ordering matters.
- Overstating alignment, no-alias, or inbounds facts for a hinted access can
  enable invalid transformations.
- Pass-pipeline order can determine whether hint metadata is preserved long
  enough for a custom pass to consume it.

## Pitfall links

- [`06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md)
- [`09-atomic-ordering-mismatch.md`](../08-pitfalls/09-atomic-ordering-mismatch.md)
- [`10-volatile-is-not-atomic.md`](../08-pitfalls/10-volatile-is-not-atomic.md)
- [`11-address-space-confusion.md`](../08-pitfalls/11-address-space-confusion.md)
- [`12-vectorization-blocked-by-aliasing.md`](../08-pitfalls/12-vectorization-blocked-by-aliasing.md)
- [`13-pass-pipeline-ordering-surprise.md`](../08-pitfalls/13-pass-pipeline-ordering-surprise.md)
