# Vertex, Edge, and Attribute Lowering

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


BCIR graph-shaped data often starts as vertices connected by edges with
attribute payloads. LLVM IR has no graph primitive, so the lowering must choose
an explicit memory and ABI shape.

## BCIR-level meaning

- **Vertex**: a stable node identity, usually backed by a resource ID, claim ID,
  or array index.
- **Edge**: a relation from one vertex to another, often represented as an
  adjacency row, packed pair, or edge-list entry.
- **Attribute**: payload attached to a vertex or edge, such as weight, opcode,
  lane, hazard-domain flags, provenance hash, or cost fields.
- **Claim link**: claims can be viewed as graph vertices whose read/write
  resource-ID arrays define dependency edges.

## Likely LLVM IR representation

- Use arrays of named structs for vertex/edge rows when the shape is stable:
  `%bcir.vertex = type { i32, i32, i64 }` and `%bcir.edge = type { i32, i32, i64 }`.
- Use `getelementptr inbounds` to select a vertex, edge, or field; use separate
  `load` instructions before arithmetic or comparisons.
- Keep payload fields in integer types with explicit extension/truncation at ABI
  boundaries.
- Use module-level globals for small static examples and pointer arguments for
  runtime-owned graph storage.
- See [`examples/vertex-edge-attribute.ll`](examples/vertex-edge-attribute.ll)
  for a minimal edge-weight lookup that assembles as a standalone module.

## Relevant runtime ABI structs/functions

- `%bcir.claim` can encode graph-like
  vertices through packed control bits, read resource IDs, write resource IDs,
  hazard domain, and immediates.
- `@bcir.claim.rd` and
  `@bcir.claim.wr` expose read and
  write resource edges from a claim.
- `%bcir.res` records resource
  identity, domain, base pointer, and bounds-like fields.
- `@bcir.registry.lookup` turns a resource
  ID into a resource table entry.
- `@bcir.gem.execute_claim` is the seed
  executor that interprets one graph/claim node against the registry table.
- Existing examples: `runtime/llvm/bcir_examples_phase3.ll`
  for claim/batch/phase globals, `runtime/llvm/bcir_examples_worklist.ll`
  for worklist-style claim arrays, and `runtime/llvm/bcir_examples_phase4_generated.ll`
  for generated resources and batches.

## Verifier risks

- Nested instruction expressions are invalid; compute GEPs, loads, compares, and
  arithmetic as separate named SSA instructions.
- Struct field indexes must match the exact named struct layout in the module.
- PHI nodes in graph walks must list exactly one incoming value per predecessor.
- Duplicate helper names collide at link time when many generated graph modules
  are combined.

## Optimization risks

- Alias ambiguity between vertex arrays, edge arrays, and attribute arrays can
  block vectorization or load-store motion.
- Incorrect `inbounds` on a GEP can turn out-of-range graph traversals into
  poison and enable unsound transforms.
- Losing stable vertex/edge identity during CSE or inlining can make diagnostics
  and provenance mapping harder.
- Debug metadata copied from source graph construction can become stale after
  lowering or scheduling rewrites.

## Pitfall links

- [`01-nested-instruction-expressions.md`](../08-pitfalls/01-nested-instruction-expressions.md)
- [`02-phi-predecessor-mismatch.md`](../08-pitfalls/02-phi-predecessor-mismatch.md)
- [`04-duplicate-symbols.md`](../08-pitfalls/04-duplicate-symbols.md)
- [`05-type-schema-drift.md`](../08-pitfalls/05-type-schema-drift.md)
- [`07-debug-metadata-bloat.md`](../08-pitfalls/07-debug-metadata-bloat.md)
- [`08-stale-debug-locations.md`](../08-pitfalls/08-stale-debug-locations.md)
- [`12-vectorization-blocked-by-aliasing.md`](../08-pitfalls/12-vectorization-blocked-by-aliasing.md)
