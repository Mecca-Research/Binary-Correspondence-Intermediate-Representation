# Vertex graph lowering through MLIR and LLVM IR

BCIR vertex/edge graphs are a good fit for MLIR because the source dialect can
carry graph topology, register-binding intent, and diagnostic hints as explicit
operations and attributes before the lowering pass chooses an ABI shape. The key
review question is not "can every BCIR fact survive as metadata?" but "which
facts are semantic values, which become runtime table fields, and which are safe
to drop?"

The worked examples for this lesson are:

- [`examples/bcir-vertex-graph.mlir`](examples/bcir-vertex-graph.mlir) — BCIR
  source-dialect sketch.
- [`examples/bcir-vertex-graph-lowered-llvm-dialect.mlir`](examples/bcir-vertex-graph-lowered-llvm-dialect.mlir)
  — LLVM-dialect MLIR shape before translation to LLVM IR.
- [`examples/bcir-vertex-graph-lowered.ll`](examples/bcir-vertex-graph-lowered.ll)
  — verifier-checkable textual LLVM IR.

For the direct LLVM-IR mapping rules behind this MLIR example, see
[`../bcir-mapping/01-vertex-edge-attribute.md`](../bcir-mapping/01-vertex-edge-attribute.md),
[`../bcir-mapping/02-register-binding.md`](../bcir-mapping/02-register-binding.md),
and
[`../bcir-mapping/10-metadata-and-diagnostics.md`](../bcir-mapping/10-metadata-and-diagnostics.md).

## Source dialect facts

The source MLIR example models a three-vertex graph with two edges. It keeps
these BCIR facts distinct:

- **Vertex IDs** are semantic graph identities. They start as `id` attributes on
  `bcir.vertex` operations and as the `vertex_ids` graph catalog.
- **Edge lists** are semantic topology. They start as `bcir.edge` operations and
  an `edge_list` graph catalog.
- **Register binding** is a lowering constraint or preference. The example uses
  `required = false`, so the binding can become a runtime binding slot and the
  named physical register preference can disappear.
- **Metadata hints** are diagnostic/profiling facts. The example records a
  source rule and a hotness hint; these may survive as non-semantic metadata,
  named metadata, or side catalogs, but correctness cannot depend on them.

## LLVM-dialect boundary

A graph-lowering pass normally rewrites custom BCIR types into LLVM-compatible
values:

```text
!bcir.vertex       -> integer ID fields or descriptor pointers
!bcir.edge         -> rows in an edge table or explicit src/dst values
!bcir.graph        -> descriptor pointer, globals, or ABI struct
register binding   -> table indexes, call arguments, target-specific hooks, or nothing when optional
metadata hints      -> custom metadata/diagnostic side records, or dropped when non-semantic
```

In the LLVM-dialect example, vertex and edge data are represented as integer
values that stand in for loads from lowered ABI tables. The optional register
preference no longer demands `%r10`; it is represented as a small binding-slot
integer that a runtime or later target-specific lowering may interpret. Comments
mark where a real translation may attach non-semantic LLVM metadata or write a
diagnostic side table.

## Textual LLVM IR boundary

The final `.ll` example makes the survival rules concrete:

- Vertex IDs survive as fields in `@bcir.vertex.table` and are loaded with
  ordinary `getelementptr` and `load` instructions.
- The edge list survives as `@bcir.edge.table`; each edge row stores source ID,
  destination ID, and an attribute payload.
- The register binding survives only as an integer binding slot. The optional
  physical-register preference intentionally lowers away because portable LLVM IR
  does not promise a named hardware register for an ordinary SSA value.
- Metadata hints survive as `!bcir.vertex`, `!bcir.edge`, and `!bcir.hint`
  attachments plus named metadata catalogs. Optimizers may ignore or drop this
  metadata, so all executable behavior is still represented by values, memory,
  and calls.

## Review checklist

When reviewing a BCIR graph-to-LLVM lowering, require an explicit answer for each
source fact:

1. Is the fact semantic, advisory, or diagnostic?
2. If semantic, where is the corresponding LLVM-dialect or LLVM IR value?
3. If advisory, what target/runtime mechanism consumes it, and is it safe to
   drop?
4. If diagnostic, what metadata or side catalog carries it, and what happens if
   optimization drops it?
5. Does register binding require an ABI or target-specific mechanism, or is it
   only a preference?
6. Are vertex/edge IDs stable after canonicalization, inlining, and table
   compaction?

## Exercises

Use this lesson before:

- [`../exercises/033-lower-mlir-graph-op-to-llvm-dialect.prompt.md`](../exercises/033-lower-mlir-graph-op-to-llvm-dialect.prompt.md)
  for a graph-load lowering review.
- [`../exercises/034-review-mlir-to-llvm-type-conversion.prompt.md`](../exercises/034-review-mlir-to-llvm-type-conversion.prompt.md)
  for type-conversion hazard review.
