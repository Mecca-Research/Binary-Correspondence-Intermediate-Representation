# 14.9 — BCIR MLIR End-to-End Walkthrough

This walkthrough follows one BCIR graph fragment from a custom MLIR dialect
sketch to canonical BCIR, LLVM dialect, and textual LLVM IR. The files are
illustrative training artifacts rather than checked MLIR tests, but the final
`.ll` is intentionally feasible LLVM IR and is covered by the repository example
verifier.

## Step 1: source-level BCIR dialect

Start with [`examples/bcir-dialect-source-sketch.mlir`](examples/bcir-dialect-source-sketch.mlir).
It keeps BCIR facts first-class:

- `%root` and `%child` have custom vertex types.
- The edge has source/destination spaces and a semantic kind.
- The `weight` attribute is named before lowering chooses a numeric ABI key.
- The HAM hint records policy, distance, and confidence.
- The register binding is optional, so lowering may keep it only when a target
  supports it.
- The runtime call is still expressed with BCIR-shaped operands.

This is the best stage for schema diagnostics and graph-aware canonicalization.

## Step 2: canonical BCIR

[`examples/bcir-canonicalized.mlir`](examples/bcir-canonicalized.mlir) shows the
same fragment after source cleanup:

- Edge direction has been normalized.
- Attribute and edge names have been resolved to stable keys.
- The HAM hint has been canonicalized into target-independent fields.
- The optional register binding is preserved as a preference but marked safe to
  drop.
- Runtime ABI intent is explicit enough for conversion patterns.

Canonical BCIR should still be readable by domain reviewers. If a graph error is
found here, diagnostics can still refer to vertices, edges, attributes, and
resource bindings directly.

## Step 3: lowered LLVM dialect

[`examples/bcir-lowered-llvm-dialect.mlir`](examples/bcir-lowered-llvm-dialect.mlir)
shows a runtime-backed lowering. The graph has become an opaque runtime pointer,
IDs and attribute keys are concrete integers, and BCIR operations have become
`llvm.call` operations:

- `@bcir_lookup_child` preserves edge traversal through the runtime ABI.
- `@bcir_prefetch_vertex` is the chosen HAM-hint lowering.
- `@bcir_get_edge_attr_f32` preserves the attribute value lookup.
- The optional register binding is represented only by a diagnostic comment in
  this documentation sketch; there is no portable LLVM-dialect physical-register
  guarantee.
- `@bcir_consume_weighted_child` is the runtime boundary that consumes the
  lowered graph result.

At this point, MLIR conversion legality should say there are no remaining BCIR
operations.

## Step 4: final textual LLVM IR

[`examples/bcir-final.ll`](examples/bcir-final.ll) is the textual LLVM IR shape:

```llvm
%child = call ptr @bcir_lookup_child(ptr %graph, i64 0, i32 3)
call void @bcir_prefetch_vertex(ptr %child, i32 2)
%weight = call float @bcir_get_edge_attr_f32(ptr %graph, i64 0, i32 7)
%status = call i32 @bcir_consume_weighted_child(ptr %graph, ptr %child, float %weight)
ret i32 %status
```

The `.ll` no longer has custom BCIR types, but it still has executable carriers
for the facts that matter at runtime: graph handle, vertex child handle, edge
kind key, attribute key, HAM prefetch request, and ABI calls.

## End-to-end survival summary

| BCIR fact | Source dialect | Canonical BCIR | LLVM dialect | Textual LLVM IR |
|---|---|---|---|---|
| Vertex identity | custom vertex type and ID attrs | fixed `i64` root ID plus graph space | `i64` ID and `!llvm.ptr` child handle | `i64` and `ptr` call operands/results |
| Edge topology | `bcir.edge` with `kind = "contains"` | `edge_kind = 3` and normalized direction | `@bcir_lookup_child(..., i32 3)` | call to `@bcir_lookup_child` |
| Attributes | `bcir.attribute` named `weight` | `attr_key = 7`, result `f32` | `@bcir_get_edge_attr_f32` | call returning `float` |
| HAM hints | `bcir.ham_hint` policy/distance/confidence | canonical prefetch distance | `@bcir_prefetch_vertex` | call to prefetch runtime helper |
| Register/resource binding | optional `bcir.bind_register` | optional preference remains documented | dropped as non-semantic preference | no portable LLVM IR artifact |
| Runtime ABI calls | symbolic BCIR consume op | resolved ABI intent | `llvm.func`/`llvm.call` | `declare`/`call` |

## Review questions

When reviewing a BCIR-to-MLIR bridge, ask:

1. Which source facts are semantic and must survive to runtime behavior?
2. Which facts are hints and may disappear with a remark?
3. Which custom types are erased, and what values replace them?
4. Which ABI calls become the stable contract between generated code and the
   BCIR runtime?
5. Which verifier catches schema drift before final `.ll` is emitted?

## See also

- [`05-type-conversion.md`](05-type-conversion.md) — type conversion survival tables.
- [`06-conversion-patterns.md`](06-conversion-patterns.md) — operation-level rewrite contracts.
- [`08-diagnostics-and-verification.md`](08-diagnostics-and-verification.md) — verification workflow.
- [`../exercises/039-lower-bcir-mlir-end-to-end.prompt.md`](../exercises/039-lower-bcir-mlir-end-to-end.prompt.md) — end-to-end review exercise.
