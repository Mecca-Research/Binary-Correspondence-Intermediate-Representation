# 14.4 — BCIR as a Custom Dialect

BCIR already has concepts that are richer than plain LLVM IR: Vertex-Edge-
Attribute structure, HAM hints, register binding, and Mixed Stride graphs. A
custom MLIR dialect can keep those concepts first-class while still providing a
clear path to LLVM IR.

This chapter is a design sketch, not a committed ABI. The important lesson is
where to place each kind of information so it survives long enough and lowers
predictably.

## Candidate dialect namespace

Use a short namespace such as `bcir`:

```mlir
%v = "bcir.vertex"() {id = 42 : i64, space = "claim"} : () -> !bcir.vertex
%e = "bcir.edge"(%v) {kind = "contains"} : (!bcir.vertex) -> !bcir.edge
%a = "bcir.attribute"(%e) {name = "weight", type = "f32"} : (!bcir.edge) -> !bcir.attr
```

In a real dialect, custom assembly forms could make this prettier, but the
generic form is good for design review because it exposes operands, attributes,
and result types directly.

## Vertex-Edge-Attribute placement

| BCIR concept | MLIR representation | Possible lowering |
|---|---|---|
| Vertex identity | `!bcir.vertex` type plus `bcir.vertex`/`bcir.vertex.lookup` ops. | Integer ID, pointer to vertex record, or runtime handle. |
| Edge identity/topology | `!bcir.edge` type and edge operations with source/destination operands. | Adjacency-list loads, runtime API calls, or descriptor fields. |
| Attributes | `bcir.attribute` ops for runtime attributes; MLIR attributes for compile-time labels. | Loads from attribute storage, constants, metadata, or runtime calls. |
| Graph space/schema | Module/function attributes or symbol ops. | Global descriptors, metadata, or runtime registration. |
| Validation facts | Verifier rules and canonicalization patterns. | Usually erased after checks or converted to runtime assertions. |

The key distinction is static vs dynamic. A vertex namespace such as `"claim"`
can be an MLIR attribute. A vertex ID loaded from input data must be an SSA
value.

## HAM hints

HAM hints are usually optimization guidance, not correctness semantics. Model
them as explicit operations or attributes so passes can inspect, merge, lower,
or drop them intentionally:

```mlir
%hint = "bcir.ham_hint"(%v) {
  distance = 2 : i32,
  policy = "prefetch",
  confidence = 0.875 : f64
} : (!bcir.vertex) -> !bcir.hint
```

Possible lowering targets:

- LLVM `llvm.prefetch` calls when the hint identifies a concrete address.
- LLVM loop/profile metadata when the hint describes control-flow likelihood.
- Runtime scheduling calls when the hint is consumed by a BCIR runtime.
- No code, if the target cannot use the hint and correctness is unchanged.

Document the fallback behavior. A hint ignored by the lowering pipeline must not
silently remove required synchronization, bounds checks, or data movement.

## Register binding

Register binding is more target-adjacent than graph topology. Keep it separate
from semantic graph operations:

```mlir
%bound = "bcir.bind_register"(%value) {
  reg_class = "gpr",
  preference = "r10",
  required = false
} : (i64) -> i64
```

Design choices:

- Use attributes for compile-time preferences such as register class or named
  physical-register preference.
- Use verifiers to reject impossible classes for the value type.
- Treat optional preferences as hints that can disappear before LLVM IR.
- Treat required bindings as ABI constraints that must lower to target-specific
  intrinsics, inline assembly constraints, calling-convention choices, or a
  backend feature—never as a vague metadata note.

LLVM IR has limited portable ways to demand a physical register. If correctness
requires a specific register, the design likely belongs near the backend ABI,
inline assembly, or target-specific lowering rather than a generic optimization
metadata attachment.

## Mixed Stride graphs

Mixed Stride graph layout can be a type/attribute pair when compile-time known,
or a descriptor value when dynamic:

```mlir
%g = "bcir.mixed_stride.graph"(%base) {
  rank = 3 : i32,
  static_strides = [1, 4, -1],
  layout = "soa"
} : (!llvm.ptr) -> !bcir.graph<rank = 3>
```

Suggested representation:

- Compile-time rank/layout: custom type parameters or attributes.
- Static strides: dense integer attributes.
- Dynamic strides: SSA operands, often carried in descriptor structs or memrefs.
- Bounds and ownership: attributes only if static; operands/runtime checks if
  input-dependent.

During lowering, Mixed Stride facts may become `memref` maps/descriptors, GEP
arithmetic, runtime descriptor fields, or metadata for analysis. The lowering
must define signedness, units (elements vs bytes), overflow behavior, and target
pointer-size interactions.

## Lowering strategy for BCIR

A practical BCIR-to-LLVM path could be:

1. **BCIR dialect**: graph operations, schema symbols, hints, register-binding
   preferences, Mixed Stride descriptors.
2. **Canonical BCIR**: normalized edge directions, folded static attributes,
   validated schema references.
3. **Structured dialects**: graph walks become `scf` loops or `cf` branches;
   attribute access becomes `memref` or runtime calls.
4. **LLVM-compatible dialects**: pointer arithmetic, descriptor structs,
   integer IDs, and concrete calls.
5. **LLVM dialect**: `llvm.func`, `llvm.load`, `llvm.call`, `llvm.br`, and
   `llvm.return`.
6. **LLVM IR**: `.ll`/bitcode consumed by existing LLVM tools.

## Pitfall checklist

Before lowering a BCIR dialect operation, ask:

- Is this fact semantic, or only a hint?
- If it is semantic, where is it represented after lowering?
- If it is a hint, is it safe to drop?
- Are vertex/edge/attribute IDs values or compile-time constants?
- Does the lowering preserve memory effects and ordering?
- Does register binding require target-specific handling?
- Are Mixed Stride units and overflow behavior specified?
- Does the LLVM ABI see the same struct/descriptor schema in every module?

## See also

- [`examples/bcir-dialect-sketch.mlir`](examples/bcir-dialect-sketch.mlir) — source-level dialect sketch.
- [`05-vertex-graph-lowering.md`](05-vertex-graph-lowering.md) — end-to-end vertex/edge graph lowering with register-binding and metadata survival notes.
- [`03-lowering-to-llvm-dialect.md`](03-lowering-to-llvm-dialect.md) — lowering mechanics and common pitfalls.
- [`../08-pitfalls/05-type-schema-drift.md`](../08-pitfalls/05-type-schema-drift.md) — low-level schema drift after lowering.
- [`../08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md) — prefetch-hint lowering hazards.
