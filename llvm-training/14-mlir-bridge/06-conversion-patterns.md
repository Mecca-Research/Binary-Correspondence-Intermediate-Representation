# 14.6 — Conversion Patterns for BCIR Operations

Conversion patterns are the mechanical bridge from one legal dialect set to the
next. In MLIR terms, a pass declares which operations are legal, which are
illegal, and which rewrite patterns can replace illegal operations with legal
ones. For BCIR, patterns should be small, auditable, and explicit about what
semantic information survives.

## Pattern shape

A BCIR conversion pattern usually does four things:

1. Match a high-level operation such as `bcir.edge.lookup`, `bcir.attribute.load`,
   `bcir.ham_hint`, or `bcir.bind_register`.
2. Ask the type converter for the lowered operand/result types.
3. Emit lower-level operations: `arith`, `memref`, `cf`, `func`, `LLVM`, or a
   runtime ABI call.
4. Replace the original operation and attach diagnostics if information is being
   intentionally dropped.

For documentation and review, write each pattern as a contract:

```text
bcir.attribute.load(edge, "weight" : f32)
  requires: edge has a descriptor-backed attribute table or runtime ABI symbol
  rewrites to: memref.load descriptor.weight_table[edge.ordinal]
           or: call @bcir_attr_f32(graph, edge_id, attr_key)
  preserves: attribute value, schema key, memory effects
  drops: source-level attribute name if attr_key is a numeric ABI enum
```

## Direct descriptor pattern

Use a descriptor pattern when the graph layout is fixed enough for generated IR
to address fields directly:

```mlir
// Source sketch.
%child = "bcir.edge.target"(%edge) : (!bcir.edge<src = "claim", dst = "blob">) -> !bcir.vertex<space = "blob", id_bits = 64>
%weight = "bcir.attribute"(%edge) {name = "weight"} : (!bcir.edge<src = "claim", dst = "blob">) -> f32
```

A lowering may rewrite that into descriptor arithmetic:

```mlir
%edge_ordinal = arith.index_cast %edge_id : i64 to index
%target = memref.load %targets[%edge_ordinal] : memref<?xi64>
%weight = memref.load %weights[%edge_ordinal] : memref<?xf32>
```

The benefit is optimization visibility: ordinary MLIR and LLVM passes can reason
about loads, indices, and control flow. The cost is ABI/schema coupling.

## Runtime-call pattern

Use a runtime-call pattern when storage, validation, concurrency, or scheduling
is runtime-owned:

```mlir
%weight = "bcir.attribute"(%edge) {name = "weight", storage = "runtime"}
  : (!bcir.edge<src = "claim", dst = "blob">) -> f32
```

Lower to an explicit call boundary:

```mlir
%key = llvm.mlir.constant(7 : i32) : i32
%weight = llvm.call @bcir_get_edge_attr_f32(%graph, %edge_id, %key)
  : (!llvm.ptr, i64, i32) -> f32
```

This preserves correctness while hiding storage layout from LLVM. The call must
carry accurate memory effects and ABI attributes so later optimization does not
reorder it unsafely.

## What survives lowering: edge topology

| Stage | Representation | What survives | What may be lost |
|---|---|---|---|
| BCIR dialect | `bcir.edge(src, dst)` with `kind`, `directed`, and typed endpoints | Source/destination spaces, direction, edge kind, verifier checks | Nothing if operands and attrs are present |
| Canonical BCIR | normalized direction and resolved edge-kind symbol | Stable topology and schema key | Source spelling and aliases such as `contains` vs canonical enum |
| Mid-level MLIR | adjacency arrays, ordinals, cursors, or runtime calls | Reachability and endpoint IDs | Custom edge type and region-level graph context |
| LLVM dialect | GEP/load sequence or ABI call | Executable endpoint lookup | Dialect verifier knowledge of valid topology |
| LLVM IR | pointer arithmetic, loads, branches, calls | Runtime topology behavior | Source graph intent unless metadata/debug records it |

## What survives lowering: HAM hints

| Stage | Representation | What survives | What may be lost |
|---|---|---|---|
| BCIR dialect | `bcir.ham_hint` op with policy, distance, confidence | Full hint vocabulary and target-independent intent | Nothing if verifier accepts only known policies |
| Canonical BCIR | merged/deduplicated hints | Strongest applicable policy and confidence | Duplicates and dominated hints |
| Mid-level MLIR | prefetchable address calculation, scheduling op, or annotation | Chosen lowering intent | Hints unsupported by target/runtime |
| LLVM dialect | `llvm.prefetch`, metadata, or runtime call | Concrete prefetch/schedule side effect when emitted | Abstract HAM model |
| LLVM IR | `llvm.prefetch`, call, or metadata | Backend-visible hint or runtime request | Any hint safely ignored by lowering |

## Register-binding pattern

Register binding needs special care because generic LLVM IR does not guarantee a
portable physical-register assignment. Treat optional bindings as preferences and
required bindings as ABI/backend constraints.

```text
bcir.bind_register(value, reg_class="gpr", preference="r10", required=false)
  optional rewrite: return value unchanged and emit a remark
  target rewrite: target intrinsic, inline-asm constraint, calling convention, or backend hook
```

A required binding pattern must fail conversion if no legal target-specific
lowering exists. Silently dropping a required binding changes semantics.

## Pattern review checklist

- Is the replacement legal in the target dialect set?
- Are memory effects and ordering equivalent?
- Does the pattern preserve diagnostic locations and useful operation names?
- Is every dropped hint optional by definition?
- Does the pattern fail loudly for required register/resource constraints?
- Are runtime-call declarations centralized so ABI drift is visible?

## See also

- [`05-type-conversion.md`](05-type-conversion.md) — type-carrier decisions used by patterns.
- [`08-diagnostics-and-verification.md`](08-diagnostics-and-verification.md) — how conversion failures should report lost facts.
- [`examples/bcir-lowered-llvm-dialect.mlir`](examples/bcir-lowered-llvm-dialect.mlir) — runtime-call-oriented LLVM-dialect lowering.
- [`../exercises/033-lower-mlir-graph-op-to-llvm-dialect.prompt.md`](../exercises/033-lower-mlir-graph-op-to-llvm-dialect.prompt.md) — graph-op lowering exercise.
