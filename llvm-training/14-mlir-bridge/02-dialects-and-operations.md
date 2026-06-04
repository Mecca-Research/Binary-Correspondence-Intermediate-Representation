# 14.2 — Dialects and Operations

A dialect is a namespace plus a set of IR definitions: operations, types,
attributes, interfaces, canonicalization patterns, parsing/printing rules,
verification rules, and conversion hooks. Dialects are how MLIR supports many
levels of abstraction without forcing every frontend into one universal op set.

## Dialect design basics

Start by choosing the abstraction level the dialect owns:

| Question | Good dialect answer |
|---|---|
| What facts must survive optimization? | Make them operations, attributes, types, or interfaces rather than comments. |
| What invariants can be verified locally? | Put them in operation/type verifiers. |
| Which values are runtime values? | Use SSA operands/results. |
| Which values are compile-time facts? | Use attributes. |
| Which nested bodies need structured semantics? | Use regions. |
| Which control-flow edges need to be explicit? | Use successors or region terminators. |
| Which data should lower away? | Define conversion patterns to standard/LLVM dialects. |

A dialect should be high-level enough to expose useful invariants, but not so
high-level that every transform must understand the full source language.

## Operation anatomy

An MLIR operation has these major parts:

```text
%results = dialect.operation(%operands) <attributes> : type-signature
```

The generic operation model includes:

- **Name**: `arith.addi`, `memref.load`, `llvm.load`, `bcir.edge`, etc.
- **Operands**: SSA values consumed by the operation.
- **Results**: SSA values produced by the operation.
- **Attributes**: immutable compile-time facts such as names, flags, shapes,
  edge kinds, register classes, or lowering policies.
- **Regions**: nested IR bodies owned by the operation.
- **Successors**: block targets for branch-like operations.
- **Traits/interfaces**: reusable properties such as symbol behavior,
  terminator requirements, memory effects, or callable/function-like behavior.

## Attributes vs operands

A common design mistake is putting runtime data in attributes or compile-time
facts in operands.

Use an **attribute** when the value is known while compiling and should be part
of the operation's static semantics:

```mlir
%h = "bcir.ham_hint"(%vertex) {level = 2 : i32, policy = "prefetch"}
     : (!bcir.vertex) -> !bcir.hint
```

Use an **operand** when the value is computed at runtime:

```mlir
%weight = memref.load %weights[%edge_id] : memref<?xf32>
%edge2 = "bcir.edge.with_weight"(%edge, %weight)
         : (!bcir.edge, f32) -> !bcir.edge
```

If the fact can be symbolic at compile time but materialized at runtime later,
consider an attribute during high-level passes and a clear lowering rule that
turns it into constants, metadata, runtime calls, or explicit memory fields.

## Types and attributes

Types describe SSA values. Attributes describe immutable facts. Both can be
custom:

```mlir
!bcir.vertex<space = "claim", id_bits = 64>
!bcir.edge<src = "claim", dst = "blob">
#bcir.ham<distance = 2, confidence = 0.875 : f64>
```

In real TableGen/C++ definitions, these custom forms need parsers, printers,
and verifiers. A verifier should reject impossible combinations early: a
register binding attribute with an unknown register class, a Mixed Stride graph
with inconsistent rank/stride counts, or an edge operation whose source and
destination vertex spaces do not match its declared type.

## Regions and block arguments

Operations with regions are powerful because they can express structured bodies:

```mlir
"bcir.walk"(%root) ({
^bb0(%v: !bcir.vertex):
  %next = "bcir.neighbor"(%v) {edge_kind = "contains"}
          : (!bcir.vertex) -> !bcir.vertex
  "bcir.yield"(%next) : (!bcir.vertex) -> ()
}) : (!bcir.vertex) -> ()
```

The block argument `%v` represents a value supplied by the parent operation. In
lower-level control-flow dialects, block arguments also replace LLVM IR PHI
nodes until final LLVM IR translation.

## Dialect implementation checklist

For a production dialect, define at least:

1. **Operations** with clear operands, results, attributes, and regions.
2. **Types** for domain values that should not be confused with raw pointers or
   integers.
3. **Attributes** for stable compile-time facts.
4. **Verification** for local invariants.
5. **Canonicalization patterns** for obvious simplifications.
6. **Interfaces** for symbol lookup, memory effects, shape inference, or
   side-effect modeling where relevant.
7. **Lowering contracts** explaining which information is preserved as LLVM
   metadata, runtime ABI fields, constants, or discarded hints.

## See also

- [`examples/bcir-dialect-sketch.mlir`](examples/bcir-dialect-sketch.mlir) — illustrative BCIR custom-dialect sketch.
- [`04-bcir-as-custom-dialect.md`](04-bcir-as-custom-dialect.md) — BCIR placement guidance.
