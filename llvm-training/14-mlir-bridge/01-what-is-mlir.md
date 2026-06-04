# 14.1 — What is MLIR?

MLIR (Multi-Level Intermediate Representation) is LLVM's infrastructure for
building **many interoperable IRs** instead of one fixed IR. It is useful when a
frontend wants to preserve structured source facts—loops, tensors, graph nodes,
layout constraints, or domain annotations—before those facts are gradually
lowered to LLVM IR and target machine code.

LLVM IR is a low-level, SSA, control-flow-and-memory IR. MLIR is a framework for
SSA IRs at multiple abstraction levels. An MLIR program can contain high-level
operations such as `linalg.matmul`, mid-level control-flow operations such as
`scf.for`, low-level memory operations such as `memref.load`, and operations in
the `llvm` dialect that closely model LLVM IR.

## Core IR hierarchy

The most important MLIR container concepts are:

| Concept | What it means | LLVM IR analogy |
|---|---|---|
| **Module** | A top-level operation, usually `builtin.module`, that owns nested IR. | LLVM module, but represented as an operation. |
| **Operation** | The universal unit of MLIR IR: it has a name, operands, results, attributes, regions, and optional successors. | Instruction, terminator, declaration, global, or even module-like construct depending on dialect. |
| **Region** | A nested list of blocks owned by an operation. Regions encode bodies of functions, loops, conditionals, graph scopes, or symbol tables. | Function body or nested control-flow body; LLVM IR has no general nested regions. |
| **Block** | A sequence of operations with zero or more block arguments. Blocks may be connected by branch-like terminators. | Basic block plus PHI-like values represented as block arguments. |
| **Attribute** | Compile-time metadata attached to operations, types, or symbols. Attributes are immutable values such as strings, integers, arrays, dictionaries, affine maps, and dialect-specific records. | LLVM constants, metadata, function attributes, and instruction flags, depending on use. |
| **Type** | The type of an SSA value, attribute payload, or symbol interface. Types may be builtin (`i32`, `index`) or dialect-specific (`memref<...>`, `!llvm.ptr`, `!bcir.vertex`). | LLVM scalar/vector/aggregate/pointer types, plus many higher-level type systems. |

## One syntax, many dialects

MLIR's textual syntax always follows the same structural model, but operation
semantics come from dialects. For example:

```mlir
module {
  func.func @add_one(%x: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %y = arith.addi %x, %c1 : i32
    return %y : i32
  }
}
```

This snippet uses:

- `builtin.module` through the shorthand `module { ... }`.
- `func.func` for a function-like operation.
- `arith.constant` and `arith.addi` for integer operations.
- `return`, whose printed form is a shorthand for the `func.return` operation.

The syntax is uniform, but the verifier, parser, printer, canonicalizations,
and lowering rules are owned by the dialects.

## Regions and blocks preserve structure

MLIR can represent nested control flow directly:

```mlir
scf.if %cond {
  %v = arith.addi %a, %b : i32
  scf.yield %v : i32
} else {
  %v = arith.subi %a, %b : i32
  scf.yield %v : i32
} : i32
```

That structure is more explicit than a flattened LLVM IR CFG. Lowering later
creates the necessary blocks, branches, and PHI-equivalent block arguments.
Preserving structure longer lets analyses reason about loops, tensor shapes,
graph topology, and domain-specific invariants before low-level details obscure
them.

## Why use MLIR in an LLVM training repo?

MLIR is not a replacement for LLVM IR. It is often the **bridge** between a rich
frontend representation and LLVM IR. A compiler may use MLIR to:

1. Parse or import source/domain IR into high-level dialects.
2. Normalize and optimize while source structure remains visible.
3. Progressively lower through standard dialects such as `scf`, `cf`, `arith`,
   `memref`, and `func`.
4. Convert into the `llvm` dialect.
5. Translate the `llvm` dialect to LLVM IR bitcode/text or continue into LLVM's
   optimization and code-generation pipeline.

For BCIR, MLIR is attractive because graph and binding concepts can remain
first-class operations and attributes until the compiler has enough target and
runtime information to lower them safely.

## See also

- [`02-dialects-and-operations.md`](02-dialects-and-operations.md) — dialect and operation design basics.
- [`03-lowering-to-llvm-dialect.md`](03-lowering-to-llvm-dialect.md) — lowering pipelines and the LLVM dialect.
- [`04-bcir-as-custom-dialect.md`](04-bcir-as-custom-dialect.md) — sketching BCIR concepts as a custom dialect.
