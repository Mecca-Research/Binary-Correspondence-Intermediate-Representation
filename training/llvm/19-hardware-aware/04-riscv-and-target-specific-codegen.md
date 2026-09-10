# RISC-V Extensions and Target-Specific Codegen

A RISC-V extension lowering is a pipeline, not just an intrinsic name:

```text
BCIR/MLIR op
  -> target-independent LLVM form or target-gated intrinsic
  -> legalization and feature check
  -> SelectionDAG/GlobalISel pattern
  -> target pseudo or machine instruction
  -> register allocation and scheduling
  -> MC encoding and object attributes
```

## Target-specific intrinsic boundary

[`examples/riscv-extension-lowering-sketch.ll`](examples/riscv-extension-lowering-sketch.ll)
uses a custom `llvm.bcir.riscv.*` declaration to keep a pulse/accumulate operation
visible. It is intentionally a design sketch, not an upstream RISC-V intrinsic.
A real implementation must:

- register the intrinsic and its overloaded types/attributes in TableGen;
- gate lowering on the exact RISC-V extension and ABI;
- validate immediate operands;
- define legal scalar/vector widths and register classes;
- select a pseudo or instruction with scheduling and hazard information;
- emit the extension in target attributes/object metadata as required;
- provide a runtime or portable expansion for targets without the extension.

Never guess the spelling or signature of an upstream `llvm.riscv.*` intrinsic.
Check the LLVM version's `IntrinsicsRISCV.td`, generated headers, tests, and
LangRef/target documentation.

## Feature dispatch

Choose feature policy explicitly:

- **AOT fixed target:** compile with the required extension and reject mismatches.
- **Multiversioning:** produce a baseline implementation and an extension-tuned
  implementation selected by a resolver or runtime feature test.
- **JIT:** query the host/target machine before retaining the intrinsic; otherwise
  rewrite to the fallback ABI before object emission.

An IR target triple does not prove that the extension is enabled. Target feature
attributes and the configured `TargetMachine` must agree.

## What belongs below LLVM IR

Instruction encoding, tied operands, early-clobber behavior, register classes,
subregister constraints, itineraries, processor resources, and pseudo expansion
belong in the target backend. Metadata may guide selection but cannot stand in
for those definitions.

## Pitfalls

- Target-specific intrinsics are not portable across architectures or LLVM
  versions.
- A syntactically valid declaration may still have no lowering.
- Generic optimization can alter operands before selection; intrinsic properties
  must describe what transformations are legal.
- Hard-coding a target policy into generic arithmetic can make fallback behavior
  incorrect or prevent other backends from recognizing the operation.

See [`../13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md)
and [`../12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md).
