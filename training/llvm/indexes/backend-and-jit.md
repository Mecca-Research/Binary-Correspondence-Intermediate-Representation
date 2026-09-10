# Index: Backend code generation, JIT, and object emission

Use this index when you know what you are trying to *do* to a program after the
optimizer is finished — select instructions, describe a target, emit objects,
compile at run time, deploy a kernel — but not which chapter covers it.

Concepts are phrased as the task, not as the chapter title, because that is how
you arrive at them: you know the problem, not the page.

| Concept | Where it is covered |
|---|---|
| Turning optimized IR into machine instructions | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) |
| Instruction selection, scheduling, and register allocation order | [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) |
| Describing a machine to the compiler as data | [`12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md) |
| Writing and reading `.td` target descriptions | [`12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md) |
| Confusing a generated file with a source file | [`08-pitfalls/15-tablegen-generated-file-confusion.md`](../08-pitfalls/15-tablegen-generated-file-confusion.md) |
| Compiling and running code at run time | [`12-backend-jit/03-orc-jit.md`](../12-backend-jit/03-orc-jit.md) |
| Standing up a minimal just-in-time compiler | [`12-backend-jit/03-orc-jit.md`](../12-backend-jit/03-orc-jit.md), [`12-backend-jit/examples/lljit-outline.cpp.md`](../12-backend-jit/examples/lljit-outline.cpp.md) |
| Assembling, relaxing, and relocating emitted code | [`12-backend-jit/04-mc-and-relocations.md`](../12-backend-jit/04-mc-and-relocations.md) |
| Why a symbol resolves at link time but not at run time | [`12-backend-jit/04-mc-and-relocations.md`](../12-backend-jit/04-mc-and-relocations.md), [`12-backend-jit/05-orc-layers.md`](../12-backend-jit/05-orc-layers.md) |
| Layering symbol lookup and lazy materialization | [`12-backend-jit/05-orc-layers.md`](../12-backend-jit/05-orc-layers.md) |
| Transforming IR on its way through a JIT | [`12-backend-jit/examples/orc-irtransformlayer-bcir.cpp.md`](../12-backend-jit/examples/orc-irtransformlayer-bcir.cpp.md) |
| Producing a symbol's definition on demand | [`12-backend-jit/examples/orc-materialization-unit-gaadmsf.cpp.md`](../12-backend-jit/examples/orc-materialization-unit-gaadmsf.cpp.md) |
| Teaching a backend an operation the target does not have | [`12-backend-jit/06-custom-bcir-intrinsics.md`](../12-backend-jit/06-custom-bcir-intrinsics.md) |
| Running BCIR kernels under a live ORC runtime | [`12-backend-jit/07-advanced-orc-runtime-integration.md`](../12-backend-jit/07-advanced-orc-runtime-integration.md) |
| Shipping compiled kernels to a running process | [`12-backend-jit/examples/dynamic-kernel-deployment-sketch.md`](../12-backend-jit/examples/dynamic-kernel-deployment-sketch.md) |
| Executing generated code on another machine or device | [`12-backend-jit/examples/remote-jitlink-heterogeneous-sketch.md`](../12-backend-jit/examples/remote-jitlink-heterogeneous-sketch.md) |
| What it costs to support a new processor family | [`12-backend-jit/08-target-backend-porting-map.md`](../12-backend-jit/08-target-backend-porting-map.md) |
| Deciding whether a target port is worth starting | [`12-backend-jit/08-target-backend-porting-map.md`](../12-backend-jit/08-target-backend-porting-map.md) |
