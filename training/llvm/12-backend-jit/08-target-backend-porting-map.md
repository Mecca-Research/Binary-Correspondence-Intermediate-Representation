# Porting a Backend Target: The Complete Map

## What this chapter is

Chapters [`01`](01-codegen-pipeline.md)–[`07`](07-advanced-orc-runtime-integration.md)
are diagnostic guides: they help you read what an existing backend did. This one
is the **map of the work** a new backend target actually requires — the file
inventory, the order, the checkpoints, and the parts that dominate the schedule.

It is deliberately a map and not a manual. Porting a target is a
multi-engineer-year project whose details live in the LLVM source tree and in
your ISA manual, and a chapter that pretended otherwise would be the least
honest page in this corpus. What a map gives you is the thing that is genuinely
hard to get anywhere else: an accurate sense of the shape and the size, so the
decision to start is an informed one.

**This repository has decided not to build one.** BCIR is a planning,
verification, artifact, and runtime layer *above* resident toolchains, and the
GO/STOP criteria are written down in
[`../../docs/BCIR_NATIVE_OBJECT_GATE.md`](../../../docs/BCIR_NATIVE_OBJECT_GATE.md)
with a feasibility verdict in
[`../../docs/research/BCIR_NATIVE_BACKEND_FEASIBILITY.md`](../../../docs/research/BCIR_NATIVE_BACKEND_FEASIBILITY.md).
The map below is also the evidence for that decision.

## The inventory

A target lives in `llvm/lib/Target/<Name>/`. The pieces, grouped by what they
answer:

### What the machine is (TableGen)

| File | Describes |
| --- | --- |
| `<Name>.td` | the umbrella: pulls in everything below |
| `<Name>RegisterInfo.td` | registers, register classes, sub-registers, allocation order |
| `<Name>InstrInfo.td` | instructions: operands, encodings, patterns, scheduling class |
| `<Name>InstrFormats.td` | encoding formats shared across instruction groups |
| `<Name>CallingConv.td` | argument and return classification |
| `<Name>Schedule.td` | scheduling model: latencies, ports, resource cycles |
| `<Name>.td` subtargets | features, CPUs, and which features each CPU has |

The register file is where most first-time ports go wrong, and the mistake is
almost always the same: register *classes* are not sets of registers, they are
sets of registers **plus an allocation order plus a type constraint**. Getting
sub-register indices and register unit overlap right is a prerequisite for the
register allocator working at all, and errors surface as inexplicable
miscompiles far later.

### What C++ has to supply

| File | Supplies |
| --- | --- |
| `<Name>TargetMachine.cpp` | the target's entry point; builds the codegen pass pipeline |
| `<Name>ISelLowering.cpp` | how SelectionDAG nodes become target operations |
| `<Name>ISelDAGToDAG.cpp` | instruction selection beyond what patterns cover |
| `<Name>InstrInfo.cpp` | copies, spills, reloads, branch analysis |
| `<Name>RegisterInfo.cpp` | frame pointer, reserved registers, frame index elimination |
| `<Name>FrameLowering.cpp` | prologue, epilogue, stack layout, unwind |
| `<Name>AsmPrinter.cpp` | MachineInstr to MCInst, and assembly output |
| `<Name>Subtarget.cpp` | feature-dependent behaviour |
| `MCTargetDesc/*` | encoding, relocations, object writing, disassembly |
| `AsmParser/*`, `Disassembler/*` | optional, and needed sooner than expected |
| `TargetInfo/*` | registration boilerplate |

`ISelLowering` is where the schedule goes. Every type and operation the target
does not natively support needs an action — legal, promote, expand, or custom —
and the `custom` cases are hand-written lowering per operation. Then the same
questions return for vectors, then for the atomics model, then for the varargs
convention.

### The rest of the toolchain

A target that only exists inside `llc` is not usable. The full set:

- Clang driver and target support: predefined macros, type sizes, ABI info,
  built-ins, inline-assembly constraints.
- The assembler and disassembler (`AsmParser`, `Disassembler`), which are also
  your best test harness — round-tripping every instruction through
  assemble/disassemble catches encoding bugs nothing else does.
- Relocations, and the linker that resolves them (`lld` support, or an existing
  one that already knows the format).
- Unwind information, or exceptions and debuggers do not work.
- A runtime library (`compiler-rt`) for the operations the target expands into
  calls: 64-bit division on a 32-bit target, soft float, atomics.
- Debug info: DWARF register numbers, frame descriptions, and a debugger that
  can consume them.

## The order that works

1. **Get an empty target to register and build.** No instructions selected,
   nothing lowered. This alone shakes out the CMake and registration wiring.
2. **Registers and the simplest instructions.** Enough to select `add` on
   integers and return.
3. **Calling convention and frame lowering.** Now functions can call functions.
   This is the first point where anything is testable end to end.
4. **The MC layer: encoding, then disassembly.** Round-trip every instruction
   you have added. Do this early — retrofitting an encoding fix after
   instruction selection depends on it is far more expensive.
5. **The long tail of `ISelLowering`.** Every unsupported type and operation,
   one at a time, driven by test failures.
6. **The scheduling model.** Correctness does not depend on it; performance
   entirely does.
7. **Vectors, atomics, exceptions, debug info.** Each is a project.

The checkpoint discipline matters more than the order: after each step there
must be a runnable test that fails before and passes after. A backend developed
without that becomes a system where nothing works and nothing localizes.

## Testing

| Layer | Tool |
| --- | --- |
| Instruction selection | `llc` + `FileCheck` on `.ll` inputs |
| Machine-level passes | `llc -stop-after=` / `-start-before=` on `.mir` |
| Encoding | `llvm-mc -show-encoding`, and assemble/disassemble round-trips |
| Relocations and objects | `llvm-readobj`, `llvm-objdump` |
| End to end | an emulator or real hardware, running a real program |
| Differential | the same source through an established target, comparing behaviour |

MIR tests are the ones people discover late. Being able to write a `.mir` file
that starts at a specific point in the pipeline turns "the register allocator
did something strange" from an afternoon into a test case. See
[`../19-hardware-aware/README.md`](../19-hardware-aware/README.md).

## Where the schedule actually goes

Ranked by how much time they consume in practice, most first:

1. **The long tail of `ISelLowering`.** Not the common operations — the
   hundreds of type/operation combinations that need `Expand` or `Custom`.
2. **The ABI.** Every corner: structs by value, varargs, alignment, returns in
   memory, callee-saved registers across every path.
3. **Encoding correctness.** Only systematic round-trip testing finds these.
4. **Vectors.** Effectively a second instruction selector.
5. **The scheduling model.** Correct code that is three times too slow is not a
   finished target.
6. **Everything downstream:** linker, unwinder, debugger, runtime library.

The pattern is that the *interesting* work — selecting instructions for a new
ISA — is a small fraction of the total, and the ABI and encoding work that
dominates is neither novel nor optional.

## When a full target is the wrong answer

Before starting, check whether something narrower answers the question:

| Instead of a target | Consider |
| --- | --- |
| a new ISA broadly similar to an existing one | a subtarget/feature of that target |
| accelerator offload | intrinsics plus a runtime call boundary — [`06-custom-bcir-intrinsics.md`](06-custom-bcir-intrinsics.md) |
| a domain-specific accelerator | an MLIR dialect lowering to an existing target — [`../18-mlir-lowering-to-llvm/README.md`](../18-mlir-lowering-to-llvm/README.md) |
| custom encodings on an existing ISA | inline assembly, or an `MCTargetDesc` extension |
| experimentation and measurement | an emulator plus the existing toolchain |

This is the reasoning behind this repository's own decision. BCIR needs
*placement, cost, and legality* over hardware, and it obtains those by planning
above a resident toolchain — which is a bounded, testable amount of work with
none of the schedule above.

## Pitfalls

- **Starting with instruction selection.** Registration, registers, and the ABI
  come first, and skipping them makes everything after untestable.
- **Register classes without correct sub-register and unit overlap.** Allocator
  bugs that look like miscompiles.
- **Postponing the disassembler.** It is your encoding test.
- **Postponing the ABI.** It reaches into every later decision.
- **A scheduling model bolted on at the end.** Correct and slow is not done.
- **No `.mir` tests.** Machine-level bugs stop being localizable.
- **Assuming the frontend is free.** Clang target support is its own project.
- **Assuming `llc` output is a program.** Linker, unwinder, and runtime library
  are all still ahead.

## BCIR notes

- The native-object question here is governed by a written gate with GO and STOP
  criteria, not by drift. That is the general lesson worth taking: an expensive
  direction gets a written decision boundary *before* work starts, so that
  starting is a decision and not an accumulation of small steps.
- BCIR does own real native objects for a bounded slice, and the boundary of
  that slice is documented rather than implied. "We do not have a general
  instruction selector" is a stronger position than a half-built one, and it is
  only credible because the gate says what would change it.
- If the gate ever opens, this map is the estimate, and the ranked schedule
  above is the part to price first.

## See also

- [`01-codegen-pipeline.md`](01-codegen-pipeline.md) — the pipeline this map builds
- [`02-tablegen.md`](02-tablegen.md) — reading target descriptions
- [`../19-hardware-aware/README.md`](../19-hardware-aware/README.md) — MIR and machine-level inspection
- [`../../docs/BCIR_NATIVE_OBJECT_GATE.md`](../../../docs/BCIR_NATIVE_OBJECT_GATE.md) — the decision boundary
- [`../../docs/research/BCIR_NATIVE_BACKEND_FEASIBILITY.md`](../../../docs/research/BCIR_NATIVE_BACKEND_FEASIBILITY.md) — the feasibility verdict
