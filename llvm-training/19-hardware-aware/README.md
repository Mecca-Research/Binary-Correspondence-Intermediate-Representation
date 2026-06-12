# Hardware-Aware LLVM Lowering

This chapter explains where hardware knowledge belongs as a BCIR or other
high-level operation moves through LLVM. The goal is not to make generic LLVM IR
pretend to be a machine description. The goal is to preserve exactly the
information that later layers need, then hand each decision to the lowest layer
that can implement it correctly.

The chapter uses **Dragon Egg** as a taxonomy for hardware-facing operations and
**GAADMSF** for graph-aware data movement, scheduling, programming pulses, and
flow execution. These are training models, not new semantics in generic LLVM IR.
The executable BCIR oracle and MLIR dialect law remain separate from this
training corpus.

## Lessons

1. [`01-dragon-egg-gaadmsf-intrinsics.md`](01-dragon-egg-gaadmsf-intrinsics.md)
   classifies operations and chooses runtime calls, custom intrinsics, or
   advisory metadata.
2. [`02-programming-pulses-and-flow-execution.md`](02-programming-pulses-and-flow-execution.md)
   models GAADMSF programming pulses and Dragon Egg flow execution.
3. [`03-calibration-governor.md`](03-calibration-governor.md) separates explicit
   calibration state from advisory governor policy.
4. [`04-riscv-and-target-specific-codegen.md`](04-riscv-and-target-specific-codegen.md)
   follows a RISC-V extension operation into target-specific lowering.
5. [`05-machineir-and-mir-customization.md`](05-machineir-and-mir-customization.md)
   identifies MIR as the backend customization layer after instruction selection.
6. [`06-register-allocation-and-memory-hints.md`](06-register-allocation-and-memory-hints.md)
   covers GAA-aware register-allocation hints and hierarchical memory hints.

## Key takeaways

- Keep portable value, memory, and control-flow semantics in ordinary LLVM IR.
- Use a runtime call for opaque device control or queue activity unless the
  backend must inspect the operation before call lowering.
- Use a registered intrinsic or target pseudo when instruction selection,
  legalization, register banks, immediate fields, hazards, or scheduling need a
  first-class operation.
- Use metadata only for advice that may be ignored without changing correctness.
- Represent calibration and governor values explicitly when execution depends on
  them; metadata may annotate policy but cannot secretly supply required state.
- LLVM SSA names such as `%pulse` and `%acc` are values, not physical registers.
- MIR is target-specific compiler state, not verifier-safe LLVM IR.
- Always retain a runtime or portable lowering when a target-specific operation
  must run on machines without the extension.

## Hardware abstraction layers

| Layer | Owns | Appropriate representation |
|---|---|---|
| BCIR/MLIR source law | Domain operations, graph topology, claims, diagnostics | Typed dialect operations and attributes |
| LLVM IR | Portable values, addresses, control flow, ABI boundaries | Instructions, calls, registered intrinsics, ignorable metadata |
| Target lowering | Legalization and target feature selection | Target intrinsics, SelectionDAG/GlobalISel patterns, target pseudos |
| Machine IR (MIR) | Register classes/banks, machine operands, scheduling, spills | Machine instructions, virtual registers, regalloc hints |
| MC/object/runtime | Encodings, relocations, queues, device protocols | MC instructions, object records, runtime calls and side tables |

A decision should move downward when it requires information unavailable at the
higher layer. For example, generic IR can express that a flow consumes a buffer,
but only the backend knows whether a RISC-V extension instruction exists and
which register class its accumulator requires.

## Decision guide

### When to use LLVM intrinsics

Use a registered generic, target-specific, or project-specific intrinsic when:

- the operation must survive as one node until instruction selection;
- operands have target register-class or register-bank constraints;
- an immediate mode field must be checked before encoding;
- legalization, hazard recognition, or machine scheduling needs the operation;
- optimizers need declared memory effects that are more precise than an opaque
  external call.

A custom `llvm.bcir.*` spelling in this corpus is a design sketch until it is
registered in LLVM TableGen and given target lowering. A target-specific
intrinsic is not portable merely because it uses valid LLVM syntax.

### When to use metadata

Use metadata for non-semantic advice such as:

- preferred calibration profile or confidence;
- desired graph-affinity group;
- cache/reuse/placement preference;
- spill cost or coalescing preference;
- diagnostics that explain the origin of a hardware policy.

If dropping the metadata would change the program's required result, memory
ordering, device command, or synchronization behavior, metadata is the wrong
representation. Use operands, control flow, an intrinsic, or a runtime call.

### When backend or MIR customization is needed

Customize the backend and possibly MIR when correctness or quality depends on:

- a new instruction encoding or target extension;
- custom legal types, register classes, subregisters, or register banks;
- pseudo expansion after register allocation;
- target hazards, itinerary/resource models, or packet/bundle formation;
- fixed physical-register constraints;
- GAA-aware coalescing, allocation priority, spill placement, or rematerialization;
- memory hierarchy decisions that require frame indices, stack slots, or final
  machine addressing modes.

IR metadata alone cannot force a physical register or guarantee a cache level.
A backend pass may consume an IR hint, but the enforceable decision is made in
MachineIR/MIR and must remain correct if the hint cannot be honored.

## Examples and verification boundary

- [`examples/gaadmsf-programming-pulse.ll`](examples/gaadmsf-programming-pulse.ll)
- [`examples/dragon-egg-flow-execution.ll`](examples/dragon-egg-flow-execution.ll)
- [`examples/calibration-governor-metadata.ll`](examples/calibration-governor-metadata.ll)
- [`examples/riscv-extension-lowering-sketch.ll`](examples/riscv-extension-lowering-sketch.ll)
- [`examples/mir-register-hint-sketch.mir.txt`](examples/mir-register-hint-sketch.mir.txt)

The `.ll` files are assembly/verifier examples. They are intentionally excluded
from portable backend smoke lowering because their calls describe custom or
target-specific contracts. The `.mir.txt` file is a review sketch: do not pass it
to `llvm-as`, and do not assume it is accepted by `llc -run-pass` without being
regenerated for the exact LLVM version and target.

## Pitfall checklist

- **Encoding target policy as generic IR semantics:** keep generic IR honest;
  isolate target choices behind lowering or advisory annotations.
- **Using metadata when codegen needs a real intrinsic:** metadata can disappear
  and cannot impose instruction-selection semantics.
- **Assuming target-specific intrinsics are portable:** feature-check and provide
  fallback lowering.
- **Confusing LLVM IR register names with physical registers:** `%x` is an SSA
  value; allocation happens later.
- **Treating MIR examples as verifier-safe LLVM IR:** MIR has a different parser,
  target context, and version-sensitive schema.

## Cross-links

- [`../bcir-mapping/07-gaadmsf-operations.md`](../bcir-mapping/07-gaadmsf-operations.md)
- [`../bcir-mapping/08-dragon-egg-operations.md`](../bcir-mapping/08-dragon-egg-operations.md)
- [`../bcir-mapping/03-mixed-stride-graphs.md`](../bcir-mapping/03-mixed-stride-graphs.md)
- [`../12-backend-jit/`](../12-backend-jit/)
- [`../reference/intrinsics-quickref.md`](../reference/intrinsics-quickref.md)
- [`../indexes/bcir-patterns.md`](../indexes/bcir-patterns.md)
