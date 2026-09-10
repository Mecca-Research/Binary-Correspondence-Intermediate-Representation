# GAA-Aware Register Allocation and Hierarchical Memory Hints

This lesson uses **GAA** as a graph-affinity/allocation hint domain: values in the
same affinity group are likely to participate in the same GAADMSF flow or Dragon
Egg machine operation. GAA hints are optimization inputs, not generic LLVM
semantics.

## GAA-aware register-allocation hints

Useful hints may describe:

- affinity groups whose values should coalesce or remain in compatible banks;
- preferred accumulator/vector/predicate banks;
- values that are expensive to spill across a programming-pulse/flow boundary;
- rematerializable setup values;
- anti-affinity when simultaneous operands must occupy distinct registers;
- live-range phase boundaries where splitting is cheap.

At LLVM IR level, attach only advisory group IDs/costs or preserve a custom
intrinsic that makes operand roles visible. A backend pass can map those hints to
virtual-register allocation order, coalescing preferences, spill weights, bank
selection, or target pseudos. It must obey correctness constraints first and may
ignore hints under pressure.

Do not claim that `%gaa_accumulator` is a physical accumulator register. The name
is diagnostic only. Fixed-register requirements belong in instruction
constraints, calling conventions, inline-asm constraints, or target lowering.

## Hierarchical memory hints

A hierarchical memory schema can state:

- logical level: near-device SRAM, L1, L2, LLC, NUMA node, HBM, or host memory;
- role: graph index, edge payload, accumulator, pulse table, or output;
- reuse scope and expected lifetime;
- stride, stream direction, and temporal/non-temporal preference;
- placement confidence and fallback;
- prefetch distance or writeback phase.

These are hints unless the address space or runtime allocation API gives them
real semantics. A backend may translate them to prefetches, cache-control
instructions, address spaces, section placement, runtime allocation flags, or
nothing. Unknown metadata must be safely ignorable.

Mixed-stride graph lowering should still compute correct byte addresses without
the hints. See
[`../bcir-mapping/03-mixed-stride-graphs.md`](../bcir-mapping/03-mixed-stride-graphs.md).

## End-to-end hint lifecycle

1. BCIR/MLIR assigns stable affinity and memory-policy attributes.
2. LLVM lowering attaches versioned metadata or intrinsic operands.
3. Optimization passes preserve, merge, or deliberately invalidate hints.
4. Target lowering maps surviving intent to MachineIR properties.
5. Regalloc and machine scheduling apply hints subject to legality/pressure.
6. Diagnostics report honored, degraded, or ignored policy.

Without steps 3 and 6, hints tend to become stale folklore. Test both the
hint-present path and the fallback after metadata stripping.

## Pitfalls

- Encoding a required placement as ignorable metadata.
- Inflating spill weights so aggressively that allocation quality regresses.
- Applying graph affinity before checking register-bank compatibility.
- Treating cache levels as uniform across targets.
- Letting metadata refer to SSA values or operations that a transform deleted.
