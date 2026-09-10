# MachineIR and MIR as the Backend Customization Layer

After instruction selection, LLVM represents target instructions and virtual
registers in MachineIR. MIR is a textual serialization used by backend tests and
`llc` pass pipelines. It is the right review layer when a requirement depends on
machine operands, register classes, frame indices, scheduling, or pseudo
instructions.

## Typical customization points

- **Target pseudo-instruction:** preserve a Dragon Egg/GAADMSF operation while
  later passes know its complete machine-level shape.
- **Register-bank/class constraints:** require accumulator, vector, predicate, or
  special-purpose registers.
- **Instruction scheduling:** model latency, resources, hazards, and bundles.
- **Pre-RA expansion:** expand a pseudo when virtual-register freedom is useful.
- **Post-RA expansion:** expand only after physical registers and spill decisions
  are known.
- **Machine function pass:** consume GAA affinity or memory-placement hints and
  translate them into target-specific priorities or annotations.

## Reading the MIR sketch

[`examples/mir-register-hint-sketch.mir.txt`](examples/mir-register-hint-sketch.mir.txt)
illustrates virtual registers, a target pseudo, and comments showing where a
GAA-aware allocation hint might be consumed. The `.mir.txt` suffix is deliberate:
this artifact is explanatory and is not guaranteed to match any installed LLVM
version's MIR schema or a real target's opcode/register-class names.

Real MIR tests should be generated from the exact `llc` build, minimized, and
run with a command such as:

```bash
llc -mtriple=<target> -run-pass=<machine-pass> input.mir -o -
```

Do not run a MIR file through `llvm-as` or `opt`. Conversely, LLVM IR `%names`
are SSA values and cannot be used to demand physical registers.

## When an IR pass is insufficient

Move the customization to the backend when it needs:

- selected opcodes or machine operands;
- live intervals and interference;
- spill weights, stack slots, or rematerialization decisions;
- physical-register aliases and reserved registers;
- final addressing modes or instruction packet constraints.

An earlier IR/MLIR pass may preserve intent with a custom intrinsic or metadata.
The backend pass must define a conservative fallback if the hint cannot be
honored and must not turn a performance preference into accidental correctness.

## MIR stability warning

MIR is intentionally close to LLVM backend internals. Opcodes, register classes,
properties, YAML fields, and pass preconditions can change. Treat prose sketches
as design aids and checked MIR tests as version-pinned compiler tests, not as a
portable interchange format.
