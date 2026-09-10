# Backend Code Generation Pipeline

LLVM IR is target-independent enough for optimizers to reason about program
semantics, but a CPU or object file needs target-specific instructions,
registers, calling conventions, relocations, and encodings. LLVM's backend
bridges that gap: it lowers optimized IR into a target-specific machine-code
representation, applies machine-level optimizations, assigns physical
resources, and emits assembly or object bytes.

Official reference: [LLVM Target-Independent Code Generator](https://llvm.org/docs/CodeGenerator.html).

## Big picture: IR to machine code

A simplified `llc`-style pipeline looks like this:

```text
LLVM IR Module
  ↓ IR legalization / target lowering setup
Instruction selection
  ↓
MachineInstr in virtual registers, usually machine SSA
  ↓
Machine-code SSA optimizations and scheduling/formation
  ↓
Register allocation and spill code insertion
  ↓
Prolog/epilog insertion and frame finalization
  ↓
Late machine-code optimization
  ↓
MC layer: MCInst, fixups, relaxation, object/assembly emission
```

The exact pass list is target- and optimization-level-dependent. Targets also
hook into many stages with lowering rules, register-bank information,
instruction itineraries, scheduling models, calling-convention code, and
assembler/object-file definitions.

## Instruction selection

Instruction selection maps LLVM IR operations to target instructions or target
pseudo-instructions. It answers questions such as:

- Which instruction implements `add i32` on this target?
- Does a load folded into an arithmetic operation form one target instruction?
- Which addressing modes are legal?
- How are calls, returns, atomics, vector operations, and intrinsics lowered?

LLVM has two major instruction-selection frameworks:

- **SelectionDAG**: the mature selector. IR is lowered to a directed acyclic
  graph of operations, legalized for the target, combined, and selected into
  target-specific nodes/instructions. Many production targets rely heavily on
  SelectionDAG.
- **GlobalISel**: a newer selector that starts from generic machine
  instructions, legalizes them, maps operands to register banks/classes, and
  selects target instructions. It is designed to be more global and easier to
  reason about in some target/backend workflows.

Both paths eventually produce **MachineInstr** objects. `MachineInstr` is the
central backend instruction representation: it can describe real target
instructions, pseudo-instructions, virtual-register operands, physical-register
operands, frame indices, target flags, and debug locations.

## Scheduling and formation

Before physical registers are fixed, LLVM may shape and order machine
instructions to expose target-friendly structure:

- combine target instructions and remove redundant copies;
- form target idioms or pseudo-instructions that will expand later;
- schedule instructions to reduce stalls, respect dependencies, and use issue
  resources well;
- prepare instructions for register allocation by controlling live ranges and
  copy placement.

Some scheduling occurs before register allocation, and some can occur after
register allocation. The target's scheduling model and itinerary data, often
produced from TableGen descriptions, guide these decisions.

## Machine-code SSA optimizations

LLVM IR is in SSA form, and much of the early machine-code representation also
uses SSA-like virtual registers: each virtual register is defined once, and use
lists can be analyzed precisely. This enables machine-level optimizations that
are too target-specific for the IR optimizer, for example:

- machine common subexpression elimination;
- copy propagation and coalescing preparation;
- dead machine-instruction elimination;
- peephole combines using target instruction semantics;
- branch folding and block placement decisions informed by target costs.

These passes operate on `MachineFunction`, `MachineBasicBlock`, and
`MachineInstr`, not on `Function`, `BasicBlock`, and `Instruction` IR objects.

## Register allocation

Register allocation maps virtual registers to physical registers. It must obey:

- register classes and banks;
- calling-convention clobbers and callee-saved rules;
- instruction operand constraints;
- live-range interference;
- target-specific reserved registers;
- spill and reload costs.

If there are not enough physical registers, the allocator inserts spills and
reloads to stack slots. After this point, many virtual-register SSA properties
are gone, and passes must reason about physical registers, aliases, subregisters,
and stack frame layout.

## Prolog/epilog insertion

Prolog/epilog insertion finalizes function entry and exit mechanics:

- create or adjust the stack frame;
- save and restore callee-saved registers;
- materialize frame-pointer or stack-pointer adjustments;
- resolve abstract frame indices to concrete stack references;
- handle target-specific unwind, red-zone, alignment, and calling-convention
  details.

This step depends on register allocation results because it must know which
callee-saved registers were actually used and which stack objects survived.

## Late machine-code optimization

Late passes clean up after register allocation and frame finalization. Common
examples include:

- peephole optimizations on physical-register instructions;
- branch relaxation or branch shortening;
- copy elimination where physical registers make copies redundant;
- target-specific pseudo-instruction expansion;
- block layout and alignment adjustments;
- hazard recognizers for targets with pipeline restrictions.

Late optimizations are powerful but constrained: they must preserve the concrete
physical-register and stack-frame decisions already made.

## Code emission and the MC layer

The final backend representation is lowered into the **MC layer**. The MC layer
contains target-independent abstractions for assembly and object emission:

- `MCInst`: a compact, lower-level instruction form suitable for printing or
  encoding;
- `MCExpr`, symbols, fixups, and relocations;
- assembler printers and object writers;
- instruction encoders and disassemblers;
- streamers that write textual assembly or binary object data.

In short: `MachineInstr` is for backend analysis and transformation;
`MCInst`/MC objects are for final assembly/object encoding.

## Varargs and ABI variance

C-style variadic functions look target-independent in the function type, but
accessing their unnamed arguments is not target-independent. The `va_arg`
instruction is lowered according to the target ABI: the backend must know how
that ABI represents `va_list`, where register-save areas live, how stack and
register arguments are classified, and how each extracted type advances the
cursor. That means two targets can accept similar-looking IR while requiring
different machine-code sequences and different frontend setup around the
`va_list` object.

A minimal verifier-safe shape is:

```llvm
declare void @llvm.va_start(ptr)
declare void @llvm.va_end(ptr)

define i32 @take_one_i32(i32 %tag, ...) {
entry:
  %ap = alloca ptr, align 8
  call void @llvm.va_start(ptr %ap)
  %value = va_arg ptr %ap, i32
  call void @llvm.va_end(ptr %ap)
  ret i32 %value
}
```

Treat examples like this as syntax and verifier guidance, not as permission to
synthesize portable varargs lowering by hand. Agents should preserve
frontend-produced varargs IR unless they intentionally model the selected
target ABI. In particular, avoid rewriting `llvm.va_start`, `llvm.va_end`,
`va_arg`, or target-shaped `va_list` storage during generic IR cleanup unless
you know the caller/callee ABI contract that code generation will apply.

For related IR syntax, see the `va_arg` entry in
[`../reference/instruction-quickref.md`](../reference/instruction-quickref.md)
and the variadic function notes in
[`../01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md).

## BCIR hardware-aware hooks

BCIR hardware-aware operations should enter the backend in a form that preserves
what instruction selection and register allocation need to know. For example, a
mixed-stride GEM tile can compute ordinary LLVM byte offsets for A, B, and C,
load register-oriented vector fragments, and then call a custom intrinsic-shaped
hook such as `@llvm.bcir.gem.mixed.stride.v4f32`. A BCIR-aware backend can select
that hook to a target pseudo-instruction; a generic JIT can rewrite it to a
runtime ABI call before codegen.

Keep these checkpoints distinct:

- IR optimization may simplify stride math and address calculations.
- Instruction selection owns the custom intrinsic or pseudo-instruction mapping.
- Register allocation owns physical placement of vector/tile fragments.
- ORC owns the policy decision to keep the intrinsic, rewrite it to a runtime
  symbol, or reject the module with a diagnostic.

See [`06-custom-bcir-intrinsics.md`](06-custom-bcir-intrinsics.md),
[`examples/custom-bcir-intrinsic-jit.ll`](examples/custom-bcir-intrinsic-jit.ll),
and [`../bcir-mapping/examples/hardware-aware-gem-lowering.ll`](../bcir-mapping/examples/hardware-aware-gem-lowering.ll)
for the full hardware-aware example.

## Where the names fit

| Term | Fits here | Mental model |
|---|---|---|
| SelectionDAG | Instruction selection and legalization | Mature graph-based IR-to-instruction selector |
| GlobalISel | Instruction selection and legalization | Generic-MachineInstr selector pipeline |
| MachineInstr | Main machine-code optimization representation | Target instruction or pseudo with virtual/physical operands |
| MC layer | Final emission | Assembly/object-level instruction encoding, fixups, relocations |

## See also

- [LLVM Target-Independent Code Generator](https://llvm.org/docs/CodeGenerator.html)
- [Writing an LLVM Backend](https://llvm.org/docs/WritingAnLLVMBackend.html)
- [`02-tablegen.md`](02-tablegen.md) for the generated target data that feeds many backend stages
- [`03-orc-jit.md`](03-orc-jit.md) for using code generation from a JIT entry point
- [`06-custom-bcir-intrinsics.md`](06-custom-bcir-intrinsics.md) for hardware-aware BCIR custom intrinsic and fallback policies
