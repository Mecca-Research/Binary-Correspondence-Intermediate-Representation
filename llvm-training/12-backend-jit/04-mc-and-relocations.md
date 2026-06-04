# MC Layer and Relocations

LLVM IR names functions and globals symbolically. The backend must turn those
names into bytes, symbol table entries, and relocation records. The MC layer is
the target-independent framework that represents this late machine-code view.

## Pipeline position

A simplified path from IR to object code is:

```text
LLVM IR
  -> instruction selection
  -> MachineInstr
  -> register allocation / late machine passes
  -> MCInst
  -> MCStreamer / MCAssembler
  -> object bytes + symbols + relocations
```

`MachineInstr` still carries compiler-only concepts such as virtual registers,
frame indices, and target pseudo-instructions. `MCInst` is closer to what an
assembler sees: opcodes, physical registers, immediates, symbols, and fixups.

## MC concepts

| Concept | Job |
| --- | --- |
| `MCInst` | A lowered target instruction with operands suitable for encoding or printing. |
| `MCOperand` | Register, immediate, expression, or symbol operand inside an `MCInst`. |
| `MCExpr` | Symbolic expression such as `foo`, `foo+4`, or target-specific relocation syntax. |
| `MCFixup` | A placeholder saying some instruction or data bytes need a value later. |
| `MCStreamer` | Emits assembly or object records from instructions, labels, and directives. |
| `MCAssembler` | Lays out fragments, resolves local fixups, and writes object relocations. |

## What is a relocation?

A relocation says: "after layout or linking knows symbol `S`, patch these bytes
using relocation kind `K`." It records four things:

1. where the patch goes: section plus offset;
2. what symbol/expression it references;
3. which target relocation kind applies, such as absolute address or PC-relative
   branch displacement;
4. any addend used in the expression.

Local assembler-time differences can often be resolved immediately. References
to external symbols, inter-section addresses, and JIT symbols normally remain as
relocations for the object linker or JIT linker.

## Why a JIT cannot find a symbol

When ORC/LLJIT reports a missing symbol, the failure usually happens after IR was
successfully compiled. Common causes:

- the IR declaration name does not match the exported runtime symbol exactly;
- C++ name mangling was expected but the IR uses an unmangled name, or vice versa;
- the symbol exists in the host process but was not added to the JIT's symbol
  generator or dylib search order;
- the compiled object contains a relocation against a helper function that was
  optimized in, legalized, or emitted by the runtime library;
- visibility or platform prefix rules changed the final object symbol name.

Debug from the object layer outward: dump symbols, inspect relocations, then add
or rename the provider.

## Practical inspection commands

```bash
llc -filetype=obj llvm-training/12-backend-jit/examples/codegen-input.ll -o /tmp/codegen-input.o
llvm-nm /tmp/codegen-input.o
llvm-objdump -dr /tmp/codegen-input.o
llvm-readobj --relocations --symbols /tmp/codegen-input.o
```

Use `-filetype=asm` first when you only need readable assembly. Use object output
when the question is symbol binding, relocation kind, section placement, or JIT
linking.

## BCIR/JIT checklist

- Keep runtime helper declarations in generated IR synchronized with the actual
  exported helper names.
- For C++ helpers, expose `extern "C"` wrappers if the IR should call unmangled
  names.
- If a helper is expected from the host process, register the host-process symbol
  generator before adding modules that reference it.
- Dump object relocations when a JIT failure names a symbol but the IR looked
  valid; the relocation table is the backend's concrete dependency list.
- Remember that target lowering may introduce new dependencies, especially for
  libcalls, atomics, vector operations, and exception/runtime support.
