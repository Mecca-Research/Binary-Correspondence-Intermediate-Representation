# TableGen for Backend Descriptions

TableGen is LLVM's domain-specific record-description language. LLVM uses it to
keep large families of related facts declarative: registers, instructions,
selection patterns, calling-convention fragments, scheduling models, diagnostic
metadata, attributes, and more.

Official references:

- [TableGen Overview](https://llvm.org/docs/TableGen/)
- [TableGen Programmer's Reference](https://llvm.org/docs/TableGen/ProgRef.html)
- [`tblgen` command guide](https://llvm.org/docs/CommandGuide/tblgen.html)

## Why LLVM uses TableGen

A backend contains many tables with repeated structure:

- instruction names, encodings, operands, assembly strings, and flags;
- register names, aliases, subregisters, and register classes;
- instruction-selection patterns that map IR-like operations to target
  instructions;
- scheduling resources, latencies, itineraries, and processor variants.

Writing all of this directly in C++ would be repetitive and error-prone.
TableGen lets target authors write compact `.td` records, then generate C++
include fragments and other data files consumed by the backend.

TableGen does **not** replace backend C++. It describes data and patterns that
backend C++ interprets, specializes, and extends.

## Basic `.td` syntax

### `class`

A `class` defines a reusable record template. Parameters make it easy to define
families of related records.

```tablegen
class ToyReg<string asmName, int encoding> {
  string AsmName = asmName;
  int Enc = encoding;
}
```

### `def`

A `def` creates a concrete record, optionally inheriting from one or more
classes.

```tablegen
def R0 : ToyReg<"r0", 0>;
def R1 : ToyReg<"r1", 1>;
```

### `let`

`let` overrides or sets fields for records. It can apply to a single record or a
block of records.

```tablegen
class ToyInst<string mnemonic> {
  string Mnemonic = mnemonic;
  bit isCommutable = 0;
}

let isCommutable = 1 in
def ADDrr : ToyInst<"add">;
```

### `multiclass` and `defm`

A `multiclass` defines a family of records. `defm` instantiates that family with
a shared prefix and parameters. Use it when a target has repeated instruction
forms such as register-register, register-immediate, and memory variants.

```tablegen
multiclass BinaryOp<string mnemonic> {
  def rr : ToyInst<mnemonic # "rr">;
  def ri : ToyInst<mnemonic # "ri">;
}

defm ADD : BinaryOp<"add">; // creates ADDrr and ADDri-like records
```

The exact generated names depend on the record suffixes and target conventions.
Keep naming patterns obvious; generated C++ diagnostics often report the expanded
record name.

## How backend target descriptions use TableGen

### Registers

Targets define physical registers, register aliases, subregister relationships,
and register classes. Generated register information feeds register allocation,
instruction verification, assembly printing/parsing, and disassembly.

Typical generated include concepts include register enum values, register-class
tables, and helper methods used by `TargetRegisterInfo`.

### Instructions

Instruction records describe operands, assembly strings, encodings, implicit
uses/defs, side effects, branch/call/load/store flags, and pseudo-instruction
properties. Generated instruction information feeds `TargetInstrInfo`, assembler
printers, encoders, decoders, and machine-instruction verification.

### Patterns

Selection patterns map source operations to target instructions. In a
SelectionDAG-oriented target, patterns often describe how DAG nodes such as adds,
loads, constants, and addressing modes become target instructions. GlobalISel can
also consume generated selector data when the target supports it.

### Scheduling data

Scheduling descriptions model processor resources, latencies, issue widths, and
variant CPUs. Machine schedulers use this data to choose instruction orderings
that are legal and profitable for the selected subtarget.

## Example `llvm-tblgen` commands

Commands vary by build tree and target, but the pattern is always:

```bash
llvm-tblgen -I llvm/include -I llvm/lib/Target/Toy \
  -gen-register-info llvm/lib/Target/Toy/Toy.td \
  -o ToyGenRegisterInfo.inc

llvm-tblgen -I llvm/include -I llvm/lib/Target/Toy \
  -gen-instr-info llvm/lib/Target/Toy/Toy.td \
  -o ToyGenInstrInfo.inc

llvm-tblgen -I llvm/include -I llvm/lib/Target/Toy \
  -gen-asm-writer llvm/lib/Target/Toy/Toy.td \
  -o ToyGenAsmWriter.inc

llvm-tblgen -I llvm/include -I llvm/lib/Target/Toy \
  -gen-dag-isel llvm/lib/Target/Toy/Toy.td \
  -o ToyGenDAGISel.inc
```

In a normal LLVM CMake build, target CMake rules run the relevant TableGen
backends and place generated `.inc` files in the build directory. Backend C++
then includes those generated fragments with macros such as
`GET_INSTRINFO_ENUM`, `GET_REGINFO_TARGET_DESC`, or target-specific include
patterns.

## Generated-file mental model

Think of TableGen as producing C++ include fragments, not standalone libraries.
A backend source file often does this:

```cpp
#define GET_INSTRINFO_CTOR_DTOR
#include "ToyGenInstrInfo.inc"
```

A different source file may include the same generated `.inc` with another macro
to request enum definitions, tables, declarations, or method bodies. Losing track
of which macro includes which section is one of the fastest ways to get confused
while reading backend code.

## Pitfalls

### Treating TableGen as a general-purpose language

TableGen has conditionals, lists, string operations, classes, and multiclasses,
but it is not a replacement for C++ or Python. Keep `.td` files declarative.
When logic becomes procedural, prefer moving behavior into backend C++ or a
purpose-built TableGen backend.

### Losing generated include files

The source tree contains `.td` inputs; the build tree contains generated `.inc`
outputs. If an IDE, grep command, or code browser only indexes the source tree,
backend C++ may appear to include missing files. Check the build directory and
the target's CMake TableGen rules before assuming the include is hand-written.

### Overusing clever multiclasses

Multiclasses reduce duplication, but deeply nested parameterized records can hide
what instruction records actually exist. Prefer readable record names and small,
composable templates over compact but opaque metaprogramming.

## Example file

See [`examples/minimal-instruction.td`](examples/minimal-instruction.td) for a
small illustrative TableGen record set. It is designed for reading, not as a full
LLVM target that can be compiled by itself.
