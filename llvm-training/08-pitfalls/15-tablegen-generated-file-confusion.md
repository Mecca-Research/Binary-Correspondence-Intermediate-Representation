# Pitfall 15 — TableGen Generated-File Confusion

## BCIR instance

| Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|
| Training-only exemplar; no affected BCIR `.ll` file recorded | Unknown | `llvm-tblgen -I <llvm-include> -I <target-dir> -gen-instr-info <target>.td -o <build>/*GenInstrInfo.inc` | Regenerate TableGen `.inc` files from `.td` inputs in the build tree instead of editing or searching only source-tree outputs. | [`12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md); [`12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md); [`10-grammar/README.md`](../10-grammar/README.md) |

## The error

When browsing only the source tree or building without the TableGen step:

```text
fatal error: 'ToyGenInstrInfo.inc' file not found
```

Or from an IDE/code-search tool:

```text
cannot open source file "X86GenRegisterInfo.inc"
```

The include may be real even though it is not checked into the source tree.

## Minimal reproducer

Backend C++ often includes generated fragments:

```cpp
#define GET_INSTRINFO_CTOR_DTOR
#include "ToyGenInstrInfo.inc"
```

The corresponding source input is a `.td` file:

```tablegen
def ADDrr : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins);
  let AsmString = "add";
}
```

A normal LLVM CMake build runs `llvm-tblgen` and places `ToyGenInstrInfo.inc` in
the build directory. If you compile the `.cpp` by hand, point the include path at
only the source tree, or search only checked-in files, it looks missing.

## Why it happens

TableGen outputs C++ include fragments, not standalone hand-authored source
files. The same generated `.inc` can expose different sections depending on
which `GET_*` macro is defined before including it. That means two includes of
one filename can intentionally expand to different declarations, tables, enums,
or method bodies.

The source-of-truth is usually a chain of target `.td` files plus shared LLVM
`.td` definitions. The generated file lives in the build tree and is tied to the
chosen CMake target, include paths, and TableGen backend option such as
`-gen-instr-info` or `-gen-register-info`.

## Fix pattern

Use the build system's TableGen rules instead of compiling backend sources by
hand. When debugging:

```sh
find build -name '*GenInstrInfo.inc' -print
find build -name '*GenRegisterInfo.inc' -print
```

Regenerate the specific file with a command shaped like:

```sh
llvm-tblgen -I llvm/include -I llvm/lib/Target/Toy \
  -gen-instr-info llvm/lib/Target/Toy/Toy.td \
  -o build/lib/Target/Toy/ToyGenInstrInfo.inc
```

When reading generated code, first identify:

1. which `.td` file is the root input;
2. which `llvm-tblgen -gen-*` backend produced the `.inc`;
3. which `GET_*` macro is active at that include site;
4. whether your IDE indexes the build directory as well as the source directory.

## BCIR-relevant note

If BCIR grows a custom backend, disassembler table, or instruction-description
generator, separate checked-in declarative inputs from generated include files.
Document the regeneration command and do not patch generated `.inc` output as if
it were the source of truth. Review diffs in both the `.td` inputs and generated
snapshots only when the project intentionally checks snapshots in.

## See also

- [`../12-backend-jit/02-tablegen.md`](../12-backend-jit/02-tablegen.md) — TableGen mental model and commands
- [`../12-backend-jit/01-codegen-pipeline.md`](../12-backend-jit/01-codegen-pipeline.md) — where generated backend tables are consumed
- [`../10-grammar/README.md`](../10-grammar/README.md) — generated grammar artifacts and source-vs-output thinking
