# What is LLVM IR?

## TL;DR

LLVM IR is a **typed**, **SSA-based**, **platform-independent**
intermediate representation that sits between source code (C, C++,
Rust, Swift, ...) and machine code. It's a structured assembly
language: low-level enough to lower efficiently to any target, but
abstract enough that the same IR runs on x86, ARM, RISC-V, GPU
targets, and more.

It exists in two interchangeable forms:

- **`.ll`** — textual, human-readable
- **`.bc`** — binary bitcode, compact

`llvm-as file.ll → file.bc`, `llvm-dis file.bc → file.ll`. Identical
semantics; pick the form that suits the tool.

## The five characteristics

1. **Static Single Assignment (SSA).** Each value is defined exactly
   once. Branches that merge values use `phi` nodes. See
   `02-ssa.md`.
2. **Strongly typed.** Every value carries a type. There's no
   untyped "register" — `i32 %x` and `i64 %x` are different things.
3. **Three-address-ish form.** Most instructions:
   `%result = op type, operand1, operand2`.
4. **Extensible.** You can add metadata, attributes, address spaces,
   target-specific intrinsics without breaking the parser.
5. **Modular.** Modules are independently compilable; `llvm-link`
   merges them.

## The hierarchy

```
Module
├── target triple, datalayout, source_filename
├── Global variables (@foo)
├── Functions (@bar)
│   └── Basic blocks (labels)
│       └── Instructions
│           └── Operands (values, constants, types)
└── Metadata (!N)
```

See `01-syntax/01-modules-functions-blocks.md` for details on each layer.

## A minimal example

```llvm
; A function that adds two i32s and returns the result.
define i32 @add(i32 %a, i32 %b) {
  %result = add i32 %a, %b
  ret i32 %result
}
```

What's going on:

- `define` introduces a function definition (vs `declare` for an
  external).
- `i32 @add` — function returns i32, named `@add`.
- `(i32 %a, i32 %b)` — two i32 parameters with SSA names.
- `%result = add i32 %a, %b` — an `add` instruction whose result is
  bound to the SSA value `%result`. The `i32` annotates the operand
  type.
- `ret i32 %result` — terminator. Every basic block ends with one.

Run it through the tools:

```bash
llvm-as add.ll -o add.bc       # text → bitcode
llvm-dis add.bc -o add2.ll     # bitcode → text (lossless)
opt -passes=verify add.bc -o /dev/null   # verify only
```

## Why care?

For an LLM agent specifically, LLVM IR is useful because:

- It's the **lingua franca** between dozens of source languages and
  dozens of CPU/GPU targets. If you can read it, you can reason about
  code regardless of the source language.
- Static analysis, optimization, and JIT all happen at the IR level.
  Knowing IR lets you understand what `clang -O3` actually does.
- Many custom IRs (including BCIR in the sibling project) ultimately
  lower to LLVM IR. Reading the lowering output tells you whether the
  custom IR was sound.

## Pitfalls

- IR is **not** assembly. There's no register allocation, no machine
  layout, no call ABI baked in. Those happen at the backend
  (`llc`).
- IR is **not** untyped. You can't `add %x, %y` without a type.
- The textual form is **not** stable across LLVM major versions for
  every construct (opaque pointers, attribute syntax). Pin a version
  if you generate IR mechanically.

## See also

- `02-ssa.md` — SSA, the most important property
- `03-ir-vs-asm-vs-other-irs.md` — how IR differs from assembly,
  Java bytecode, CIL, SPIR-V, GIMPLE
- `01-syntax/01-modules-functions-blocks.md` — the structural hierarchy
- `examples/simple-add.ll` — the example above
