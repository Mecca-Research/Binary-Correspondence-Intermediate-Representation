# Modules, Functions, and Basic Blocks

## TL;DR

LLVM IR has three structural layers:

- **Module** — top-level container, one per compilation unit. Has
  globals, functions, and metadata.
- **Function** — `@`-prefixed callable unit, contains basic blocks.
- **Basic block** — `label:`-prefixed straight-line sequence of
  instructions, ending in exactly one terminator.

Names: `@` = global (visible across functions in the module, possibly
across modules), `%` = local (function-scoped SSA name).

## Module

A module is what `llvm-as` reads from a single `.ll` file. Its
top-level constructs:

```llvm
source_filename = "example.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Globals
@counter = global i32 0, align 4
@.msg    = private unnamed_addr constant [13 x i8] c"Hello, ABI\0A\00"

; External declarations
declare i32 @printf(ptr, ...)

; Function definitions
define i32 @main() {
  ; ...
  ret i32 0
}

; Module-level metadata
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Dwarf Version", i32 4}
```

Key fields:

- **`source_filename`** — informational, used in diagnostics.
- **`target datalayout`** — tells `opt`/`llc` how to size and align
  types. Required for some passes; omit and you'll get warnings.
- **`target triple`** — `<arch>-<vendor>-<sys>-<env>`. Strings like
  `x86_64-pc-linux-gnu`, `aarch64-apple-darwin`,
  `riscv64-unknown-elf`.
- **Globals** (`@foo = global T value`) — module-scope storage.
- **Constants** (`@foo = constant T value`) — same, but immutable.
- **Function declarations** (`declare`) — external; resolved at link
  time.
- **Function definitions** (`define`) — internal; has a body.
- **Module-level metadata** — named `!llvm.*` lists.

## Function

```llvm
define <linkage> <retattrs> <type> @name (<params>) <fnattrs> {
  entry:
    ...
    ret <type> <val>
}
```

Minimal:

```llvm
define i32 @add(i32 %a, i32 %b) {
  %r = add i32 %a, %b
  ret i32 %r
}
```

With linkage and attributes:

```llvm
define internal i32 @add(i32 %a, i32 %b) nounwind readnone {
  %r = add i32 %a, %b
  ret i32 %r
}
```

Components:

| Slot | Meaning | Examples |
|---|---|---|
| Linkage | Visibility/merging | `internal`, `private`, `external`, `weak`, `linkonce_odr` |
| Return attrs | Apply to return value | `zeroext`, `signext`, `noalias`, `nonnull` |
| Calling conv | ABI | `ccc` (default), `fastcc`, `coldcc`, `tailcc`, etc. |
| Type | Return type | `void`, `i32`, `ptr`, `<4 x float>` |
| Name | Function name | `@add`, `@"some.name"` |
| Params | Comma-separated typed params | `(i32 %a, ptr nocapture %p)` |
| Function attrs | Apply to function | `nounwind`, `readonly`, `alwaysinline`, `noinline` |

`declare` (no body) uses the same shape, just without the `{ ... }`.

### Parameter attributes

Each parameter can have attributes that constrain or describe it:

```llvm
define void @copy(ptr noalias %dst, ptr noalias %src, i64 %n) {
  ; ...
}
```

Common ones:

| Attribute | Meaning |
|---|---|
| `noalias` | Pointer doesn't alias other `noalias` pointers |
| `nocapture` | The callee doesn't keep the pointer past the call |
| `readonly` | The callee doesn't write through this pointer |
| `nonnull` | The pointer is not null |
| `dereferenceable(N)` | At least N bytes of the pointer are dereferenceable |
| `signext` / `zeroext` | Smaller-than-word integer extension on ABI |
| `byval(T)` | Pass-by-value of the pointed-to type |
| `sret(T)` | Output parameter (struct return) |

## Basic block

A basic block is a labeled sequence of instructions ending in exactly
one terminator. Terminators are: `ret`, `br` (conditional and
unconditional), `switch`, `indirectbr`, `invoke`, `callbr`, `resume`,
`catchswitch`, `catchret`, `cleanupret`, `unreachable`.

```llvm
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %positive, label %nonpositive

positive:
  ret i32 1

nonpositive:
  ret i32 0
```

Rules:

- **Exactly one terminator**, at the end. If you have `br ... ret ...`
  in the same block, the verifier rejects it.
- **First block of a function has no predecessors** other than the
  function entry. Conventionally named `entry`.
- **Phi nodes (if any) come first** in the block, before any other
  instruction.
- **All paths must end in a terminator** — falling off the end of a
  function is undefined behavior; the verifier catches it.

## Relationship

```
Module
  ├── global @g1
  ├── global @g2
  ├── function @f1
  │     ├── basic block entry
  │     │     ├── instruction
  │     │     ├── instruction
  │     │     └── terminator (br/ret/...)
  │     ├── basic block bb1
  │     │     └── ...
  │     └── basic block bb2
  │           └── ...
  └── function @f2
        └── ...
```

## Full module example

See `examples/module-anatomy.ll`.

## Pitfalls

- **Forgetting the terminator.** Every basic block needs one.
  Verifier: *"Block does not have a terminator instruction"*.

- **Multiple terminators in one block.** The verifier rejects this
  too.

- **Naming a local `%foo` that conflicts with a label `%foo`.** Local
  identifiers share a namespace with labels. Just pick distinct names.

- **Forgetting `entry:` label when the first block has predecessors.**
  Function-entry blocks can't be the target of a `br` from another
  block. If you need that, introduce a new block.

- **Using `@` for a local.** `@` is always module-scope. `%` is always
  function-scope.

- **Mixing `i32 %x` and `i64 %x`** in the same scope. Each SSA name
  has exactly one definition with one type. Reusing the name with a
  different type is a parse error.

## See also

- `00-foundations/02-ssa.md` — why each `%name` is defined once
- `02-instruction-format.md` — the shape of individual instructions
- `03-comments-metadata.md` — `;` and `!N`
- `04-memory/03-global-variables.md` — global linkage, visibility,
  TLS, address spaces
- `reference/glossary.md` — every identifier prefix in one table
