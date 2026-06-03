# Global and Local Constants

## TL;DR

"Constant" in LLVM IR can mean two things:

1. **A `constant` global** — a module-scope, immutable, named value
   stored in (typically) read-only memory.
2. **A constant operand inline in an instruction** — a literal
   `i32 42`, `float 1.0`, or a `getelementptr` constant expression
   computed at compile time.

There is no `constant` local with a `%` prefix — local SSA values
are always defined by an instruction. If a local is constant-valued
(e.g., `%x = add i32 1, 2`), the optimizer will fold it.

## Global constants

```llvm
@PI         = constant double 0x400921FB54442D18
@MAGIC      = constant i32 0xCAFEBABE
@.fmt       = private unnamed_addr constant [4 x i8] c"%d\0A\00"
@LOOKUP     = constant [3 x i32] [i32 10, i32 20, i32 30]
```

Anatomy:

```
@<name> = [<linkage>] [<visibility>] [<dll-storage>] [<thread-local>]
          [<unnamed_addr>] [<addrspace>]
          constant <type> <initializer>
          [, section "..."] [, comdat ...] [, align N]
```

Key contrasts with `global`:

| | `constant` | `global` |
|---|---|---|
| Mutable? | No (UB to write) | Yes |
| Placed in | Read-only section (`.rodata`) | Read-write section (`.data` / `.bss`) |
| Initializer required? | Yes | No (defaults to undef/zero) |

```llvm
@a = global i32 42         ; .data, mutable
@b = constant i32 42       ; .rodata, immutable
```

## Local "constants" (folded SSA values)

There's no special syntax to declare a local constant. You write an
instruction that produces a value, and the optimizer (or a human
reader) treats it as constant if all operands are.

```llvm
define i32 @demo(i32 %x) {
  %k = add i32 7, 0        ; effectively the constant 7
  %y = mul i32 %x, %k      ; equivalent to mul i32 %x, 7 after folding
  ret i32 %y
}
```

At any optimization level above `-O0`, the `%k = add i32 7, 0` is
folded into the literal `7` at the use site.

## Constant expressions

LLVM permits some compile-time computation inside operand position
using **constant expressions**. The most common ones:

```llvm
; Pointer-to-pointer bitcast inside an initializer
@aliased = constant ptr bitcast (ptr @foo to ptr)

; GEP at compile time
@second_elem_ptr = constant ptr getelementptr inbounds (
  [3 x i32], ptr @LOOKUP, i32 0, i32 1)

; Arithmetic at compile time
@neg = constant i32 sub (i32 0, i32 42)     ; -42

; A pointer-to-int conversion
@addr = constant i64 ptrtoint (ptr @foo to i64)
```

Constant expressions look like instructions but appear in operand
position, parenthesized. They're evaluated by the assembler/linker,
not at runtime.

**Important:** constant expressions are a limited grammar. You
*cannot* nest arbitrary instructions in operand position:

```llvm
; WRONG — `xor` of an SSA value cannot be a constant expression operand
%y = or i1 (xor i1 %x, true), %z
```

See [`../08-pitfalls/01-nested-instruction-expressions.md`](../08-pitfalls/01-nested-instruction-expressions.md) for the
real-world version of this mistake.

## Differences at a glance

| Property | Global constant `@C` | Inline literal `i32 5` | Folded local `%x = add 2, 3` |
|---|---|---|---|
| Has a symbol? | Yes | No | No |
| Has an address? | Yes | No | No |
| Has a type? | Yes | Yes | Yes |
| Visible to linker? | Yes | No | No |
| Lifetime | Program | Per-use | Per-function |

## When to use which

- **Global constant** — anything you might want to reference by
  address: format strings, lookup tables, vtables, large arrays of
  data, shared configuration.

- **Inline literal** — short numeric values: loop bounds, masks,
  bit-field shifts, branch conditions. No reason to introduce a
  symbol for `42`.

- **Constant expression** — initializing one global from another,
  computing a pointer-to-pointer at module load, or building a
  globally-visible derived value.

## Example: combining

```llvm
@global_int    = constant i32 42
@global_string = constant [13 x i8] c"Hello, World\00"

define i32 @main() {
  ; Inline literal use of global constant
  %v = load i32, ptr @global_int, align 4

  ; Inline literal
  %r = add i32 %v, 100

  ; Folded local (post-opt)
  %two_x = mul i32 %r, 2

  ret i32 %two_x
}
```

## Pitfalls

- **Writing to a `constant`.** UB. Most platforms will segfault.

- **Confusing `constant` with `const` from C/C++.** A `constant`
  global is more like a literal pool entry than a `const`-qualified
  variable.

- **Trying to use a SSA value in a constant expression.** Only
  constant operands are allowed in constant expressions.

- **Forgetting `private unnamed_addr` on internal string literals.**
  Without it, the symbol is exported and addresses are
  uniqued — bigger binary.

- **Initialization with the wrong shape.** `constant [3 x i32] [i32
  1, i32 2]` doesn't have three elements. Parser error.

## See also

- [`../01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) — how globals fit into
  the module
- [`../04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) — `global` vs `constant`, linkage,
  visibility
- [`../08-pitfalls/01-nested-instruction-expressions.md`](../08-pitfalls/01-nested-instruction-expressions.md) — limit of
  constant expressions
