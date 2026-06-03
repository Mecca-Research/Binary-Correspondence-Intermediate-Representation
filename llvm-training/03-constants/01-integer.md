# Integer Constants

## TL;DR

An integer constant is written as `<type> <value>`, where `<type>` is
any integer type and `<value>` is a (possibly negative) decimal literal
or a hex literal in `0xNN` form (signed `s0x`/unsigned `u0x` for wide
values).

```llvm
i32 42
i32 -1
i64 1000000000000
i1  1            ; true
i1  0            ; false
i32 0x7FFFFFFF   ; INT_MAX
```

## Syntax

```
<type> <decimal-literal>
<type> 0x<hex-digits>
<type> u0x<hex-digits>     ; explicitly unsigned interpretation
<type> s0x<hex-digits>     ; explicitly signed   interpretation
```

The `u0x`/`s0x` forms only matter for integer types wider than 64 bits
(`i128` and beyond) where decimal would be unwieldy.

## Common widths

| Type | Range (signed) | Range (unsigned) |
|---|---|---|
| `i1` | 0..1 (bool) | 0..1 |
| `i8` | -128..127 | 0..255 |
| `i16` | -32768..32767 | 0..65535 |
| `i32` | -2³¹..2³¹-1 | 0..2³²-1 |
| `i64` | -2⁶³..2⁶³-1 | 0..2⁶⁴-1 |

LLVM doesn't tag a value as signed or unsigned — the operation does
(see [`../02-types/01-primitive-types.md`](../02-types/01-primitive-types.md)).

## Where integer constants appear

In any context where an integer-typed operand is expected.

```llvm
; Arithmetic
%x = add i32 5, 10                    ; both constants
%y = mul i32 %x, 2                    ; one constant operand

; Comparisons
%cmp = icmp slt i32 %n, 100           ; "n less than 100, signed"

; Initializers
@counter = global i32 0
@flags   = global i32 0xCAFEBABE

; Control flow
br i1 1, label %always_true, label %dead    ; constant-folded branch

; Switch labels
switch i32 %v, label %default [
  i32 0, label %case_zero
  i32 1, label %case_one
  i32 100, label %case_hundred
]

; GEP indices
%p = getelementptr inbounds [10 x i32], ptr %arr, i32 0, i32 3
```

## i1 (booleans)

```llvm
i1 1        ; true
i1 0        ; false
true        ; shorthand — only valid in some places
false       ; shorthand
```

`true`/`false` keywords work in some operand contexts but not all.
When in doubt, write `i1 1` / `i1 0`.

## Arbitrary-width integers

LLVM supports any width 1..2²³-1:

```llvm
i27 12345
i100 0x123456789abcdef0123456789
i256 0
```

These are useful for big-integer arithmetic, bit-precise hash functions,
and DSL backends. Most code only uses the common widths above.

## Pitfalls

- **Out-of-range constant.** `i8 256` doesn't fit. The verifier will
  reject (or sometimes silently truncate, depending on version — don't
  rely on it).

- **Missing type.** `add 5, 10` doesn't parse. The instruction needs
  `add i32 5, 10`.

- **Using `true`/`false` where `i1 1`/`i1 0` is required.**
  In most positions, both work. In a struct/array initializer the
  bare keyword may not be accepted; spell it out.

- **Negative constant in an unsigned context.** `-1` of type `i32` has
  the bit pattern `0xFFFFFFFF`. If you then `udiv` it, you're dividing
  by 4294967295. The compiler doesn't warn — that's by design.

## See also

- [`../02-types/01-primitive-types.md`](../02-types/01-primitive-types.md) — integer types in general
- [`04-global-vs-local.md`](04-global-vs-local.md) — using integer constants as initializers
- [`../reference/instruction-quickref.md`](../reference/instruction-quickref.md) — operations that take integer
  operands
