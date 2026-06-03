# Primitive Types

## TL;DR

LLVM IR primitive types:

| Type | Meaning |
|---|---|
| `iN` | Integer of N bits (1 ≤ N ≤ 2²³). Common: `i1`, `i8`, `i16`, `i32`, `i64`, `i128` |
| `half`, `bfloat`, `float`, `double`, `fp128`, `x86_fp80`, `ppc_fp128` | IEEE/architecture-specific floats |
| `void` | No value (only valid as function return type or for `ret void`) |
| `ptr` | Generic (opaque) pointer; address space can be attached |
| `label` | Basic-block label (used by branches) |
| `token` | Opaque control value (used by EH and coroutine intrinsics) |
| `metadata` | Metadata reference (only as function-parameter type for intrinsics) |
| `x86_mmx` | Legacy x86 MMX (largely deprecated) |

## Integer types: `iN`

```llvm
i1     %b      ; boolean
i8     %byte
i16    %word
i32    %int
i64    %long
i128   %big
i27    %weird  ; LLVM allows non-power-of-2 widths
```

- **Signed vs unsigned is in the operation, not the type.** `i32` is
  bit-pattern; `sdiv` interprets it signed, `udiv` interprets it
  unsigned.
- **Comparisons** use signed (`slt`, `sle`, `sgt`, `sge`) or unsigned
  (`ult`, `ule`, `ugt`, `uge`) predicates.
- **`i1` is the boolean type.** Conditions for `br i1`, results of
  `icmp`/`fcmp`, etc.

Operations on integers:
- Arithmetic: `add`, `sub`, `mul`, `udiv`, `sdiv`, `urem`, `srem`
- Bitwise: `and`, `or`, `xor`, `shl`, `lshr`, `ashr`
- Comparison: `icmp <pred>`
- Conversion: `trunc`, `zext`, `sext`, `inttoptr`, `bitcast`,
  `sitofp`, `uitofp`

## Floating-point types

| Type | Width | Format |
|---|---|---|
| `half` | 16 | IEEE 754 binary16 |
| `bfloat` | 16 | Brain float (1-8-7) |
| `float` | 32 | IEEE 754 binary32 |
| `double` | 64 | IEEE 754 binary64 |
| `fp128` | 128 | IEEE 754 binary128 |
| `x86_fp80` | 80 | x87 extended precision |
| `ppc_fp128` | 128 | PowerPC double-double |

```llvm
float  3.14
double 1.0e10
half   1.0
```

Operations use `f`-prefixed instructions (`fadd`, `fsub`, ...) and
the `fcmp` comparison. They accept fast-math flags (see
`reference/instruction-quickref.md`).

## `void`

Only valid as:
- The return type of a function: `define void @do_thing()`.
- The argument to `ret`: `ret void` (with no value).

You can't have a `void` SSA value. `void %x` is nonsense.

## `ptr` — the opaque pointer

Modern LLVM (≥ 15) uses a single **opaque pointer** type for every
pointer:

```llvm
ptr           ; pointer in default address space (0)
ptr addrspace(1)   ; pointer in address space 1
```

The pointee type is not part of the pointer's type. Operations that
need to know the pointee type spell it out at the operation site:

```llvm
%v = load i32, ptr %p, align 4         ; load an i32 from %p
store i32 %v, ptr %p, align 4          ; store an i32 to %p
%gep = getelementptr i32, ptr %arr, i32 5    ; advance by 5 i32s
```

Before LLVM 15, every pointer carried its pointee: `i32*`, `[8 x
float]*`, etc. The opaque-pointer transition unified these. New IR
should use `ptr`; old `i32*`-style still parses but is deprecated.
See `02-types/03-opaque-and-pointer-types.md`.

## `label`

The type of a basic-block reference:

```llvm
br label %loop
br i1 %cond, label %t, label %f
```

You never declare a `label` value yourself; it's implicit in
terminator operands.

## `token`

Opaque control value used by exception-handling intrinsics and
coroutines:

```llvm
%cs = catchswitch within none [label %handler] unwind to caller
%cp = catchpad within %cs [...]
```

`%cs` and `%cp` are tokens. The `phi` rule says all incoming values
must have the same type; tokens have additional restrictions (you
generally can't phi over them). Mostly you'll only see them in
EH-using code.

## `metadata`

A type that appears only as a parameter to certain intrinsics like
`llvm.dbg.value`:

```llvm
call void @llvm.dbg.value(metadata i32 %x, metadata !12, metadata !DIExpression())
```

You don't define normal values of this type.

## Combining primitive types

```llvm
define i32 @demo(i32 %a, float %b, ptr %c) {
entry:
  %x      = add i32 %a, 10            ; integer arithmetic
  %y      = fadd float %b, 3.14       ; float arithmetic
  %loaded = load i32, ptr %c, align 4 ; pointer dereference (typed at use)
  %sum    = add i32 %x, %loaded
  ret i32 %sum
}
```

## Pitfalls

- **`i32` vs `i64` mismatch.** A common cause of verifier complaints
  is using a 32-bit constant or value in a 64-bit context. Cast with
  `zext` (unsigned) or `sext` (signed).

- **Treating `i1` as `i8` or vice versa.** They are different types.
  Convert via `zext` or `trunc`.

- **`void`-typed values.** You can't bind one: `%x = call void
  @do_thing()` is a parse error. Drop the `%x = `.

- **Using `ptr` with `align` smaller than the natural alignment of
  the access type.** LLVM allows misaligned loads but generates
  slower code (and may even trap on strict targets). Use the natural
  alignment when in doubt.

- **Pre-LLVM-15 typed pointers in new code.** `i32*` still parses but
  is deprecated. Use `ptr`.

- **Confusing `half` and `bfloat`.** Same bit-width, different
  formats. Not interchangeable.

## See also

- `02-composite-types.md` — structs, arrays, vectors
- `03-opaque-and-pointer-types.md` — `ptr` in detail
- `03-constants/01-integer.md` — integer literals
- `03-constants/02-floating-point.md` — float literals (decimal and hex)
- `reference/instruction-quickref.md` — operations on each type
