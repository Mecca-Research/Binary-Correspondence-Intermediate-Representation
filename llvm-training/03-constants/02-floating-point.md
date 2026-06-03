# Floating-Point Constants

## TL;DR

Float constants come in two flavors:

- **Decimal**: `float 3.14`, `double 1.5e10`, `float -0.0`
- **Hexadecimal** (bit-exact): `double 0x4000000000000000`,
  `float 0x4049000000000000`

Hex is the only way to express bit-exact NaN, denormals, and
otherwise-unrepresentable values portably.

## Types

| Type | Width | Format |
|---|---|---|
| `half` | 16 | IEEE binary16 |
| `bfloat` | 16 | Brain float (1-8-7) |
| `float` | 32 | IEEE binary32 |
| `double` | 64 | IEEE binary64 |
| `fp128` | 128 | IEEE binary128 |
| `x86_fp80` | 80 | x87 extended (1+15+64) |
| `ppc_fp128` | 128 | PowerPC double-double |

## Decimal literals

```llvm
float  3.14
float  -0.5
float  1.0e-10
double 2.718281828459045
double -1.5
```

Notes:

- Always include the decimal point. `float 1` is an *integer* 1
  forced into a `float` context — actually it doesn't parse; you need
  `float 1.0`.
- Scientific notation works: `1.0e10`, `-3.5e-4`.
- The decimal literal is silently widened to the type's representation.

## Hexadecimal literals

For bit-exact precision (e.g., emitting denormals, NaN patterns,
exact rounding), use hex:

```llvm
double 0x4000000000000000          ; exactly 2.0
double 0x3FF0000000000000          ; exactly 1.0
double 0x7FF8000000000000          ; quiet NaN
double 0x7FF0000000000000          ; +inf
double 0xFFF0000000000000          ; -inf
double 0x0000000000000001          ; smallest positive denormal

float  0x4049000000000000          ; ~3.141592 (NB: hex is 64-bit, truncated to float)
```

Special prefixes:

| Prefix | Type | Width of hex digits |
|---|---|---|
| `0x` (plain) | `double`, `float` (widened/narrowed) | 16 |
| `0xK` | `x86_fp80` | 20 |
| `0xL` | `fp128` | 32 |
| `0xM` | `ppc_fp128` | 32 |
| `0xH` | `half` | 4 |
| `0xR` | `bfloat` | 4 |

## Where they appear

```llvm
; Arithmetic
%y = fadd float %x, 3.14
%z = fmul double 1.0e-3, %y

; Comparisons
%cmp = fcmp olt float %x, 0.0

; Initializers
@pi = constant double 0x400921FB54442D18    ; π to double precision

; Vector constants
@v  = constant <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>
```

## Pitfalls

- **Integer literal in float context.** `float 1` — write `float 1.0`.
  The parser doesn't auto-widen.

- **Using `0xNNNNNNNN` (8 hex digits) for a `float`.** Plain `0x`
  literals are 16 hex digits = 64 bits, sized to `double`. To get a
  bit-exact `float`, use the `0x` form with 16 digits anyway — the
  bottom bits are simply ignored when narrowed:
  ```llvm
  float 0x3FF0000000000000   ; bit pattern of double 1.0; narrowed to float 1.0
  ```

- **Treating `0.0` and `-0.0` as equal.** They compare equal under
  `fcmp oeq` and `fcmp ueq`, but their bit patterns differ. Tests
  that care about sign of zero must inspect bits.

- **NaN comparisons.** `fcmp oeq nan, nan` is false. Use `uno` to
  check unorderedness.

- **Fast-math flags change semantics.** `fadd fast` allows the
  optimizer to ignore NaN/Inf, reorder, and use approximations.
  Don't sprinkle `fast` on safety-critical code.

## See also

- `02-types/01-primitive-types.md` — float types in general
- `reference/instruction-quickref.md` — fast-math flags table
