# Composite Types: Structs, Arrays, Vectors

## TL;DR

| Type | Shape | Indexed by | Memory layout |
|---|---|---|---|
| Struct | `{ T1, T2, ... }` or named `%S` | Constant integer field index | Sequential, with target padding |
| Array | `[N x T]` | Runtime integer (in element units) | Sequential, no padding |
| Vector | `<N x T>` | Runtime integer (lane index) | SIMD-packed, no padding |

You access composite-typed memory with `getelementptr` (GEP) and
composite-typed *values* (as SSA) with `extractvalue`/`insertvalue`
(structs/arrays) or `extractelement`/`insertelement`/`shufflevector`
(vectors).

## Struct types

```llvm
; Literal struct (anonymous, unique by structural identity)
{ i32, float, ptr }

; Named struct (declared at module scope)
%Person = type { i32, float, ptr }
```

The two forms differ:

- **Literal**: structurally typed, can appear inline anywhere.
- **Named**: declared once, referenced by name; the compiler may
  forward-declare it.

### Packed struct

`<{ ... }>` skips alignment padding:

```llvm
%PackedHeader = type <{ i32, i8, i64 }>
```

The unpacked equivalent would insert padding to align the `i64` to
8 bytes; the packed form crams them adjacent.

### Accessing struct fields

Through memory (pointer):

```llvm
%Person = type { i32, float, ptr }

define void @set_age(ptr %p, i32 %v) {
  %age_p = getelementptr inbounds %Person, ptr %p, i32 0, i32 0
  store i32 %v, ptr %age_p, align 4
  ret void
}
```

Breakdown of `getelementptr`:
- First index (`i32 0`) — step over `%Person` instances. `0` = no
  step; just look at this one.
- Second index (`i32 0`) — pick field 0 (the `i32 age`).

For field 1 (the `float`): second index `i32 1`. For field 2 (the
`ptr`): `i32 2`.

Through SSA values:

```llvm
%pair = insertvalue { i32, i32 } undef, i32 7, 0
%pair2 = insertvalue { i32, i32 } %pair, i32 9, 1
%first = extractvalue { i32, i32 } %pair2, 0       ; -> 7
%second = extractvalue { i32, i32 } %pair2, 1      ; -> 9
```

Indices here are constant compile-time integers, *not* SSA values.

## Array types

```llvm
[10 x i32]               ; ten i32s
[4 x [4 x float]]        ; 4x4 matrix
[3 x { i32, ptr }]       ; array of structs
```

Indexed dynamically:

```llvm
define i32 @get(ptr %arr, i32 %i) {
  %p = getelementptr inbounds [10 x i32], ptr %arr, i32 0, i32 %i
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
```

The two GEP indices have different meanings, just like with structs:

- First index: steps over `[10 x i32]` units. `0` means "stay at
  this array".
- Second index: picks an element within the array.

If `%arr` actually points to a sequence of arrays (e.g., the start of
a `[10 x [10 x i32]]`), the first index can be non-zero to skip
between rows.

### Initialized array constant

```llvm
@xs = global [3 x i32] [i32 1, i32 2, i32 3]
```

Or zero-initialized:

```llvm
@buf = global [1024 x i32] zeroinitializer
```

## Vector types

```llvm
<4 x float>           ; 128-bit SIMD: four floats
<8 x i32>             ; 256-bit SIMD: eight i32s
<16 x i8>             ; 128-bit SIMD: sixteen bytes
```

Vectors are not arrays — they're hardware SIMD registers. Operations
on vectors are **element-wise**:

```llvm
define <4 x float> @vadd(<4 x float> %a, <4 x float> %b) {
  %r = fadd <4 x float> %a, %b   ; lane 0, 1, 2, 3 each independently added
  ret <4 x float> %r
}
```

### Per-lane access

```llvm
%elem  = extractelement <4 x float> %v, i32 2     ; get lane 2
%new   = insertelement  <4 x float> %v, float 1.0, i32 2  ; replace lane 2

; Shuffle: build a new vector from two source vectors via a mask
%shuf  = shufflevector <4 x float> %a, <4 x float> %b,
                       <4 x i32> <i32 0, i32 1, i32 4, i32 5>
                       ; result lanes: a[0], a[1], b[0], b[1]
```

### Scalable vectors

LLVM 9+ supports vectors whose length is target-dependent (e.g., ARM
SVE):

```llvm
<vscale x 4 x i32>     ; "vscale-many" groups of 4
```

You'll only see these in SVE/SVE2 codegen. Mostly ignore.

## Comprehensive example

```llvm
; A row-major 4x4 matrix wrapped in a struct
%Matrix = type { i32, i32, [4 x <4 x float>] }

define float @trace(ptr %m) {
entry:
  br label %loop

loop:
  %i        = phi i32 [ 0, %entry ], [ %i_next, %loop ]
  %sum      = phi float [ 0.0, %entry ], [ %new_sum, %loop ]
  ; Get pointer to lane %i of row %i in the array-of-vectors at field index 2
  %elem_p   = getelementptr inbounds %Matrix, ptr %m,
                            i32 0, i32 2, i32 %i, i32 %i
  %elem     = load float, ptr %elem_p, align 4
  %new_sum  = fadd float %sum, %elem
  %i_next   = add i32 %i, 1
  %done     = icmp eq i32 %i_next, 4
  br i1 %done, label %exit, label %loop

exit:
  ret float %new_sum
}
```

That GEP has four indices: step the struct (0, stay), pick field 2
(the `[4 x <4 x float>]`), pick row `%i`, pick column `%i`. The final
pointer is to a `float`, which is what we load.

## Pitfalls

- **Off-by-one on the leading GEP index.** It's "step over the *type*
  at the base", not "first field". Set it to 0 when you have a
  pointer to a single instance.

- **Confusing `[N x T]` with `<N x T>`.** Brackets are arrays
  (no parallelism semantics); angle brackets are vectors (lane-wise).

- **Mismatched struct field count between modules.** If module A
  declares `%S = type { i32, i32 }` and module B declares
  `%S = type { i32, i32, i32 }`, `llvm-link` cannot unify them.
  See `08-pitfalls/05-type-schema-drift.md` for a real instance from
  BCIR.

- **Storing into a packed struct as if it were unpacked.** Packed
  structs have no padding; offsets differ from the unpacked form.

- **Using `extractelement`/`insertelement` on arrays or
  `extractvalue`/`insertvalue` on vectors.** They are not
  interchangeable. The verifier rejects.

- **Forgetting `inbounds` on GEPs that should have it.** Without
  `inbounds`, the optimizer assumes the GEP might overflow and is
  more conservative. With `inbounds`, overflow is UB and the
  optimizer can be aggressive.

## See also

- `01-primitive-types.md` — the building blocks
- `03-opaque-and-pointer-types.md` — pointers to composite types
- `04-memory/01-alloca.md` — allocating composite types on the stack
- `04-memory/02-load-store.md` — loading/storing fields
- `08-pitfalls/05-type-schema-drift.md` — what breaks when struct
  layouts disagree
