# Exact C Lowering Rules

Every claim below is asserted by
[`../tools/verify-frontend-lowering.py`](../tools/verify-frontend-lowering.py)
against [`examples/struct-layout.c`](examples/struct-layout.c) and
[`examples/control-flow-lowering.c`](examples/control-flow-lowering.c).
Output shown is `clang 18.1.3`, target `x86_64-unknown-linux-gnu`.

## Locals: every declaration is an `alloca` at `-O0`

```llvm
define dso_local i32 @padded_value(ptr noundef %0) {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ...
```

Even a parameter gets a stack slot it is immediately stored into. This is
deliberate: it keeps `-O0` IR a direct transliteration of the source, which is
what debuggers and `-O0` semantics need. `mem2reg` removes it at `-O1` and
above, which is visible in the checked `-O1` case:

| Level | `loop_sum` contains `alloca` |
| --- | --- |
| `-O0` | yes |
| `-O1` | no |

Do not read "the frontend emits terrible IR" into this. It emits *literal* IR.

## Aggregates: the frontend picks a struct kind, the data layout does the rest

```c
struct Padded { char tag; int value; char flag; };            // size 12
struct Packed { char tag; int value; char flag; } __attribute__((packed));
union  Word   { unsigned u; float f; unsigned char bytes[4]; };
```

```llvm
%struct.Padded = type { i8, i32, i8 }
%struct.Packed = type <{ i8, i32, i8 }>
%union.Word    = type { i32 }
```

Three rules in three lines:

1. **Natural padding is implicit.** `{ i8, i32, i8 }` with the module's data
   layout already means offsets 0/4/8 and size 12. The frontend materializes
   explicit padding fields only when natural alignment cannot express the ABI's
   answer (over-aligned members, some bit-field layouts, some Microsoft ABI
   cases).
2. **`packed` is a different struct kind**, `<{ ... }>`, not an attribute. Its
   consequence shows up at every access:

   ```llvm
   ; packed_value
   %5 = load i32, ptr %4, align 1     ; not align 4
   ```

   That `align 1` is the whole cost of `packed`: the backend must assume an
   unaligned access.
3. **A union is storage for its largest member**, with the type of whichever
   member is convenient. Access through a different member is a raw
   reinterpretation of that storage:

   ```llvm
   ; union_bits_of_float: store as float, load as i32
   store float %4, ptr %3, align 4
   %5 = load i32, ptr %3, align 4
   ```

For the frontend's own layout table — offsets, sizes, alignments — use
`clang -Xclang -fdump-record-layouts -c file.c -o /dev/null`.

## Bit-fields do not exist in IR

This is the clearest case of frontend erasure. `unsigned count : 12` at bit
offset 3 becomes, on read:

```llvm
%4 = load i32, ptr %3, align 4
%5 = lshr i32 %4, 3
%6 = and i32 %5, 4095
```

and on write, a read-modify-write of the whole storage unit:

```llvm
%7  = load i32, ptr %6, align 4
%8  = and i32 %5, 4095
%9  = shl i32 %8, 3
%10 = and i32 %7, -32761       ; clear the field
%11 = or  i32 %10, %9          ; insert the new value
store i32 %11, ptr %6, align 4
```

Consequences that matter in real code:

- **A bit-field write touches neighbouring fields' storage.** In C11 and later,
  adjacent bit-fields in the same storage unit are one memory location for the
  purposes of the memory model, so concurrent writes to two bit-fields in the
  same unit race. Separate them with a zero-width `:0` field if they must be
  written concurrently.
- **`volatile` bit-fields are still read-modify-write**, so the access count is
  not what the source suggests.
- Grepping IR for `bitfield` finds nothing. The concept is gone; only the shifts
  and masks remain.

## Control flow

### The conditional operator and short-circuit operators are branches

`p != 0 && n > 0` cannot become `and i1 %a, %b`: the right operand must not be
evaluated when the left is false. The frontend emits a branch, always:

```llvm
; guarded_load: br i1 present, `and i1` absent
```

The gate asserts both halves — the branch is present *and* the bitwise form is
absent. Only the optimizer may later flatten this, and only when it can prove
the right operand is safe to speculate.

### `switch` stays a `switch`

```llvm
switch i32 %2, label %sw.default [
  i32 0, label %sw.bb
  i32 1, label %sw.bb1
  i32 2, label %sw.bb2
  i32 17, label %sw.bb3
]
```

The frontend does not decide jump table versus binary search versus if-else
chain; it emits the `switch` and the backend chooses. Sparse cases are still a
`switch` in IR.

### Loops are the obvious blocks

`for` becomes init / cond / body / inc / exit blocks with names like
`for.cond`, `for.body`, `for.inc`, `for.end`. The block names are frontend
conventions and make `-O0` IR far easier to read than the numbered form —
`-fno-discard-value-names` keeps them on a release build where they would
otherwise be dropped.

## `volatile` is an instruction flag

```llvm
; read_twice
%3 = load volatile i32, ptr %2, align 4
%5 = load volatile i32, ptr %4, align 4
```

Two loads, still two loads at `-O1` (the gate checks this). The type qualifier
is a *source* concept; what survives into IR is the `volatile` marker on each
access. If the marker is missing, no amount of `volatile` in the source will
stop the optimizer — and a tool that rewrites IR must carry the flag or it will
silently merge accesses the program required.

## Other C constructs, briefly

| C construct | IR result |
| --- | --- |
| `a[i]` on `int *a` | `getelementptr inbounds i32, ptr %a, i64 %idx` after `sext i32 %i to i64` |
| `int a[static 8]` parameter | `ptr noundef align 4 dereferenceable(32)` |
| `restrict` pointer parameter | `noalias` on the argument |
| string literal | `private unnamed_addr constant [N x i8] c"..."` |
| `static` local | a module-level global with internal linkage |
| VLA | `alloca` with a runtime element count, plus stack save/restore |
| `_Atomic` load/store | `load atomic` / `store atomic` with an ordering |
| designated initializer | a constant aggregate, or `memcpy` from one |
| large struct assignment | `llvm.memcpy` |
| `inline` (C99) | may emit no symbol at all in that TU |

`restrict` becoming `noalias` is worth pausing on: it is a promise the *source*
made, forwarded to the optimizer as a fact. If the promise is false, the
resulting miscompile is a source bug, and it will look like an optimizer bug.

## Integer promotion and conversion

The usual arithmetic conversions happen in Sema, so IR shows their result, not
their derivation:

| Source | IR |
| --- | --- |
| `char + char` | two `sext ... to i32`, `add i32`, then `trunc` on assignment |
| signed overflow | `add nsw` — the `nsw` *is* the UB, made explicit |
| unsigned overflow | `add` with no flag — defined wraparound |
| `(unsigned)x` from `int` | no instruction; the bits are unchanged |
| `(double)i` | `sitofp` |
| `(int)d` | `fptosi` — UB if out of range, and no trap |

`nsw` on signed arithmetic is the single most consequential frontend annotation
in C: it hands the optimizer permission to assume no overflow, which is exactly
what makes signed-overflow UB *observable* as a miscompile rather than a wrap.

## Pitfalls

- **Expecting `packed` to be free.** Every access becomes `align 1`.
- **Assuming a bit-field write only touches its own bits.** It rewrites the
  whole storage unit.
- **Treating `-O0` IR as representative.** It is a transliteration; measure and
  reason at `-O1` or above.
- **Losing `volatile` or `atomic` markers when rewriting IR.** The type
  qualifier is not in the IR; only the flag is.
- **Reading `and i1` as a short-circuit.** If you see it, an optimizer produced
  it after proving speculation safe — the frontend never emits it.
- **Assuming `restrict`/`noalias` was verified.** Nobody checked it.
- **Assuming struct layout is portable.** It is target ABI, per triple.

## BCIR notes

- The erasures on this page are exactly the facts a correspondence IR must
  carry *itself* if it wants them back later. Once a bit-field is shifts and
  masks, no downstream analysis can recover "this was a 12-bit field" without
  metadata that says so.
- BCIR's lane-typed and registry-first design exists to keep such facts
  addressable rather than inferred. When emitting IR from BCIR, treat every
  source-level fact you intend to rely on downstream as something to encode
  explicitly — a metadata node, an attribute, or an ABI-visible type — not as
  something to re-derive.
- `restrict`→`noalias` is the frontend's version of the alias-fact question GEM+
  slice G9 answered on the BCIR side: emit the fact only when an analysis
  established it, never as an optimistic default.

## See also

- [`04-cxx-lowering-rules.md`](04-cxx-lowering-rules.md) — the C++ additions
- [`05-abi-and-target-lowering.md`](05-abi-and-target-lowering.md) — signatures and calling conventions
- [`../04-memory/README.md`](../04-memory/README.md) — `alloca`/`load`/`store`/`getelementptr`
- [`../11-concurrency/README.md`](../11-concurrency/README.md) — `volatile` versus atomic
