# `load` and `store`

## TL;DR

```llvm
%v = load   <type>, ptr <addr> [, align N] [, !meta]
store <type> <val>, ptr <addr> [, align N] [, !meta]
```

`load` reads a typed value from the address pointed at; `store`
writes one. With opaque pointers, the **load/store specifies the
access type** (the pointer doesn't carry a pointee type).

Atomic variants add `atomic` and an ordering keyword.

## load

```llvm
%v_i32   = load i32,   ptr %p, align 4
%v_f     = load float, ptr %p, align 4
%v_v4    = load <4 x i32>, ptr %p, align 16
%v_with_meta = load i32, ptr %p, align 4, !tbaa !3, !range !4
```

Optional clauses:

| Clause | Meaning |
|---|---|
| `volatile` | Don't optimize away or reorder (for MMIO etc.) |
| `atomic` | Treat as an atomic load — ordering required |
| `syncscope("...")` | Limit ordering to a sync scope (e.g., `singlethread`) |
| `<ordering>` | `unordered`, `monotonic`, `acquire`, `seq_cst` |
| `align N` | Alignment guarantee |

## store

```llvm
store i32 42, ptr %p, align 4
store <4 x float> %v, ptr %p, align 16
store volatile i32 0, ptr @hw_register, align 4
```

Same optional clauses as `load`. `store` has **no result** (it's not
on the left of `=`).

## Atomic variants

```llvm
; Atomic load
%v = load atomic i32, ptr %p seq_cst, align 4

; Atomic store
store atomic i32 1, ptr %p release, align 4

; Single-thread-scoped
%v = load atomic i32, ptr %p syncscope("singlethread") monotonic, align 4
```

Orderings (weakest to strongest):

| Ordering | Use case |
|---|---|
| `unordered` | Almost no guarantees; for race-free loads/stores you want the optimizer to leave alone |
| `monotonic` | Coherent per-location, but no inter-location ordering |
| `acquire` (loads only) | Subsequent loads/stores can't be moved before |
| `release` (stores only) | Prior loads/stores can't be moved after |
| `acq_rel` | Both (for `atomicrmw`/`cmpxchg`) |
| `seq_cst` | Sequentially consistent — strongest |

For full atomic semantics, see also `atomicrmw`, `cmpxchg`, and
`fence` in the reference.

## Examples

### Local variable

```llvm
define i32 @inc(i32 %x) {
  %tmp = alloca i32, align 4
  store i32 %x, ptr %tmp, align 4
  %v   = load i32, ptr %tmp, align 4
  %r   = add i32 %v, 1
  ret i32 %r
}
```

### Array element

```llvm
define i32 @get(ptr %arr, i32 %i) {
  %p = getelementptr inbounds i32, ptr %arr, i32 %i
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
```

### Struct field

```llvm
%Person = type { i32, float, ptr }

define float @get_height(ptr %p) {
  %h_p = getelementptr inbounds %Person, ptr %p, i32 0, i32 1
  %h   = load float, ptr %h_p, align 4
  ret float %h
}
```

### Volatile (e.g., MMIO)

```llvm
@HW_STATUS = external addrspace(1) global i32

define i32 @read_hw() {
  %v = load volatile i32, ptr addrspace(1) @HW_STATUS, align 4
  ret i32 %v
}
```

## Alignment

`align N` is a *guarantee* you give the compiler about the pointer's
alignment. If you lie, the program is UB.

| Type | Natural alignment (typical) |
|---|---|
| `i8` | 1 |
| `i16` | 2 |
| `i32` | 4 |
| `i64` | 8 |
| `float` | 4 |
| `double` | 8 |
| `<4 x float>` | 16 |
| `<8 x float>` | 32 |
| pointer | 8 (on 64-bit), 4 (on 32-bit) |

You can specify *less* than natural alignment to perform a misaligned
load/store — the compiler will emit the slower sequence.

## Metadata attachments

The most common ones:

| Metadata | Effect |
|---|---|
| `!tbaa !N` | Type-based alias analysis class |
| `!alias.scope`, `!noalias` | Scoped noalias |
| `!nontemporal !0` | Streaming/non-cached hint |
| `!invariant.load` | Loaded value won't change |
| `!nonnull !0` | (load only) Loaded pointer isn't null |
| `!range !N` | (load only) Result is in `[lo, hi)` |
| `!align !N` | (load only) Loaded pointer is N-aligned |

See [`../01-syntax/03-comments-metadata.md`](../01-syntax/03-comments-metadata.md).

## Pitfalls

- **Mismatched access type and stored data.** Storing `i64 42` and
  then `load i32, ptr %p` returns the low or high half depending on
  endianness — usually a bug.

- **Lying about alignment.** `align 16` on a pointer that's actually
  4-byte aligned: classic UB. Hard to track down.

- **Skipping `volatile` for MMIO.** The optimizer may coalesce or
  reorder loads/stores to the same address; device registers must
  never be reordered.

- **Atomic + missing ordering.** `load atomic i32, ptr %p` without
  an ordering keyword doesn't parse.

- **Atomic over a too-large type.** Most targets only support atomic
  load/store up to pointer width. `load atomic i256` won't lower.

- **Forgetting that `store` has no result.** `%x = store ...` doesn't
  parse.

## See also

- [`01-alloca.md`](01-alloca.md) — where `alloca` returns a pointer suitable for
  load/store
- [`03-global-variables.md`](03-global-variables.md) — loading/storing globals
- [`../02-types/03-opaque-and-pointer-types.md`](../02-types/03-opaque-and-pointer-types.md) — why the access type
  lives on the load/store
- [`../reference/instruction-quickref.md`](../reference/instruction-quickref.md) — `atomicrmw`, `cmpxchg`,
  `fence`
