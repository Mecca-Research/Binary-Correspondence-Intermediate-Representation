# Advanced IR 01 — Common Intrinsics

## TL;DR

LLVM intrinsics are calls whose semantics are known to LLVM itself.
They are written as ordinary `declare` + `call` pairs, but their names,
attributes, and overloaded suffixes are part of the contract:

```llvm
declare void @llvm.memcpy.p0.p0.i64(ptr noalias, ptr noalias, i64, i1 immarg)
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)
```

Use intrinsics when the operation has optimizer-visible meaning that a
normal helper function would hide. In BCIR-style runtime IR this most
often means memory operations, overflow-checked arithmetic, lifetime
markers, and explicit prefetch hints.

## Memory-transfer intrinsics

### `llvm.memcpy`

`llvm.memcpy` copies bytes from a source object to a destination object.
The source and destination ranges must not overlap; use `llvm.memmove`
if overlap is possible.

Opaque-pointer IR still encodes the address-space overloads and the size
integer in the intrinsic name:

```llvm
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly,
                                    ptr noalias nocapture readonly,
                                    i64,
                                    i1 immarg)
```

Name components:

- `p0` — destination pointer is in address space 0.
- `p0` — source pointer is in address space 0.
- `i64` — the length argument has type `i64`.
- The final `i1` is the volatile flag and should normally be `false`.

BCIR note: `memcpy` is a good fit for fixed-format record movement or
copying serialized blobs when the source and destination are known not
to alias. The optimizer can reason about the copy size and may inline,
combine, or remove it.

### `llvm.memmove`

`llvm.memmove` is the overlap-safe form:

```llvm
declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly,
                                     ptr nocapture readonly,
                                     i64,
                                     i1 immarg)
```

Use it when shifting data within the same allocation or ring buffer.
Do not use `memcpy` and merely hope the ranges do not overlap; that is a
semantic promise to LLVM.

### `llvm.memset`

`llvm.memset` fills a memory range with one byte value:

```llvm
declare void @llvm.memset.p0.i64(ptr nocapture writeonly,
                                 i8,
                                 i64,
                                 i1 immarg)
```

The pointer address space and length integer type are overloaded in the
name. The fill byte is always `i8`.

## Overflow-checked arithmetic

The checked arithmetic intrinsics return a two-field struct:

```llvm
{ T result, i1 overflow_flag }
```

Common additions:

```llvm
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64)
declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64)
```

- `llvm.uadd.with.overflow.<T>` treats operands as unsigned.
- `llvm.sadd.with.overflow.<T>` treats operands as signed two's-complement.
- The IR type is still just `iN`; signedness lives in the intrinsic name.

The overflow flag is a normal `i1` value. Branch on it, store it, or
convert it as needed. Do not replace checked arithmetic with `add nuw`
or `add nsw`: those flags assert that overflow cannot occur and produce
poison if the assertion is false.

## Lifetime intrinsics

Lifetime intrinsics mark the period during which a stack or temporary
object is live:

```llvm
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
```

The size argument is `immarg`: it must be a compile-time constant.
Use `-1` only when the size is unknown. These markers are optimization
hints; they do not allocate or deallocate memory.

BCIR note: lifetime markers are useful around scratch buffers created by
frontends or hand-written helper IR. They help stack coloring and dead
store elimination but should not be used to model ownership or runtime
validity checks.

## Prefetch

`llvm.prefetch` asks the target to fetch a cache line before a later
load or store:

```llvm
declare void @llvm.prefetch(ptr, i32 immarg, i32 immarg, i32 immarg)

; read prefetch, high locality, data cache
call void @llvm.prefetch(ptr %p, i32 0, i32 3, i32 1)
```

The integer arguments are all `immarg`:

| Argument | Typical values |
|---|---|
| `rw` | `0` read, `1` write |
| `locality` | `0` no temporal locality through `3` high locality |
| `cache type` | `1` data cache, `0` instruction cache on targets that distinguish them |

BCIR note: prefetch can matter for predictable scans over claim tables,
blob pages, or work queues. It is only a hint. It is not portable as a
performance guarantee, and a backend may ignore it.

## How overloaded intrinsic declarations are encoded

LLVM does not infer the overloaded form from the call site. The
intrinsic name is the selected overload:

| Operation | Example declaration | Encoded overloads |
|---|---|---|
| Copy bytes | `@llvm.memcpy.p0.p0.i64` | destination pointer AS, source pointer AS, length type |
| Fill bytes | `@llvm.memset.p0.i64` | destination pointer AS, length type |
| Checked add | `@llvm.uadd.with.overflow.i32` | integer type |
| Lifetime | `@llvm.lifetime.start.p0` | pointer address space |
| Math | `@llvm.sqrt.f64` | floating type |

With opaque pointers, `ptr` no longer carries a pointee type, but
intrinsics still need address-space and scalar/vector type overloads in
their names. A mismatch between the suffix and the formal parameter
list is a verifier error or, worse, an accidental declaration of a name
that LLVM does not recognize as the intended intrinsic.

## Finding canonical signatures

Use primary LLVM sources instead of copying a random declaration from an
old `.ll` file:

1. **LLVM LangRef** — start at the intrinsic's LangRef section for
   semantics and examples.
2. **TableGen definitions** — inspect `llvm/include/llvm/IR/Intrinsics.td`
   in the LLVM source tree for generic intrinsics.
3. **Generated intrinsic tables** — when working inside an LLVM build,
   generated files encode the resolved overload and attribute data.
4. **Your toolchain** — produce IR with `clang -S -emit-llvm` for a
   minimal C/C++ example, then run `llvm-as` and `opt -passes=verify`.

The TableGen declaration tells you which operands are overloaded, which
parameters are `immarg`, and which memory/side-effect attributes LLVM
expects.

## Pitfalls

- **Wrong overloaded suffix/signature.** `@llvm.memcpy.p0.p0.i32` takes
  an `i32` length. Calling it with `i64 %n` is not the same intrinsic.
- **Non-constant `immarg`.** `llvm.prefetch` and `lifetime.*` require
  literal constants for their immediate arguments.
- **Assuming intrinsics are ordinary functions.** Do not define an
  intrinsic body. Declare it and call it.
- **Assuming an intrinsic is portable.** Generic intrinsics have stable
  IR semantics, but their lowering and performance are target-dependent.
