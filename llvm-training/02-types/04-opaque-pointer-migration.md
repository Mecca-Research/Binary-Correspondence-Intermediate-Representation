# Opaque Pointer Migration

## TL;DR

Modern LLVM IR uses the opaque pointer spelling `ptr` instead of typed
pointer spellings such as `i32*`, `%Node*`, or `[16 x i8]*`.

The pointer value still has pointer properties, especially its address
space, but it no longer stores a pointee type. Operations that access
memory or compute typed offsets now spell the relevant type at the use
site:

```llvm
%v = load i32, ptr %p, align 4
store i32 %v, ptr %p, align 4
%field = getelementptr %Struct, ptr %p, i32 0, i32 1
```

Read opaque pointers as: **the pointer says where; the instruction says
what kind of object is accessed or indexed.**

## Why LLVM moved away from typed pointers

Older LLVM IR wrote a pointee type into every pointer type:

```llvm
i32*          ; pointer whose spelling says "points to i32"
%Pair*        ; pointer whose spelling says "points to %Pair"
[8 x i8]*     ; pointer whose spelling says "points to an 8-byte array"
```

That looked helpful, but in practice it created more noise than truth:

- **The pointee type was often redundant.** A `load` already names the
  loaded type, and a `store` already names the stored value type.
- **The pointee type was not a reliable memory-object model.** A pointer
  may be used to access bytes, fields, whole aggregates, or different
  views of the same allocation at different program points.
- **Typed pointers forced no-op pointer bitcasts.** Frontends and
  optimizers had to insert `bitcast` instructions just to satisfy pointer
  spelling, even when the machine address did not change.
- **IR consumers overfit to pointer spelling.** Analyses could accidentally
  infer layout or aliasing facts from `i32*` versus `%Pair*` instead of
  inspecting the instruction that actually performs the access.

Opaque pointers remove the fake precision. The official LLVM Opaque
Pointers documentation describes this as a simplification analogous to
removing signedness from integer types: the key property moved from the
container type into the operations that need it.

## Where the pointee type moved

### `load`

The loaded type is the first operand of the instruction:

```llvm
; Opaque-pointer form
%v = load i32, ptr %p, align 4
```

Do not ask, "What does `%p` point to?" Ask, "What type does this `load`
read here?" In this example, the access type is `i32`.

### `store`

The stored value carries the access type:

```llvm
; Opaque-pointer form
store i32 %v, ptr %p, align 4
```

The pointer operand is just the destination address. The value operand
says this store writes an `i32`.

### `getelementptr`

`getelementptr` still needs an element type to compute offsets and field
indices, but that type is now explicit in the instruction:

```llvm
; Opaque-pointer form
%field = getelementptr %Struct, ptr %p, i32 0, i32 1
```

The `%Struct` argument is not recovered from `ptr`; it is part of the GEP
itself. If you choose the wrong type here, you compute the wrong offset.

### `call` and `invoke`

Function signatures also carry the meaningful types. A pointer argument
is `ptr`, while the callee declaration and the argument list describe how
that pointer is used:

```llvm
declare void @consume_i32_ptr(ptr)

define void @caller(ptr %p) {
entry:
  call void @consume_i32_ptr(ptr %p)
  ret void
}
```

If the callee loads from `%p`, inspect the callee body or its API contract;
do not reconstruct a pointee type from the argument spelling.

## Before/after examples

The legacy examples below are intentionally shown as historical reference.
They may not assemble with a modern LLVM toolchain. See
[`examples/typed-pointer-before.ll.txt`](examples/typed-pointer-before.ll.txt)
for the complete non-modern reference and
[`examples/opaque-pointer-after.ll`](examples/opaque-pointer-after.ll) for a
verifier-clean modern module.

### 1. Typed pointer load/store

Before, the pointer operand repeated the pointee type:

```llvm
; Legacy typed-pointer IR
%v = load i32, i32* %p, align 4
store i32 %v, i32* %q, align 4
```

After, the load/store operation keeps the access type and the pointer is
opaque:

```llvm
; Modern opaque-pointer IR
%v = load i32, ptr %p, align 4
store i32 %v, ptr %q, align 4
```

### 2. Typed pointer GEP

Before, the GEP's base pointer spelling carried the apparent aggregate
pointee type:

```llvm
; Legacy typed-pointer IR
%field = getelementptr inbounds %Pair, %Pair* %p, i32 0, i32 1
%v = load i32, i32* %field, align 4
```

After, the GEP names `%Pair` directly and returns `ptr`:

```llvm
; Modern opaque-pointer IR
%field = getelementptr inbounds %Pair, ptr %p, i32 0, i32 1
%v = load i32, ptr %field, align 4
```

### 3. Typed pointer bitcast chains

Before, changing the apparent pointee type required pointer-to-pointer
`bitcast`s even though the address stayed the same:

```llvm
; Legacy typed-pointer IR
%bytes = bitcast %Pair* %p to i8*
%as_i32 = bitcast i8* %bytes to i32*
%v = load i32, i32* %as_i32, align 4
```

After, the same address is a `ptr`; the access instruction names the type:

```llvm
; Modern opaque-pointer IR
%v = load i32, ptr %p, align 4
```

If you still need byte-wise addressing, keep the GEP element type explicit:

```llvm
%byte3 = getelementptr i8, ptr %p, i64 3
%b = load i8, ptr %byte3, align 1
```

### 4. Opaque pointer equivalent module shape

A migrated function signature typically changes from typed pointer
arguments to `ptr` arguments:

```llvm
%Pair = type { i32, i32 }

define i32 @sum_pair(ptr %p) {
entry:
  %a.ptr = getelementptr inbounds %Pair, ptr %p, i32 0, i32 0
  %b.ptr = getelementptr inbounds %Pair, ptr %p, i32 0, i32 1
  %a = load i32, ptr %a.ptr, align 4
  %b = load i32, ptr %b.ptr, align 4
  %sum = add i32 %a, %b
  ret i32 %sum
}
```

The IR is not less typed. The types have moved to the instructions where
they are semantically relevant.

## Migration checklist

Use this checklist when converting old IR, frontend output, tests, or an IR
consumer from typed pointers to opaque pointers.

1. **Remove assumptions based on pointer spelling.**
   - Replace checks like "is this an `i32*`?" with checks of the operation
     that uses the pointer.
   - Treat `ptr` as an address value, not as a promise about the object behind
     the address.

2. **Inspect load/store/GEP/call signatures for the actual access type.**
   - `load i32, ptr %p` reads an `i32`.
   - `store %Pair %v, ptr %p` writes a `%Pair` value.
   - `getelementptr %Pair, ptr %p, ...` indexes according to `%Pair` layout.
   - `call void @f(ptr %p)` tells you only that a pointer is passed; inspect
     `@f`'s declaration, definition, attributes, and API contract for meaning.

3. **Avoid reconstructing pointee types from `ptr`.**
   - Do not invent a single "real" pointee type for a pointer SSA value.
   - A pointer may be used by multiple operations with different access types,
     especially around byte buffers, unions, object headers, or serialization
     layouts.

4. **Delete pointer-to-pointer bitcast chains that only changed spelling.**
   - `bitcast ptr %p to ptr` is unnecessary.
   - Keep real conversions such as `ptrtoint`, `inttoptr`, and address-space
     conversions only when the semantics require them.

5. **Preserve address-space information.**
   - `i32 addrspace(1)*` migrates to `ptr addrspace(1)`, not plain `ptr`.
   - Use `addrspacecast` only for intentional cross-address-space conversion.

6. **Run modern LLVM tools on migrated modules.**
   - Assemble textual IR with `llvm-as`.
   - Run the verifier with `opt -passes=verify -disable-output`.
   - Optionally run `opt -S` to confirm that unnecessary bitcasts disappeared
     and the printed IR is in modern opaque-pointer form.

## Pitfalls

### Losing address-space information

Opaque pointers are not all identical. The address space remains part of the
pointer type:

```llvm
; Legacy typed-pointer IR
i32 addrspace(1)* %global_p

; Correct migration
ptr addrspace(1) %global_p

; Incorrect migration: silently moved to address space 0
ptr %global_p
```

This matters on GPU targets, non-integral pointer targets, and any target whose
datalayout assigns different meanings or sizes to different address spaces.

### Confusing pointer type with memory object type

A `ptr` value is not an object schema. This is legal IR shape:

```llvm
%tag = load i8, ptr %p, align 1
%payload = getelementptr i8, ptr %p, i64 8
%word = load i64, ptr %payload, align 8
```

The same base pointer participates in `i8` and `i64` accesses. The memory
object's real source-language layout may be known to a frontend or ABI, but it
is not encoded in the spelling `ptr`.

### Keeping unnecessary bitcasts after migration

Typed-pointer IR often accumulated chains like `%T* -> i8* -> i32*`. After
migration, these may become meaningless `ptr -> ptr` casts. Remove them unless
they express something real:

- Keep `addrspacecast` when changing address spaces intentionally.
- Keep `ptrtoint`/`inttoptr` when the IR really crosses between pointers and
  integers.
- Remove pointer `bitcast`s whose only purpose was to satisfy old pointee-type
  spelling.

## Official references

- [LLVM Opaque Pointers documentation](https://llvm.org/docs/OpaquePointers.html)
- [LLVM Language Reference: Pointer Type](https://llvm.org/docs/LangRef.html#pointer-type)

## See also

- [`03-opaque-and-pointer-types.md`](03-opaque-and-pointer-types.md) — concepts behind opaque types and opaque pointers
- [`02-composite-types.md`](02-composite-types.md) — struct/array layout and GEP indexing
- [`../04-memory/02-load-store.md`](../04-memory/02-load-store.md) — explicit access types on loads and stores
- [`../04-memory/04-address-spaces.md`](../04-memory/04-address-spaces.md) — preserving `addrspace(N)` during migration
