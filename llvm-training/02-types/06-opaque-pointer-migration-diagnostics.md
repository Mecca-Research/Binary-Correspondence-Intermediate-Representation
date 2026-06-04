# Opaque Pointer Migration Diagnostics

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

## Verification commands

```sh
llvm-as llvm-training/02-types/examples/opaque-pointer-after.ll -o /tmp/opaque-pointer-after.bc
opt -passes=verify -disable-output /tmp/opaque-pointer-after.bc
opt -S llvm-training/02-types/examples/opaque-pointer-after.ll -o -
```

## Official references

- [LLVM Opaque Pointers documentation](https://llvm.org/docs/OpaquePointers.html)
- [LLVM Language Reference: Pointer Type](https://llvm.org/docs/LangRef.html#pointer-type)

## See also

- [`03-opaque-and-pointer-types.md`](03-opaque-and-pointer-types.md) — concepts behind opaque types and opaque pointers
- [`02-composite-types.md`](02-composite-types.md) — struct/array layout and GEP indexing
- [`../04-memory/02-load-store.md`](../04-memory/02-load-store.md) — explicit access types on loads and stores
- [`../04-memory/04-address-spaces.md`](../04-memory/04-address-spaces.md) — preserving `addrspace(N)` during migration
