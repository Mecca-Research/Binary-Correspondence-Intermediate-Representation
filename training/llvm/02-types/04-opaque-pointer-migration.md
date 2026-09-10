# Opaque Pointer Migration

## TL;DR

Modern LLVM IR uses the opaque pointer spelling `ptr` instead of typed pointer spellings such as `i32*`, `%Node*`, or `[16 x i8]*`. The pointer value still has pointer properties, especially its address space, but it no longer stores a pointee type.

Operations that access memory or compute typed offsets now spell the relevant type at the use site:

```llvm
%v = load i32, ptr %p, align 4
store i32 %v, ptr %p, align 4
%field = getelementptr %Struct, ptr %p, i32 0, i32 1
```

Read opaque pointers as: **the pointer says where; the instruction says what kind of object is accessed or indexed.**

## Lookup dispatcher

| Need | Read |
|---|---|
| Why typed pointers went away | This file, below |
| Where access/index types moved for `load`, `store`, `getelementptr`, `call`, and `invoke` | [`05-opaque-pointer-migration-patterns.md`](05-opaque-pointer-migration-patterns.md) |
| Migration checklist and operation-by-operation rewrite rules | [`05-opaque-pointer-migration-patterns.md`](05-opaque-pointer-migration-patterns.md) |
| Address-space, object-schema, and leftover-bitcast pitfalls | [`06-opaque-pointer-migration-diagnostics.md`](06-opaque-pointer-migration-diagnostics.md) |
| Historical typed-pointer snippets and modern equivalents | [`07-opaque-pointer-migration-examples.md`](07-opaque-pointer-migration-examples.md) |
| Complete before/after modules | [`examples/typed-pointer-before.ll.txt`](examples/typed-pointer-before.ll.txt), [`examples/opaque-pointer-after.ll`](examples/opaque-pointer-after.ll) |

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

## Official references

- [LLVM Opaque Pointers documentation](https://llvm.org/docs/OpaquePointers.html)
- [LLVM Language Reference: Pointer Type](https://llvm.org/docs/LangRef.html#pointer-type)

## See also

- [`03-opaque-and-pointer-types.md`](03-opaque-and-pointer-types.md) — concepts behind opaque types and opaque pointers
- [`02-composite-types.md`](02-composite-types.md) — struct/array layout and GEP indexing
- [`../04-memory/02-load-store.md`](../04-memory/02-load-store.md) — explicit access types on loads and stores
- [`../04-memory/04-address-spaces.md`](../04-memory/04-address-spaces.md) — preserving `addrspace(N)` during migration
