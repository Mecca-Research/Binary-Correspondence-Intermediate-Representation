# Quickref: Opaque Pointers

## Mental model

- Pointer values print as `ptr` or `ptr addrspace(N)`; they do **not** encode a pointee type.
- The pointee or access type now lives on the operation: `load i32, ptr %p`, `store i32 %v, ptr %p`, `getelementptr %T, ptr %base, ...`, or a function/call signature.
- Address spaces are still part of the pointer type and must be preserved exactly.

## Edit checklist

- Replace typed-pointer spellings such as `i32*` with `ptr` only after moving the element type to loads, stores, GEPs, allocas, calls, and constants.
- Replace API code that asks a pointer for its element type with operation-specific types, such as load/store value type, GEP source element type, or callee function type.
- Remove no-op pointer-to-pointer bitcasts that only changed pointee type; keep address-space casts only when semantically required.
- Check constants and globals: array/string initializers still have concrete aggregate types even when references to them are opaque `ptr`.

## Common verifier/parser failures

| Symptom | Likely fix |
| --- | --- |
| Old text contains `i8*`, `%T*`, or `<4 x i32>*` | Rewrite pointer operands as `ptr` and add explicit operation types. |
| GEP result indexes the wrong struct/array shape | Use the real source element type in `getelementptr`. |
| Address-space mismatch between producer and consumer | Preserve `ptr addrspace(N)` or use the correct `addrspacecast`. |
| Pass code expects pointer element type | Thread the element/access/function type from the instruction or ABI schema. |

## Deep links

- [`../02-types/README.md`](../02-types/README.md)
- [`../02-types/04-opaque-pointer-migration.md`](../02-types/04-opaque-pointer-migration.md)
- [`../02-types/05-opaque-pointer-migration-patterns.md`](../02-types/05-opaque-pointer-migration-patterns.md)
- [`../04-memory/02-load-store.md`](../04-memory/02-load-store.md)
- [`../08-pitfalls/11-address-space-confusion.md`](../08-pitfalls/11-address-space-confusion.md)
