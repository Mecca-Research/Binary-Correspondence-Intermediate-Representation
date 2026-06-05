# Types: Primitive, Composite, and Pointer IR

This chapter explains how LLVM IR represents values with primitive types,
aggregate/vector composites, and pointer values. It also covers the modern
opaque-pointer model and the migration work needed when older typed-pointer IR
or APIs are updated for LLVM versions where `ptr` no longer carries a pointee
type.

## Chapter dispatcher

| Need | Read |
| --- | --- |
| Integers, floating-point types, `void`, labels, tokens, and basic type spelling | [`01-primitive-types.md`](01-primitive-types.md) |
| Arrays, structs, packed structs, vectors, scalable vectors, and aggregate access basics | [`02-composite-types.md`](02-composite-types.md) |
| Pointer spelling, address spaces, opaque pointers, and where pointee type information now lives | [`03-opaque-and-pointer-types.md`](03-opaque-and-pointer-types.md) |
| Migration overview for replacing typed-pointer assumptions with explicit operation types | [`04-opaque-pointer-migration.md`](04-opaque-pointer-migration.md) |
| Concrete migration patterns for loads, stores, GEPs, calls, bitcasts, and aliases | [`05-opaque-pointer-migration-patterns.md`](05-opaque-pointer-migration-patterns.md) |
| Diagnostics and review checks for catching stale typed-pointer assumptions | [`06-opaque-pointer-migration-diagnostics.md`](06-opaque-pointer-migration-diagnostics.md) |
| Before/after migration examples showing old typed-pointer IR and modern opaque-pointer IR | [`07-opaque-pointer-migration-examples.md`](07-opaque-pointer-migration-examples.md) |

## Opaque pointer migration path

Use this focused path when updating old LLVM IR, frontend code, or pass code
that still assumes pointer values encode their pointee types:

1. Start with [`04-opaque-pointer-migration.md`](04-opaque-pointer-migration.md)
   for the migration goals and the key mental model change.
2. Apply the edit patterns in
   [`05-opaque-pointer-migration-patterns.md`](05-opaque-pointer-migration-patterns.md)
   to move type facts onto instructions and APIs that still require them.
3. Use [`06-opaque-pointer-migration-diagnostics.md`](06-opaque-pointer-migration-diagnostics.md)
   as a checklist for verifier errors, parser errors, and code-review smells.
4. Compare your result with the before/after walkthroughs in
   [`07-opaque-pointer-migration-examples.md`](07-opaque-pointer-migration-examples.md).

## See also

- [`../04-memory/02-load-store.md`](../04-memory/02-load-store.md) — load and
  store syntax that carries explicit access types with opaque pointers.
- [`02-composite-types.md`](02-composite-types.md) — GEP source element
  types and aggregate indexing for composite values.
- [`../08-pitfalls/11-address-space-confusion.md`](../08-pitfalls/11-address-space-confusion.md)
  — address-space mistakes that are easy to miss during pointer migration.
