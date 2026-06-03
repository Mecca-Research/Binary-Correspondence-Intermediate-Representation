# Pitfall 05 — Type Schema Drift Across Modules

## The error

```
llvm-link: error: linking type '%bcir.blob.header': type definitions differ
```

or, more subtly, the link succeeds but downstream verification fails
because two modules disagree on field count or types.

## What's happening

A **named type** (`%S = type { ... }`) declared in two different
modules must have the **same structural body**. When `llvm-link`
merges them, identical bodies are unified into one type. **Different
bodies** cause an error or — depending on linker version — produce
*two* types named `%S` and `%S.0`, leading to confusing downstream
errors.

## Minimal reproducer

`a.ll`:
```llvm
%Header = type {
  i32,    ; magic
  i16,    ; major
  i16,    ; minor
  i64,    ; offset
  i64,    ; size
  i64,    ; payload_offset
  i64     ; payload_size
}        ; 7 fields

define i32 @use_header(ptr %h) {
  %p = getelementptr %Header, ptr %h, i32 0, i32 0
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
```

`b.ll`:
```llvm
%Header = type {
  i32,    ; magic
  i16,    ; major
  i16,    ; minor
  i64,    ; offset
  i64,    ; size
  i64,    ; payload_offset
  i64,    ; payload_size
  i64     ; checksum
}        ; 8 fields — different!

define i64 @get_checksum(ptr %h) {
  %p = getelementptr %Header, ptr %h, i32 0, i32 7
  %v = load i64, ptr %p, align 8
  ret i64 %v
}
```

After link, the type is no longer unified. `a.ll`'s consumers see a
7-field struct; `b.ll`'s consumers see an 8-field struct; they
disagree on the offset of every field after `i32 magic`. The GEP
indices into the "same" type produce wrong offsets in one of them.

## Fix

**Single source of truth.** Pick one module to define the named type
canonically, and have every other module include the same definition
(or, in the case of code generators, share the type definition).

Practical patterns:

### (a) Schema header module

Create one module (`schema.ll`) that contains only type definitions:

```llvm
; schema.ll — the canonical types, shared across the project
%Header = type { i32, i16, i16, i64, i64, i64, i64, i64 }
%View   = type { ptr, ptr, ptr, ptr, ptr, ptr, i64, ptr, i64 }
```

Then every module that uses these declares the same body (just copy
the lines, or for tooling: link `schema.ll` into every artifact
first).

### (b) Code-generator emits types from one source

If you generate IR programmatically, define each named type *once*
in your code-generator, and reuse the resulting type object across
every module emission.

## The real BCIR instance

`%bcir.blob.header` and `%bcir.blob.view` were defined in three
files with inconsistent shapes:

| File | `%bcir.blob.header` fields | `%bcir.blob.view` fields |
|---|---|---|
| `bcir_registry_schema.ll` (old) | 7 | 8 |
| `bcir_blob_schema.ll` (old) | 8 | 9 |
| `bcir_blob_verify.ll` (old) | 8 | 9 |

`llvm-link` could not unify across these. Fixed in commit `1f62e86`
by aligning all three to the canonical 8-field header and 9-field
view (verified in the post-fix state by grepping):

```
=== bcir_registry_schema.ll ===   8 fields header,  9 fields view
=== bcir_blob_schema.ll       ===  8 fields header,  9 fields view
=== bcir_blob_verify.ll       ===  8 fields header,  9 fields view
```

## How to detect early

Grep all `.ll` files in a project for a type name and compare bodies:

```bash
# Look at every definition of %bcir.blob.header
grep -B0 -A20 '%bcir.blob.header = type' runtime/llvm/*.ll
```

When introducing a new named type, **declare it once, in a shared
file** — don't re-declare in every consumer.

Even better: tool the build so that all consumers `#include` the
single canonical schema file, or that the code generator emits the
shared types into every output.

## Related

- **Different alignment / packed-ness for "same" type.**
  `%S = type { i32, i64 }` (8-byte aligned, has padding) vs
  `%S = type <{ i32, i64 }>` (packed, no padding) — same field
  count, different layouts. Pure drift bait.

- **Opaque vs concrete.** `%S = type opaque` in one module and
  `%S = type { i32, i64 }` in another link cleanly — the concrete
  body wins. Fine as long as the consumers of the opaque view don't
  GEP into the type.

## See also

- `02-types/02-composite-types.md` — struct layout
- `02-types/03-opaque-and-pointer-types.md` — opaque types
- `08-pitfalls/04-duplicate-symbols.md` — the symbol-level analogue
