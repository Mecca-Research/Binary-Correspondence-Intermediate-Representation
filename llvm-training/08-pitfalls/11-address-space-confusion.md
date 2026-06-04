# Pitfall 11 — Address-Space Confusion

## BCIR instance

| Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|
| Training-only exemplar; no affected BCIR `.ll` file recorded | Unknown | `opt -passes=verify <bcir-address-space-output>.ll -o /dev/null` | Preserve pointer address spaces through loads, stores, GEPs, casts, and helper signatures. | [`04-memory/04-address-spaces.md`](../04-memory/04-address-spaces.md); [`02-types/03-opaque-and-pointer-types.md`](../02-types/03-opaque-and-pointer-types.md); [`04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) |

## The error

When a value in one address space is used where another is required:

```text
'%p' defined with type 'ptr addrspace(1)' but expected 'ptr'
```

When a generator tries to bitcast between address spaces:

```text
invalid cast opcode for cast from 'ptr addrspace(1)' to 'ptr'
```

The exact wording varies by tool and LLVM version, but the important clue is the
pair of pointer types: `ptr` and `ptr addrspace(N)` are not the same type.

## Minimal reproducer

```llvm
@gpu_global = addrspace(1) global i32 0, align 4

define i32 @bad() {
entry:
  ; @gpu_global has type ptr addrspace(1), not ptr.
  %v = load i32, ptr @gpu_global, align 4      ; ❌ wrong pointer type
  ret i32 %v
}
```

Another common bad shape:

```llvm
define ptr @bad_cast(ptr addrspace(1) %p) {
entry:
  %q = bitcast ptr addrspace(1) %p to ptr      ; ❌ use addrspacecast
  ret ptr %q
}
```

## Why it happens

With opaque pointers, the pointee type no longer appears in the pointer type, but
the address space still does. `ptr`, `ptr addrspace(1)`, and `ptr addrspace(3)`
are distinct pointer types that may have different representations, sizes,
legal operations, and target lowering rules.

Address-space numbers are target-defined. A frontend or lifter that assumes
`addrspace(1)` always means the same memory region will eventually generate IR
that is valid for one target and wrong for another.

## Fix pattern

Keep address spaces in your type model and spell them at every pointer use:

```llvm
@gpu_global = addrspace(1) global i32 0, align 4

define i32 @good() {
entry:
  %v = load i32, ptr addrspace(1) @gpu_global, align 4
  ret i32 %v
}
```

Use `addrspacecast` for cross-space conversions only when the target semantics
allow such a conversion:

```llvm
define ptr @generic_view(ptr addrspace(1) %p) {
entry:
  %q = addrspacecast ptr addrspace(1) %p to ptr
  ret ptr %q
}
```

Do not use `bitcast` to change address spaces. Do not merge pointers from
different address spaces in one `phi` unless you first convert them to a common,
semantically valid address space.

## BCIR-relevant note

BCIR often needs to represent memory classes recovered from a binary: RAM,
thread-local storage, MMIO, stack, GPU global/shared/private memory, or segmented
regions. Preserve that memory-space distinction in IR pointer types or in an
explicit side table. Accidentally normalizing every pointer to `ptr` can erase
information the backend, verifier, or correspondence checker needs.

## See also

- [`../04-memory/04-address-spaces.md`](../04-memory/04-address-spaces.md) — address-space syntax and mental model
- [`../02-types/03-opaque-and-pointer-types.md`](../02-types/03-opaque-and-pointer-types.md) — opaque pointers still carry address space
- [`../04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) — `addrspace()` on globals
