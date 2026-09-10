# Pitfall 04 — Duplicate Symbol Definitions Across Modules

> ⚠️ **Retired / historical material.** This guide references the early **LLVM-IR-schema runtime**
> (`runtime/llvm/`, since removed). BCIR's current representation is the **MLIR dialect**
> (`mlir/include/BCIR/`) + the **C runtime** (`runtime/c/`); see `docs/PARITY.md`,
> `docs/kernel/HETEROGENEOUS_CHANNELS.md`, and `docs/BCIR_LANGREF.md`. Kept for historical context — do
> **not** follow the `runtime/llvm/` paths below.
<!-- allow-retired-paths -->


## BCIR instance

| Affected BCIR file(s) | Commit | Failing tool command | Fix summary | Related training chapters |
|---|---|---|---|---|
| `runtime/llvm/bcir_gem_seed.ll`; `runtime/llvm/bcir_worklist.ll` | `1f62e86` | `llvm-link runtime/llvm/bcir_gem_seed.ll runtime/llvm/bcir_worklist.ll -S -o /dev/null` | Keep one definition of `@execute_worklist` and make the other module declare it. | [`04-memory/03-global-variables.md`](../04-memory/03-global-variables.md); [`05-type-schema-drift.md`](05-type-schema-drift.md) |

## The error

```
llvm-link: error: linking module flags 'PIC Level': IDs have conflicting values in 'a.ll' and 'b.ll'
```

or, more commonly:

```
llvm-link: ERROR: Linking globals named '@foo': symbol multiply defined!
```

## What's happening

When you `llvm-link` multiple modules, each `define`d global function
or variable must have a unique name across the merged set, **unless**
its linkage allows merging (e.g., `weak`, `linkonce`, `available_externally`).

The most common case: two modules each `define`d the same function
under the assumption they wouldn't be linked together — until the
build pipeline started linking them.

## Minimal reproducer

`a.ll`:
```llvm
define void @execute_worklist(ptr %ctx) {
  ret void
}
```

`b.ll`:
```llvm
define void @execute_worklist(ptr %ctx) {
  ret void
}
```

```
$ llvm-link a.ll b.ll -S -o /dev/null
llvm-link: ERROR: Linking globals named '@execute_worklist': symbol multiply defined!
```

## Fix options

Choose one based on intent:

### (a) Keep one definition, declare in the other

If only one of the modules should *own* the implementation, the other
should `declare` it:

`a.ll` (the implementation):
```llvm
define void @execute_worklist(ptr %ctx) {
  ret void
}
```

`b.ll` (the consumer):
```llvm
declare void @execute_worklist(ptr)
```

### (b) Allow merging via linkage

If both modules legitimately define the same function (e.g., a
header-defined inline function emitted into every translation unit),
use `linkonce_odr`:

```llvm
define linkonce_odr void @execute_worklist(ptr %ctx) {
  ret void
}
```

Both modules can define it; the linker picks one and discards the
others. "ODR" = One Definition Rule; you assert all definitions are
semantically identical.

### (c) Rename one

If they're actually different functions, rename. Usually the easiest
fix when the collision is accidental.

## The real BCIR instance

Two LLVM IR files both defined `@bcir.gem.execute_worklist`:

- `runtime/llvm/bcir_gem_seed.ll:107`
- `runtime/llvm/bcir_worklist.ll:9`

Both were linked together by `validate_phase3.sh` and
`validate_phase4.sh`, breaking the pipeline.

Fixed in commit `1f62e86` by demoting the `bcir_gem_seed.ll` copy to
a `declare`:

```llvm
; runtime/llvm/bcir_gem_seed.ll
declare void @bcir.gem.execute_worklist(ptr, ptr, i64, ptr)
```

The definition lives only in `bcir_worklist.ll`.

## How to detect early

Before linking, you can grep across modules:

```bash
for f in *.ll; do
  echo "=== $f ==="
  grep -E '^define ' "$f" | awk '{print $NF}' | awk -F'(' '{print $1}'
done | sort | uniq -d
```

Anything that shows up multiple times is a collision risk. Run this in
CI alongside `llvm-link --only-needed`.

## Related issues

- **Different types for same-named symbol.** Worse than a redefinition
  — the linker may silently pick one, leading to runtime miscompiles.
- **Different linkage for same-named symbol.** `external` in one
  module, `private` in another won't link cleanly.
- **Internal symbol leaking across modules.** A `private` global
  shouldn't even be visible to the linker; if you're seeing collisions
  involving `private` names, check your linkage.

## See also

- [`../04-memory/03-global-variables.md`](../04-memory/03-global-variables.md) — linkage types table
- [`05-type-schema-drift.md`](05-type-schema-drift.md) — the type-level analogue
