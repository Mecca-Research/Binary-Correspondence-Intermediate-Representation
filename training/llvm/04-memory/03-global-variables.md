# Global Variables

## TL;DR

```llvm
@counter        = global i32 0, align 4
@const_value    = constant i32 42
@msg            = private unnamed_addr constant [6 x i8] c"hi!\0A\00"
@tls            = thread_local global i32 0
@gpu_mem        = addrspace(1) global [256 x i32] zeroinitializer
@hidden_export  = hidden global i32 0
```

Globals live at module scope, persist for the program's lifetime,
and are reachable via `@name`. Linkage, visibility, thread-locality,
address space, alignment, and section can all be specified.

## Full grammar (abridged)

```
@<name> = [<linkage>] [<preemption>] [<visibility>] [<dll-storage>]
          [<thread-local>] [<unnamed-addr>] [<addrspace>]
          [<externally_initialized>]
          (global | constant) <type> [<initializer>]
          [, section "..."] [, comdat ...] [, align N]
          [, !meta ...]
```

## Linkage types

Controls who can see and merge the symbol.

| Linkage | Visible to other modules? | Merge rules |
|---|---|---|
| `private` | No | Stripped from symbol table |
| `internal` | No | Static — module-local |
| `available_externally` | Yes (declaration only emitted) | Definition is "available" but not emitted; the linker resolves to a real definition elsewhere |
| `linkonce` | Yes | One definition wins; can be discarded |
| `linkonce_odr` | Yes | Same, but caller asserts ODR (One Definition Rule) |
| `weak` | Yes | One definition wins; preserved even if unreferenced |
| `weak_odr` | Yes (ODR-respecting) | |
| `common` | Yes (uninitialized only) | Tentative definitions merge |
| `appending` | Yes | Special: array-typed globals are concatenated |
| `extern_weak` | Yes (declaration) | Resolves to null if no definition exists |
| `external` | Yes | The default for exported symbols |

Default linkage (no keyword): `external` for declarations, `external`
for definitions of named globals.

## Visibility

| Visibility | Meaning |
|---|---|
| `default` | Visible from other shared objects (default) |
| `hidden` | Not visible from other shared objects |
| `protected` | Visible but not overridable |

```llvm
@public_export  = global i32 0                  ; default
@hidden_symbol  = hidden global i32 0
```

## DLL storage (Windows)

```llvm
@imported = external dllimport global i32
@exported = dllexport global i32 0
```

## Preemption

```llvm
@local      = dso_local global i32 0       ; cannot be preempted at runtime
@preemptable = dso_preemptable global i32 0 ; may be replaced by a stronger definition
```

`dso_local` enables more aggressive optimization (the address is
known at link time).

## Thread-local storage (TLS)

```llvm
@tls         = thread_local global i32 0
@tls_le      = thread_local(localexec) global i32 0
```

TLS models (mostly relevant for shared libraries):

- `localdynamic` — TLS in a dynamically loaded library
- `initialexec` — TLS resolved at process startup
- `localexec` — fastest; only valid in the main executable

## Address space

```llvm
@global_mem = addrspace(1) global i32 0     ; GPU global memory, typically
@constant_mem = addrspace(4) global i32 0   ; GPU constant memory, typically
```

The numeric mapping is target-defined. See
[`04-address-spaces.md`](04-address-spaces.md).

## `global` vs `constant`

```llvm
@a = global i32 42         ; mutable, .data
@b = constant i32 42       ; immutable, .rodata
```

`global` may or may not have an initializer:
```llvm
@x = global i32 42         ; initialized
@y = global i32 0          ; zero
@z = global i32 undef      ; uninitialized
@arr = global [1024 x i32] zeroinitializer
```

`constant` must have an initializer.

## Initialization

Initializers can be:

- A literal (`i32 42`, `float 3.14`)
- A composite literal (`[3 x i32] [i32 1, i32 2, i32 3]`)
- `zeroinitializer` (zero all bytes)
- `undef` (uninitialized, optimizer may choose anything)
- A constant expression (`ptr getelementptr (...)`)
- Another global's address (`ptr @other`)

```llvm
@a = global i32 42
@p = global ptr @a                  ; points to @a
@arr_of_ptrs = global [2 x ptr] [
  ptr @a,
  ptr getelementptr inbounds ([3 x i32], ptr @table, i32 0, i32 1)
]
@table = constant [3 x i32] [i32 10, i32 20, i32 30]
```

## Section and alignment

```llvm
@data    = global i32 0, section ".mydata", align 4
@aligned = global i32 0, align 16
```

`section` places the global in a specific output-file section.
Useful for embedded targets that need specific memory regions.

## Comdat (Windows/COFF)

Allows multiple modules to define the same global; one survives at
link time:

```llvm
$mygroup = comdat any
@x = global i32 0, comdat($mygroup)
```

## Reading and writing

```llvm
@counter = global i32 0, align 4

define void @increment() {
  %v   = load i32, ptr @counter, align 4
  %vp1 = add i32 %v, 1
  store i32 %vp1, ptr @counter, align 4
  ret void
}
```

Globals are referenced by name (`@counter`); the value of that name
is its address (a `ptr`). Use `load`/`store` to access the underlying
data.

## Pitfalls

- **Modifying a `constant`.** UB.

- **Forgetting `section` on embedded targets.** A peripheral register
  needs the right section to land at the right address.

- **`private` on a global that's referenced from another module.**
  The linker won't see it.

- **TLS in shared library code without the right TLS model.**
  Performance varies by ~10× across `localexec` (fastest) and
  `localdynamic` (slowest).

- **Forgetting `dso_local` for symbols you know are local.** The
  optimizer is more conservative without it, generating slower code
  on PIC targets.

- **`comdat` mismatch.** Different modules declaring the same comdat
  with different selection kinds (`any`, `largest`, etc.) breaks
  linking.

## See also

- [`../03-constants/04-global-vs-local.md`](../03-constants/04-global-vs-local.md) — global constants in detail
- [`04-address-spaces.md`](04-address-spaces.md) — `addrspace()`
- [`02-load-store.md`](02-load-store.md) — using `load`/`store` with globals
- [`../01-syntax/01-modules-functions-blocks.md`](../01-syntax/01-modules-functions-blocks.md) — globals at module
  scope
