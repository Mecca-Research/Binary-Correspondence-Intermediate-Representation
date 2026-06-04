# Opaque Types and Pointer Types

## TL;DR

Two related but distinct concepts:

- **Opaque type** (`%T = type opaque`) — a named type whose body is
  not specified at the declaration site. Forward declaration.
- **Opaque pointer** (`ptr`) — a pointer whose pointee type is not
  part of the pointer's type. Universal in LLVM ≥ 15.

Both reduce coupling between the IR and consumers, at the cost of
moving "what type is this?" decisions to the use site.

## Opaque types

```llvm
; Declare without a body
%MyStruct = type opaque

; Functions can take pointers to opaque types
declare void @work_on_struct(ptr %s)

; Later in the same or another module, define the body
%MyStruct = type { i32, ptr, [16 x i8] }
```

Use cases:

- **Forward declarations** — break circular references between named
  types.
- **Implementation hiding** — expose `%MyStruct` as opaque in a public
  interface module; define it concretely in the implementation.

In modern LLVM IR with opaque pointers, opaque types are less needed
than they used to be — `ptr` already hides the pointee — but they
still help when consumers need a *named* abstract type for type
checking or for matching named types across modules.

## Opaque pointers — the modern default

```llvm
ptr                    ; pointer in default address space (0)
ptr addrspace(1)       ; pointer in global address space
ptr addrspace(3)       ; pointer in shared/local address space
```

The pointer doesn't carry a pointee type. Every operation that needs
to know what the pointer points at writes the type at the operation
site:

```llvm
%v   = load i32, ptr %p, align 4         ; loading an i32
%w   = load float, ptr %p, align 4       ; loading a float (different access kind through same ptr)
store i64 42, ptr %p, align 8            ; storing an i64

%p4  = getelementptr i32, ptr %p, i32 5  ; advance by 5 * sizeof(i32)
%pf  = getelementptr float, ptr %p, i32 5 ; advance by 5 * sizeof(float)
```

The `ptr` itself is the same; the operation says how to interpret
what's behind it.

### Why opaque pointers?

Before LLVM 15, every pointer was typed: `i32*`, `[10 x float]*`,
`{ i32, i64 }*`. This:

- **Bloated the IR** with redundant type info that every load/store
  *already* specified.
- **Required `bitcast`s** to change the apparent type of a pointer,
  which were no-ops at the machine level but cluttered the IR.
- **Confused alias analysis** — the optimizer needs to reason about
  *what kind of access happens*, not what shape the pointer claims.

Opaque pointers (`ptr`) collapse the model: a pointer is a pointer is
a pointer.

### Legacy typed pointer syntax

Still parses in current LLVM, but treated as `ptr`:

```llvm
i32*                   ; deprecated; same as `ptr`
[10 x i32]*            ; deprecated
{ i32, ptr }*          ; deprecated
```

The `bitcast`s that used to be needed to change typed-pointer
varieties are no longer required. New IR should never write typed
pointers.

### Address spaces and `ptr`

The address space attaches to the pointer:

```llvm
ptr                  ; default (0): generic
ptr addrspace(1)     ; e.g., GPU global memory
ptr addrspace(3)     ; e.g., GPU shared memory
ptr addrspace(4)     ; e.g., constant memory
```

Address spaces are target-dependent. The numbering is convention; the
datalayout string can declare attributes per address space.

To convert between address spaces:

```llvm
%g = addrspacecast ptr %p to ptr addrspace(1)
```

See [`../04-memory/04-address-spaces.md`](../04-memory/04-address-spaces.md).

## Combining opaque types and opaque pointers

```llvm
%MyStruct = type opaque

declare ptr @create_struct()
declare void @destroy_struct(ptr)
declare void @use_struct(ptr)

define i32 @main() {
  %s = call ptr @create_struct()
  call void @use_struct(ptr %s)
  call void @destroy_struct(ptr %s)
  ret i32 0
}
```

This compiles and links fine. The implementation of `%MyStruct`
appears only where the structure is dereferenced.

## Pitfalls

- **Mixing opaque-pointer and typed-pointer IR.** A module
  half-converted to opaque pointers may produce confusing errors.
  Convert wholesale.

- **Forgetting `getelementptr`'s type argument.** With typed pointers,
  GEP read the pointer's pointee type to know the stride; with opaque
  pointers, you spell out the *element type* at the GEP site. Get
  this wrong and your stride is off:
  ```llvm
  ; Wrong: pretending the array is of i64s
  %p = getelementptr i64, ptr %arr_of_i32, i32 5    ; strides by 40 bytes, not 20
  ; Right:
  %p = getelementptr i32, ptr %arr_of_i32, i32 5    ; strides by 20 bytes
  ```

- **Trying to declare a typed pointer at use site.** `i32* %p` in a
  function signature is the old syntax. Use `ptr %p`; the load/store
  carries the access type.

- **Confusing `bitcast` of pointers with cross-address-space casts.**
  Within one address space, `ptr` to `ptr` is a no-op; you wouldn't
  bitcast. Across address spaces, use `addrspacecast`.

- **Opaque struct used where its body is required.** If a function
  needs `sizeof(%MyStruct)` or accesses a field, the opaque
  declaration alone won't suffice — you need the body in scope.

## See also

- [`01-primitive-types.md`](01-primitive-types.md) — `ptr` and `void` listed as primitives
- [`02-composite-types.md`](02-composite-types.md) — structs, arrays, vectors
- [`04-opaque-pointer-migration.md`](04-opaque-pointer-migration.md) — migrating legacy typed-pointer IR to modern `ptr` IR
- [`../04-memory/02-load-store.md`](../04-memory/02-load-store.md) — load/store through `ptr` with
  explicit access type
- [`../04-memory/04-address-spaces.md`](../04-memory/04-address-spaces.md) — `addrspace(N)` and
  `addrspacecast`
