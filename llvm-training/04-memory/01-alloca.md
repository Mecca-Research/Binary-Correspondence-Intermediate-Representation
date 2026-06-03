# Stack Allocation: `alloca`

## TL;DR

`alloca` reserves stack memory for a value (or N copies of one) and
returns a `ptr` to it. Memory is released automatically when the
function returns. Used for **local variables** (especially mutable
ones) and short-lived buffers.

```llvm
%x        = alloca i32                       ; one i32
%x_align  = alloca i32, align 4              ; one i32, 4-byte aligned
%arr      = alloca [10 x i32]                ; ten i32s, all contiguous
%arr_dyn  = alloca i32, i32 %n               ; %n i32s — runtime-sized
%struct   = alloca %MyStruct, align 8
%in_addrs = alloca i32, addrspace(5)         ; alloca in address space 5 (e.g., GPU local)
```

## Syntax

```
%name = alloca <type> [, <num-elements>] [, align N] [, addrspace(N)]
```

- **`<type>`** — element type. The allocated region holds `N` copies
  of this.
- **`<num-elements>`** (optional) — integer value (constant or SSA).
  Defaults to 1.
- **`align N`** (optional) — alignment of the returned pointer.
- **`addrspace(N)`** (optional) — the address space the pointer lives
  in.

The result type is always `ptr` (or `ptr addrspace(N)`).

## When to use

- **Mutable locals** that don't fit in SSA cleanly. Frontends typically
  start by `alloca`ing every local, then let `mem2reg` promote them
  to SSA registers.
- **Small fixed-size arrays** for scratch buffers.
- **Capturing by reference** in functions that need a stable address
  for a local.

## When NOT to use

- **Large or unbounded buffers** — you'll overflow the stack.
- **Memory that outlives the function** — return values, heap data.
  Use `malloc`/`new`/equivalent for those.
- **Already-SSA values** — if the variable has a single definition,
  bind it to an SSA name directly. No need to round-trip through
  memory.

## Example: mutable local

```llvm
define i32 @sum_array(ptr %arr, i32 %n) {
entry:
  %sum_p = alloca i32, align 4
  store i32 0, ptr %sum_p, align 4
  br label %loop

loop:
  %i        = phi i32 [ 0, %entry ], [ %next_i, %loop ]
  %sum_old  = load i32, ptr %sum_p, align 4
  %elem_p   = getelementptr inbounds i32, ptr %arr, i32 %i
  %elem     = load i32, ptr %elem_p, align 4
  %sum_new  = add i32 %sum_old, %elem
  store i32 %sum_new, ptr %sum_p, align 4
  %next_i   = add i32 %i, 1
  %done     = icmp eq i32 %next_i, %n
  br i1 %done, label %exit, label %loop

exit:
  %final = load i32, ptr %sum_p, align 4
  ret i32 %final
}
```

After `-mem2reg`, this turns into pure SSA with a `phi` for the
running sum, no `alloca` at all.

## Example: variable-length array (VLA)

```llvm
define void @scratch(i32 %n) {
entry:
  %buf = alloca i32, i32 %n     ; reserves n * sizeof(i32) bytes
  ; ... use %buf ...
  ret void
}
```

The compiler emits a stack-pointer adjustment; the memory is released
on return.

## Alignment

```llvm
%x = alloca i32, align 4         ; 4-byte aligned
%y = alloca i64, align 8         ; 8-byte aligned (natural)
%z = alloca i32, align 16        ; over-aligned (16 bytes for an i32)
```

Use natural alignment unless the value will be accessed with vector
or atomic operations that require more.

## `alloca` and address spaces

```llvm
%local = alloca i32, addrspace(5)    ; GPU local/private memory, typically
```

Default is address space 0 (generic). Other address spaces are used
on targets with multiple memory regions (CUDA, OpenCL, SPIR).

## Inalloca and other rare flavors

`alloca` has rarely-used variants:

- **`inalloca`** — special form for x86 calling conventions that pass
  arguments via stack memory; you'll see it in the output of certain
  frontends, almost never write it manually.
- **`swifterror`** — a special alloca that participates in Swift's
  error-return ABI.

Skip these unless you're working on those specific platforms.

## Pitfalls

- **Putting `alloca` in a loop.** Each iteration allocates more
  stack; over enough iterations, you stack-overflow. Put allocas in
  the entry block.

- **Returning a pointer to an `alloca`.** The pointer is dead the
  moment the function returns. Classic dangling-pointer bug.

- **Forgetting `align` for atomic or vector access.** Misaligned
  atomics are UB on most targets.

- **Allocating an unbounded amount.** `alloca i8, i32 %n` for
  attacker-controlled `%n` is a stack-smash vulnerability.

- **Treating `alloca` as `malloc`.** `alloca` is *stack*, not heap.
  No `free` is needed (or possible).

## See also

- [`02-load-store.md`](02-load-store.md) — accessing the allocated memory
- [`../00-foundations/02-ssa.md`](../00-foundations/02-ssa.md) — why mem2reg removes most allocas
- [`../02-types/03-opaque-and-pointer-types.md`](../02-types/03-opaque-and-pointer-types.md) — the result type
- [`04-address-spaces.md`](04-address-spaces.md) — `addrspace(N)` on alloca
