# Address Spaces and Pointers

## TL;DR

LLVM IR can model multiple, disjoint memory regions per target via
**address spaces**: a numeric tag attached to pointer types and
allocations. `addrspace(0)` is the default (generic CPU memory).
Other numbers are target-defined, used heavily by GPU and embedded
targets.

```llvm
ptr                   ; address space 0 (generic)
ptr addrspace(1)      ; e.g., GPU global memory
ptr addrspace(3)      ; e.g., GPU shared/local memory
ptr addrspace(4)      ; e.g., GPU constant memory

@g = addrspace(1) global i32 0           ; global in addrspace 1
%x = alloca i32, addrspace(5)             ; alloca returning addrspace(5) pointer

%cast = addrspacecast ptr %p to ptr addrspace(1)   ; cross-space cast
```

## Why address spaces?

A single 64-bit virtual address space on a CPU is the simplest case.
But many real targets have multiple distinct memory regions:

- **GPU** — global, shared (per-workgroup), local/private (per-thread),
  constant, image/texture
- **Embedded** — flash, RAM, peripheral registers, EEPROM
- **FPGA** — distinct on-chip memory banks
- **Accelerators** — host memory vs device memory

Address spaces let the compiler:
- Know a pointer can only legitimately point into one region
- Pick the right load/store instruction for that region (cached vs
  uncached, different bandwidth)
- Reject illegal casts statically

## Syntax

### On a pointer type

```llvm
ptr                ; addrspace(0), the default
ptr addrspace(N)
```

### On a global

```llvm
@g = addrspace(1) global i32 0
```

### On an alloca

```llvm
%x = alloca i32, addrspace(5)
```

### On a function parameter or return

```llvm
define ptr addrspace(1) @get_gpu_mem(ptr addrspace(1) %arr) {
  ; ...
}
```

## Conventional numbering

The numbering is **target-defined**. By convention:

| AS | Common interpretation |
|---|---|
| 0 | Generic / default |
| 1 | GPU global memory (CUDA, OpenCL) |
| 2 | GPU constant memory |
| 3 | GPU shared / local (per workgroup) |
| 4 | GPU constant (varies) |
| 5 | GPU private (per thread) |
| 6 | Generic GPU pointer (in some targets) |

You can verify by looking at the target's clang/LLVM headers or the
target's `DataLayout` string. Do not assume cross-target consistency.

## `addrspacecast`

Convert a pointer from one address space to another. This is **not**
a no-op: the address representation may differ.

```llvm
define void @demo(ptr %generic, ptr addrspace(1) %global) {
  ; Cross-space casts:
  %g1 = addrspacecast ptr %generic       to ptr addrspace(1)
  %g2 = addrspacecast ptr addrspace(1) %global to ptr

  ; You CANNOT cast across spaces with bitcast (different representations):
  ; %bad = bitcast ptr %generic to ptr addrspace(1)   ; WRONG
  ret void
}
```

Use `bitcast` only within the same address space; `addrspacecast`
between them.

## DataLayout interactions

The `target datalayout` string can declare per-address-space pointer
sizes and alignments:

```llvm
target datalayout = "e-p:64:64-p1:32:32-i64:64-S128"
                       ^^^^^^   ^^^^^^^
                       AS 0     AS 1
```

In this layout, AS 0 pointers are 64-bit, AS 1 pointers are 32-bit.
That's a real thing — e.g., GPU global pointers might be 64-bit and
shared pointers 32-bit on the same target.

When emitting IR, you mostly don't worry about this; just attach the
right `addrspace(N)` and trust the datalayout to size them.

## Example: GPU kernel with multiple memory spaces

```llvm
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @gpu_kernel(
    ptr addrspace(1) %global_in,
    ptr addrspace(1) %global_out
) {
entry:
  %local = alloca i32, addrspace(5)
  store i32 0, ptr addrspace(5) %local, align 4

  %v = load i32, ptr addrspace(1) %global_in, align 4
  store i32 %v, ptr addrspace(1) %global_out, align 4
  ret void
}
```

## Pitfalls

- **`bitcast` across address spaces.** Use `addrspacecast`. The
  verifier rejects same-bitwidth cross-space `bitcast`.

- **Assuming AS numbering is portable.** What's AS 1 on AMDGPU might
  not be AS 1 on NVPTX.

- **Forgetting alignment for AS-specific loads.** GPU shared memory
  is often natively 4-aligned; misaligned access can be slow or trap.

- **Mixing pointers from different address spaces in `phi`.** Each
  phi's incoming values must share the *exact* type, including
  address space.

- **Calling a function via a pointer in the wrong AS.** Function
  pointers usually live in AS 0; address-spaced function pointers
  exist on some targets but not all.

## See also

- `02-types/03-opaque-and-pointer-types.md` — opaque `ptr` carrying
  an address space
- `03-global-variables.md` — `addrspace()` on globals
- `01-alloca.md` — `alloca i32, addrspace(N)`
- `01-syntax/01-modules-functions-blocks.md` — datalayout
