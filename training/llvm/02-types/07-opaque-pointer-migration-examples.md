# Opaque Pointer Migration Examples

## Before/after examples

The legacy examples below are intentionally shown as historical reference.
They may not assemble with a modern LLVM toolchain. See
[`examples/typed-pointer-before.ll.txt`](examples/typed-pointer-before.ll.txt)
for the complete non-modern reference and
[`examples/opaque-pointer-after.ll`](examples/opaque-pointer-after.ll) for a
verifier-clean modern module.

### 1. Typed pointer load/store

Before, the pointer operand repeated the pointee type:

```llvm
; Legacy typed-pointer IR
%v = load i32, i32* %p, align 4
store i32 %v, i32* %q, align 4
```

After, the load/store operation keeps the access type and the pointer is
opaque:

```llvm
; Modern opaque-pointer IR
%v = load i32, ptr %p, align 4
store i32 %v, ptr %q, align 4
```

### 2. Typed pointer GEP

Before, the GEP's base pointer spelling carried the apparent aggregate
pointee type:

```llvm
; Legacy typed-pointer IR
%field = getelementptr inbounds %Pair, %Pair* %p, i32 0, i32 1
%v = load i32, i32* %field, align 4
```

After, the GEP names `%Pair` directly and returns `ptr`:

```llvm
; Modern opaque-pointer IR
%field = getelementptr inbounds %Pair, ptr %p, i32 0, i32 1
%v = load i32, ptr %field, align 4
```

### 3. Typed pointer bitcast chains

Before, changing the apparent pointee type required pointer-to-pointer
`bitcast`s even though the address stayed the same:

```llvm
; Legacy typed-pointer IR
%bytes = bitcast %Pair* %p to i8*
%as_i32 = bitcast i8* %bytes to i32*
%v = load i32, i32* %as_i32, align 4
```

After, the same address is a `ptr`; the access instruction names the type:

```llvm
; Modern opaque-pointer IR
%v = load i32, ptr %p, align 4
```

If you still need byte-wise addressing, keep the GEP element type explicit:

```llvm
%byte3 = getelementptr i8, ptr %p, i64 3
%b = load i8, ptr %byte3, align 1
```

### 4. Opaque pointer equivalent module shape

A migrated function signature typically changes from typed pointer
arguments to `ptr` arguments:

```llvm
%Pair = type { i32, i32 }

define i32 @sum_pair(ptr %p) {
entry:
  %a.ptr = getelementptr inbounds %Pair, ptr %p, i32 0, i32 0
  %b.ptr = getelementptr inbounds %Pair, ptr %p, i32 0, i32 1
  %a = load i32, ptr %a.ptr, align 4
  %b = load i32, ptr %b.ptr, align 4
  %sum = add i32 %a, %b
  ret i32 %sum
}
```

The IR is not less typed. The types have moved to the instructions where
they are semantically relevant.

## See also

- [`examples/typed-pointer-before.ll.txt`](examples/typed-pointer-before.ll.txt) — complete legacy typed-pointer reference.
- [`examples/opaque-pointer-after.ll`](examples/opaque-pointer-after.ll) — verifier-clean modern opaque-pointer module.
- [`05-opaque-pointer-migration-patterns.md`](05-opaque-pointer-migration-patterns.md) — operation-by-operation migration rules.
