# Opaque Pointer Migration Patterns

## Where the pointee type moved

### `load`

The loaded type is the first operand of the instruction:

```llvm
; Opaque-pointer form
%v = load i32, ptr %p, align 4
```

Do not ask, "What does `%p` point to?" Ask, "What type does this `load`
read here?" In this example, the access type is `i32`.

### `store`

The stored value carries the access type:

```llvm
; Opaque-pointer form
store i32 %v, ptr %p, align 4
```

The pointer operand is just the destination address. The value operand
says this store writes an `i32`.

### `getelementptr`

`getelementptr` still needs an element type to compute offsets and field
indices, but that type is now explicit in the instruction:

```llvm
; Opaque-pointer form
%field = getelementptr %Struct, ptr %p, i32 0, i32 1
```

The `%Struct` argument is not recovered from `ptr`; it is part of the GEP
itself. If you choose the wrong type here, you compute the wrong offset.

### `call` and `invoke`

Function signatures also carry the meaningful types. A pointer argument
is `ptr`, while the callee declaration and the argument list describe how
that pointer is used:

```llvm
declare void @consume_i32_ptr(ptr)

define void @caller(ptr %p) {
entry:
  call void @consume_i32_ptr(ptr %p)
  ret void
}
```

If the callee loads from `%p`, inspect the callee body or its API contract;
do not reconstruct a pointee type from the argument spelling.

## Migration checklist

Use this checklist when converting old IR, frontend output, tests, or an IR
consumer from typed pointers to opaque pointers.

1. **Remove assumptions based on pointer spelling.**
   - Replace checks like "is this an `i32*`?" with checks of the operation
     that uses the pointer.
   - Treat `ptr` as an address value, not as a promise about the object behind
     the address.

2. **Inspect load/store/GEP/call signatures for the actual access type.**
   - `load i32, ptr %p` reads an `i32`.
   - `store %Pair %v, ptr %p` writes a `%Pair` value.
   - `getelementptr %Pair, ptr %p, ...` indexes according to `%Pair` layout.
   - `call void @f(ptr %p)` tells you only that a pointer is passed; inspect
     `@f`'s declaration, definition, attributes, and API contract for meaning.

3. **Avoid reconstructing pointee types from `ptr`.**
   - Do not invent a single "real" pointee type for a pointer SSA value.
   - A pointer may be used by multiple operations with different access types,
     especially around byte buffers, unions, object headers, or serialization
     layouts.

4. **Delete pointer-to-pointer bitcast chains that only changed spelling.**
   - `bitcast ptr %p to ptr` is unnecessary.
   - Keep real conversions such as `ptrtoint`, `inttoptr`, and address-space
     conversions only when the semantics require them.

5. **Preserve address-space information.**
   - `i32 addrspace(1)*` migrates to `ptr addrspace(1)`, not plain `ptr`.
   - Use `addrspacecast` only for intentional cross-address-space conversion.

6. **Run modern LLVM tools on migrated modules.**
   - Assemble textual IR with `llvm-as`.
   - Run the verifier with `opt -passes=verify -disable-output`.
   - Optionally run `opt -S` to confirm that unnecessary bitcasts disappeared
     and the printed IR is in modern opaque-pointer form.

## See also

- [`04-opaque-pointer-migration.md`](04-opaque-pointer-migration.md) — migration dispatcher.
- [`06-opaque-pointer-migration-diagnostics.md`](06-opaque-pointer-migration-diagnostics.md) — verifier and review pitfalls.
- [`07-opaque-pointer-migration-examples.md`](07-opaque-pointer-migration-examples.md) — before/after snippets.
