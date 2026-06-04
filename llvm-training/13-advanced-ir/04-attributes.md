# Function and Parameter Attributes

Attributes are compact facts attached to functions, parameters, return values,
call sites, and memory operations. They are not comments: optimizers and codegen
use them to remove work, choose ABI lowering, and prove aliasing or trapping
properties. Incorrect attributes can make otherwise valid IR miscompile.

## Where attributes appear

```llvm
; Return attr, function attrs, parameter attrs, and a call-site attr.
declare noalias ptr @malloc(i64 noundef) nounwind allocsize(0)

define noundef i32 @read_first(ptr noalias readonly align 4 dereferenceable(4) %p) nounwind {
entry:
  %v = load i32, ptr %p, align 4
  ret i32 %v
}
```

Common locations:

- **Return attributes**: before the return type, for facts about the returned
  value (`noalias`, `nonnull`, `noundef`, `zeroext`, `signext`).
- **Parameter attributes**: after the parameter type, for facts about one
  argument (`noalias`, `readonly`, `align`, `dereferenceable`, ABI extension
  attributes).
- **Function attributes**: after the function header or declaration, for facts
  about the whole function (`nounwind`, `memory(read)`, `willreturn`).
- **Call-site attributes**: on a specific `call` or `invoke`, when the fact is
  true for that call even if the callee declaration is less precise.

## Memory behavior attributes

Prefer the modern `memory(...)` form when you can state effects precisely:

```llvm
declare i32 @pure_hash(ptr) memory(read)
declare void @touch_arg(ptr) memory(argmem: readwrite)
declare void @no_effects() memory(none)
```

Useful meanings:

| Attribute | Meaning | Typical use |
| --- | --- | --- |
| `memory(none)` / legacy `readnone` | Does not read or write memory visible to the IR memory model. | Pure arithmetic helper. |
| `memory(read)` / legacy `readonly` | Reads memory but does not write it. | Hashing, lookup, validation. |
| `memory(argmem: readwrite)` | May only access memory reachable from pointer arguments. | Buffer transforms with no global access. |
| `inaccessiblememonly` style effects | Only touches memory invisible to the IR program. | Runtime bookkeeping with no alias to program pointers. |

`readonly` can also appear on a pointer parameter to say memory reached through
that parameter is only read by the function. Do not use it if the callee writes
through aliases or stores into the object indirectly.

## Pointer and aliasing attributes

| Attribute | Scope | Optimizer promise |
| --- | --- | --- |
| `noalias` on return | Returned pointer does not alias any other live pointer visible to the caller, malloc-style. |
| `noalias` on parameter | During the call, accesses through that parameter do not alias accesses through other pointer parameters or globals in the specified way. |
| `nonnull` | Pointer is never null. A null value becomes poison/undefined for uses depending on context. |
| `noundef` | Value is not poison or undef. Useful at ABI boundaries where C/C++ undefined values must not leak. |
| `align N` | Pointer value is at least `N`-byte aligned. More specific than only aligning a load/store. |
| `dereferenceable(N)` | At least `N` bytes can be safely loaded from the pointer for the dynamic lifetime required by the attribute. |
| `dereferenceable_or_null(N)` | Either null or safely dereferenceable for `N` bytes. |

Be conservative. `dereferenceable(16)` is stronger than "usually points to a
16-byte object"; it permits speculative loads in places where a fault or trap
would otherwise be observable.

## Control-flow and exception attributes

- `nounwind`: the function does not unwind the stack. Use it for C helpers only
  if they cannot throw, longjmp as an unwind, or trigger language-level unwinds.
- `willreturn`: the function eventually returns to the caller if it has defined
  behavior. Do not put it on infinite loops or functions that terminate the
  process on normal inputs.
- `noreturn`: the function never returns normally, such as `abort`.
- `mustprogress`: loops in the function must make observable progress; this can
  justify deleting empty infinite loops.

## ABI attributes

ABI attributes describe how frontend source types cross the target ABI. They are
part of the call signature contract, not optimization hints.

| Attribute | Use |
| --- | --- |
| `byval(<ty>)` | Pass a copy of an aggregate object by pointer according to ABI rules. |
| `sret(<ty>)` | Hidden structure-return pointer used for aggregate returns. |
| `inreg` | Prefer/register-class ABI placement for an argument or return where the target ABI supports it. |
| `zeroext`, `signext` | Caller/callee agree to zero-extend or sign-extend narrow integer values. |
| `inalloca(<ty>)`, `preallocated(<ty>)` | Specialized argument-allocation protocols used by selected ABIs. |
| `swiftself`, `swifterror` | Language ABI hooks; do not invent them outside the matching frontend ABI. |

Mismatched ABI attributes between declarations and definitions are a link-time
or runtime bug waiting to happen. Keep declarations emitted by different modules
byte-for-byte compatible for ABI-relevant attributes.

## Call-site refinement pattern

If a generic declaration is shared but one call has stronger local facts, attach
attributes at the call site:

```llvm
declare i32 @consume(ptr)

define i32 @caller(ptr %p) {
entry:
  %v = call i32 @consume(ptr nonnull align 8 dereferenceable(8) %p)
  ret i32 %v
}
```

This tells optimizers about this call without lying about all possible callers of
`@consume`.

## BCIR checklist

When lowering BCIR-style operations to LLVM IR:

1. Put ABI attributes (`sret`, `byval`, `zeroext`, `signext`) in the type-lowering
   layer, where target ABI knowledge belongs.
2. Put semantic attributes (`noalias`, `readonly`, `memory(read)`) only after the
   verifier or proof layer can justify them for all dynamic executions.
3. Prefer load/store `align` for a known access alignment and parameter `align`
   only when every use of the pointer value can rely on that alignment.
4. Treat `dereferenceable` as a speculation permission, not a bounds comment.
5. Keep declaration and definition attributes synchronized across runtime `.ll`
   files to avoid cross-module drift.
