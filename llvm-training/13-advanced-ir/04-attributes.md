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

`readonly` can also appear on a pointer parameter to say memory reached through
that parameter is only read by the function. Do not use it if the callee writes
through aliases or stores into the object indirectly.

## Modern `memory(...)` effects

The modern spelling describes **which memory locations** a function or call may
access and **how** it may access them. If no location is named, the access applies
to all memory the optimizer models for ordinary IR. Named locations narrow the
promise:

- `argmem`: memory reachable from pointer arguments passed to the call.
- `inaccessiblemem`: memory not otherwise reachable by LLVM IR pointers, such as
  allocator state, runtime-internal handles, or JIT bookkeeping.

Common forms:

| Attribute | Promise | Use only when... |
| --- | --- | --- |
| `memory(none)` | The call does not read or write any modeled memory. | The result depends only on SSA operands and constants, and the call has no hidden state or externally visible side effects. |
| `memory(read)` | The call may read memory but may not write it. | It performs validation, hashing, lookup, or observation without writes to buffers, globals, errno-like state, logs, or runtime caches. |
| `memory(write)` | The call may write memory but may not read it. | It initializes or clobbers storage without reading old contents or hidden state. This is rarer than it looks. |
| `memory(argmem: read)` | The call may read only memory reachable from pointer arguments. | It inspects caller-provided buffers and does not read globals, tables, caches, or inaccessible runtime state. |
| `memory(argmem: readwrite)` | The call may read or write only memory reachable from pointer arguments. | It transforms caller-provided buffers and does not touch global or runtime-internal state. |
| `memory(inaccessiblemem: readwrite)` | The call may read or write only inaccessible memory. | It updates runtime-private state and cannot access caller pointers, globals, or other IR-visible storage. |
| `memory(argmem: readwrite, inaccessiblemem: readwrite)` | The call may mutate argument-reachable memory and runtime-private state, but not globals or other memory. | A wrapper writes result buffers and also updates runtime-owned bookkeeping. |
| `memory(argmem: read, inaccessiblemem: readwrite)` | The call reads argument buffers and mutates only runtime-private state. | A runtime query consumes caller data and records profile/cache data without writing the caller's buffers. |

`memory(...)` composes by location: omitted locations are promised untouched. A
BCIR wrapper that writes `%dst` and increments a runtime statistic is not
`memory(argmem: readwrite)` if the statistic is an IR-visible global, and it is
not `memory(inaccessiblemem: readwrite)` if `%dst` is a normal pointer argument.
When the implementation detail is uncertain, omit the memory attribute or use a
less precise whole-memory effect such as `memory(readwrite)`.

Legacy attributes are coarser aliases for common whole-function effects:

| Legacy spelling | Modern equivalent | Important limitation |
| --- | --- | --- |
| `readnone` | `memory(none)` | Strongest promise: no reads, no writes, and no hidden runtime state visible to the memory model. |
| `readonly` on a function | `memory(read)` | Allows reads from any memory, not just argument buffers; forbids every write. |
| `writeonly` on a function | `memory(write)` | Allows writes to any memory; forbids reads, including reads of old destination contents or hidden state. |

Parameter attributes with the same names are different contracts. For example,
`ptr readonly %p` constrains accesses through `%p`; it does not make the whole
function `memory(read)`. Likewise, `ptr writeonly %dst` does not prove that a
runtime wrapper never reads globals, flags, caches, or other arguments.

### BCIR runtime-wrapper miscompile examples

Over-promising a runtime wrapper lets optimization legally remove, merge, or
move calls around loads and stores. The IR still verifies, but the declared
contract no longer matches the runtime boundary.

This wrapper writes through `%dst`, but `memory(none)` says it has no memory
effects. A caller that ignores the status can have the call deleted, leaving a
later load from `%dst` to see the old value:

```llvm
declare i32 @bcir.rt.add_i32(ptr, i32, i32, i32)

; Wrong: the runtime writes through %dst.
define i32 @bcir.map.op.add_i32_wrapper(ptr %dst, i32 %lhs, i32 %rhs) memory(none) {
entry:
  %status = call i32 @bcir.rt.add_i32(ptr %dst, i32 %lhs, i32 %rhs, i32 1)
  ret i32 %status
}

define i32 @caller(ptr %slot) {
entry:
  %ignored = call i32 @bcir.map.op.add_i32_wrapper(ptr %slot, i32 40, i32 2)
  %value = load i32, ptr %slot, align 4
  ret i32 %value
}
```

A safer declaration for the wrapper is at least `memory(argmem: write)` if it
only stores the result, or `memory(argmem: readwrite)` if it may read the old
destination, input buffers, or descriptors reachable from arguments. Omit the
attribute if the out-of-line runtime declaration is shared with implementations
that also read globals or mutate process-visible state.

This declaration promises only argument-memory effects, but many BCIR runtimes
also maintain counters, diagnostics, caches, or last-error slots. If those are
IR-visible globals, an optimizer may reorder global accesses across the call:

```llvm
@bcir.last_status = external global i32

declare i32 @bcir.rt.mul_i32(ptr, ptr, ptr)

; Wrong if the runtime also stores @bcir.last_status or a diagnostic global.
declare i32 @bcir.map.op.mul_i32_checked(ptr, ptr, ptr) memory(argmem: readwrite)

define i32 @read_status_after(ptr %dst, ptr %lhs, ptr %rhs) {
entry:
  %status = call i32 @bcir.map.op.mul_i32_checked(ptr %dst, ptr %lhs, ptr %rhs)
  %last = load i32, ptr @bcir.last_status, align 4
  %combined = add i32 %status, %last
  ret i32 %combined
}
```

If the global is real IR memory, use a whole-memory write-capable effect such as
`memory(readwrite)` or split the API so the pure buffer transform and the
diagnostic update have separate declarations. If the bookkeeping is genuinely
inaccessible to IR, a combined form like `memory(argmem: readwrite,
inaccessiblemem: readwrite)` is accurate and still gives the optimizer useful
alias information.

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
