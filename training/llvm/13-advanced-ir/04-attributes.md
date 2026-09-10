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


## Calling conventions

A calling convention is ABI-significant: it controls how arguments and return
values are assigned to registers or stack slots, which registers must be
preserved, and whether special tail-call rules apply. The convention is written
after linkage/visibility and before the return type on a `define` or `declare`,
and the same convention must be written at each direct call site unless the
default C convention is intended. The basic function shape is introduced in
[Modules, Functions, and Basic Blocks](../01-syntax/01-modules-functions-blocks.md).

```llvm
declare fastcc i32 @add_fast(i32, i32)

define i32 @caller(i32 %a, i32 %b) {
entry:
  %sum = call fastcc i32 @add_fast(i32 %a, i32 %b)
  ret i32 %sum
}
```

| Convention | Typical purpose | Notes for handwritten or lowered IR |
| --- | --- | --- |
| `ccc` | The default C calling convention. | Use for C ABI interoperability, variadic C functions, and runtime entry points that are exported to ordinary object code. It may be omitted because it is the default, but spelling it can make ABI intent explicit. |
| `fastcc` | A target-chosen fast convention for internal calls. | Optimizes register assignment and call overhead, but is not a stable external ABI. Use only when every declaration, definition, and call site is controlled together. |
| `coldcc` | Calls to cold paths such as error or bailout helpers. | Optimized for preserving caller-side state and reducing hot-path overhead rather than callee speed. Good for rare runtime exits, not for frequently executed helpers. |
| `tailcc` | Functions designed for guaranteed tail-call style lowering where supported. | Useful for functional-language runtimes or dispatch loops that require bounded stack growth. Pair with structurally valid tail-position calls. |
| `ghccc` | The Glasgow Haskell Compiler calling convention. | Specialized for GHC's runtime model and register use. Do not use it for generic BCIR helpers unless the whole runtime ABI is intentionally GHC-compatible. |
| `swiftcc` | Swift language ABI calls. | Carries Swift-specific ABI expectations, including interactions with Swift parameter attributes. Do not copy it just because a helper was originally emitted by Swift. |
| `swifttailcc` | Swift tail-call ABI calls. | A Swift-specific convention for tail-call-heavy paths. It is not a generic replacement for `tailcc`. |
| `cc <n>` | A numbered target-specific or frontend-specific convention. | The meaning of `<n>` is not self-describing in textual IR. Use only when the producer and consumer agree on the exact convention number and target support. |

Calling conventions are part of the call boundary contract. A BCIR runtime
wrapper that declares `ccc` but calls or defines the implementation as `fastcc`,
`swiftcc`, or a numbered `cc <n>` can pass the same LLVM verifier type checks in
some textual contexts yet still lower to incompatible machine-code call
sequences. Keep the wrapper declaration, definition, and every generated call in
lockstep with the runtime boundary policy described in
[Runtime Call Boundaries](../bcir-mapping/09-runtime-call-boundaries.md).

## Tail-call markers

LLVM calls can carry a tail-call marker before `call`. The marker describes what
the IR producer requires or forbids; it is separate from the function's calling
convention, although some conventions make tail calls more practical.

| Marker | Meaning | Use only when... |
| --- | --- | --- |
| `tail` | A request or hint that the call may be lowered as a tail call. | The call is in tail position and preserving stack growth is beneficial, but correctness does not depend on the transformation. |
| `musttail` | A requirement that the call be lowered as a tail call. | The next instruction is the matching `ret`, the caller/callee signatures and ABI-impacting attributes satisfy LLVM's strict `musttail` rules, and unbounded stack growth would be a correctness bug. |
| `notail` | A prohibition on tail-call optimization for this call. | The frame, return address, stack trace shape, sanitizer behavior, or runtime instrumentation must remain visible across the call. |

```llvm
declare i32 @step(i32)
declare void @record_and_return(i32)

define i32 @tail_hint(i32 %x) {
entry:
  %next = tail call i32 @step(i32 %x)
  ret i32 %next
}

define i32 @tail_required(i32 %x) {
entry:
  %next = musttail call i32 @step(i32 %x)
  ret i32 %next
}

define void @keep_frame(i32 %x) {
entry:
  notail call void @record_and_return(i32 %x)
  ret void
}
```

For BCIR runtime wrappers, `musttail` is risky unless the wrapper is deliberately
designed as a transparent trampoline. ABI-impacting attributes such as `sret`,
`byval`, `inreg`, `inalloca`, and `preallocated` must line up with the call and
return rules. A copied `musttail` marker on a wrapper that adds a status code,
reorders hidden arguments, or changes a structure-return pointer turns a small
IR annotation mistake into a hard verifier failure or an ABI-invalid tail jump.

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
part of the call signature contract, not optimization hints. The same source-level
function type may lower differently on different targets, so these attributes
belong in the target-aware type-lowering layer rather than in a late cleanup pass.

| Attribute | Use |
| --- | --- |
| `byval(<ty>)` | Pass a copy of an aggregate object by pointer according to ABI rules. |
| `sret(<ty>)` | Hidden structure-return pointer used for aggregate returns. |
| `inreg` | Prefer/register-class ABI placement for an argument or return where the target ABI supports it. |
| `zeroext`, `signext` | Caller/callee agree to zero-extend or sign-extend narrow integer values. |
| `inalloca(<ty>)`, `preallocated(<ty>)` | Specialized argument-allocation protocols used by selected ABIs. |
| `swiftself`, `swifterror` | Language ABI hooks; do not invent them outside the matching frontend ABI. |

### ABI attribute examples

`sret(<ty>)` marks a hidden pointer where the callee stores an aggregate return.
The IR-level return type is usually `void`, but the source-level result lives in
the caller-provided storage:

```llvm
%Pair = type { i64, i64 }

declare void @make_pair(ptr noalias sret(%Pair) align 8, i64, i64)

define void @use_sret(ptr %out) {
entry:
  call void @make_pair(ptr sret(%Pair) align 8 %out, i64 1, i64 2)
  ret void
}
```

`byval(<ty>)` says the pointer argument is passed as a by-value aggregate copy
under the target ABI. The callee receives a pointer-shaped IR value, but ABI
lowering must protect the caller's original object from callee writes through the
formal parameter:

```llvm
%Pair = type { i64, i64 }

declare i64 @sum_pair(ptr byval(%Pair) align 8)

define i64 @use_byval(ptr %pair) {
entry:
  %sum = call i64 @sum_pair(ptr byval(%Pair) align 8 %pair)
  ret i64 %sum
}
```

`inalloca(<ty>)` is an argument-allocation protocol: the caller builds an
argument frame object and passes it to a callee that consumes that allocation. It
is target- and ABI-specific, and it cannot be casually combined with other
argument-storage attributes such as `sret` or `byval` on the same argument:

```llvm
%Args = type { ptr, i32 }

declare void @consume_frame(ptr inalloca(%Args))

define void @use_inalloca(ptr %p, i32 %n) {
entry:
  %frame = alloca inalloca %Args, align 8
  %slot0 = getelementptr %Args, ptr %frame, i32 0, i32 0
  store ptr %p, ptr %slot0, align 8
  %slot1 = getelementptr %Args, ptr %frame, i32 0, i32 1
  store i32 %n, ptr %slot1, align 4
  call void @consume_frame(ptr inalloca(%Args) %frame)
  ret void
}
```

`preallocated(<ty>)` is another explicit call-argument allocation protocol.
Non-`musttail` calls that use it carry a `preallocated` operand bundle tying the
call to setup/argument/teardown intrinsics; this makes it unsuitable for ad-hoc
manual copying between wrappers:

```llvm
%Pair = type { i64, i64 }

declare token @llvm.call.preallocated.setup(i32)
declare ptr @llvm.call.preallocated.arg(token, i32)
declare void @takes_preallocated(ptr preallocated(%Pair))

define void @use_preallocated(i64 %a, i64 %b) {
entry:
  %tok = call token @llvm.call.preallocated.setup(i32 1)
  %arg = call ptr @llvm.call.preallocated.arg(token %tok, i32 0) preallocated(%Pair)
  %field0 = getelementptr %Pair, ptr %arg, i32 0, i32 0
  store i64 %a, ptr %field0, align 8
  %field1 = getelementptr %Pair, ptr %arg, i32 0, i32 1
  store i64 %b, ptr %field1, align 8
  call void @takes_preallocated(ptr preallocated(%Pair) %arg) [ "preallocated"(token %tok) ]
  ret void
}
```

`inreg` requests target-specific register treatment for an argument or return. It
only has meaning where the target ABI defines one, and every declaration and call
must agree on the same placement contract:

```llvm
declare inreg i32 @small_result(ptr inreg)

define inreg i32 @forward_inreg(ptr inreg %ctx) {
entry:
  %r = call inreg i32 @small_result(ptr inreg %ctx)
  ret i32 %r
}
```

Mismatched ABI attributes between declarations and definitions are a link-time
or runtime bug waiting to happen. Keep declarations emitted by different modules
byte-for-byte compatible for ABI-relevant attributes. For BCIR runtime wrappers,
copying a callee prototype without its hidden `sret` result, dropping `byval`
alignment, moving an `inalloca` frame argument, omitting a `preallocated` operand
bundle, or adding `inreg` to only one side of a call can make the wrapper and the
runtime disagree about where bits live. That disagreement can corrupt caller
stack slots, pass stale pointers instead of aggregate copies, leak temporary
argument storage across a boundary, or return status values in registers the
caller never reads. Treat ABI attributes as part of the stable boundary contract
covered by [Runtime Call Boundaries](../bcir-mapping/09-runtime-call-boundaries.md),
not as decoration that can be inferred after the wrapper is generated.

## BCIR runtime-wrapper signature drift

Runtime wrappers often look like mechanical copies of lower-level runtime
prototypes, but the wrapper's LLVM function type and ABI attributes are the
actual contract seen by the optimizer, code generator, linker, and JIT. Copying
only the visible scalar types is not enough. The return type, pointer address
spaces, variadic marker, calling convention, tail-call marker, parameter order,
hidden result pointer, extension attributes, and ABI storage attributes must all
match the runtime implementation and every generated call site.

Common failure modes include:

- A wrapper drops an `sret` parameter and returns a pointer or status scalar
  instead, so the caller and callee disagree about where the aggregate result is
  written.
- A generated call copies the argument list but not `byval(<ty>)` or its
  alignment, so the callee may write through storage the caller expected to be a
  private aggregate copy.
- A trampoline preserves a `musttail` marker while adding diagnostics, status
  translation, or reordered hidden ABI arguments, invalidating the exact tail-call
  signature relationship.
- A JIT declaration uses `ccc` and a runtime object file was built for `fastcc`,
  `swiftcc`, or a target-specific `cc <n>`, so registers and stack slots are
  interpreted under different conventions.

When the BCIR lowering layer cannot prove the exact signature and ABI attributes,
prefer a conservative wrapper with a simple C ABI and explicit loads/stores over
a clever declaration copied from an unrelated frontend. Then document the stable
boundary in the same place as the runtime declaration; see
[Runtime Call Boundaries](../bcir-mapping/09-runtime-call-boundaries.md).

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

1. Put ABI attributes (`sret`, `byval`, `inalloca`, `preallocated`, `inreg`,
   `zeroext`, `signext`) in the type-lowering layer, where target ABI knowledge
   belongs.
2. Put semantic attributes (`noalias`, `readonly`, `memory(read)`) only after the
   verifier or proof layer can justify them for all dynamic executions.
3. Prefer load/store `align` for a known access alignment and parameter `align`
   only when every use of the pointer value can rely on that alignment.
4. Treat `dereferenceable` as a speculation permission, not a bounds comment.
5. Keep declaration and definition attributes synchronized across runtime `.ll`
   files to avoid cross-module drift.
