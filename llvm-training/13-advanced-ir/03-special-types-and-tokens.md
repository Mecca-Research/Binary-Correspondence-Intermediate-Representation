# Advanced IR 03 — Special Types and Tokens

## TL;DR

Most LLVM IR values are integers, pointers, floating-point values,
structs, arrays, or vectors. Advanced IR also contains restricted types
that exist to model compiler internals or target-specific hardware:

| Type | Usual role |
|---|---|
| `token` | Opaque control dependency for EH, coroutine, and statepoint-like intrinsics |
| `metadata` | Operand type for debug and analysis intrinsics |
| `half` | IEEE 754 binary16 value |
| `bfloat` | bfloat16 value with 8 exponent bits and 7 fraction bits |
| `x86_amx` | Target extension type for Intel AMX tile values |
| `<vscale x N x T>` | Scalable vector with runtime-dependent lane count |

Do not treat these as ordinary storage-friendly scalar types. They have
extra verifier and backend constraints.

## `token`

`token` is an opaque value that preserves a relationship between
operations without exposing a normal data representation. It appears in
exception handling, coroutines, convergence, garbage-collector
statepoints, and other advanced intrinsics.

Important restrictions:

- You cannot inspect the bits of a token.
- You generally cannot put tokens through `phi` or `select`.
- You cannot load, store, or allocate ordinary memory of type `token`.
- Token-producing operations usually have strict placement rules.

A simplified statepoint-style outline:

```llvm
declare token @llvm.experimental.gc.statepoint.p0(i64 immarg, i32 immarg,
                                                  ptr, i32 immarg, i32 immarg,
                                                  ...)

%tok = call token (i64, i32, ptr, i32, i32, ...)
       @llvm.experimental.gc.statepoint.p0(i64 0, i32 0, ptr elementtype(void ()) @callee,
                                           i32 0, i32 0, i32 0, i32 0)
```

The token is not the program result. It is a handle consumed by related
intrinsics or passes.

## Coroutine tokens and lowering phases

Coroutine IR is another place where `token` values are intentionally
opaque. Frontends such as Clang usually emit a **presplit** coroutine as
one ordinary function containing coroutine intrinsics. LLVM coroutine
lowering then rewrites that function into a ramp function plus outlined
resume, destroy, and cleanup paths. In other words, the token does not
model application data; it lets the coroutine passes keep the pieces of
one coroutine tied together while optimization and outlining happen.

Key switched-resume coroutine intrinsics:

| Intrinsic | Token/pointer role | Phase notes |
|---|---|---|
| `llvm.coro.id` | Produces the identity `token` for one coroutine. | Presplit IR should have one identity call for the coroutine. Later passes can fill in fields such as the coroutine function address and outlined function table. |
| `llvm.coro.begin` | Consumes the identity token and returns the coroutine frame/handle pointer. | Marks frame setup. The handle is the value used by resume/destroy operations and by later frame accesses. |
| `llvm.coro.suspend` | Consumes a `token` from `llvm.coro.save`, or `token none`, and returns a small state code. | Marks a suspension point. The following branch or `switch` usually distinguishes suspended, resumed, and destroyed paths. |
| `llvm.coro.end` | Consumes the coroutine handle and a result token, usually `token none`. | Marks the point where access to the frame ends and control may return to the caller/resumer. |

**Presplit** IR is the frontend-facing form: one coroutine-shaped function
still contains normal control flow plus `llvm.coro.*` markers and usually
has the `presplitcoroutine` function attribute. **Split** IR is the
lowered form after coroutine passes have built the frame and outlined the
continuations. At that point, the original body may be represented as a
ramp function and separate resume/destroy functions rather than one
source-like function.

A minimal switched-resume outline is provided in
[`examples/coroutine-outline.ll`](examples/coroutine-outline.ll). For a
quick list of coroutine-related intrinsics, see
[`intrinsics-quickref.md#coroutine-intrinsics`](../reference/intrinsics-quickref.md#coroutine-intrinsics).

BCIR agent guidance: review coroutine IR for verifier-sensitive token
flow, preserve existing `llvm.coro.*` structure when transforming nearby
code, and check that coroutine lowering has run before backend codegen.
Do not hand-author full coroutine lowering unless you are deliberately
writing a focused LLVM coroutine test; the details are frontend-, ABI-,
and pass-pipeline-sensitive.

## GC statepoints and relocation semantics

Garbage-collected runtimes sometimes need the optimizer and backend to make a
safepoint explicit: the generated code must record which managed pointers are
live, call or poll the runtime, and then continue with pointer values that may
have moved. LLVM models this with the GC statepoint intrinsic family instead of
ordinary calls.

| Intrinsic / concept | Role |
|---|---|
| `llvm.experimental.gc.statepoint` | Produces a `token` for one safepoint-like call or poll. The token ties later GC intrinsics to that exact program point; it is not the callee's normal return value. |
| `llvm.experimental.gc.relocate` | Consumes the statepoint token plus indices into the live pointer set and returns the post-statepoint address for one base/derived pointer pair. |
| Live pointer set | The managed pointers that the GC must know about across the statepoint, usually carried in the statepoint's `"gc-live"` operand bundle. |

A moving collector may update object addresses while the statepoint executes.
After such a safepoint, the old SSA pointer is only the pre-statepoint address.
Use `llvm.experimental.gc.relocate` for each live managed pointer that remains
needed, and use the relocated result for loads, stores, comparisons, and calls
after the statepoint. Reusing the original pointer after relocation can mean
reading a stale address, hiding a live root from the GC lowering pipeline, or
letting optimizations reason about the wrong value across the safepoint.

A small relocation outline is provided in
[`examples/gc-statepoint-relocate.ll`](examples/gc-statepoint-relocate.ll). It
shows a statepoint with two live pointers: a base object and a derived field
pointer. The `gc.relocate` calls use indices into that live set, and the
post-statepoint load uses the relocated derived pointer rather than the original
GEP result.

BCIR agent guidance: treat statepoint tokens and relocated pointers as a single
contract. If you move, clone, or delete statepoint-adjacent code, keep the
`"gc-live"` set and every `gc.relocate` index synchronized, and audit downstream
uses so that post-statepoint managed-pointer uses refer to relocated SSA values.

## `metadata` as an intrinsic parameter type

`metadata` is the IR type used by some intrinsics to accept either a
metadata node or a value wrapped as metadata. Debug intrinsics are the
most common example:

```llvm
declare void @llvm.dbg.value(metadata, metadata, metadata)

call void @llvm.dbg.value(metadata i32 %x,
                          metadata !12,
                          metadata !DIExpression())
```

This does not create a normal SSA value of type `metadata`. It is a
special operand channel for compiler information. See the metadata
chapter for node syntax and debug-info details.

## `half` and `bfloat`

Both `half` and `bfloat` are 16-bit floating-point types, but they are
not interchangeable:

| Type | Meaning | Common use |
|---|---|---|
| `half` | IEEE 754 binary16 | GPUs, vector units, storage, ML kernels |
| `bfloat` | bfloat16 | ML workloads that prefer float32-like exponent range |

Pitfalls:

- The same 16 raw bits mean different numbers in each type.
- Some targets support storage but not native arithmetic.
- Legalization may promote arithmetic to `float` or use library calls.
- Overloaded intrinsics encode the type: `llvm.sqrt.f16` and a bfloat
  form, when available for an intrinsic, are different overloads.

## `x86_amx`

`x86_amx` is a target extension type used to represent Intel AMX tile
register values inside x86-specific intrinsics. Treat it as a backend
interface type, not a portable data structure.

Guidelines:

- Only use it in x86-specific code paths.
- Require the appropriate AMX target features.
- Prefer compiler-generated IR unless you are writing backend-facing
  tests or a carefully gated target module.
- Do not put AMX-specific IR in generic BCIR runtime modules without a
  portable fallback.

## Scalable vectors

A scalable vector has syntax:

```llvm
<vscale x 4 x i32>
```

The vector has a runtime-dependent number of lanes equal to `vscale * 4`.
This is used by architectures such as AArch64 SVE and RISC-V V where the
hardware vector length is not fixed at compile time.

Implications:

- The exact byte size is not a normal compile-time integer.
- Some operations that require fixed sizes are restricted.
- Intrinsic overload names may encode scalable vector element/count
  information.
- Code that assumes `<4 x i32>` and `<vscale x 4 x i32>` have the same
  ABI behavior is wrong.

## BCIR guidance

For portable BCIR material, keep special types at the boundary:

- Use `metadata` for debug and optimization annotations only.
- Avoid manually authoring token-heavy EH/coroutine/statepoint IR unless
  the chapter or test is specifically about that subsystem.
- Use `half`/`bfloat` only when the data format is part of the schema or
  target contract.
- Put `x86_amx` and target extension types in target-gated modules.
- Document scalable-vector assumptions when vector length affects memory
  layout or ABI.

## Pitfalls

- **Treating `token` as a normal first-class value.** Tokens have
  first-class-like syntax in some places but are deliberately opaque and
  restricted.
- **Passing normal values where `metadata` operands are required.** Use
  `metadata i32 %x` or metadata nodes as the intrinsic syntax requires.
- **Confusing `half` and `bfloat`.** Same storage width, different
  numerical format.
- **Using `x86_amx` without AMX features.** This is target-specific IR.
- **Assuming scalable vectors have fixed byte sizes.** The lane count is
  runtime-dependent.
