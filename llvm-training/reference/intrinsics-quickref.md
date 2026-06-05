# Intrinsics Quick Reference

Use this broad category page when you know the intrinsic family and need the spelling pattern, signature shape, or main caveat. For declaration rules, mangling guidance, and pitfalls, start with [`intrinsics.md`](intrinsics.md).

## Common categories

### Math

| Intrinsic | Lowers to |
|---|---|
| `llvm.sqrt.<T>` | `sqrt` |
| `llvm.sin.<T>`, `llvm.cos.<T>`, `llvm.tan.<T>` | libm calls |
| `llvm.exp.<T>`, `llvm.exp2.<T>`, `llvm.log.<T>`, `llvm.log2.<T>`, `llvm.log10.<T>` | libm |
| `llvm.pow.<T>(x, y)` | `pow` |
| `llvm.powi.<T>(x, i)` | Integer-power form |
| `llvm.fabs.<T>` | `fabs` |
| `llvm.copysign.<T>` | `copysign` |
| `llvm.floor.<T>`, `llvm.ceil.<T>`, `llvm.trunc.<T>`, `llvm.round.<T>`, `llvm.rint.<T>`, `llvm.nearbyint.<T>` | Rounding modes |
| `llvm.fma.<T>(a, b, c)` | Fused multiply-add |
| `llvm.fmuladd.<T>` | Multiply-add that may fuse |
| `llvm.minnum.<T>`, `llvm.maxnum.<T>` | IEEE 754 min/max |
| `llvm.minimum.<T>`, `llvm.maximum.<T>` | NaN-propagating min/max |

For strict FP exception and rounding semantics, use the constrained forms in
[Constrained floating-point intrinsics](#constrained-floating-point-intrinsics).

### Bit manipulation

| Intrinsic | Effect |
|---|---|
| `llvm.ctlz.<T>(v, is_zero_poison)` | Count leading zeros |
| `llvm.cttz.<T>(v, is_zero_poison)` | Count trailing zeros |
| `llvm.ctpop.<T>(v)` | Population count |
| `llvm.bswap.<T>(v)` | Byte swap |
| `llvm.bitreverse.<T>(v)` | Bit reverse |
| `llvm.fshl.<T>(a, b, n)` | Funnel shift left |
| `llvm.fshr.<T>(a, b, n)` | Funnel shift right |

The `is_zero_poison` argument to `ctlz`/`cttz` is `immarg`; see
[`../08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md).

### Overflow-checked arithmetic

Overflow intrinsics return a `{T, i1}` struct: `{result, overflow_flag}`. They
are preferable to guessing with wrap flags when the source language needs both
the wrapped result and an explicit overflow test. The `extractvalue` uses are
aggregate instructions; see [`instruction-quickref.md#aggregate-instructions`](instruction-quickref.md#aggregate-instructions).

| Intrinsic | Signature pattern | Meaning |
|---|---|---|
| `llvm.uadd.with.overflow.<T>` | `{T, i1} (T, T)` | Unsigned addition plus carry/overflow flag. |
| `llvm.sadd.with.overflow.<T>` | `{T, i1} (T, T)` | Signed addition plus overflow flag. |
| `llvm.usub.with.overflow.<T>` | `{T, i1} (T, T)` | Unsigned subtraction plus borrow/overflow flag. |
| `llvm.ssub.with.overflow.<T>` | `{T, i1} (T, T)` | Signed subtraction plus overflow flag. |
| `llvm.umul.with.overflow.<T>` | `{T, i1} (T, T)` | Unsigned multiplication plus overflow flag. |
| `llvm.smul.with.overflow.<T>` | `{T, i1} (T, T)` | Signed multiplication plus overflow flag. |

```llvm
%r = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %a, i32 %b)
%sum = extractvalue { i32, i1 } %r, 0
%ovf = extractvalue { i32, i1 } %r, 1
```

### Memory intrinsics

Memory intrinsics are not ordinary library calls: the optimizer recognizes them,
may inline them, may combine/split them, and may lower them to target moves or
runtime calls. Address-space suffixes (`p0`, `p1`, ...) and integer length
suffixes (`i32`, `i64`) are part of overloaded names. See
[`../13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md)
for examples and [`../08-pitfalls/11-address-space-confusion.md`](../08-pitfalls/11-address-space-confusion.md)
for pointer address-space mistakes.

| Intrinsic | Signature sketch | Effect | Key constraints |
|---|---|---|---|
| `llvm.memcpy.p<dst-as>.p<src-as>.<len-ty>` | `void (ptr dst, ptr src, <len-ty> len, i1 isvolatile)` | Copy `len` bytes from `src` to `dst`. | Source and destination must not overlap unless language semantics make overlap irrelevant; use `memmove` for overlap. Alignment is normally expressed as parameter attributes. |
| `llvm.memmove.p<dst-as>.p<src-as>.<len-ty>` | `void (ptr dst, ptr src, <len-ty> len, i1 isvolatile)` | Copy `len` bytes safely when ranges may overlap. | Usually less optimizable than `memcpy` because overlap must be preserved. |
| `llvm.memset.p<dst-as>.<len-ty>` | `void (ptr dst, i8 value, <len-ty> len, i1 isvolatile)` | Fill `len` bytes at `dst` with one byte value. | Fill value is an `i8`; larger typed fills are represented as bytes. |
| `llvm.memcpy.inline.p<dst-as>.p<src-as>.<len-ty>` | `void (ptr dst, ptr src, <len-ty> imm-len, i1 isvolatile)` | Inline-only fixed-size copy. | Length is an `immarg`, so it must be a compile-time constant. |
| `llvm.memcpy.element.unordered.atomic.*` / related forms | Targeted memory-copy families with element atomicity | Copies elements without imposing synchronization order. | Use only when the element-atomic semantics are required; for inter-thread synchronization, use atomic instructions in [`../11-concurrency/`](../11-concurrency/). |
| `llvm.masked.load.<vec>.p<as>` | `<vec> (ptr p, i32 imm-align, <N x i1> mask, <vec> passthru)` | Loads only active lanes and uses `passthru` for inactive lanes. | Masked-off lanes must not perform memory reads; profitability depends heavily on target masked-load support. |
| `llvm.masked.store.<vec>.p<as>` | `void (<vec> value, ptr p, i32 imm-align, <N x i1> mask)` | Stores only active lanes. | Masked-off lanes must not write memory; see advanced vectorization examples for predicated store IR. |

**Volatile caveat:** the `isvolatile` flag makes the memory operation observable,
but it does not make it atomic or synchronizing. See
[`../08-pitfalls/10-volatile-is-not-atomic.md`](../08-pitfalls/10-volatile-is-not-atomic.md).

### Lifetime intrinsics

Lifetime markers communicate object lifetime to optimizers. They are hints about
when a memory region is dead or live; they do not allocate memory and they do not
run constructors or destructors.

| Intrinsic | Signature sketch | Effect | Notes |
|---|---|---|---|
| `llvm.lifetime.start.p<as>` | `void (i64 imm-size, ptr addrspace(as) p)` | The bytes at `p` become live. | Size is an `immarg`; use `-1` only when the size is unknown by convention. |
| `llvm.lifetime.end.p<as>` | `void (i64 imm-size, ptr addrspace(as) p)` | The bytes at `p` are no longer live. | After this marker, optimizers may treat previous stored values as dead until a new start. |
| `llvm.invariant.start.p<as>` | `ptr (i64 imm-size, ptr addrspace(as) p)` | Starts an invariant region and returns a descriptor token-like pointer. | More specialized than lifetime markers; use only when memory truly remains immutable. |
| `llvm.invariant.end.p<as>` | `void (ptr descriptor, i64 imm-size, ptr addrspace(as) p)` | Ends a matching invariant region. | Must correspond to the descriptor returned by `invariant.start`. |

Because size operands are `immarg`, dynamic sizes trigger the pitfall described
in [`../08-pitfalls/06-immarg-violation.md`](../08-pitfalls/06-immarg-violation.md).

### Atomic primitives and optimizer hints

Almost always you'll use the `atomicrmw`, `cmpxchg`, and `fence`
instructions directly rather than intrinsics. The quick-reference table for
atomic instruction forms is in
[`instruction-quickref.md#atomic-instruction-forms-and-ordering-constraints`](instruction-quickref.md#atomic-instruction-forms-and-ordering-constraints),
and the concurrency chapter is in [`../11-concurrency/`](../11-concurrency/).

| Intrinsic | Effect |
|---|---|
| `llvm.assume(i1)` | Assume the condition is true; violating the assumption permits optimization as UB. |
| `llvm.expect.<T>(val, expected)` | Branch-probability hint. |
| `llvm.expect.with.probability.<T>(val, expected, probability)` | Branch-probability hint with explicit probability metadata-like information. |
| `llvm.trap()` | Unconditional trap. |
| `llvm.debugtrap()` | Like `trap` but reserved for debugger breakpoints. |

### Vector reduction intrinsics

Vector reductions fold lanes into one scalar. They are common in Loop Vectorizer
outputs; see [`../09-vectorization/README.md`](../09-vectorization/README.md)
and the vector instruction patterns in
[`instruction-quickref.md#vector-instructions`](instruction-quickref.md#vector-instructions).

| Intrinsic | Result | Meaning / caveat |
|---|---|---|
| `llvm.vector.reduce.add.<T>(vec)` | Element type of `vec` | Integer lane sum. |
| `llvm.vector.reduce.mul.<T>(vec)` | Element type | Integer lane product. |
| `llvm.vector.reduce.and.<T>(vec)` | Element type | Bitwise AND across lanes. |
| `llvm.vector.reduce.or.<T>(vec)` | Element type | Bitwise OR across lanes. |
| `llvm.vector.reduce.xor.<T>(vec)` | Element type | Bitwise XOR across lanes. |
| `llvm.vector.reduce.smin.<T>(vec)` / `llvm.vector.reduce.smax.<T>(vec)` | Element type | Signed integer min/max. |
| `llvm.vector.reduce.umin.<T>(vec)` / `llvm.vector.reduce.umax.<T>(vec)` | Element type | Unsigned integer min/max. |
| `llvm.vector.reduce.fadd.<T>(start, vec)` | FP element type | Floating-point sum. Reassociation legality depends on FP flags/semantics. |
| `llvm.vector.reduce.fmul.<T>(start, vec)` | FP element type | Floating-point product. |
| `llvm.vector.reduce.fmin.<T>(vec)` / `llvm.vector.reduce.fmax.<T>(vec)` | FP element type | FP min/max using LLVM's non-NaN-propagating min/max semantics. |
| `llvm.vector.reduce.fminimum.<T>(vec)` / `llvm.vector.reduce.fmaximum.<T>(vec)` | FP element type | NaN-propagating minimum/maximum variants. |

If a loop fails to vectorize because memory dependencies are unclear, see
[`../08-pitfalls/12-vectorization-blocked-by-aliasing.md`](../08-pitfalls/12-vectorization-blocked-by-aliasing.md).

### Prefetch and cache-related intrinsics

These intrinsics are hints or low-level hooks. They do not create correctness
requirements for ordinary memory, and they do not replace atomics. Target support
and lowering quality vary, especially for cache-line operations.

| Intrinsic | Signature sketch | Purpose | Important operands |
|---|---|---|---|
| `llvm.prefetch` | `void (ptr p, i32 rw, i32 locality, i32 cache)` | Hint that memory at `p` will be used soon. | `rw`: `0` read, `1` write. `locality`: `0` no temporal locality through `3` high temporal locality. `cache`: `0` data cache, `1` instruction cache. Integer operands are `immarg`. |
| `llvm.clear_cache` | `void (ptr begin, ptr end)` | Flush/clear instruction cache for a byte range when the target needs it. | Useful for JITs or generated code before execution. See [`../12-backend-jit/03-orc-jit.md`](../12-backend-jit/03-orc-jit.md). |
| `llvm.readcyclecounter` | integer result | Reads a target cycle counter when available. | Timing-oriented; not a memory-ordering primitive. |
| `llvm.x86.sse2.clflush` and newer x86 cache-line forms | Target-specific declarations under `llvm.x86.*` | Emit x86 cache-line flush/writeback operations when the subtarget supports them. | Guard with target features and keep fallbacks; see [Target-specific intrinsic naming patterns](#target-specific-intrinsic-naming-patterns). |

### Constrained floating-point intrinsics

The `llvm.experimental.constrained.*` family models floating-point operations
when rounding mode and exception behavior must be explicit, as with strict FP
language modes. These calls usually take metadata operands such as rounding mode
and exception behavior. They are more restrictive than ordinary `fadd`/`fmul`
and the unconstrained math intrinsics.

| Intrinsic family | Rough operation | Ordinary counterpart |
|---|---|---|
| `llvm.experimental.constrained.fadd.<T>` / `fsub` / `fmul` / `fdiv` / `frem` | Strict binary FP arithmetic | `fadd`, `fsub`, `fmul`, `fdiv`, `frem` |
| `llvm.experimental.constrained.fma.<T>` | Strict fused multiply-add | `llvm.fma.<T>` |
| `llvm.experimental.constrained.sqrt.<T>` | Strict square root | `llvm.sqrt.<T>` |
| `llvm.experimental.constrained.pow.<T>`, `powi`, `sin`, `cos`, `exp`, `log`, ... | Strict libm-like operations | Unconstrained math intrinsics/libcalls |
| `llvm.experimental.constrained.fptrunc`, `fpext` | Strict FP width conversions | `fptrunc`, `fpext` |
| `llvm.experimental.constrained.fptosi`, `fptoui`, `sitofp`, `uitofp` | Strict FP/integer conversions | Conversion instructions |
| `llvm.experimental.constrained.fcmp`, `fcmps` | Strict FP comparisons | `fcmp` |

Use constrained forms consistently within strict regions; mixing strict and
unconstrained operations can accidentally grant optimizers freedom that the
source language did not grant.


### Stackmap and patchpoint intrinsics

`llvm.experimental.stackmap` and `llvm.experimental.patchpoint.*` are
JIT/deoptimization/runtime-patching hooks. They assemble as intrinsic calls, but
their useful meaning comes from target-emitted stackmap records and a runtime
that can consume those records.

| Intrinsic | Signature sketch | Purpose | Important operands |
|---|---|---|---|
| `llvm.experimental.stackmap` | `void (i64 id, i32 shadow_bytes, ...)` | Record a generated-code location and the machine locations of live values. | `id` is runtime-owned. `shadow_bytes` reserves a patching/shadow region. Variadic operands are live values for the side table. |
| `llvm.experimental.patchpoint.<ret>` | `<ret> (i64 id, i32 num_bytes, ptr target, i32 num_args, ...)` | Emit a patchable call-shaped or placeholder site and a stackmap record. | `target`/`num_args` describe the optional call target and call arguments; trailing operands are stackmap-only live values. |

See [`../12-backend-jit/06-custom-bcir-intrinsics.md`](../12-backend-jit/06-custom-bcir-intrinsics.md)
and [`../12-backend-jit/examples/stackmap-patchpoint.ll`](../12-backend-jit/examples/stackmap-patchpoint.ll)
for a minimal JIT-oriented example and verification caveats.

### GC statepoint intrinsics

`llvm.experimental.gc.statepoint` marks a runtime safepoint and produces the
`token` consumed by related GC intrinsics. `llvm.experimental.gc.relocate` uses
that token and indices into the statepoint live pointer set, commonly the
`"gc-live"` operand bundle, to produce post-safepoint managed pointers. Do not
reuse pre-statepoint managed pointer SSA values after relocation; use the
relocated values. See
[`../13-advanced-ir/03-special-types-and-tokens.md#gc-statepoints-and-relocation-semantics`](../13-advanced-ir/03-special-types-and-tokens.md#gc-statepoints-and-relocation-semantics)
and
[`../13-advanced-ir/examples/gc-statepoint-relocate.ll`](../13-advanced-ir/examples/gc-statepoint-relocate.ll).

### Target-specific intrinsic naming patterns

Target-specific intrinsics live under target namespaces and are declared only
when the backend knows their names. They are not portable IR contracts. For more
context, see [`../13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md).

| Namespace / pattern | Examples | Meaning / caveat |
|---|---|---|
| `llvm.x86.<isa>.<op>` | `llvm.x86.sse2.*`, `llvm.x86.avx.*`, `llvm.x86.avx2.*`, `llvm.x86.avx512.*` | x86 SIMD, scalar, cache, crypto, and miscellaneous operations. The middle component often names the ISA extension or instruction family. |
| `llvm.x86.*.<width or suffix>` | names containing `128`, `256`, `512`, `ps`, `pd`, `epi32`, `si`, etc. | Name fragments often encode vector width and element interpretation, but the declaration in `IntrinsicsX86.td` is authoritative. |
| `llvm.aarch64.*` | NEON, SVE, SME, system instructions | Requires AArch64 backend support and usually feature checks. |
| `llvm.arm.*` | ARM/Thumb/NEON/M-profile helpers | Similar feature and ABI caveats as AArch64. |
| `llvm.amdgcn.*` | AMDGPU operations, address spaces, waves, LDS/global memory | Often tied to GPU address spaces and subtarget features. |
| `llvm.nvvm.*` | NVPTX/NVIDIA GPU operations | Intended for NVPTX lowering; not portable to host targets. |
| `llvm.riscv.*` | RISC-V scalar/vector/crypto helpers | Frequently depends on enabled extensions. |
| `llvm.ppc.*`, `llvm.s390.*`, `llvm.wasm.*` | Backend-specific helpers | Use only behind target-specific lowering paths. |

**x86 guidance:** prefer target-independent IR first (`shufflevector`, vector
arithmetic, `llvm.fma`, reductions, memory intrinsics). Reach for `llvm.x86.*`
only when you need a specific instruction semantic that generic IR cannot state
or when implementing a target-specific builtin. Calling an x86 intrinsic on a
non-x86 target, or without the required feature, fails later in the pipeline; see
[`../13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md).

### Custom backend intrinsics

Custom intrinsic names such as `llvm.bcir.*` are backend contracts, not portable
LLVM guarantees. They should be declared exactly as the backend registered them
and paired with a runtime fallback when a generic JIT or non-BCIR target may see
the module. For BCIR hardware-aware lowering, the typical shape is a
register-oriented vector/tile payload plus an immediate mode operand:

| Pattern | Signature sketch | Purpose | Caveat |
|---|---|---|---|
| `llvm.bcir.gem.mixed.stride.v4f32` | `<4 x float> (<4 x float> a, <4 x float> b, <4 x float> acc, i32 immarg mode)` | Preserve a mixed-stride GEM tile as one selection-visible operation. | Requires BCIR-aware backend support or an ORC rewrite to a runtime function such as `@bcir.runtime.gem.v4f32`. |

See [`../12-backend-jit/06-custom-bcir-intrinsics.md`](../12-backend-jit/06-custom-bcir-intrinsics.md)
for declaration, metadata, and JIT policy guidance.

### Coroutine intrinsics

`llvm.coro.id`, `llvm.coro.begin`, `llvm.coro.suspend`,
`llvm.coro.resume`, etc. — used by `clang -fcoroutines-ts` and
similar frontends. Rarely written by hand.

### Exception handling

`llvm.eh.typeid.for`, `llvm.eh.exceptionpointer`,
`llvm.eh.exceptioncode`, `llvm.eh.sjlj.setjmp`,
`llvm.eh.sjlj.longjmp`, etc. EH pads and tokens are summarized in
[`instruction-quickref.md#other-and-special-constructs`](instruction-quickref.md#other-and-special-constructs)
and [`../13-advanced-ir/03-special-types-and-tokens.md`](../13-advanced-ir/03-special-types-and-tokens.md).

### Debug info

| Intrinsic | Effect |
|---|---|
| `llvm.dbg.declare(metadata, !var, !expr)` | Variable lives at the pointed address |
| `llvm.dbg.value(metadata, !var, !expr)` | Variable has this value here |
| `llvm.dbg.addr(metadata, !var, !expr)` | Variable address (DI v2) |

These are erased before codegen; they only affect debug output. Avoid metadata
bloat and stale locations; see [`../08-pitfalls/07-debug-metadata-bloat.md`](../08-pitfalls/07-debug-metadata-bloat.md)
and [`../08-pitfalls/08-stale-debug-locations.md`](../08-pitfalls/08-stale-debug-locations.md).

## See also

- [`intrinsics.md`](intrinsics.md) — broad reference, declaration rules, and pitfalls.
- [`instruction-quickref.md`](instruction-quickref.md) — instruction syntax that pairs with intrinsic results.
- [`../13-advanced-ir/01-common-intrinsics.md`](../13-advanced-ir/01-common-intrinsics.md) — runnable common intrinsic examples.
- [`../quickref/advanced-ir.md`](../quickref/advanced-ir.md) — short pre-edit checklist for intrinsic-heavy or contract-heavy IR.
- [`../13-advanced-ir/02-target-specific-intrinsics.md`](../13-advanced-ir/02-target-specific-intrinsics.md) — target-specific intrinsic namespaces and feature requirements.
- [`../12-backend-jit/06-custom-bcir-intrinsics.md`](../12-backend-jit/06-custom-bcir-intrinsics.md) — custom BCIR intrinsic declarations and JIT fallback policy.
